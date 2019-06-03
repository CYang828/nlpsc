# encoding:utf-8

import csv
import time

from .dataset import Dataset, DatasetHeader
from .util.file import get_files
from .document import file2document, Document
from .util.aio import AIOPoolWrapper
from .util.process import ProcessPoolWrapper
from .core import NLPShortcutCore, producer, aio


__all__ = ['NLPShortcut']


# async def _load_data_from_file(files, corpus,  lang, fn):
#     with CommandProcessBar(total=count_files, desc='read_files process') as pbar:
#         pbar.update(1)
#         for name, path in files:
#             try:
#                 text = await aio_read_file(path, 'r', encoding='utf-8-sig')
#             except UnicodeDecodeError as e:
#                 print('{}'.format(e))
#                 continue
#             document = make_document(text, lang=lang, name=name)
#             corpus.add(document)
#     return corpus
#
#
# def load_data_from_memory(text):
#     """
#     从字面量加载数据，反回document对象
#
#     :argument
#         text: 字面量
#     """
#
#     document = make_document(text)
#     return document
#
#
# def load_data_from_net():
#     """
#     从网络中加载数据，反回document对象
#
#     :argument
#         text: 字面量
#     """
#     pass


class NLPShortcut(NLPShortcutCore):

    def __init__(self, name=None):
        super(NLPShortcut, self).__init__()
        self.name = name
        self._dataset = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lazy_call()

    def lazy_call(self):
        """真实执行操作的方法"""

        pools = []

        # 遍历整个调用树，运行真实的处理函数
        try:
            for bridge in self.deepdive(self):
                bridge.nlpsc = self
                r = bridge.call()
                if isinstance(r, (ProcessPoolWrapper, AIOPoolWrapper)):
                    pools.append(r)
                self._dataset = bridge.return_obj
                self._dataset.name = self.name

            # 等待cpu和io的pool中的任务完成
            for pool in pools:
                while not pool.tasks_finished():
                    time.sleep(0.5)
        except KeyboardInterrupt:
            print('Ctrl + C -- exit nlpsc!')
            exit(0)
        finally:
            ProcessPoolWrapper.stop()
            AIOPoolWrapper.stop()

    @aio
    @producer(topic="load_dataset_from_file")
    def lazy_load_dataset_from_file(self, fin, lang='zh', fn=None, header=None) -> Dataset:
        """从文件或文件夹中加载语料库

        :argument
            fin:
                输入的数据，可以是文件路径或者文件所在的目录
            lang:
                语言类型 (zh, en)
            fn:
                如果fin是目录，会根据filter来过滤加载文件，默认加载全部文件
            header:
                `nlpsc.dataset.DatasetHeader` 对象"""

        self._dataset.add_header(header)

        files = get_files(fin, fn)
        for name, path in files:
            self.produce(file2document,
                         path,
                         'r',
                         encoding='utf-8-sig',
                         lang=lang)
        print('load files finished, get a dataset!')

    @aio
    @producer(topic="load_dataset_from_tsv")
    def lazy_load_dataset_from_tsv(self, fin, lang='zh', quotechar=None):
        """Reads a tab separated value file."""
        with open(fin, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            headers = next(reader)
            # TODO：这里返回dataset
            # Example = namedtuple('Example', headers)
            #
            # examples = []
            # for line in reader:
            #     example = Example(*line)
            #     examples.append(example)
            # return examples

    @aio
    @producer(topic='load_dataset_from_dump')
    def lazy_load_dataset_from_dump(self, fin, lang='zh', fn=None) -> Dataset:
        """从dump的文件中加载语料库

        :argument
            fin:
                输入的数据，可以是文件路径或者文件所在的目录
            lang:
                语言类型 (zh, en)
            fn:
                如果fin是目录，会根据filter来过滤加载文件，默认加载全部文件"""

        print('load dump file finished, get a dataset!')

    def get_dataset(self):
        """获取语料库"""
        return self._dataset

    def serving(self):
        """提供对外的http服务，在线进行语料库的标注等工作"""
        pass











