# encoding:utf-8

# import asyncio

from .core import callable_register, NLPShortcutCore, deepdive, producer, io, cpu
from .corpus import Corpus
from .document import make_document
from .util.thread import ThreadWrapper
from .util.file import get_files, aio_read_file
from .util.tool import CommandProcessBar, uniqueid
from .util.process import ProcessPoolWrapper


__all__ = ['NLPShortcut']


def load_data_from_file(fin, lang='zh', fn=None):
    """
    从文件中加载数据，放回document对象

    :argument
        fin:
            输入的数据，可以是文件路径或者文件所在的目录
        lang:
            语言类型 (zh, en)
        filter:
            如果fin是目录，会根据filter来过滤加载文件，默认加载全部文件
    """
    return NLPShortcut().load_data_from_file(fin, lang=lang, fn=fn)


# async def _load_data_from_file(count_files, files, corpus,  lang, fn):
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

    def __init__(self):
        super(NLPShortcut, self).__init__()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lazy_call()

    def lazy_call(self):
        """真实执行具体操作的方法"""

        try:
            print(__package__)
            for i, bridge in enumerate(deepdive(self)):
                print("{} bridge".format(i))
                worker = bridge.call()

            print(worker)
            print('main process stop')
        except KeyboardInterrupt:
            print('exit nlpsc')
        finally:
            ProcessPoolWrapper.stop()

    def a(self):
        import time
        time.sleep(1)
        print('load data produce: a')
        return 'a'

    @io
    @producer(topic="test")
    def __load_data_from_file(self) -> Corpus:
        # corpus = Corpus()
        # files = get_files(fin, fn)
        # count_total_files = len(files)
        # loop = asyncio.get_event_loop()
        # try:
        #     corpus = loop.run_until_complete(_load_data_from_file(count_total_files, files, corpus, lang, fn))
        # finally:
        #     loop.close()
        # return corpus
        print('load')
        for i in range(10):
            producer.test.produce(self.a)
        print('load data finish')








