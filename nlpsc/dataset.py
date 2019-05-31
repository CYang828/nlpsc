# encoding:utf-8

import os
import random

from .error import NLPSCError
from .document import Document
from .util.tool import uniqueid
from .util.file import gen_filename
from .core import NLPShortcutCore, producer, aio, cpu
from .util.python import get_runtime_function_name
from .util.file import aio_write_file


class Dataset(object):
    """用来做annotations"""
    pass


class Dataset(NLPShortcutCore):
    """数据集对象"""

    def __init__(self, name=None):
        super(Dataset, self).__init__()

        self.id = uniqueid()
        # 数据集名称
        self.name = name if name else self.id
        # 数据集中文档个数
        self.size = 0
        self._documents = {}
        self.header = None

    def __iter__(self):
        return self.iter()

    def __call__(self, *args, **kwargs):
        return self

    def add(self, document):
        """向语料集中增加文档对象"""

        if not isinstance(document, Document):
            print('please add nlpsc.document.Document to Corpus')
            raise NLPSCError

        # TODO:这个算法可以使用bloom filter优化
        self._documents[document.id] = document
        self.size += 1
        return self

    def iter(self):
        """遍历语料集中的文档"""
        for document in self.documents:
            yield document

    def sample(self, n=1) -> Dataset:
        """随机抽样展示数据"""

        if len(self.documents) == 0:
            print('corpus is empty, please add some document')
        else:
            n = n if n <= self.size else self.size
            sample_idx = random.sample(range(self.size), n)
            for idx in sample_idx:
                i = list(self._documents.keys())[idx]
                print(self._documents[i])
        return self

    def set_header(self, header):
        """设置dump成文件时的header"""
        self.header = '\t'.join(header)

    async def write_header(self, output_dir, filename):
        if self.header:
            header = '\t'.join(self.header)
            await aio_write_file(output_dir, filename, header, pattern='a+')

    @cpu
    @producer('document_paragraph')
    def lazy_iter_paragraph(self) -> Dataset:
        self.iter_process(self.documents, get_runtime_function_name())
        return self

    @cpu
    @producer('document_sentence')
    def lazy_iter_sentence(self) -> Dataset:
        self.iter_process(self.documents, get_runtime_function_name())
        return self

    @cpu
    @producer('document_word')
    def lazy_iter_word(self) -> Dataset:
        self.iter_process(self.documents, get_runtime_function_name())
        return self

    @cpu
    @producer(topic='document_clean')
    def lazy_iter_clean(self) -> Dataset:
        for document in self.consume():
            self.produce(document.clean)
            self.add(document)
        print('finish clean')
        return self

    @cpu
    @producer('document_preprocess')
    def lazy_iter_preprocess(self, fn) -> Dataset:

        self.iter_process(self.documents, get_runtime_function_name(), fn)
        return self

    @cpu
    @producer(topic='document_tokenize')
    def lazy_iter_tokenize(self, tokenizer=None, userdict=None) -> Dataset:
        print('start token')
        for document in self.consume():
            self.produce(document.tokenize, tokenizer_opt=tokenizer, userdict=userdict)
            self.add(document)
        print('end token')
        return self

    @cpu
    @producer(topic='document_stopword')
    def lazy_iter_stopword(self, stopwordict=None) -> Dataset:
        print('document_stopword')
        for document in self.consume():
            self.produce(document.stopword, stopwordict=stopwordict)
            self.add(document)
        print('stopword finished')
        return self

    @cpu
    @producer(topic='document_represent')
    def lazy_iter_represent(self) -> Dataset:
        for document in self.consume():
            self.produce(document.represent)
            self.add(document)
        return self

    @cpu
    @producer('document_literal')
    def lazy_iter_literal(self, word_delimiter=' ') -> Dataset:
        self.iter_process(self.documents, get_runtime_function_name(),
                          word_delimiter=word_delimiter)
        return self

    @aio
    @producer('document_dump')
    def lazy_iter_dump(self, outdir, header=None, is_structured=False, prefix="dump", suffix="nlpsc",
                       paragraph_delimiter='</p>', sentence_delimiter='</s>', word_delimiter=' ') -> Dataset:

        if header:
            self.set_header(header)

        # 如果文件已经存在，删除该文件
        dump_filename = gen_filename(self.name, prefix=prefix, suffix=suffix)
        path = os.path.join(outdir, dump_filename)
        if os.path.exists(path):
            os.remove(path)
        self.produce(self.write_header, outdir, dump_filename)
        for document in self.consume():
            self.produce(document.dump,
                         outdir, dump_filename,
                         is_structured=is_structured, prefix=prefix, suffix=suffix,
                         paragraph_delimiter=paragraph_delimiter, sentence_delimiter=sentence_delimiter,
                         word_delimiter=word_delimiter)
        return self

    @property
    def documents(self):
        return list(self._documents.values())

