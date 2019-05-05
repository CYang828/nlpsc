# encoding:utf-8

import os
import random

from .error import NLPSCError
from .document import Document
from .util.tool import uniqueid
from .util.file import gen_filename
from .util.python import get_runtime_function_name
from .core import NLPShortcutCore, producer, aio, cpu, wait


class Corpus(object):
    """用来做annotations"""
    pass


class Corpus(NLPShortcutCore):
    """语料集对象"""

    def __init__(self, name=None):
        super(Corpus, self).__init__()

        self.id = uniqueid()
        # 文档名称
        self.name = name if name else self.id
        # 文档集中文档个数
        self.size = 0
        self._documents = {}

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

    def sample(self, n=1) -> Corpus:
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

    @cpu
    @producer('document_paragraph')
    def __iter_paragraph(self) -> Corpus:
        self.iter_process(self.documents, get_runtime_function_name())
        return self

    @cpu
    @producer('document_sentence')
    def __iter_sentence(self) -> Corpus:
        self.iter_process(self.documents, get_runtime_function_name())
        return self

    @cpu
    @producer('document_word')
    def __iter_word(self) -> Corpus:
        self.iter_process(self.documents, get_runtime_function_name())
        return self

    @cpu
    @producer(topic='document_clean')
    def __iter_clean(self) -> Corpus:
        for document in self.consume():
            document.clean()
            self.add(document)
            self.produce(document)
        print('finish clean')
        return self

    @cpu
    @producer('document_preprocess')
    def __iter_preprocess(self, fn) -> Corpus:

        self.iter_process(self.documents, get_runtime_function_name(), fn)
        return self

    @cpu
    @producer(topic='document_tokenize')
    def __iter_tokenize(self, tokenizer=None, userdict=None) -> Corpus:
        print('start token')
        for document in self.consume():
            print(userdict)
            document.tokenize(tokenizer=tokenizer,
                              userdict=userdict)
            self.produce(document)
            self.add(document)
            print(document)
        print('end token')
        return self

    @cpu
    @producer(topic='document_stopword')
    def __iter_stopword(self, stopwordict=None) -> Corpus:
        print('document_stopword')
        for document in self.consume():
            document.stopword(stopwordict)
            self.add(document)
            self.produce(document)
        print('stopword finished')
        return self

    @cpu
    @producer('document_literal')
    def __iter_literal(self, word_delimiter=' ') -> Corpus:
        self.iter_process(self.documents, get_runtime_function_name(),
                          word_delimiter=word_delimiter)
        return self

    @aio
    @producer('document_dump')
    def __iter_dump(self, outdir, is_structured=False, prefix="dump", suffix="nlpsc",
                    paragraph_delimiter='</p>', sentence_delimiter='</s>', word_delimiter=' ') -> Corpus:
        # 如果文件已经存在，删除该文件
        dump_filename = gen_filename(self.name, prefix=prefix, suffix=suffix)
        path = os.path.join(outdir, dump_filename)
        if os.path.exists(path):
            os.remove(path)
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
