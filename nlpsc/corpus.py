# encoding:utf-8

import sys
import random

from .error import NLPSError
from .document import Document
from .util.python import get_runtime_function_name
from .core import NLPShortcutCore, consumer, producer, io, cpu


class Corpus(object):
    """用来做annotations"""
    pass


class Corpus(NLPShortcutCore):
    """语料集对象"""

    def __init__(self):
        super(Corpus, self).__init__()
        # 文档集中文档个数
        self._size = 0
        self._documents = []

    def __iter__(self):
        return self.iter()

    def __call__(self, *args, **kwargs):
        return self

    def add(self, document):
        """向语料集中增加文档对象"""
        if not isinstance(document, Document):
            print('please add nlpsc.document.Document to Corpus')
            raise NLPSError
        self._documents.append(document)
        self._size += 1
        return self

    def iter(self):
        """遍历语料集中的文档"""
        for document in self._documents:
            yield document

    def sample(self, n=1):
        """随机抽样展示数据"""
        if len(self._documents) == 0:
            print('corpus is empty, please add some document')
        else:
            n = n if n <= self._size else self._size
            sample_idx = random.sample(range(self._size), n)
            for idx in sample_idx:
                print(self._documents[idx])
        return self

    def __iter_paragraph(self) -> Corpus:
        self.iter_process(self._documents, get_runtime_function_name())
        return self

    def __iter_sentence(self) -> Corpus:
        self.iter_process(self._documents, get_runtime_function_name())
        return self

    def __iter_word(self) -> Corpus:
        self.iter_process(self._documents, get_runtime_function_name())
        return self

    @cpu
    @consumer(topic='test')
    @producer(topic='clean')
    def __iter_clean(self) -> Corpus:
        for a, i in enumerate(consumer.test.consume()):
            print('clean consume: {}'.format(i))
            producer.clean.produce('b')
            print('clean produce [{}]: {}'.format(a+1, 'b'))

        self.iter_process(self._documents, get_runtime_function_name())
        return self

    def __iter_preprocess(self, fn) -> Corpus:

        self.iter_process(self._documents, get_runtime_function_name(), fn)
        return self

    def __iter_tokenize(self, tokenizer=None, userdict=None) -> Corpus:
        self.iter_process(self._documents, get_runtime_function_name(),
                          tokenizer=tokenizer, userdict=userdict)
        return self

    @cpu
    @consumer(topic='clean')
    def __iter_stopword(self, stopwordict=None) -> Corpus:
        print('stopword')
        print(consumer.clean.consume())
        for i in consumer.clean.consume():
            print('stopword consume: {}'.format(i))
        self.iter_process(self._documents, get_runtime_function_name(),
                          stopwordict=stopwordict)
        return self

    def __iter_literal(self, word_delimiter=' ') -> Corpus:
        self.iter_process(self._documents, get_runtime_function_name(),
                          word_delimiter=word_delimiter)
        return self

    def __iter_dump(self, output_dir, is_structured=False, prefix="dump", suffix="nlpsc",
                    paragraph_delimiter='</p>', sentence_delimiter='</s>', word_delimiter=' ') -> Corpus:
        self.iter_process(self._documents, get_runtime_function_name(),
                          output_dir, is_structured=is_structured, prefix=prefix, suffix=suffix,
                          paragraph_delimiter=paragraph_delimiter,
                          sentence_delimiter=sentence_delimiter, word_delimiter=word_delimiter)
        return self
