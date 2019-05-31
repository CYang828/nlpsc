# encoding:utf-8

import abc
import json

import numpy as np

from .vocabulary import Vocabulary


class Transformer(metaclass=abc.ABCMeta):
    """数据转换器基类

    :argument
        dataset: 要使用的数据集
        vocab_path: 词典文件，如果不指定，在需要使用词典时，会根据数据集自动生成
        do_lower_case: 是否转换成小写
        random_seed: shuffle时使用的随机种子，
                     如果指定值，则shuffle时的顺序保持一个固定的乱序
                     如果不指定值，则每次shuffle的数据顺序都不相同
        in_tokens：是否使用token数来进行batch_size的计算"""

    def __init__(self, dataset=None, vocab_path=None, do_lower_case=True,
                 random_seed=None, in_tokens=False, label_map_config=None):
        self.dataset = dataset
        self.vocab = None

        if dataset:
            self.size = dataset.size

        if vocab_path:
            self._vocab_path = vocab_path
            self.create_vocab()

        self._do_lower_case = do_lower_case
        self.in_tokens = in_tokens

        # 加载用户自定义标签
        if label_map_config:
            with open(label_map_config) as f:
                self.label_map = json.load(f)
        else:
            self.label_map = None

        # 初始化shuffle时使用的随机种子数
        np.random.seed(random_seed)

    @abc.abstractmethod
    def document2input(self, document):
        """用来把单个document转换成输入对象"""

    def _document2input(self, document):
        example = self.document2example(document)
        model_input = self.example2input(example)
        return model_input

    @abc.abstractmethod
    def document2example(self, document):
        """把document转换成该当前模型需要的样本对象"""
        pass

    @abc.abstractmethod
    def example2input(self, example):
        """把数据的样本对象转换成模型的输入"""
        pass

    @abc.abstractmethod
    def batch_inputs_generator(self, batch_size=1, epoch=1, shuffle=False):
        """数据生成器方法（用户自定义）
           需要在子类中实现，此方法必须返回一个生成器"""

    def _batch_inputs_generator(self, batch_size=1, epoch=1, shuffle=False):
        documents = self._epoch(epoch)
        if shuffle:
            self._shuffle()
        return self._batch(documents, batch_size)

    def create_vocab(self):
        """创建vocab的词典

        如果指定了vocab的文件，从文件中读取生成
        如果没有指定，则根据当前数据集中的数据生成词典"""

        if not self.vocab:
            self.vocab = Vocabulary()
            if self._vocab_path:
                self.vocab.load_vocab(self._vocab_path)
            else:
                self.vocab.auto_from_dataset(self.dataset)

    def _batch(self, documents, batch_size):
        if self.in_tokens:
            batch_documents, max_len = [], 0
            for document in documents:
                # 找到本次batch中的最长序列长度
                max_len = max(max_len, len(document.text))
                if (len(batch_documents)+1) * max_len > batch_size:
                    batch_examples = map(self.document2example, batch_documents)
                    batch_inputs = map(self.example2input, batch_examples)
                    yield batch_inputs
                else:
                    batch_documents.append(document)
        else:
            batch = len(documents)//batch_size
            remain = len(documents) % batch_size
            batch = batch+1 if remain else batch
            for i in range(batch):
                batch_documents = documents[batch_size*i: batch_size*(i+1)]
                batch_examples = map(self.document2example, batch_documents)
                batch_inputs = map(self.example2input, batch_examples)
                yield list(batch_inputs)

    def _epoch(self, n=1):
        documents = self.dataset.documents * n
        return documents

    @staticmethod
    def _shuffle(documents):
        np.random.shuffle(documents)
