# encoding:utf-8

import abc

from .vocabulary import Vocabulary


class Transformer(metaclass=abc.ABCMeta):
    """数据转换器基类

    :argument
        dataset: 要使用的数据集
        vocab_path: 词典文件，如果不指定，在需要使用词典时，会根据数据集自动生成
        do_lower_case: 是否转换成小写
        random_seed: shuffle时使用的随机种子，
                     如果指定值，则shuffle时的顺序保持一个固定的乱序
                     如果不指定值，则每次shuffle的数据顺序都不相同"""

    def __init__(self, dataset, vocab_path=None, do_lower_case=True, random_seed=None):
        self.dataset = dataset
        self._vocab_path = vocab_path
        self.vocab = None
        self._do_lower_case = do_lower_case
        self._random_seed = random_seed

    @abc.abstractmethod
    def data_generator(self, batch_size=1, epoch=1, shuffle=False):
        """数据生成器方法（用户自定义）
           需要在子类中实现，此方法必须返回一个生成器"""

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

    def _batch(self):
        pass

    def _epoch(self):
        pass

    def _shuffle(self):
        pass
