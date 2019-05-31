# encoding:utf-8

import collections

from .util.file import get_default_path
from .util.python import convert_to_unicode


class Vocabulary(object):
    """词典对象"""

    def __init__(self):
        # 结构{'token': id}
        self.vocab = None
        # 结构{'id': token}
        self.inv_vocab = None

    def load_vocab(self, vocab_file):
        """根据文件生成字典"""
        vocab = collections.OrderedDict()
        fin = open(vocab_file, encoding='utf-8')
        for num, line in enumerate(fin):
            items = convert_to_unicode(line.strip()).split("\t")
            if len(items) > 2:
                break
            token = items[0]
            index = items[1] if len(items) == 2 else num
            token = token.strip()
            vocab[token] = int(index)
        self.vocab = vocab
        self.inv_vocab = {v: k for k, v in vocab.items()}
        return self

    def auto_from_dataset(self, dataset: object) -> object:
        """根据当前的数据集生成"""
        pass

    def tokens2ids(self, tokens):
        """将token换成id"""
        output = []
        for token in tokens:
            output.append(self.vocab[token])
        return output

    def ids2tokens(self, ids):
        """将id转换成token"""
        output = []
        for i in ids:
            output.append(self.inv_vocab[i])
        return output

    def items(self):
        for k, v in self.vocab.items():
            yield k, v


def get_default_vocabulary():
    return Vocabulary().load_vocab(get_default_path('ernie/vocab.txt'))
