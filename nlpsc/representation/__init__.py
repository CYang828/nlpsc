# encoding:utf-8

from ..selector import Selector


class Representation(Selector):
    """文本表示类

    word2vec: 忽略上下文的词表示，无法解决多义词的问题
    bert: 使用fine tune的方式进行下游任务的训练 （SOTA）
    ernie: 百度根据bert和中文的一些特点进行优化后的模型 （SOTA）"""

    _support_models = {'word2vec': '_word2vec', 'bert': '_bert', 'ernie': '_ernie'}
    _default_settings = {'model': 'ernie', 'target': 'token_embedding'}
    _support_settings = {'model', 'token_embedding', 'sentence_embedding'}

    def __init__(self, **setting):
        super(Representation, self).__init__(**setting)

    def configuration(self,  **setting):
        pass

    def repr(self, text):
        """获取文本表示向量"""
        return self.exe(text)

    def finetune(self):
        """finetune表示模型"""
        pass

    def _word2vec(self):
        pass

    def _bert(self):
        pass

    def _ernie(self):
        """构建ernie对象"""
        from ..representation.ernie import PaddleErnieInferModel
        return PaddleErnieInferModel(**self._settings)
