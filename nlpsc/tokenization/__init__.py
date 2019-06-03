# encoding:utf-8


from ..tokenization.lac import PaddleLACInferModel


class Tokenization(object):
    """默认使用lac进行分词

    如果想使用jieba：请执行`pip install jieba`
    如果想使用pkuseg：请执行`pip install pkuseg`
    """

    tokenizer_option = ('jieba', 'pkuseg', 'lac', 'en', 'zh_char')

    def __init__(self):
        self._tokenizer = PaddleLACInferModel()

    def configuration(self, **setting):
        tokenizer = setting.get('tokenizer', 'lac')
        userdict = setting.get('userdict')
        parallel = setting.get('parallel', 1)
        if tokenizer == 'jieba':
            try:
                import jieba
            except ModuleNotFoundError:
                print('if you want to use jieba tokenizer, please use `pip install jieba` first')
            self._tokenizer = jieba
            if userdict:
                jieba.load_userdict(userdict)
            if parallel > 1:
                jieba.enable_parallel(parallel)
        elif tokenizer == 'pkuseg':
            try:
                import pkuseg
            except ModuleNotFoundError:
                print('if you want to use pkuseg tokenizer, please use `pip install pkuseg` first')
            self._tokenizer = pkuseg.pkuseg(user_dict=userdict)
        elif tokenizer == 'lac':
            self._tokenizer = PaddleLACInferModel()
        elif tokenizer == 'zh_char':
            from .char_tokenizer import FullTokenizer
            self._tokenizer = FullTokenizer()
        elif tokenizer == 'en':
            from .char_tokenizer import CharTokenizer
            self._tokenizer = CharTokenizer()

    def cut(self, literal):
        return list(self._tokenizer.cut(literal))


def get_tokenizer():
    tokenizer = Tokenization()
    return tokenizer



