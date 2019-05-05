# encoding:utf-8

"""
使用paddle的lac可以进一步提升分词、词性标注、命名识别的准确性
TODO：后续集成
"""

import jieba
try:
    import pkuseg
except ModuleNotFoundError as e:
    print('if you want to use pkuseg tokenizer, please use `pip install pkuseg` first')


class Tokenization(object):

    tokenizer_option = ('jieba', 'pkuseg')

    def __init__(self):
        self._tokenizer = jieba

    def configuration(self, **setting):
        tokenizer = setting.get('tokenizer', 'jieba')
        userdict = setting.get('userdict')
        parallel = setting.get('parallel', 1)

        if tokenizer == 'jieba':
            self._tokenizer = jieba
            if userdict:
                jieba.load_userdict(userdict)
            if parallel > 1:
                jieba.enable_parallel(parallel)
        elif tokenizer == 'pkuseg':
            self._tokenizer = pkuseg.pkuseg(user_dict=userdict)

    def cut(self, literal):
        return self._tokenizer.cut(literal)


if __name__ == '__main__':
    p = Tokenization()
    p.configuration(tokenizer='pkuseg')
    print(p.cut('一元二次方程在初中数学中有着很重要的地位'))
    print(p.cut('this is me'))


