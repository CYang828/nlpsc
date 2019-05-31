# encoding:utf-8

from nlpsc.tokenization import Tokenization


class TestTokenizer(object):

    def test_lac_tokenizer(self):
        tokenizer = Tokenization()
        tokenizer.configuration(tokenizer='lac')
        print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
        # lac并不支持英文，英文请使用
        print(tokenizer.cut("this is a piece of text, the meaning of it isn't important"))

    def test_zh_char_tokenizer(self):
        tokenizer = Tokenization()
        tokenizer.configuration(tokenizer='zh_char')
        print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
        # 不支持
        print(tokenizer.cut("this is a piece of text, the meaning of it isn't important"))

    def test_en_char_tokenizer(self):
        tokenizer = Tokenization()
        tokenizer.configuration(tokenizer='en')
        # 不支持
        print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
        print(tokenizer.cut("this is a piece of text, the meaning of it isn't important"))

    def test_jieba_tokenizer(self):
        tokenizer = Tokenization()
        tokenizer.configuration(tokenizer='jieba')
        print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
        print(tokenizer.cut("this is a piece of text, the meaning of it isn't important"))

    def test_pkuseg_tokenizer(self):
        tokenizer = Tokenization()
        tokenizer.configuration(tokenizer='pkuseg')
        print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
        print(tokenizer.cut("this is a piece of text, the meaning of it isn't important"))
