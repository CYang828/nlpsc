# coding:utf-8

import os
from inspect import isfunction

from .error import NLPSCError
from .util.tool import uniqueid
from .tokenization import get_tokenizer
from .util.file import aio_read_file
from .preprocessing.text import clean_text
from .util.file import aio_write_file
from .representation import Representation


async def file2document(path, pattern, encoding, lang):
    """文件转换程文档"""
    path = os.path.abspath(path)
    text = await aio_read_file(path, pattern, encoding)
    document = Document(text, path=path, lang=lang)
    return document


def make_document(text, lang, name=None):
    """将文本转换成文章对象"""
    return Document(text, lang, name)


def list2words(l):
    """列表转换word对象列表"""
    return [Word(i) for i in l]


class Document(object):
    """文章对象"""

    support_langs = ('zh', 'en')
    _stopwordict = []

    def __init__(self, text, lang='zh', path=None, name=None, dataset=None):
        if lang not in self.support_langs:
            print('langs {} can be supported now, please check it'.format(self.support_langs))
            raise NLPSCError

        self.id = uniqueid()
        # 文档名称
        self.name = name if name else self.id
        # 文档标签
        self.label = None
        # 文档路径
        self.path = path
        # 文档中保存的字面量
        self.ori = self.text = text.strip()
        # 文档语言
        self.lang = lang
        # 调用方法栈
        self._call_stack = []

        # 段落
        self._paragraphs = []
        # 句子
        self._sentences = []
        # 词
        self._words = []
        # 字符
        self._chars = []

        # 文档所属数据集
        self.dataset = dataset

    def __str__(self):
        if self._words:
            show_thing = '\t' + ' | '.join(map(str, self._words[:10]))
        else:
            show_thing = self.text

        return '<nlpsc.document.Document -> {} ({} ......) >'.format(self.name, show_thing[:100])

    def __short__(self):
        return '<nlpsc.document.Document -> {}>'.format(self.name)

    def paragraph(self):
        """文章段落切割"""
        self._paragraphs = [Paragraph(text) for text in self.text.split('\n') if text]
        return self

    def sentence(self):
        """文章句子切割，需要依赖于段落切割"""
        pass

    def word(self):
        """文章词切割，需要依赖于句子和段落切割"""
        pass

    def paragraph(self):
        """文章段落化"""
        self._paragraphs = [Paragraph(text) for text in self.text.split('\n') if text]
        return self

    def clean(self):
        """文档字面量清洗"""
        self.text = clean_text(self.text)
        return self

    def preprocess(self, fn):
        """
        数据预处理

        :argument
            fn: 预处理的自定义函数，自定函数需要返回预处理后的字面量结果
        """

        if isfunction(fn):
            self.text = fn(self.text)
            return self
        else:
            print("preprocess argument is a function, please check it!")
            raise NLPSCError

    def tokenize(self, tokenizer_opt=None, userdict=None):
        """分词

        :argument
            tokenizer: 分词器，目前中文tokenizer支持jieba（默认）、pkuseg
            userdict: 用户词典
        """

        if userdict:
            # 首先寻找default目录，是否存在该文件
            guess_path = os.path.join(os.path.dirname(__file__), 'default/userdict/{}'.format(userdict))
            if os.path.exists(guess_path):
                userdict = guess_path
            print('load tokenize userdict: {}'.format(userdict))

        tokenizer = get_tokenizer()
        tokenizer.configuration(tokenizer=tokenizer_opt, userdict=userdict)
        self._words = list2words(list(tokenizer.cut(self.text)))
        return self

    def stopword(self, stopwordict=None):
        """停用词处理

        :argument
            停用词典路径
        """

        if not self._stopwordict:
            self._load_stopwordict(stopwordict)

        remain_words = []
        for idx, word in enumerate(self._words):
            if word.text and word.text not in self._stopwordict:
                remain_words.append(word)
        del self._words
        self._words = remain_words
        return self

    def _load_stopwordict(self, stopwordict):
        """加载停用词典"""
        stopwordict = stopwordict if stopwordict else os.path.join(os.path.dirname(__file__), 'default/stopwords.txt')
        print('load stopword: {}'.format(stopwordict))
        self._stopwordict = [line.strip() for line in open(stopwordict, 'r', encoding='utf-8').readlines()]

    def represent(self):
        """向量表示

        如果使用word2vec这种静态的文本表示模型，对分词后的结果进行repr是个不错的选择
        文本表示的方式普遍分成两种，一种特征集成（Feature Ensemble）另一种微调（Fine-tuning）模式。
        但是如果是使用ELMO、GPT、Bert、Ernie这类的模型，则使用finetune的方式会更好"""
        # self._represent.configuration()
        # if self._words:
        #     for word in self._words:
        #         self._represent.repr(word)
        pass

    def literal(self, word_delimiter=' '):
        if self._words:
            literal_text = word_delimiter.join([word.text for word in self._words])
        else:
            literal_text = self.text
        return literal_text

    async def dump(self, output_dir, filename, is_structured=False, prefix="dump", suffix="nlpsc",
                         paragraph_delimiter='</p>', sentence_delimiter='</s>', word_delimiter=' '):
        """将document对象dump到文件中

        :argument
            output_dir: dump的目录
            is_structured: 是否进行结构化dump,结构化dump会调用结构化方法进行结构化
            prefix: dump文件前缀
            suffix: dump文件后缀
            paragraph_delimiter: 段落分隔符
            sentence_delimiter: 句子分隔符
            word_delimiter: 词分隔符
        """

        if is_structured:
            # 需要先进行结构化切割
            pass
        else:
            # 指根据当前document被处理的情况进行dump
            dump_text = self.literal(word_delimiter)
            dump_text = '{}||{}\n'.format(self.path if self.path else '', dump_text)
            dump_path = await aio_write_file(output_dir, filename, dump_text, pattern='a+')
            print('{:60} dump to {}'.format(self.__short__(), dump_path))
        return self


class Paragraph(object):
    """段落对象"""

    def __init__(self, text):
        self.text = text
        self._nlp = None
        self.sentences = None

    def __str__(self):
        return '<Paragraph>'

    def sentence(self):
        """段落句子切割"""
        pass

    @property
    def nlp(self):
        return self._nlp

    @nlp.setter
    def nlp(self, v):
        self._nlp = v
        self.sentences = self._make_sentence()

    def _make_sentence(self):
        """生成句子"""
        sentence = ''
        sentences = []
        words = []
        for i, token in enumerate(self._nlp.tokens):
            if token.text in ('.', '!', '?'):
                sentence += token.text
                sentences.append(Sentence(sentence, words, self._nlp.tokens))
                sentence = ''
                words = []
            else:
                sentence += token.text + ' '
                words.append(token.text)

        # 如果最后一句没有终止符号
        if sentence:
            sentences.append(Sentence(sentence, words, self._nlp.tokens))
        return sentences


class Sentence(object):
    """句子对象

    简单的语句切分方式为通过一些标点符号来进行，但是这样做会有局限性。
    英文中有些时候'.'并不一定代表是句子的结束，中文中可能也存在这种情况"""

    def __init__(self, text, words, tokens):
        self.text = text.strip()
        self.words = [Word(word, token) for word, token in zip(words, tokens)]

    def __str__(self):
        return '<Sentence>'


class Word(object):
    """单词对象"""

    def __init__(self, text):
        self.text = text.strip()

    def __str__(self):
        return '{}'.format(self.text)





