# encoding:utf-8

from nlpsc.shortcut import NLPShortcut


class TestFlow(object):

    def test_synthesis_flow(self):
        from nlpsc.document import Document

        with NLPShortcut(name='数学语料库') as ns:
            ns.load_corpus_from_file('test_data/') \
                .iter_clean() \
                .iter_tokenize(tokenizer='lac', userdict='math-chinese.txt') \
                .iter_stopword() \
                .iter_dump('output/')

        for document in ns.get_dataset().iter():
            assert isinstance(document, Document)
