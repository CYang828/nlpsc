# encoding:utf-8

from nlpsc.shortcut import NLPShortcut


with NLPShortcut(name='数学语料库') as ns:
    ns.load_corpus_from_file('../../test_docs') \
         .iter_clean() \
         .iter_tokenize(userdict='math-chinese.txt') \
         .iter_stopword() \
         .iter_dump('output/')

for document in ns.get_corpus().iter():
    print(document)
