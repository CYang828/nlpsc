# encoding:utf-8

from nlpsc.shortcut import NLPShortcut


with NLPShortcut(name='数学语料库') as ns:
    ns.load_corpus_from_file('../../test_docs') \
         .iter_clean() \
         .iter_tokenize(tokenizer='jieba', userdict='math-chinese.txt') \
         .iter_stopword() \
         .iter_dump('output/')

for document in ns.get_corpus().iter():
    print(document)


# from nlpsc.tokenization.lac import PaddleLACModel
#
#
# for s in PaddleLACModel().infer(['我爱北京天安门',
#                                  '画蛇添足',
#                                  '语言模型预训练的优点是高效性，其提高了很多 NLP 任务的水准。',
#                                  '作者认为现有的技术严重的限制了预训练表示的能力，对于 fine-tuning 方法来说，尤为如此。',
#                                  'Python在进行编码方式之间的转换时，会将 unicode 作为“中间编码”，但 unicode 最大只有128那么长，所以这里当尝试将 ascii 编码字符串转换成”中间编码” unicode 时由于超出了其范围，就报出了如上错误。将Python的默认编码方式修改为utf-8即可，在py文件开头加入以下代码：']):
#     print(s)

