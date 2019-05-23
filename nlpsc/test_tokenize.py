from nlpsc.tokenization.char_tokenizer import FullTokenizer, CharTokenizer
from nlpsc.vocabulary import Vocabulary

t = '说明：由学生身边熟悉的事物引入新课，容易激发学生的好奇心和求知欲，' \
            '同时又容易使学生产生亲切感，从而带着良好的学习状态进入新课的学习。'


vocab = Vocabulary().load_vocab('default/ernie/vocab.txt')
token1 = FullTokenizer(vocab)
print(token1.tokenize(t))

token2 = CharTokenizer(vocab)
print(token2.tokenize(t))
