from nlpsc.representation.ernie.transformer import ErnieExtractEmbeddingTransformer
from nlpsc.document import Document


d = Document(text='我是正好洛杉矶格拉斯哥', lang='zh')
print(ErnieExtractEmbeddingTransformer(vocab_path='default/ernie/vocab.txt').document2input(d))
