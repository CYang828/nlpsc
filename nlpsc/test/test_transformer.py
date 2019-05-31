from nlpsc.representation.ernie.transformer import ErnieExtractEmbeddingTransformer, SequenceLabelTransformer, \
    ErnieClassifyTransformer
from nlpsc.document import Document


def test_transformer():
    d = Document(text='我是正好洛杉矶格拉斯哥', lang='zh')
    print(ErnieExtractEmbeddingTransformer().document2input(d))
    print(SequenceLabelTransformer(label_map_config='../default/ernie/label_map.json').document2input(d))
    print(ErnieClassifyTransformer(vocab_path='../default/ernie/vocab.txt').document2input(d))
