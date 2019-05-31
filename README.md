<p align="center">
  <img width="380" src="/assets/nlpsc.png">
</p>

## 介绍
[nlpsc](https://github.com/BSlience/nlpsc) 是一个nlp初学者可以快速学习理论并实践，nlp进阶者快速验证模型的项目。
这里有学习时的笔记、验证的代码和大家的思考，你也可以把你的文字和代码PR给我，让更多的人能在学习NLP的过程中少走弯路。
目前深度学习框架选用的是[Paddle](https://github.com/PaddlePaddle/Paddle)和[tensorflow](https://github.com/tensorflow/tensorflow)

## 笔记
这里是一些学习笔记。

[A Brief Note of Representation](notes/A%20Brief%20Note%20of%20Representation.md)

## 实战
### 文档处理
可以按照自己需求，随意的定制文档处理流程，快速高效的完成数据处理。
```python
from nlpsc.shortcut import NLPShortcut


with NLPShortcut(name='数学语料库') as ns:
    ns.load_corpus_from_file('test_data/') \
        .iter_clean() \
        .iter_tokenize(tokenizer='lac', userdict='math-chinese.txt') \
        .iter_stopword() \
        .iter_dump('output/')

for document in ns.get_dataset().iter():
    print(document)
```


### 深度学习模型相关
模型finetune
```python
import paddle.fluid as fluid
from nlpsc.dataset import Dataset
from nlpsc.document import Document

from nlpsc.representation.ernie import PaddleErniePretrainedModel, ErnieClassifyTransformer

# 创建数据集
dataset = Dataset(name='测试数据集')
dataset.header = ['label', 'text_a']
d = Document(text='我是正好洛杉矶格拉斯哥', lang='zh')
dataset.add(d)

# 定义数据transformer，用来将数据集中的数据转换成模型可计算的形式
generator = ErnieClassifyTransformer(dataset=dataset).batch_inputs_generator(epoch=1,
                                                                             shuffle=False)
# 定义模型
with PaddleErniePretrainedModel() as ernie_model:
    # 定义reader
    with ernie_model.define_reader(generator) as reader:
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, ernie_model.max_seq_len, 1], [-1, ernie_model.max_seq_len, 1],
                    [-1, ernie_model.max_seq_len, 1], [-1, ernie_model.max_seq_len, 1], [-1, 1],
                    [-1, 1]],
            dtypes=['int64', 'int64', 'int64', 'float32', 'int64', 'int64'],
            lod_levels=[0, 0, 0, 0, 0, 0],
            name='train_reader',
            use_double_buffer=True)

        src_ids, sent_ids, pos_ids, input_mask, labels, qids = fluid.layers.read_file(pyreader)
        reader.connect_with_model(pyreader, src_ids=src_ids, sent_ids=sent_ids, pos_ids=pos_ids,
                                  input_mask=input_mask, labels=labels, qids=qids)

    # 定义finetune的网络
    with ernie_model.define_finetune():
        cls_feats = ernie_model.model.get_pooled_output()
        cls_feats = fluid.layers.dropout(
            x=cls_feats,
            dropout_prob=0.1,
            dropout_implementation="upscale_in_train")
        logits = fluid.layers.fc(
            input=cls_feats,
            size=2,
            param_attr=fluid.ParamAttr(
                name="cls_out_w",
                initializer=fluid.initializer.TruncatedNormal(scale=0.02)),
            bias_attr=fluid.ParamAttr(
                name="cls_out_b", initializer=fluid.initializer.Constant(0.)))

    # 定义loss
    with ernie_model.define_loss():
        ce_loss, probs = fluid.layers.softmax_with_cross_entropy(
            logits=logits, label=labels, return_softmax=True)
        loss = fluid.layers.mean(x=ce_loss)

    # 定义优化函数
    with ernie_model.define_optimizer():
        optimizer = fluid.optimizer.Adam(learning_rate=.1)

        ernie_model.train(epoch=2)
```

### 各种nlp的工具

#### 分词
中文分词（基于ernie）
```python
from nlpsc.tokenization import Tokenization

tokenizer = Tokenization()
tokenizer.configuration(tokenizer='lac')
print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
```

中文单字分割
```python
from nlpsc.tokenization import Tokenization

tokenizer = Tokenization()
tokenizer.configuration(tokenizer='zh_char')
print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
```

英文单词分割
```python
from nlpsc.tokenization import Tokenization

tokenizer = Tokenization()
tokenizer.configuration(tokenizer='en')
print(tokenizer.cut('这是一个测试文档，文档的意义并不重要，重要的是怎么才能凑够数字'))
```

更多分词相关请查看 [分词用例](nlpsc/test/test_tokenizer.py)

#### 词向量


## 许可
[MIT](LICENSE.md)

Copyright on (c) 2019-present bslience
