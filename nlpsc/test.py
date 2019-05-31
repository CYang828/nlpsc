# encoding:utf-8

# from nlpsc.shortcut import NLPShortcut
#
#
# with NLPShortcut(name='数学语料库') as ns:
#     ns.load_corpus_from_file('../../test_docs') \
#          .iter_clean() \
#          .iter_tokenize(tokenizer='lac', userdict='math-chinese.txt') \
#          .iter_stopword() \
#          .iter_dump('output/')
#
# for document in ns.get_dataset().iter():
#     print(document)


# from nlpsc.tokenization.lac import PaddleLACModel
#
#
# for s in PaddleLACModel().infer(['我爱北京天安门',
#                                  '画蛇添足',
#                                  '语言模型预训练的优点是高效性，其提高了很多 NLP 任务的水准。',
#                                  '作者认为现有的技术严重的限制了预训练表示的能力，对于 fine-tuning 方法来说，尤为如此。',
#                                  'Python在进行编码方式之间的转换时，会将 unicode 作为“中间编码”，但 unicode 最大只有128那么长，所以这里当尝试将 ascii 编码字符串转换成”中间编码” unicode 时由于超出了其范围，就报出了如上错误。将Python的默认编码方式修改为utf-8即可，在py文件开头加入以下代码：']):
#     print(s)


# from nlpsc.representation.ernie import PaddleErnieInferModel
#
#
# PaddleErnieInferModel().infer(['我爱北京我是北京天安门'])
#
# print(PaddleErnieInferModel().token_embedding('方法随机').shape)
# print(PaddleErnieInferModel().sentence_embedding('方法随机sssssssssssss').shape)

# import os
#
import paddle.fluid as fluid
from nlpsc.representation.ernie import PaddleErniePretrainedModel, ErnieClassifyTransformer
from nlpsc.dataset import Dataset
from nlpsc.document import Document


# 指定数据集dataset
# dataset = ns.get_dataset()
dataset = Dataset(name='测试数据集')
dataset.header = ['label', 'text_a']
d = Document(text='我是正好洛杉矶格拉斯哥', lang='zh')
d.label = 0
dataset.add(d)
d = Document(text='我是正好洛杉矶格拉斯哥', lang='zh')
d.label = 0
dataset.add(d)


# 定义一个数据transformer，用来将数据集中的数据转换成模型可计算的形式
generator = ErnieClassifyTransformer(dataset=dataset).batch_inputs_generator(epoch=1,
                                                                             shuffle=False)

# # 定义一个模型
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

        (src_ids, sent_ids, pos_ids, input_mask, labels, qids) = fluid.layers.read_file(pyreader)
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

    # 可视化当前网络 -> 方便校对,尤其是对输入输入出的描述，要清晰，易懂
    # 执行预模型相关的操作
    # 执行过程因该是一个先注册后执行的过程，这样可以分析用户想要做的事情，然后做些分析，最后展示用户想要的呈现方式
    ernie_model.train(fetch_list=[loss])
    # ernie_model.infer()
    # ernie_model.evaluate()
#
#     # 一般在做模型的时候，我们两种场景，
#     # 一种是在实验阶段，希望能够快速验证想法，特点是；不需要太多的数据和机器，快速并且可视化
#     # 第二种是写成产代码，需要高性能、高吞吐，特点是：数据量和需要的计算量都很大
