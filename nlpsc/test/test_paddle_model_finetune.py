# encoding:utf-8

import paddle.fluid as fluid
from nlpsc.dataset import Dataset
from nlpsc.document import Document

from nlpsc.representation.ernie import PaddleErniePretrainedModel, ErnieClassifyTransformer


def test_model_finetune():
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
