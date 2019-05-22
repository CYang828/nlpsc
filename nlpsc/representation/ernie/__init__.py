# encoding:utf-8

import numpy as np
import paddle.fluid as fluid

from ...reader import Reader
from ...error import NLPSCError
from ...util.file import get_default_path
from .ernie import ErnieModel, ErnieConfig
from ...representation.ernie.util import split_text
from ...model import PaddleInferModel, PaddlePretrainedModel, modelcontext
from .transformer import ErnieBaseTransformer, ErnieClassifyTransformer, SequenceLabelTransformer, ErnieExtractEmbeddingTransformer


class PaddleErnieInferModel(PaddleInferModel):
    """Ernie推断模型"""

    def __init__(self, use_gpu=False, max_seq_len=128,
                 do_lower_case=True, pretrained_model=None):
        super(PaddleInferModel, self).__init__(use_gpu)
        self._vocab_path = get_default_path('ernie/vocab.txt')
        self._max_seq_len = max_seq_len
        self._do_lower_case = do_lower_case
        self._reader = ErnieExtractEmbeddingTransformer(vocab_path=self._vocab_path,
                                                        max_seq_len=max_seq_len,
                                                        do_lower_case=do_lower_case)
        pretrained_path = pretrained_model if pretrained_model\
            else get_default_path('pretrained-models/ernie-inference/')
        self.load_inference(pretrained_path)

    def infer(self, texts):
        """推断text的成分

        :argument
            texts: 文本列表
        :return
            results
        """

        erine_input = self._reader.convert_example_to_erine_input(texts)
        cls_emb, unpad_top_layer_emb = self._exe.run(self._inference_program,
                                                     feed={self._feed_target_names[0]: erine_input[0],
                                                           self._feed_target_names[1]: erine_input[1],
                                                           self._feed_target_names[2]: erine_input[2],
                                                           self._feed_target_names[3]: erine_input[3],
                                                           self._feed_target_names[4]: erine_input[4]},
                                                     fetch_list=self._fetch_targets,
                                                     return_numpy=False)
        return np.array(cls_emb), np.array(unpad_top_layer_emb)

    def token_embedding(self, text):
        """字符级别的embedding

        输入单句: "好好学习，天天向上"
        预处理后的单句为: "[CLS]好好学习，天天向上[SEP]"，共 11 个 token
        那么返回的 Embedding 矩阵 W 就是 11 * 768 的矩阵，
        其中 W[0][:] 就是 "[CLS]" 对应的 embedding，W[1][:] 表示 "好" 对应的 embedding"""

        texts = split_text(text, self._max_seq_len)
        return self.infer(texts)[1][1:-1]

    def sentence_embedding(self, text):
        """句子级别的embedding"""
        texts = split_text(text, self._max_seq_len)
        return self.infer(texts)[0]


class PaddleErniePretrainedModel(PaddlePretrainedModel):
    """Ernie预训练模型"""

    def __init__(self, use_gpu=False, use_fp16=False, ernie_config_path=None,
                 init_checkpoint_path=None, init_pretrained_params_path=None, max_seq_len=512):
        self.ernie_config_path = ernie_config_path if ernie_config_path \
            else get_default_path('ernie/ernie_config.json')
        self.use_fp16 = use_fp16
        self.max_seq_len = max_seq_len
        super(PaddleErniePretrainedModel, self).__init__(use_gpu, init_checkpoint_path, init_pretrained_params_path)

    @modelcontext
    def create_reader(self, generator):
        """创建文件读取器"""
        self.generator = generator
        pyreader = fluid.layers.py_reader(
            capacity=50,
            shapes=[[-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1],
                    [-1, self.max_seq_len, 1], [-1, self.max_seq_len, 1], [-1, 1]],
            dtypes=['int64', 'int64', 'int64', 'float', 'int64'],
            lod_levels=[0, 0, 0, 0, 0],
            name='ernie_reader',
            use_double_buffer=True)

        src_ids, sent_ids, pos_ids, input_mask, seq_lens = fluid.layers.read_file(pyreader)
        ernie_reader = Reader(pyreader, src_ids=src_ids, sent_ids=sent_ids, pos_ids=pos_ids,
                              input_mask=input_mask, seq_lens=seq_lens)
        self.reader = ernie_reader
        return ernie_reader

    @modelcontext
    def create_model(self):
        """定义ernie网络"""
        if not self.reader:
            print('please call create reader function first')
            raise NLPSCError

        ernie_config = ErnieConfig(self.ernie_config_path)
        ernie_model = ErnieModel(
            src_ids=self.reader.src_ids,
            position_ids=self.reader.pos_ids,
            sentence_ids=self.reader.sent_ids,
            input_mask=self.reader.input_mask,
            config=ernie_config,
            use_fp16=self.use_fp16)
        return ernie_model

    def train(self, epoch, lr=None, save_inference=False, save_checkpoint=False):
        pass

    def evaluate(self):
        pass

    def infer(self):
        pass

    def bulletin_board(self):
        pass

