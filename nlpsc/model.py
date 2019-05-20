# encoding:utf-8

import os

import numpy as np
import paddle.fluid as fluid


class PaddleModel(object):
    """paddle模型基类"""

    def __init__(self, use_gpu=False):
        if use_gpu:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        self._exe = fluid.Executor(place)


class PaddleInferModel(PaddleModel):
    """paddle推理模型基类"""

    def __init__(self, use_gpu=False):
        super(PaddleInferModel, self).__init__(use_gpu)
        self._inference_program = None
        self._feed_target_names = None
        self._fetch_targets = None

    def load_inference(self, inference_path):
        """inference模型加载"""
        [self._inference_program, self._feed_target_names, self._fetch_targets] = \
            fluid.io.load_inference_model(dirname=inference_path, executor=self._exe)

    def infer(self, *args, **kwargs):
        """模型推理"""
        raise NotImplemented


class PaddlePretrainedModel(PaddleModel):
    """paddle预训练模型基类"""

    def __init__(self, use_gpu=False):
        super(PaddlePretrainedModel, self).__init__(use_gpu)
        # 数据读取器
        self.reader = None
        # 模型
        self.model = None

    def init_params(self, init_checkpoint_path=None, init_pretrained_params_path=None):
        """初始化参数"""
        init_checkpoint_path and self.load_checkpoint(init_checkpoint_path=init_checkpoint_path)
        (init_pretrained_params_path and not init_checkpoint_path) \
        and self.load_pretrained_params(pretrained_params_path=init_pretrained_params_path)

    def create_model(self):
        """模型创建

        :return
            reader: 数据读取器"""
        raise NotImplementedError

    def finetune(self):
        """模型微调"""
        raise NotImplemented

    def train(self, *args, **kwargs):
        """模型训练"""
        raise NotImplementedError

    def load_checkpoint(self, init_checkpoint_path, use_fp16=False):
        """加载checkpoint"""
        assert os.path.exists(
            init_checkpoint_path), "[%s] cann't be found." % init_checkpoint_path

        def existed_persitables(var):
            if not fluid.io.is_persistable(var):
                return False
            return os.path.exists(os.path.join(init_checkpoint_path, var.name))

        fluid.io.load_vars(
            self._exe,
            init_checkpoint_path,
            main_program=self._main_program,
            predicate=existed_persitables)
        print("Load model from {}".format(init_checkpoint_path))

        if use_fp16:
            self._cast_fp32_to_fp16()

    def load_pretrained_params(self,
                               pretrained_params_path,
                               use_fp16=False):
        """加载pretrained参数"""
        assert os.path.exists(pretrained_params_path
                              ), "[%s] cann't be found." % pretrained_params_path

        def existed_params(var):
            if not isinstance(var, fluid.framework.Parameter):
                return False
            return os.path.exists(os.path.join(pretrained_params_path, var.name))

        fluid.io.load_vars(
            self._exe,
            pretrained_params_path,
            main_program=self._main_program,
            predicate=existed_params)
        print("Load pretraining parameters from {}.".format(
            pretrained_params_path))

        if use_fp16:
            self._cast_fp32_to_fp16()

    def _cast_fp32_to_fp16(self):
        print("Cast parameters to float16 data format.")
        for param in self._main_program.global_block().all_parameters():
            if not param.name.endswith(".master"):
                param_t = fluid.global_scope().find_var(param.name).get_tensor()
                data = np.array(param_t)
                if param.name.find("layer_norm") == -1:
                    param_t.set(np.float16(data).view(np.uint16), self._exe.place)
                master_param_var = fluid.global_scope().find_var(param.name +
                                                                 ".master")
                if master_param_var is not None:
                    master_param_var.get_tensor().set(data, self._exe.place)
