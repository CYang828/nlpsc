# encoding:utf-8

import os
from functools import wraps

import numpy as np
import paddle.fluid as fluid
from .reader import Reader


def modelcontext(fn):
    """模型上下文装饰器，用来定义模型范围"""

    @wraps(fn)
    def wrapper(obj, *args, **kwargs):
        with fluid.program_guard(obj.main_program, obj.startup_program):
            r = fn(obj, *args, **kwargs)
        return r
    return wrapper


class PaddleModel(object):
    """paddle模型基类"""

    def __init__(self, use_gpu=False):
        if use_gpu:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        self.exe = fluid.Executor(place)


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
    """paddle预训练模型基类

    generator：用来真实读取，组装，转换，batch的组件，需要用户针对不同的任务定制
    reader: 用来进行数据切分，转换成模型需要类型的抽象组件
    """

    def __init__(self, use_gpu=False, init_checkpoint_path=None, init_pretrained_params_path=None):
        super(PaddlePretrainedModel, self).__init__(use_gpu)
        # 数据生成器
        self.generator = None
        # 数据读取器
        self.reader = None
        # 模型
        self.model = None
        # 模型program
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        # 模型参数初始化
        self.init_checkpoint_path = init_checkpoint_path
        self.init_pretrained_params_path = init_pretrained_params_path

        self._flag_reader = False
        self._flag_loss = False
        self._flag_optimizer = False

    def define_reader(self, generator):
        """定义reader"""
        obj = self
        self.generator = generator
        self._flag_reader = True

        class _Reader(object):

            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                obj.init_params()

            @staticmethod
            def connect_with_model(pyreader, **kwargs):
                obj.reader = Reader(pyreader, **kwargs)

        return _Reader()

    def init_params(self):
        """初始化参数"""
        self.init_checkpoint_path and self.load_checkpoint(init_checkpoint_path=self.init_checkpoint_path)
        (self.init_pretrained_params_path and not self.init_checkpoint_path) \
        and self.load_pretrained_params(pretrained_params_path=self.init_pretrained_params_path)

    def define_model(self):
        """模型创建

        :return
            reader: 数据读取器"""
        raise NotImplementedError

    def define_finetune(self):
        """定义finetune网络"""
        obj = self

        class _Finetune(object):
            def __enter__(self):
                obj.model = obj.define_model()
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                obj.init_params()

        return _Finetune()

    def define_loss(self):
        """定义损失函数"""
        obj = self
        self._flag_loss = True

        class _LossFunction(object):
            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return _LossFunction()

    def define_optimizer(self):
        """定义优化函数"""
        obj = self
        self._flag_optimizer = True

        class _OptimizerFunction(object):
            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return _OptimizerFunction()

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
            main_program=self.main_program,
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
            main_program=self.main_program,
            predicate=existed_params)
        print("Load pretraining parameters from {}.".format(
            pretrained_params_path))

        if use_fp16:
            self._cast_fp32_to_fp16()

    def _cast_fp32_to_fp16(self):
        print("Cast parameters to float16 data format.")
        for param in self.main_program.global_block().all_parameters():
            if not param.name.endswith(".master"):
                param_t = fluid.global_scope().find_var(param.name).get_tensor()
                data = np.array(param_t)
                if param.name.find("layer_norm") == -1:
                    param_t.set(np.float16(data).view(np.uint16), self._exe.place)
                master_param_var = fluid.global_scope().find_var(param.name +
                                                                 ".master")
                if master_param_var is not None:
                    master_param_var.get_tensor().set(data, self._exe.place)
