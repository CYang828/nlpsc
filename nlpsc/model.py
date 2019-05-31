# encoding:utf-8

import os
import abc
from functools import wraps

import numpy as np
import paddle.fluid as fluid
from .reader import Reader
from .core import NLPShortcutCore
from .error import NLPSCError


def modelcontext(fn):
    """模型上下文装饰器，用来定义模型范围"""

    @wraps(fn)
    def wrapper(obj, *args, **kwargs):
        with fluid.program_guard(obj.main_program, obj.startup_program):
            with fluid.unique_name.guard():
                if fn.__name__ in ('define_reader', 'define_model', 'define_finetune',
                                   'define_loss', 'define_optimizer'):
                    # define标志位设置为true
                    setattr(obj, '_flag_{}'.format(fn.__name__), True)
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
            fluid.io.load_inference_model(dirname=inference_path, executor=self.exe)

    def infer(self, *args, **kwargs):
        """模型推理"""
        raise NotImplemented


class PaddlePretrainedModel(NLPShortcutCore, PaddleModel):
    """
    paddle预训练模型基类

    Attributes:
    -----------
    reader: nlpsc.reader.Reader
        数据读取器,用来把transformer中提供的数据，转换成模型需要的tensor组件，
        默认为define_reader的返回值，如果没有自定哦define_reader，
        则需要使用connect_with_model手动建立reader和model的关系
    model:
         模型对象，define_model函数的返回值，如果define_model无返回则默认为self
    loss:
         损失函数，默认为define_loss的返回值，如果没有自定哦define_loss，
         则需要使用connect_with_model手动建立loss和model的关系
    optimizer:
         优化函数，默认为define_optimizer的返回值，如果没有自定哦define_optimizer，
         则需要使用connect_with_model手动建立optimizer和model的关系
    """

    def __init__(self, use_gpu=False, init_checkpoint_path=None, init_pretrained_params_path=None):
        NLPShortcutCore.__init__(self)
        PaddleModel.__init__(self, use_gpu)

        # 数据生成器
        self._generator = None
        # 数据读取器
        self.reader = None
        # 模型对象
        self.model = self
        # 损失函数
        self.loss = None
        # 优化函数
        self.optimizer = None
        # 模型program
        self.main_program = fluid.Program()
        self.startup_program = fluid.Program()
        # 模型参数初始化
        self._init_checkpoint_path = init_checkpoint_path
        self._init_pretrained_params_path = init_pretrained_params_path

    @modelcontext
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.lazy_call()

    def _pre_exe(self):
        """网络执行前检查"""

        if not self._flag_define_reader:
            print('please call define_reader first!')
            raise NLPSCError

        if not self._flag_define_model:
            print('please call define_model first!')
            raise NLPSCError

    def lazy_call(self):
        """懒加载"""
        self._pre_exe()
        for bridge in self.deepdive():
            bridge.call(add_bridge=False)

    @modelcontext
    def define_reader(self, generator):
        """定义reader"""
        obj = self
        self._generator = generator

        class _Reader(object):

            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()
                fluid.unique_name.guard().__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            @staticmethod
            def connect_with_model(pyreader, **kwargs):
                obj.reader = Reader(pyreader, self._generator, **kwargs)

        return _Reader()

    @modelcontext
    def define_model(self):
        """定义主要模型网络"""
        obj = self

        class _Model(object):

            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()
                fluid.unique_name.guard().__enter__()
                return self

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

        return _Model()

    @modelcontext
    def define_finetune(self):
        """定义finetune网络"""
        obj = self

        class _Finetune(object):
            def __enter__(self):
                obj.model = obj.define_model()
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()
                fluid.unique_name.guard().__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                obj.init_params()

        return _Finetune()

    @modelcontext
    def define_loss(self):
        """定义损失函数"""
        obj = self

        class _LossFunction(object):
            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()
                fluid.unique_name.guard().__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            @staticmethod
            def connect_with_model(loss):
                obj.loss = loss

        return _LossFunction()

    @modelcontext
    def define_optimizer(self):
        """定义优化函数"""
        obj = self

        class _OptimizerFunction(object):
            def __enter__(self):
                fluid.program_guard(obj.main_program, obj.startup_program).__enter__()
                fluid.unique_name.guard().__enter__()

            def __exit__(self, exc_type, exc_val, exc_tb):
                pass

            @staticmethod
            def connect_with_model(optimizer):
                obj.optimizer = optimizer

        return _OptimizerFunction()

    def init_params(self, init_checkpoint_path=None, init_pretrained_params_path=None):
        """初始化参数

        如果两个参数都存在，则优先使用checkpoint进行初始化"""

        self._init_checkpoint_path = init_checkpoint_path
        self._init_pretrained_params_path = init_pretrained_params_path

        # 初始化网络参数
        self.exe.run(self.startup_program)
        self._init_checkpoint_path and self._load_checkpoint(init_checkpoint_path=self._init_checkpoint_path)
        (self._init_pretrained_params_path and not self._init_checkpoint_path) \
        and self._load_pretrained_params(pretrained_params_path=self._init_pretrained_params_path)

    def lazy_train(self, epoch=10, fetch_list=None, return_numpy=True):
        """模型训练"""
        step = 0
        for i in range(epoch):
            self.reader.read()
            while True:
                try:
                    fetch_ret = self.exe.run(self.main_program,
                                             fetch_list=fetch_list,
                                             return_numpy=return_numpy)
                    step += 1
                    print(step)
                except fluid.core.EOFException:
                    self.reader.reset()
                    break

    @abc.abstractmethod
    def __infer(self, fetch_list=None, return_numpy=True):
        """模型训练"""

    @abc.abstractmethod
    def __evaluate(self):
        """模型训练"""

    @abc.abstractmethod
    def __watcher(self):
        """打开可视化页面"""

    @abc.abstractmethod
    def __save(self, path, step=10):
        """模型保存相关"""

    def _load_checkpoint(self, init_checkpoint_path, use_fp16=False):
        """加载checkpoint"""
        assert os.path.exists(
            init_checkpoint_path), "[%s] can't be found." % init_checkpoint_path

        def existed_persitables(var):
            if not fluid.io.is_persistable(var):
                return False
            return os.path.exists(os.path.join(init_checkpoint_path, var.name))

        fluid.io.load_vars(
            self.exe,
            init_checkpoint_path,
            main_program=self.main_program,
            predicate=existed_persitables)
        print("Load model from {}".format(init_checkpoint_path))

        if use_fp16:
            self._cast_fp32_to_fp16()

    def _load_pretrained_params(self,
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
            self.exe,
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
