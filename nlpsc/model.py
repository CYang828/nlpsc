# encoding:utf-8


import paddle.fluid as fluid


class PaddleModel(object):
    """paddle模型基类"""

    def __init__(self, use_gpu=False):
        # 创建执行器
        if use_gpu:
            place = fluid.CUDAPlace(0)
        else:
            place = fluid.CPUPlace()
        self._exe = fluid.Executor(place)

        self._inference_program = None
        self._feed_target_names = None
        self._fetch_targets = None

    def load(self, pretrain_path):
        """预训练模型加载"""
        [self._inference_program, self._feed_target_names, self._fetch_targets] = \
            fluid.io.load_inference_model(dirname=pretrain_path, executor=self._exe)

    def infer(self, *args, **kwargs):
        """模型推理"""
        raise NotImplementedError
