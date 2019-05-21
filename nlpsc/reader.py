# encoding:utf-8


class Reader(object):
    """数据读取器"""

    def __init__(self, paddle_reader, **kwargs):
        self._reader = paddle_reader
        for k, v in kwargs.items():
            setattr(self, k, v)

    def read(self, data_generator):
        """读取数据

        从一个generator中遍历数据"""
        self._reader.decorate_tensor_provider(data_generator)
        self._reader.start()

    def reset(self):
        """数据读取完成，reader复位"""
        self._reader.reset()


