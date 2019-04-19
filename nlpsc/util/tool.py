# encoding:utf-8

import uuid
import time

from tqdm import tqdm


class timing(object):
    """计时器"""

    __unitfactor = {'s': 1,
                    'ms': 1000,
                    'us': 1000000}

    def __init__(self, unit='s', precision=4):
        self.start = None
        self.end = None
        self.total = 0
        self.unit = unit
        self.precision = precision

    def __enter__(self):
        if self.unit not in timing.__unitfactor:
            raise KeyError('Unsupported time unit.')
        self.start = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        self.total = (self.end - self.start) * timing.__unitfactor[self.unit]
        self.total = round(self.total, self.precision)
        return False

    def __str__(self):
        return '{total}{unit}'.format(total=self.total, unit=self.unit)


class FuncBridge(object):

    unique_obj_cache = {}

    def __init__(self, calling, fn_name, fn_desc=None):
        self._calling = calling
        self._fn_name = fn_name
        self._fn_args = ()
        self._fn_kwargs = {}
        self._fn_desc = fn_desc if fn_desc else '{} processing'.format(self._fn_name)
        fn_return_type = getattr(self._calling, self._fn_name).__annotations__['return']
        fn_return_type_name = fn_return_type.__class__.__name__

        if fn_return_type_name not in self.unique_obj_cache.keys():
            self._fn_return_obj = fn_return_type()
            self.unique_obj_cache[fn_return_type_name] = self._fn_return_obj
        else:
            self._fn_return_obj = self.unique_obj_cache[fn_return_type_name]

    def __call__(self, *args, **kwargs):
        self._fn_args = args
        self._fn_args = kwargs
        return self._fn_return_obj

    def __str__(self):
        return '<FuncBridge {} {} {} {}>'.format(self._calling, self._fn_name, self._fn_args, self._fn_kwargs)

    def call(self):
        return getattr(self._calling, self._fn_name)(*self._fn_args, **self._fn_kwargs)

    @property
    def return_obj(self):
        return self._fn_return_obj


class IterFnBridge(object):
    def __init__(self, calling, fn_list, fn_desc):
        self._calling = calling
        self._fn_list = fn_list
        self._fn_desc = fn_desc

    def __call__(self, *args, **kwargs):
        pass
        # count_fn = len(self._fn_list)
        # with CommandProcessBar(total=count_fn, desc=self._fn_desc) as pbar:
        #     for fn in self._fn_list:
        #         fn(*args, **kwargs)
        #         pbar.update(1)
        # return self._calling


class CommandProcessBar(tqdm):
    """控制台进度条"""

    pass


def uniqueid(bit=64):
    """获取64位requesid"""

    return uuid.uuid1().int >> bit