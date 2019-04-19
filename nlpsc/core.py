# encoding:utf-8

import time
import inspect
from queue import Empty
from functools import wraps

from .error import NLPSError
from .util.tool import FuncBridge
from .util.process import TaskWrapper
from .util.thread import ThreadWrapper
from .util.process import ProcessPoolWrapper, Manager


# 框架中的唯一进程池，进程的个数为当前机器cpu个数
_process_pool = None
# 注册cpu装饰器的函数
_cpu_deco_register = []
# 注册io装饰器的函数
_io_deco_register = []


def _sugar_lazy_call():
    """懒加载方法转调用规则"""
    pass


def _sugar_magic_call(cls_name, fn_name):
    """系统内建方法转调用规则"""
    if not fn_name.startswith(cls_name):
        real_fn_name = '_{}{}'.format(cls_name, fn_name)
    else:
        real_fn_name = fn_name
    return real_fn_name


def deepdive(obj):
    """深入查看当前的调用链，返回的调用链都是bridge"""

    if isinstance(obj, NLPShortcutCore):
        for bridge in obj.call_stack:
            yield bridge

        for bridge in obj.call_stack:
            if bridge.return_obj and bridge.return_obj is not obj:
                yield from deepdive(bridge.return_obj)
    else:
        print("'deepdive' dive in 'NLPShortcutCore' object")
        raise AttributeError


def callable_register(fn):
    """函数调用注册器
    XXX方法的目的是调用本对象的__XXX方法
    """

    def is_function_valid(cls, func):
        if fn not in dir(cls):
            return True
        else:
            print("AttributeError: '{}' object has no attribute '{}'".format(cls.__class__.__name__,
                                                                             func))
            raise AttributeError

    @wraps(fn)
    def wrapper(obj, fn_name):

        # 调用python内建方法
        if fn_name.startswith('__') and fn_name.endswith('__'):
            raise AttributeError
        elif isinstance(obj, NLPShortcutCore):
            # 如果被调用的方法在定义类中已经存在则不需要进行方法名的转换
            # 调用框架定义方法
            if fn_name.startswith('__'):
                # 系统内置方法和框架内置方法，立即执行
                real_fn_name = '_{}{}'.format(obj.__class__.__name__, fn_name)
                return getattr(obj, real_fn_name)()
            elif fn_name.startswith('_'):
                # 函数的私有变量
                raise AttributeError
            else:
                # 懒加载执行的方法
                real_fn_name = '_{}__{}'.format(obj.__class__.__name__, fn_name)
                if is_function_valid(obj, real_fn_name):
                    bridge = FuncBridge(obj, real_fn_name)
                    obj.add_bridge(bridge)
                    return fn(obj, bridge)
        else:
            print("'@callable_register' decorate 'NLPShortcutCore' object")
            raise AttributeError

    return wrapper


def io(fn):
    """io装饰器

    函数执行会被放入单线程异步中执行
    实现使用asyncio的携程能力

    """

    @wraps(fn)
    def wrapper(obj, *args, **kwargs):
        global _io_deco_register
        _cpu_deco_register.append((obj.__class__.__name__, fn.__name__))
        # thread = ThreadWrapper(name=fn.__name__, target=fn, args=args, kwargs=kwargs, frequency=1)
        # thread.start()
    return wrapper


def cpu(fn):
    """cpu装饰器

    函数执行将被放入进程池中,默认开启进程数量为cpu数量
    """

    @wraps(fn)
    def wrapper(obj, *args, **kwargs):
        """装饰器外层会在声明期间运行，
        这样复制的内存空间中还没有将需要的包import完，
        放到执行期就不会有问题
        """

        global _cpu_deco_register
        _cpu_deco_register.append((obj.__class__.__name__, fn.__name__))

        # fork进程时要知道当前内存状态,
        # 可以使用dir()查看当前内存中间中都import了哪些包
        # 初始化进程池，进程个数为cpu个数
        global _process_pool
        if ProcessPoolWrapper.in_main() and not _process_pool:
            _process_pool = ProcessPoolWrapper()

        fn(obj)
        return _process_pool

    return wrapper


class producer(object):
    """生产者装饰器

    将返回值放入指定队列中
    """

    _queues = {}

    def __init__(self, topic, maxsize=0):
        self._process = None
        self._queue = None
        self._obj = None

        if topic in self._queues.keys():
            _queue= self._queues[topic]
        else:
            _manager = Manager()
            _queue = _manager.JoinableQueue(maxsize)

            self._queues[topic] = _queue
        self._queue = _queue
        setattr(producer, topic, self)

    def __call__(self, fn):

        @wraps(fn)
        def wrapper(obj, *args, **kwargs):
            self._obj = obj
            r = fn(obj, *args, **kwargs)

            def while_tasks_finished():
                ProcessPoolWrapper.processing()
                # 所有的任务都执行完后将控制每个函数的
                self.produce('__end__')
                time.sleep(0.1)
            # 等待所有任务被完成的线程
            ThreadWrapper('while_tasks_finished',
                          target=while_tasks_finished,
                          frequency=1).start()
            return r
        return wrapper

    def produce(self, task, *args, **kwargs):

        # 如果task不是一个可执行的任务,则直接将该结果行参数放入队列中
        if not hasattr(task, '__call__'):
            self._queue.put(task)
            return

        # 检测是否被cpu装饰器装饰过
        global _cpu_deco_register
        cls_name = self._obj.__class__.__name__
        fn_name = inspect.stack()[1][3]
        # 如果该函数使用cpu装饰，则将produce的内容放到进程池中进行
        if (cls_name, fn_name) in _cpu_deco_register:
            task_wrapper = TaskWrapper(queue=self._queue,
                                       task=task)
            _process_pool.apply_async_task(task_wrapper)
        elif (cls_name, fn_name) in _io_deco_register:
            return
        else:
            self._queue.put(task)
            return


class consumer(producer):
    """消费者装饰器

    改变执行期从指点队列中获取
    """

    def __init__(self, topic):
        super(consumer, self).__init__(topic)
        setattr(consumer, topic, self)

    def __call__(self, fn):

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper

    def consume(self):
        while True:
            try:
                v = self._queue.get(timeout=0.01)
                if v == '__end__':
                    break
                yield v
            except Empty:
                continue


class NLPShortcutCore(object):
    """继承该类的子类不能重载__setstate__和__getstate__，
    这些类在传递到multiprocess.Queue中时会进行pickle操作，
    操作不当会导致无法恢复传递时的值
    """

    def __init__(self):
        self.call_stack = []
        self.child_obj = None

    @callable_register
    def __getattr__(self, fn):
        return fn

    def add_bridge(self, bridge):
        self.call_stack.append(bridge)

    @staticmethod
    def iter_process(objs, fn_name, *args, **kwargs):
        real_fn_name = fn_name[5:]
        for obj in objs:
            getattr(obj, real_fn_name)(args, kwargs)

