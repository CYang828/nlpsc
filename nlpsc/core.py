# encoding:utf-8

import time
import inspect
import collections
from queue import Empty
from functools import wraps

from .error import NLPSCError
from .util.thread import ThreadWrapper
from .util.process import ProcessPoolWrapper, Manager, ProcessTaskWrapper
from .util.aio import AIOPoolWrapper, AIOTaskWrapper


class DecoRegister(object):
    """装饰器注册器"""

    _register = []

    @classmethod
    def register(cls, cls_name, fn_name):
        cls._register.append((cls_name, fn_name))

    @classmethod
    def is_register(cls, cls_name, fn_name):
        if (cls_name, fn_name) in cls._register:
            return True
        else:
            return False


class DecoCpuRegister(DecoRegister):
    """cpu 装饰器注册器"""

    _register = []
    _pool = None

    @staticmethod
    def initialize_pool():
        if ProcessPoolWrapper.in_main() and not DecoCpuRegister._pool:
            DecoCpuRegister._pool = ProcessPoolWrapper()
        return DecoCpuRegister._pool

    @staticmethod
    def pool():
        if not DecoCpuRegister._pool:
            DecoCpuRegister.initialize_pool()
        return DecoCpuRegister._pool


class DecoAioRegister(DecoRegister):
    """io 装饰器注册器"""

    _register = []
    _pool = None

    @staticmethod
    def initialize_pool():
        if ProcessPoolWrapper.in_main() and not DecoAioRegister._pool:
            DecoAioRegister._pool = AIOPoolWrapper()
        return DecoAioRegister._pool

    @staticmethod
    def pool():
        if not DecoAioRegister._pool:
            DecoAioRegister.initialize_pool()
        return DecoAioRegister._pool


class DecoProducerRegister(DecoRegister):

    _register = []
    _topic_producers = {}
    _call_producers = collections.OrderedDict()

    @staticmethod
    def register_topic(topic, prod):
        DecoProducerRegister._topic_producers[topic] = prod

    @staticmethod
    def get_producer_by_topic(topic):
        return DecoProducerRegister._topic_producers[topic]

    @staticmethod
    def register_call(cls_name, fn_name, prod):
        DecoProducerRegister._call_producers[(cls_name, fn_name)] = prod

    @staticmethod
    def get_last_producer(cls_name, fn_name, last=0):
        calls = list(DecoProducerRegister._call_producers.keys())
        try:
            call_idx = calls.index((cls_name, fn_name))
        except ValueError:
            call_idx = -1

        call_name = calls[call_idx-last]
        return DecoProducerRegister._call_producers[call_name]


def wait(fn):
    """wait装饰器

    等待之前函数执行完成
    """

    @wraps(fn)
    def wrapper(obj, bridge, *args, **kwargs):
        for bri in bridge.nlpsc.chain.iter_previous_bridges(bridge):
            print(bri.waiting_thread.isAlive())
            while bri.waiting_thread.isAlive():
                time.sleep(0.1)
            print('previous', bri)
        print('停止等待')
        return fn(obj, *args, **kwargs)
    return wrapper


def aio(fn):
    """io装饰器

    函数执行会被放入单线程异步中执行
    实现使用asyncio的携程能力

    """

    @wraps(fn)
    def wrapper(obj, bridge, *args, **kwargs):
        DecoAioRegister.register(obj.__class__.__name__, fn.__name__)
        # 初始化协程任务处理池
        pool = DecoAioRegister.initialize_pool()
        fn(obj, bridge,  *args, **kwargs)
        return pool
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

        DecoCpuRegister.register(obj.__class__.__name__, fn.__name__)
        # fork进程时要知道当前内存状态,
        # 可以使用dir()查看当前内存中间中都import了哪些包
        # 初始化进程池，进程个数为cpu个数
        pool = DecoCpuRegister.initialize_pool()
        fn(obj, *args, **kwargs)
        return pool

    return wrapper


class producer(object):
    """生产者装饰器

    将返回值放入指定队列中
    """

    # key: topic
    # value: producer queue
    _queues = {}

    def __init__(self, topic=None, maxsize=0):
        self._process = None
        self.queue = None
        self.obj = None
        self.fn = None
        self.topic = topic

        # 使用同一个topic的装饰器只使用同一个queue
        if topic in self._queues.keys():
            _queue = self._queues[topic]
        else:
            _manager = Manager()
            _queue = _manager.Queue(maxsize)
            self._queues[topic] = _queue
        self.queue = _queue

    def __call__(self, fn):

        @wraps(fn)
        def wrapper(obj, bridge,  *args, **kwargs):
            self.obj = obj
            self.fn = fn
            cls_name = obj.__class__.__name__
            fn_name = fn.__name__

            # 将当前的producer对象放入注册器中
            DecoProducerRegister.register_topic(self.topic, self)
            DecoProducerRegister.register(cls_name, fn_name)
            DecoProducerRegister.register_call(cls_name, fn_name, self)

            r = fn(obj, *args, **kwargs)

            def while_tasks_finished():
                """该线程是用来通知当前生产者结束使用的"""
                if DecoCpuRegister.is_register(cls_name, fn_name):
                    ProcessPoolWrapper.processing()
                elif DecoAioRegister.is_register(cls_name, fn_name):
                    AIOPoolWrapper.processing()

                # 生产中的任务都结束后，在生产者队列中插入结束信号
                self.queue.put('__end__')
                time.sleep(0.1)

            if DecoCpuRegister.is_register(cls_name, fn_name) or \
                    DecoAioRegister.is_register(cls_name, fn_name):
                # 等待所有任务被完成的线程
                producer_wait_thread = ThreadWrapper('producer_wait_thread',
                                                     target=while_tasks_finished,
                                                     frequency=1)
                producer_wait_thread.start()
                # 将等待线程注册到bridge中
                bridge.waiting_thread = producer_wait_thread
                return producer_wait_thread
            else:
                # 没有aio和cpu装饰器，正常结束
                self.queue.put('__end__')
                return r

        return wrapper

    @staticmethod
    def _produce(prod, task, *args, **kwargs):
        # 如果task不是一个可执行的任务,则直接将该结果行参数放入队列中
        if not hasattr(task, '__call__'):
            prod.queue.put(task)
            return

        cls_name = prod.obj.__class__.__name__
        fn_name = prod.fn.__name__
        if DecoCpuRegister.is_register(cls_name, fn_name):
            task_wrapper = ProcessTaskWrapper(queue=prod.queue,
                                              task=task,
                                              args=args,
                                              kwargs=kwargs)
            print('cpu放入任务')
            DecoCpuRegister.pool().apply_async_task(task_wrapper)
        elif DecoAioRegister.is_register(cls_name, fn_name):
            task_wrapper = AIOTaskWrapper(queue=prod.queue,
                                          task=task,
                                          args=args,
                                          kwargs=kwargs)
            print('aio放入任务')
            DecoAioRegister.pool().apply_async_task(task_wrapper)
        else:
            print('normal task')
            prod.queue.put(task)

    @staticmethod
    def produce(cls_name, fn_name, task, *args, **kwargs):
        """向最后一个注册的producer中生产任务"""

        prod = DecoProducerRegister.get_last_producer(cls_name, fn_name)
        producer._produce(prod, task, *args, **kwargs)

    @staticmethod
    def produce_by_topic(topic, task, *args, **kwargs):
        """向指定topic中生产任务"""

        prod = DecoProducerRegister.get_producer_by_topic(topic)
        producer._produce(prod, task, *args, **kwargs)

    def collect(self):
        """收集队列中的结果信息"""
        pass


class consumer(producer):
    """消费者装饰器

    改变执行期从指点队列中获取
    """

    @staticmethod
    def _consume(prob):
        while True:
            try:
                v = prob.queue.get(timeout=0.01)
                if v == '__end__':
                    break
                yield v
            except Empty:
                continue

    @staticmethod
    def consume(cls_name, fn_name):
        """获取与其连接的生产者队列消费迭代器"""

        if DecoProducerRegister.is_register(cls_name, fn_name):
            prod = DecoProducerRegister.get_last_producer(cls_name, fn_name, 1)
        else:
            prod = DecoProducerRegister.get_last_producer(cls_name, fn_name, 0)
        return consumer._consume(prod)

    @staticmethod
    def consume_by_topic(topic):
        """根据topic的名称来获取队列消费迭代器"""

        prod = DecoProducerRegister.get_producer_by_topic(topic)
        return consumer._consume(prod)


class FuncBridge(object):
    """调用方法时，实际在使用该类进行调用的方法的注册"""

    unique_obj_cache = {}

    def __init__(self, calling, fn_name, fn_desc=None):
        self.nlpsc = None
        self._calling = calling
        self._fn_name = fn_name
        self._fn_args = ()
        self._fn_kwargs = {}
        self._fn_desc = fn_desc if fn_desc else '{} processing'.format(self._fn_name)
        fn_return_type = getattr(self._calling, self._fn_name).__annotations__.get('return')
        fn_return_type_name = fn_return_type.__class__.__name__

        self._fn_return_obj = None
        if fn_return_type_name not in self.unique_obj_cache.keys():
            if fn_return_type:
                self._fn_return_obj = fn_return_type()
                self.unique_obj_cache[fn_return_type_name] = self._fn_return_obj
        else:
            self._fn_return_obj = self.unique_obj_cache[fn_return_type_name]

        self._waiting_thread = None

    def __call__(self, *args, **kwargs):
        self._fn_args = args
        self._fn_kwargs = kwargs
        return self._fn_return_obj

    def __str__(self):
        return '<FuncBridge {} {} {} {}>'.format(self._calling, self._fn_name, self._fn_args, self._fn_kwargs)

    def call(self, add_bridge=True):
        if add_bridge:
            return getattr(self._calling, self._fn_name)(self, *self._fn_args, **self._fn_kwargs)
        else:
            return getattr(self._calling, self._fn_name)(*self._fn_args, **self._fn_kwargs)

    @property
    def return_obj(self):
        return self._fn_return_obj

    @property
    def cls_name(self):
        return self._calling.__class__.__name__

    @property
    def fn_name(self):
        return self._fn_name

    @property
    def waiting_thread(self):
        return self._waiting_thread

    @waiting_thread.setter
    def waiting_thread(self, thread):
        self._waiting_thread = thread


def _sugar_lazy_call(fn_name):
    """懒加载方法转调用规则"""

    return 'lazy_{}'.format(fn_name)


def _sugar_magic_call(cls_name, fn_name):
    """系统内建方法转调用规则"""

    return '_{}{}'.format(cls_name, fn_name)


def callable_register(fn):
    """函数调用注册器
    用于把用户调用的方法转换成bridge，lazy_call时使用
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
            # 如果被调用的方法在定义类中已经存在,则不需要进行方法名的转换
            # 调用框架定义方法
            if fn_name.startswith('__'):
                # 系统内置方法和框架内置方法，立即执行
                real_fn_name = _sugar_magic_call(obj.__class__.__name__, fn_name)
                return getattr(obj, real_fn_name)()
            else:
                # 懒加载执行的方法
                real_fn_name = _sugar_lazy_call(fn_name)
                if is_function_valid(obj, real_fn_name):
                    bridge = FuncBridge(obj, real_fn_name)
                    obj.add_bridge(bridge)
                    return bridge
        else:
            print("'@callable_register' decorate 'NLPShortcutCore' object")
            raise AttributeError

    return wrapper


class _BridgeChain(object):

    def __init__(self):
        self._chains = collections.OrderedDict()

    def register(self, bridge):
        self._chains[bridge.fn_name] = bridge

    def get_last_bridge(self, bridge, last=1):
        pass

    def get_next_bridge(self, bridge, nex=1):
        pass

    def iter_previous_bridges(self, bridge):
        """返回当前bridge之前bridge的迭代器"""

        ks = list(self._chains.keys())
        p = ks.index(bridge.fn_name)
        for k in ks[:p]:
            yield self._chains[k]

    def iter(self):
        for bridge in self._chains.values():
            yield bridge


class NLPShortcutCore(object):
    """继承该类的子类不能重载__setstate__和__getstate__，
    这些类在传递到multiprocess.Queue中时会进行pickle操作，
    操作不当会导致无法恢复传递时的值
    """

    def __init__(self):
        self.call_stack = []
        self.child_obj = None
        self.chain = _BridgeChain()

    @callable_register
    def __getattr__(self, fn):
        return fn

    def deepdive(self, obj=None):
        """深入查看当前的调用链，返回的调用链都是bridge

        如果obj为空，则从当前对象开始遍历"""

        if not obj:
            obj = self

        if isinstance(obj, NLPShortcutCore):
            for bridge in obj.call_stack:
                self.chain.register(bridge)
                yield bridge

            for bridge in obj.call_stack:
                if bridge.return_obj and bridge.return_obj is not obj:

                    yield from self.deepdive(bridge.return_obj)
        else:
            print("'deepdive' dive in 'NLPShortcutCore' object")
            raise NLPSCError

    def add_bridge(self, bridge):
        self.call_stack.append(bridge)

    def produce(self, task, *args, **kwargs):
        cls_name = self.__class__.__name__
        fn_name = inspect.stack()[1].function
        producer.produce(cls_name, fn_name, task, *args, **kwargs)

    def consume(self):
        cls_name = self.__class__.__name__
        fn_name = inspect.stack()[1].function
        return consumer.consume(cls_name, fn_name)

    def iter_bridge(self):
        yield from self.chain.iter()

    @staticmethod
    def iter_process(objs, fn_name, *args, **kwargs):
        real_fn_name = fn_name[5:]
        for obj in objs:
            getattr(obj, real_fn_name)(args, kwargs)

