# encoding:utf-8

from threading import Thread, Event


class ThreadWrapper(Thread):
    """封装系统thread，方便线程停止"""

    _threads = []

    def __init__(self, name, target, args=(), kwargs={}, period=0, frequency=-1):
        """初始化线程

        :parameter:
         - `name`: 线程名
         - `target`: 任务函数,线程名会作为参数传递给task
         - `period`: 执行时间间隔
         - `frequency`： 执行次数，-1为永远执行，默认为永远执行
         - `args` `kwargs`: 函数执行参数
        """

        self._event = Event()
        self._period = period
        self._frequency = frequency
        self._threads.append(self)
        super(ThreadWrapper, self).__init__(target=target, name=name, args=args, kwargs=kwargs)

    def __str__(self):
        return '<ThreadWrapper | {name}>'.format(name=self.getName())

    def run(self):
        """运行函数,可以通过start开始线程,该函数会被自动调用"""
        while not self._event.isSet() and self._frequency:
            self._event.wait(self._period)
            self._target(*self._args, **self._kwargs)
            self._frequency -= 1

    def join(self, timeout=0):
        """结束当前线程"""

        self._event.set()
        Thread.join(self, timeout)

    @classmethod
    def stop(cls, timeout=None):
        """等待所有线程执行完毕并结束线程"""

        for thread in cls._threads:
            thread.join(timeout)
