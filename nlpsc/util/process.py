# coding:utf-8

"""进程模块"""

import os
import time

from queue import Empty
from multiprocessing import Process, cpu_count, JoinableQueue, Event, Manager, Value, Pool


class TaskWrapper(object):
    """进程池处理任务使用的包装器"""

    def __init__(self, queue, task, args=(), kwargs={}):
        self._queue = queue
        self._task = task
        self._args = args
        self._kwargs = kwargs

    def __call__(self, *args, **kwargs):
        r = self._task(*args, **kwargs)
        self._queue.put(r)
        return r


class ProcessWrapper(Process):

    proceed = Event()
    queue = JoinableQueue()

    def __init__(self):
        super(ProcessWrapper, self).__init__()

    def run(self):
        while not self.proceed.is_set():
            # print('在循环')
            try:
                tp, task, args, kwargs = self.queue.get(timeout=0.1)
            except Empty:
                continue

            if tp == 'task':
                print('执行', task)
                r = task()
                self.queue.task_done()
                print('执行完成', r)

            time.sleep(0.1)

        print('结束进程')


class ProcessPoolWrapper(object):

    _pid = os.getppid()
    print('主进程id', _pid)
    _processpool = []

    def __init__(self):
        print('当前进程id', os.getppid())
        if os.getppid() == self._pid:
            self.size = cpu_count()
            print('进程池初始化 - {}'.format(self.size))
            for _ in range(self.size):
                p = ProcessWrapper()
                self._processpool.append(p)
                p.start()

    @staticmethod
    def apply_async_task(task, args=(), kwargs={}):
        print('put queue start')
        ProcessWrapper.queue.put(('task',
                                  task,
                                  args,
                                  kwargs))
        print('put queue end')

    @staticmethod
    def tasks_finished():
        return ProcessWrapper.queue.empty()

    @staticmethod
    def in_main():
        if os.getppid() == ProcessPoolWrapper._pid:
            return True
        else:
            return False

    @staticmethod
    def processing():
        """阻塞等待任务队列中的所有任务处理完成"""
        ProcessWrapper.queue.join()

    @staticmethod
    def stop():
        """等待所有进程执行完成，并通知进程结束"""

        # join每个进程
        ProcessWrapper.proceed.set()




