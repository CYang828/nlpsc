# coding:utf-8

"""进程模块"""

import os
import time
from queue import Empty
from multiprocessing import Process, cpu_count, JoinableQueue, Event, Manager


class ProcessTaskWrapper(object):
    """进程池处理任务使用的包装器"""

    def __init__(self, queue, task, args=(), kwargs={}):
        # 结果存储队列
        self._queue = queue
        self._task = task
        self._args = args
        self._kwargs = kwargs

    def __call__(self):
        r = self._task(*self._args, **self._kwargs)
        self._queue.put(r)
        return r

    def __str__(self):
        return '<ProcessTaskWrapper {} {} {} {}>'.format(self._queue, self._task, self._args, self._kwargs)


class ProcessWrapper(Process):

    proceed = Event()
    queue = JoinableQueue()

    def __init__(self):
        super(ProcessWrapper, self).__init__()

    def run(self):
        while not self.proceed.is_set():
            try:
                tp, task = self.queue.get(timeout=0.1)
                print('接收任务')
            except Empty:
                continue

            if tp == 'task':
                r = task()
                self.queue.task_done()
                print('执行完成', r)

            time.sleep(0.1)


class ProcessPoolWrapper(object):

    _pid = os.getppid()
    _processpool = []

    def __init__(self):
        if os.getppid() == self._pid:
            self.size = cpu_count()
            for _ in range(self.size):
                p = ProcessWrapper()
                self._processpool.append(p)
                p.start()
            print('process pool initialize - core num {}'.format(self.size))

    @staticmethod
    def apply_async_task(task):
        ProcessWrapper.queue.put_nowait(('task', task))

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




