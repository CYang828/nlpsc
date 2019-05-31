# encoding:utf-8

import time
import asyncio
from asyncio import QueueEmpty

from .thread import ThreadWrapper
from ..gl import TIME_FAST_LOOP_TIMEOUT_SECOND



class AIOTaskWrapper(object):
    """异步处理任务使用的包装器"""

    def __init__(self, queue, task, args=(), kwargs={}):
        # 结果存储队列
        self._queue = queue
        self._task = task
        self._args = args
        self._kwargs = kwargs

    async def __call__(self):
        r = await self._task(*self._args, **self._kwargs)
        self._queue.put(r)
        return r

    def __str__(self):
        return '<AIOTaskWrapper {} {} {} {}>'.format(self._queue, self._task, self._args, self._kwargs)


class AIOPoolWrapper(object):

    queue = None
    _event = False
    _loop = None

    def __init__(self, size=100):
        self.size = size
        self._workers = []
        AIOPoolWrapper._loop = asyncio.new_event_loop()
        ThreadWrapper(name='aio workers thread', target=self._initialize_works, frequency=1).start()

    def _initialize_works(self):
        asyncio.set_event_loop(self._loop)
        AIOPoolWrapper.queue = asyncio.Queue()
        self._workers = [self._worker() for _ in range(self.size)]
        asyncio.get_event_loop().run_until_complete(asyncio.wait(self._workers))

    async def _worker(self):
        while not self._event:
            try:
                tp, task = AIOPoolWrapper.queue.get_nowait()
            except QueueEmpty:
                await asyncio.sleep(TIME_FAST_LOOP_TIMEOUT_SECOND)
                continue

            if tp == 'task':
                r = await task()
                self.queue.task_done()
                await asyncio.sleep(TIME_FAST_LOOP_TIMEOUT_SECOND)
                return r

    @staticmethod
    def apply_async_task(task):
        asyncio.run_coroutine_threadsafe(AIOPoolWrapper.queue.put(('task', task)), AIOPoolWrapper._loop)

    @staticmethod
    def processing():
        future = asyncio.run_coroutine_threadsafe(AIOPoolWrapper.queue.join(), AIOPoolWrapper._loop)
        while not future.done():
            time.sleep(0.1)

    @staticmethod
    def stop():
        time.sleep(1)
        AIOPoolWrapper._event = True

    @staticmethod
    def tasks_finished():
        return AIOPoolWrapper.queue.empty()




