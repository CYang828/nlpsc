# encoding:utf-8

from .bottle import run


class VBoard(object):

    def __init__(self):
        self._start = False
        self._index = None

    def watch(self, host='0.0.0.0', port=12306):
        if self._index:
            print('please open browser: http://{host}:{port}{index}'.format(host=host,
                                                                            port=port,
                                                                            index=self._index))

        if not self._start:
            run(host=host, port=port)
