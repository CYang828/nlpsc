# encoding:utf-8

from .bottle import run, route, Bottle


class VBoard(Bottle):

    def __init__(self):
        super(VBoard, self).__init__()
        self.index_uri = None

    def serve(self, host='0.0.0.0', port=12306, debug=False):
        if self.index_uri:
            print('please open browser: http://{host}:{port}{index}'.format(host=host,
                                                                            port=port,
                                                                            index=self.index_uri))
        self.run(host=host, port=port, debug=debug)



