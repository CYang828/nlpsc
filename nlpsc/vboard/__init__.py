# encoding:utf-8

import os

from .bottle import run, route, Bottle, static_file


class VBoard(Bottle):

    def __init__(self):
        super(VBoard, self).__init__()
        self.index_uri = None
        self.route('/js/<filename>', method='GET', callback=self.js)
        self.route('/css/<filename>', method='GET', callback=self.css)

    @staticmethod
    def js(filename):
        current_path = os.path.dirname(os.path.abspath(__file__))
        return static_file(filename, root=os.path.join(current_path, 'static/js'))

    @staticmethod
    def css(filename):
        current_path = os.path.dirname(os.path.abspath(__file__))
        return static_file(filename, root=os.path.join(current_path, 'static/css'))

    def serve(self, host='0.0.0.0', port=12306, debug=False):
        if self.index_uri:
            print('please open browser: http://{host}:{port}{index}'.format(host=host,
                                                                            port=port,
                                                                            index=self.index_uri))
        self.run(host=host, port=port, debug=debug)



