# encoding:utf-8

import json

from . import VBoard
from ..util.color import yield_color
from .bottle import get, post, view, request, template
# from ...nlpsc.dataset import DatasetHeader


class DatasetVBoard(VBoard):
    """数据集"""

    def __init__(self, dataset):
        self.index_url = '/dataset'
        super(DatasetVBoard, self).__init__()
        self.dataset = dataset
        self.route('/dataset', method='GET', callback=self.index)

    @staticmethod
    def _markpens(fields=None, labels=None):
        fields = fields if fields else []
        labels = labels if labels else []
        markpen_classes = []
        markpen_options = []
        markpen_items = []

        if fields or labels:
            for idx, v in enumerate(fields+['|']+labels+['|']):
                if v == '|':
                    markpen_items.append("'|'")
                else:
                    color = yield_color(idx)
                    markpen_name = 'markpen{number}'.format(number=idx)
                    markpen_class = ".{class_name}\n{{ background-color: {color}; }}".format(class_name=markpen_name,
                                                                                             color=color)
                    markpen_classes.append(markpen_class)
                    markpen_option = {'model': markpen_name, 'class': markpen_name,
                                      'title': v, 'color': color, 'type': 'marker'}
                    markpen_options.append(markpen_option)
                    markpen_items.append("'highlight:"+markpen_name+"'")
            return '\n'.join(markpen_classes), json.dumps(markpen_options), ','.join(markpen_items) + ','
        else:
            return '', '', ''

    def index(self):
        """首页渲染"""

        # # 初始化工具栏区域
        # if not self._dataset.header:
        #     header = DatasetHeader()
        # else:
        #     header = self._dataset.header
        #
        # for field in header.iter_field():
        #     pass
        #
        # for label in header.iter_label():
        #     pass
        #
        # # 初始化文档区域
        # index = request.query.index
        markpen_classes, markpen_options, markpen_items = self._markpens(fields=['主语', '宾语'],
                                                                         labels=['鱼类'])


        #
        # # 初始化标注区
        return template('dataset_template.html',
                        data='hello world',
                        markpen_classes=markpen_classes,
                        markpen_options=markpen_options,
                        markpen_items=markpen_items)

    @post('/dataset/field/add')
    def add_field(self):
        """增加field"""
        pass

    @post('/dataset/label/add')
    def add_label(self):
        """增加label"""
        pass

    @post('/dataset/field/mark')
    def mark_field(self):
        pass

    @post('/dataset/label/mark')
    def mark_label(self):
        pass

    @post('/dataset/document/confirm')
    def confirm(self):
        pass

# @get('/dataset')
# @view('dataset_template')
# def index():
#     return
