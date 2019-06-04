# encoding:utf-8

from . import VBoard
from .bottle import get, post, view, request
from ...nlpsc.dataset import DatasetHeader


class DatasetVBoard(VBoard):
    """数据集"""

    def __init__(self, dataset):
        super(DatasetVBoard, self).__init__()
        self._dataset = dataset
        self._index = '/dataset'

    @get('/dataset')
    @view('dataset_template')
    def index(self):
        """首页渲染"""


        # 初始化工具栏区域
        if not self._dataset.header:
            header = DatasetHeader()
        else:
            header = self._dataset.header

        for field in header.iter_field():
            pass

        for label in header.iter_label():
            pass

        # 初始化文档区域
        index = request.query.index

        # 初始化标注区

        return dict()

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


