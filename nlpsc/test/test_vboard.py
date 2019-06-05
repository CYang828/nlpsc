# encoding:utf-8

from nlpsc.dataset import Dataset
from nlpsc.vboard.dataset import DatasetVBoard


class TestVBoard(object):

    def test_dataset_vboard(self):
        # from nlpsc.vboard.dataset import index
        from ..vboard import bottle
        bottle.TEMPLATE_PATH.append('../vboard/views/')

        dataset = Dataset(name='测试数据集')
        dataset.add_header('F-no(int) F-text_a F-text_b L-label1(list) L-label2')

        DatasetVBoard(dataset).serve()



