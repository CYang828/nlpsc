# encoding:utf-8

from nlpsc.document import Document
from nlpsc.dataset import DatasetHeader, Dataset


class TestDataset(object):

    def test_dataset_header(self):
        header = DatasetHeader.from_str('F-no(int) F-text_a F-text_b L-label1(list) L-label2')
        print(header)

        for field, tp in header.iter_field():
            print(field, tp)

        for label, tp in header.iter_label():
            print(label, tp)

    def test_manual_dataset(self):
        dataset = Dataset(name='测试数据集')
        dataset.add_header('F-no(int) F-text_a F-text_b L-label1(list) L-label2')
        print(dataset.header)
        print(dataset.size)

    def test_load_dataset_from_abnormal_files(self):
        pass

    def test_load_dataset_from_dump_files(self):
        pass
