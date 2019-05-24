# encoding:utf-8

import numpy as np
from itertools import chain
from collections import namedtuple

from ..util.python import to_str
from ..model import PaddleInferModel
from ..util.file import get_default_path
from ..representation.ernie.util import split_text


class PaddleLACInferModel(PaddleInferModel):
    """lac 语言分析模型

    超过max_seq_len的文本会被截断"""

    def __init__(self, use_gpu=False, max_seq_len=128,
                 do_lower_case=True, random_seed=0, pretrained_model=None):
        super(PaddleLACInferModel, self).__init__(use_gpu)
        self._vocab_path = get_default_path('ernie/vocab.txt')
        self._label_map_config = get_default_path('ernie/label_map.json')
        self._random_seed = random_seed
        self._max_seq_len = max_seq_len
        self._do_lower_case = do_lower_case
        from ..representation.ernie.transformer import SequenceLabelTransformer
        self.transformer = SequenceLabelTransformer(vocab_path=self._vocab_path,
                                                    label_map_config=self._label_map_config,
                                                    max_seq_len=self._max_seq_len,
                                                    do_lower_case=self._do_lower_case,
                                                    in_tokens=False,
                                                    random_seed=self._random_seed)
        self._load_dataset()
        pretrained_path = pretrained_model if pretrained_model\
            else get_default_path('pretrained-models/lac-inference/')
        self.load_inference(pretrained_path)

    def _load_dataset(self):
        id2word_dict = dict([(str(word_id), word) for word, word_id in self._reader.vocab.items()])
        id2label_dict = dict([(str(label_id), label) for label, label_id in self._reader.label_map.items()])
        Dataset = namedtuple("Dataset", ["id2word_dict", "id2label_dict"])
        self._dataset = Dataset(id2word_dict, id2label_dict)

    def infer(self, texts, segment=False):
        """推断text的成分

        :argument
            texts: 文本列表
        :return
            results
        """

        from ..dataset import Dataset
        from ..document import Document
        dataset = Dataset()
        for text in texts:
            dataset.add(Document(text))
        self.transformer.dataset = dataset
        erine_input = self.transformer.example2input(texts)
        words, crf_decode = self._exe.run(self._inference_program,
                                          feed={self._feed_target_names[0]: erine_input[0],
                                                self._feed_target_names[1]: erine_input[1],
                                                self._feed_target_names[2]: erine_input[2],
                                                self._feed_target_names[3]: erine_input[3],
                                                self._feed_target_names[4]: erine_input[5]},
                                          fetch_list=self._fetch_targets,
                                          return_numpy=False)
        results = self._parse_result(words, crf_decode, self._dataset)

        if segment:
            return self.segment(results)
        else:
            return results

    @staticmethod
    def _parse_result(words, crf_decode, dataset):
        """解析结果为明文"""
        offset_list = (crf_decode.lod())[0]
        words = np.array(words)
        crf_decode = np.array(crf_decode)
        batch_size = len(offset_list) - 1
        batch_out_str = []
        for sent_index in range(batch_size):
            sent_out_str = ""
            sent_len = offset_list[sent_index + 1] - offset_list[sent_index]
            for tag_index in range(sent_len):
                index = tag_index + offset_list[sent_index]
                cur_word_id = str(words[index][0])
                cur_tag_id = str(crf_decode[index][0])
                cur_word = dataset.id2word_dict[cur_word_id]
                cur_tag = dataset.id2label_dict[cur_tag_id]
                sent_out_str += cur_word + u"/" + cur_tag + u" "
            sent_out_str = to_str(sent_out_str.strip())
            batch_out_str.append(sent_out_str)
        return batch_out_str

    @staticmethod
    def segment(results):
        for result in results:
            char_stack = []
            tokens = []
            chars = result.split(' ')

            for char in chars:
                pslice = char.find('/')
                word = char[:pslice]
                tag = char[pslice+1:]
                pturning = tag.find('-')
                tag_head = tag[:pturning]
                tag_tail = tag[pturning + 1:]

                if word in ('[CLS]', '[UNK]'):
                    continue

                if len(char_stack) > 0 and (tag_head != char_stack[-1][1]):
                    token = ''.join([char[0].strip('#') for char in char_stack])
                    char_stack.clear()
                    tokens.append(token)
                char_stack.append((word, tag_head, tag_tail))
            yield tokens

    def cut(self, text):
        """分词

        如果文本超过max_seq_len, 文本会被切分成多段文本进行分词"""

        texts = split_text(text, self._max_seq_len)
        return list(chain.from_iterable(list(self.infer(texts, segment=True))))
