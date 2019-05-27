# encoding:utf-8

import abc
from collections import namedtuple

import numpy as np

from .padding import pad_batch_data
from ...transformer import Transformer
from ...tokenization.char_tokenizer import FullTokenizer
from ...util.python import convert_to_unicode


class ErnieBaseTransformer(Transformer):
    """BaseReader for classify and sequence labeling task"""
    def __init__(self,
                 vocab_path=None,
                 dataset=None,
                 label_map_config=None,
                 max_seq_len=512,
                 do_lower_case=True,
                 in_tokens=False,
                 random_seed=None):
        super(ErnieBaseTransformer, self).__init__(dataset=dataset, vocab_path=vocab_path, do_lower_case=do_lower_case,
                                                   random_seed=random_seed, in_tokens=in_tokens,
                                                   label_map_config=label_map_config)
        self.max_seq_len = max_seq_len
        self.tokenizer = FullTokenizer(self.vocab, do_lower_case=do_lower_case)

        # padding过程中使用的标签
        self.pad_id = self.vocab.vocab["[PAD]"]
        self.cls_id = self.vocab.vocab["[CLS]"]
        self.sep_id = self.vocab.vocab["[SEP]"]

    def document2example(self, document):
        if self.dataset:
            Example = namedtuple('Example', self.dataset.header)
        else:
            Example = namedtuple('Example', ['text_a', 'label'])
        return Example(document.text, label=document.label)

    def example2input(self, example):
        tokens_a = self.tokenizer.tokenize(example.text_a)
        # token_b是用来做与下一句相关的模型时使用的
        tokens_b = None
        if hasattr(example, 'text_b'):
            tokens_b = self.tokenizer.tokenize(example.text_b)

        # 截断操作，使模型的输入长度小雨max_seq_len
        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            self._truncate_seq_pair(tokens_a, tokens_b, self.max_seq_len - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > self.max_seq_len - 2:
                tokens_a = tokens_a[0:(self.max_seq_len - 2)]

        # The convention in BERT/ERNIE is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0     0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = []
        text_type_ids = []
        tokens.append("[CLS]")
        text_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            text_type_ids.append(0)
        tokens.append("[SEP]")
        text_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                text_type_ids.append(1)
            tokens.append("[SEP]")
            text_type_ids.append(1)

        token_ids = self.vocab.tokens2ids(tokens)
        position_ids = list(range(len(token_ids)))

        if self.label_map:
            label_id = self.label_map[example.label]
        else:
            label_id = example.label

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_id', 'qid'])

        qid = None
        if hasattr(example, 'qid'):
            qid = example.qid

        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_id=label_id,
            qid=qid)

        return record

    @abc.abstractmethod
    def _pad_batch_records(self, batch_records):
        pass

    def batch_inputs_generator(self, batch_size=1, epoch=1, shuffle=False):
        """return generator which yields batch data for pyreader"""

        def produce():
            for batch_records in self._batch_inputs_generator(batch_size, epoch, shuffle):
                print(batch_records)
                yield self._pad_batch_records(batch_records)
        return produce

    def document2input(self, document):
        model_input = self._document2input(document)
        return self._pad_batch_records([model_input])

    @staticmethod
    def _truncate_seq_pair(tokens_a, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_a) + len(tokens_b)
            if total_length <= max_length:
                break
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop()
            else:
                tokens_b.pop()


class ErnieClassifyTransformer(ErnieBaseTransformer):
    """ClassifyReader"""

    def document2example(self, document):
        if self.dataset:
            Example = namedtuple('Example', self.dataset.header)
            print(Example(text_a=document.text, label=document.label))
            return Example(text_a=document.text, label=document.label)
        else:
            Example = namedtuple('Example', ['text_a', 'label'])
            print(Example(text_a=document.text, label=0))
            return Example(text_a=document.text, label=0)

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_labels = [record.label_id for record in batch_records]
        batch_labels = np.array(batch_labels).astype("int64").reshape([-1, 1])
        print(batch_token_ids, batch_position_ids)
        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids, pad_idx=self.pad_id, return_input_mask=True, return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, batch_labels, seq_lens
        ]

        return return_list


class SequenceLabelTransformer(ErnieBaseTransformer):
    """SequenceLabelReader"""

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]
        batch_label_ids = [record.label_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, batch_seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)
        padded_label_ids = pad_batch_data(
            batch_label_ids, pad_idx=len(self.label_map) - 1)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, padded_label_ids, batch_seq_lens
        ]
        return return_list

    def _reseg_token_label(self, tokens, labels):
        assert len(tokens) == len(labels)
        ret_tokens = []
        ret_labels = []
        for token, label in zip(tokens, labels):
            sub_token = self.tokenizer.tokenize(token)
            if len(sub_token) == 0:
                continue
            ret_tokens.extend(sub_token)
            ret_labels.append(label)
            if len(sub_token) < 2:
                continue
            sub_label = label
            if label.startswith("B-"):
                sub_label = "I-" + label[2:]
            ret_labels.extend([sub_label] * (len(sub_token) - 1))

        assert len(ret_tokens) == len(ret_labels)
        return ret_tokens, ret_labels

    def example2input(self, example):
        tokens = convert_to_unicode(example.text_a).split(u" ")
        labels = convert_to_unicode(example.label).split(u"") if example.label else ['']
        tokens, labels = self._reseg_token_label(tokens, labels)

        if len(tokens) > self.max_seq_len - 2:
            tokens = tokens[0:(self.max_seq_len - 2)]
            labels = labels[0:(self.max_seq_len - 2)]

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        token_ids = self.vocab.tokens2ids(tokens)
        position_ids = list(range(len(token_ids)))
        text_type_ids = [0] * len(token_ids)
        no_entity_id = len(self.label_map) - 1
        labels = [label if label in self.label_map else u"O" for label in labels]
        label_ids = [no_entity_id] + [
            self.label_map[label] for label in labels
        ] + [no_entity_id]

        Record = namedtuple(
            'Record',
            ['token_ids', 'text_type_ids', 'position_ids', 'label_ids'])
        record = Record(
            token_ids=token_ids,
            text_type_ids=text_type_ids,
            position_ids=position_ids,
            label_ids=label_ids)
        return record


class ErnieExtractEmbeddingTransformer(ErnieBaseTransformer):
    """ExtractEmbeddingReader"""

    def _pad_batch_records(self, batch_records):
        batch_token_ids = [record.token_ids for record in batch_records]
        batch_text_type_ids = [record.text_type_ids for record in batch_records]
        batch_position_ids = [record.position_ids for record in batch_records]

        # padding
        padded_token_ids, input_mask, seq_lens = pad_batch_data(
            batch_token_ids,
            pad_idx=self.pad_id,
            return_input_mask=True,
            return_seq_lens=True)
        padded_text_type_ids = pad_batch_data(
            batch_text_type_ids, pad_idx=self.pad_id)
        padded_position_ids = pad_batch_data(
            batch_position_ids, pad_idx=self.pad_id)

        return_list = [
            padded_token_ids, padded_text_type_ids, padded_position_ids,
            input_mask, seq_lens
        ]

        return return_list

