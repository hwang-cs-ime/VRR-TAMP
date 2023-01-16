import sys
import cv2
import torch
import random
import copy
import numpy as np
import os.path as osp
import torch.utils.data as data

sys.path.append('.')
import utils
import re

from pytorch_pretrained_bert.tokenization import BertTokenizer
from utils.transforms import letterbox

sys.modules['utils'] = utils
cv2.setNumThreads(0)


def read_examples(input_line, unique_id):
    """Read a list of `InputExample`s from an input file."""
    examples = []
    line = input_line  # reader.readline()
    line = line.strip()
    text_a = None
    text_b = None
    m = re.match(r"^(.*) \|\|\| (.*)$", line)
    if m is None:
        text_a = line
    else:
        text_a = m.group(1)
        text_b = m.group(2)
    examples.append(
        InputExample(unique_id=unique_id, text_a=text_a, text_b=text_b))
    return examples


# Bert text encoding
class InputExample(object):
    def __init__(self, unique_id, text_a, text_b):
        self.unique_id = unique_id
        self.text_a = text_a
        self.text_b = text_b


class InputFeatures(object):
    """
    A single set of features of data.
    """
    def __init__(self, unique_id, tokens, input_ids, input_mask, input_type_ids):
        self.unique_id = unique_id
        self.tokens = tokens
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.input_type_ids = input_type_ids


def convert_examples_to_features(examples, seq_length, tokenizer):
    """
    Loads a data file into a list of `InputBatch`s.
    """
    features = []
    for (ex_index, example) in enumerate(examples):
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)

        if tokens_b:
            # Modifies `tokens_a` and `tokens_b` in place so that the total length is less than the specified length. Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > seq_length - 2:
                tokens_a = tokens_a[0:(seq_length - 2)]
        tokens = []
        input_type_ids = []
        tokens.append("[CLS]")
        input_type_ids.append(0)
        for token in tokens_a:
            tokens.append(token)
            input_type_ids.append(0)
        tokens.append("[SEP]")
        input_type_ids.append(0)

        if tokens_b:
            for token in tokens_b:
                tokens.append(token)
                input_type_ids.append(1)
            tokens.append("[SEP]")
            input_type_ids.append(1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        while len(input_ids) < seq_length:
            input_ids.append(0)
            input_mask.append(0)
            input_type_ids.append(0)

        assert len(input_ids) == seq_length
        assert len(input_mask) == seq_length
        assert len(input_type_ids) == seq_length
        features.append(
            InputFeatures(
                unique_id=example.unique_id,
                tokens=tokens,
                input_ids=input_ids,
                input_mask=input_mask,
                input_type_ids=input_type_ids))
    return features


class DatasetNotFoundError(Exception):
    pass


class VGDataset(data.Dataset):

    def __init__(self, data_root, dataset='vrd', imsize=640, transform=None, split='train', max_query_len=10, bert_model='bert-base-uncased'):
        self.images = []
        self.data_root = data_root
        self.dataset = dataset
        self.imsize = imsize
        self.query_len = max_query_len
        self.transform = transform
        self.split = split
        self.tokenizer = BertTokenizer.from_pretrained(bert_model, do_lower_case=True)
        if self.dataset == 'vrd':
            self.dataset_root = osp.join(self.data_root, 'vrd')
            # directory of all images
            self.im_dir = osp.join(self.dataset_root, '{}_images'.format(split))
            # annotation file
            annotation_file = 'vrd_{}.pt'.format(split)

        elif self.dataset == 'vg':
            self.dataset_root = osp.join(self.data_root, 'vg')
            self.im_dir = osp.join(self.dataset_root, 'images')
            annotation_file = 'vg_{}.pt'.format(split)

        annotation_path = osp.join(self.dataset_root, annotation_file)
        self.images += torch.load(annotation_path)

    def pull_item(self, idx):
        img_file, _, sub_bbox, _, obj_bbox, relationship = self.images[idx]
        # box format: to x1, y1, x2, y2

        sub_bbox = np.array(sub_bbox, dtype=int)  # x y x y
        obj_bbox = np.array(obj_bbox, dtype=int)  # x y x y

        img_path = osp.join(self.im_dir, img_file)
        img = cv2.imread(img_path)

        if img.shape[-1] > 1:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.stack([img] * 3)
        return img, relationship, sub_bbox, obj_bbox

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, relationship, sub_bbox, obj_bbox = self.pull_item(idx)
        relationship = relationship.lower()

        mask = np.zeros_like(img)
        # should be inference, or specified training
        img, mask, ratio, dw, dh = letterbox(img, mask, self.imsize)
        sub_bbox[0], sub_bbox[2] = sub_bbox[0] * ratio + dw, sub_bbox[2] * ratio + dw
        sub_bbox[1], sub_bbox[3] = sub_bbox[1] * ratio + dh, sub_bbox[3] * ratio + dh

        obj_bbox[0], obj_bbox[2] = obj_bbox[0] * ratio + dw, obj_bbox[2] * ratio + dw
        obj_bbox[1], obj_bbox[3] = obj_bbox[1] * ratio + dh, obj_bbox[3] * ratio + dh

        # Norm, to tensor
        if self.transform is not None:
            img = self.transform(img)
        # encode relationship to bert input
        examples = read_examples(relationship, idx)
        features = convert_examples_to_features(examples=examples, seq_length=self.query_len, tokenizer=self.tokenizer)
        word_id = features[0].input_ids
        word_mask = features[0].input_mask
        return img, mask, np.array(word_id, dtype=int), np.array(word_mask, dtype=int), np.array(sub_bbox, dtype=np.float32), np.array(obj_bbox, dtype=np.float32)
