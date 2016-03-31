import json
import os
from pprint import pprint

import h5py
import numpy as np
import sys

from configs.get_config import Config


class DataSet(object):
    def __init__(self, name, batch_size, data, idxs, idx2id):
        self.name = name
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.data = data
        self.idxs = idxs
        self.idx2id = idx2id
        self.num_examples = len(idxs)
        self.num_full_batches = int(self.num_examples / self.batch_size)
        self.num_all_batches = self.num_full_batches + int(self.num_examples % self.batch_size > 0)
        self.reset()

    def get_num_batches(self, partial=False):
        return self.num_all_batches if partial else self.num_full_batches

    def get_batch_idxs(self, partial=False):
        assert self.has_next_batch(partial=partial), "End of data, reset required."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if partial and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        return cur_idxs

    def get_next_labeled_batch(self, partial=False):
        cur_idxs = self.get_batch_idxs(partial=partial)
        batch = [[each[i] for i in cur_idxs] for each in self.data]
        self.idx_in_epoch += len(cur_idxs)
        return batch

    def has_next_batch(self, partial=False):
        if partial:
            return self.idx_in_epoch < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        np.random.shuffle(self.idxs)


def read_data(params, mode):
    print("loading {} data ... ".format(mode))
    data_dir = params.data_dir

    fold_path = params.fold_path
    fold = json.load(open(fold_path, 'r'))
    if mode in ['train', 'test']:
        cur_image_ids = fold[mode]
    elif mode == 'val':
        cur_image_ids = fold['test']
    else:
        raise Exception()

    sents_path = os.path.join(data_dir, "sents.json")
    facts_path = os.path.join(data_dir, "facts.json")
    answers_path = os.path.join(data_dir, "answers.json")
    images_path = os.path.join(data_dir, "images.h5")
    image_ids_path = os.path.join(data_dir, "image_ids.json")

    sentss_dict = json.load(open(sents_path, "r"))
    facts_dict = json.load(open(facts_path, "r"))
    answers_dict = json.load(open(answers_path, "r"))
    images_h5 = h5py.File(images_path, 'r')
    all_image_ids = json.load(open(image_ids_path, 'r'))
    image_id2idx = {id_: idx for idx, id_ in enumerate(all_image_ids)}

    batch_size = params.batch_size
    sentss, answers, factss, images = [], [], [], []
    idx = 0
    idx2id = []
    for image_id in cur_image_ids:
        if image_id not in sentss_dict or image_id not in facts_dict:
            continue
        facts = facts_dict[image_id]
        image = images_h5['data'][image_id2idx[image_id]]
        for sent_id, (sents, answer) in enumerate(zip(sentss_dict[image_id], answers_dict[image_id])):
            sentss.append(sents)
            answers.append(answer)
            factss.append(facts)
            images.append(image)
            idx2id.append([image_id, sent_id])
            idx += 1

    data = [sentss, factss, images, answers]
    idxs = np.arange(len(answers))
    data_set = DataSet(mode, batch_size, data, idxs, idx2id)
    print("done")
    return data_set


if __name__ == "__main__":
    params = Config()
    params.batch_size = 2
    params.train = True
