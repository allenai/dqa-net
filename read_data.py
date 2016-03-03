import json
import os
from pprint import pprint

import numpy as np
import sys


class DataSet(object):
    def __init__(self, name, batch_size, data, idxs, include_leftover=False):
        self.name = name
        self.num_epochs_completed = 0
        self.idx_in_epoch = 0
        self.batch_size = batch_size
        self.data = data
        self.include_leftover = include_leftover
        self.idxs = idxs
        self.num_examples = len(idxs)
        self.num_batches = self.num_examples / self.batch_size + int(self.include_leftover)
        self.reset()

    def get_next_labeled_batch(self):
        assert self.has_next_batch(), "End of data, reset required."
        from_, to = self.idx_in_epoch, self.idx_in_epoch + self.batch_size
        if self.include_leftover and to > self.num_examples:
            to = self.num_examples
        cur_idxs = self.idxs[from_:to]
        batch = [[each[i] for i in cur_idxs] for each in self.data]
        self.idx_in_epoch += self.batch_size
        return batch

    def has_next_batch(self):
        if self.include_leftover:
            return self.idx_in_epoch < self.num_examples
        return self.idx_in_epoch + self.batch_size <= self.num_examples

    def complete_epoch(self):
        self.reset()
        self.num_epochs_completed += 1

    def reset(self):
        self.idx_in_epoch = 0
        np.random.shuffle(self.idxs)


def read_data(name, params, data_dir):
    sents_path = os.path.join(data_dir, "sents.json")
    relations_path = os.path.join(data_dir, "relations.json")
    answers_path = os.path.join(data_dir, "answers.json")
    vocab_path = os.path.join(data_dir, "vocab.json")
    meta_data_path = os.path.join(data_dir, "meta_data.json")

    sents_dict = json.load(open(sents_path, "rb"))
    relations_dict = json.load(open(relations_path, "rb"))
    answer_dict = json.load(open(answers_path, "rb"))

    batch_size = params.batch_size
    question_ids = sorted(sents_dict.keys())
    sentss = [sents_dict[id_] for id_ in question_ids]
    relationss = [relations_dict[id_] for id_ in question_ids]
    answers = [answer_dict[id_] for id_ in question_ids]
    data = [sentss, relationss, answers]
    idxs = range(len(question_ids))
    include_leftover = not params.train
    data_set = DataSet(name, batch_size, data, idxs, include_leftover=include_leftover)
    return data_set


if __name__ == "__main__":
    class Params(object): pass
    params = Params()
    params.batch_size = 2
    params.train = True
    ds = read_data('temp', params, sys.argv[1])
    for num_batches_completed in range(ds.num_batches):
        sents_batch, relations_batch, answer_batch = ds.get_next_labeled_batch()
    pprint(sents_batch)
    pprint(relations_batch)
    pprint(answer_batch)
