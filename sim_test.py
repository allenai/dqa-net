import argparse
import json
import os

import itertools
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from utils import get_pbar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("first_dir")
    parser.add_argument("second_dir")
    return parser.parse_args()

def sim_test(args):
    first_dir = args.first_dir
    second_dir = args.second_dir
    first_sents_path = os.path.join(first_dir, "sents.json")
    second_sents_path = os.path.join(second_dir, "sents.json")
    vocab_path = os.path.join(first_dir, "vocab.json")
    vocab = json.load(open(vocab_path, 'r'))
    inv_vocab = {idx: word for word, idx in vocab.items()}
    first_sents = json.load(open(first_sents_path, "r"))
    second_sents = json.load(open(second_sents_path, "r"))
    diff_dict = defaultdict(int)
    pbar = get_pbar(len(first_sents)).start()
    i = 0
    for first_id, sents1 in first_sents.items():
        text1 = sent_to_text(inv_vocab, sents1[0])
        min_second_id, diff = min([[second_id, cdiff(sents1, sents2, len(vocab))] for second_id, sents2 in second_sents.items()],
                                  key=lambda x: x[1])
        text2 = sent_to_text(inv_vocab, second_sents[min_second_id][0])
        diff_dict[diff] += 1
        """
        if diff <= 3:
            print("%s, %s, %d" % (text1, text2, diff))
        """
        pbar.update(i)
        i += 1
    pbar.finish()
    json.dump(diff_dict, open("diff_dict.json", "w"))

def sent_to_text(vocab, sent):
    return " ".join(vocab[idx] for idx in sent)

def sent_to_bow(sent, l):
    out = np.zeros([l])
    for idx in sent:
        out[idx] = 1.0
    return out

def temp():
    a = {"0.0": 128, "1.0": 61, "2.0": 181, "3.0": 152, "4.0": 170, "5.0": 144, "6.0": 128, "7.0": 120, "8.0": 70, "9.0": 50, "10.0": 44, "11.0": 22, "12.0": 19, "13.0": 17, "14.0": 3, "15.0": 4, "16.0": 3, "18.0": 2, "22.0": 1, "24.0": 1, "27.0": 1}
    keys = map(int, a.keys())
    plt.plot(keys, [a[key] for key in keys])



def diff(sent1, sent2, l):
    return np.sum(np.abs(sent_to_bow(sent1, l) - sent_to_bow(sent2, l)))

def cdiff(sents1, sents2, l):
    return min(diff(sent1, sent2, l) for sent1, sent2 in itertools.product(sents1, sents2))

if __name__ == "__main__":
    ARGS = get_args()
    sim_test(ARGS)