"""
pairs.json structure:
{'text_pairs': {image_id: [[[x, y], [w1, w2, ... , wn]], ...],
 'max_sent_size': N,
}
"""
import argparse
import json
import os

import numpy as np
from PIL import Image
import progressbar as pb

from utils import tokenize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--map_xlen", default=16, type=int)
    parser.add_argument("--map_ylen", default=16, type=int)
    return parser.parse_args()


def annos2pairs(data_dir, map_xlen, map_ylen):
    vocab_path = os.path.join(data_dir, "vocab.json")
    annos_dir = os.path.join(data_dir, "annotations")
    images_dir = os.path.join(data_dir, "images")
    pairs_path = os.path.join(data_dir, "pairs.json")
    vocab = json.load(open(vocab_path, "rb"))
    anno_names = os.listdir(annos_dir)
    out_dict = {'text_pairs': {}}
    max_sent_size = 0
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(anno_names))
    pbar.start()
    for i, anno_name in enumerate(anno_names):
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, "rb"))
        image_name, _ = os.path.splitext(anno_name)
        image_id = os.path.splitext(image_name)[0]
        image_path = os.path.join(images_dir, image_name)
        image = Image.open(image_path, 'r')
        image_xlen, image_ylen = image.size
        pairs, cur_max_sent_size = _anno2pairs(vocab, anno, image_xlen, image_ylen, map_xlen, map_ylen)
        for key, d in out_dict.iteritems():
            d[image_id] = pairs[key]
        max_sent_size = max(max_sent_size, cur_max_sent_size)
        pbar.update(i)
    pbar.finish()
    out_dict['max_sent_size'] = max_sent_size

    print("dumping json ...")
    json.dump(out_dict, open(pairs_path, "wb"))


def _scale(x, y, ratio):
    return ratio * x, ratio * y


def _get_rect_center(rect):
    return np.mean(np.array(rect), 0)


def _anno2pairs(vocab_dict, anno, image_xlen, image_ylen, map_xlen, map_ylen):
    longer = 'x' if float(image_xlen)/image_ylen > float(map_xlen)/map_ylen else 'y'
    ratio = float(map_xlen)/image_xlen if longer == 'x' else float(map_ylen)/image_ylen

    text_list = []
    max_sent_size = 0
    for label, d in anno['text'].iteritems():
        rect = d['rectangle']
        text = d['value']
        x, y = _get_rect_center(rect)
        scaled = map(int, map(round, _scale(x, y, ratio)))
        words = tokenize(text)
        sent = [vocab_dict[word] if word in vocab_dict else 0 for word in words]
        text_list.append([scaled, sent])
        max_sent_size = max(max_sent_size, len(sent))

    m = {'text_pairs': text_list}
    return m, max_sent_size

if __name__ == "__main__":
    ARGS = get_args()
    annos2pairs(ARGS.data_dir, ARGS.map_xlen, ARGS.map_ylen)