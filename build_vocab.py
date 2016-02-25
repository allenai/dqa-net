import argparse
from collections import defaultdict

import json
import os
import progressbar as pb

from utils import tokenize


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("target_dir")
    parser.add_argument("--min_count", type=int, default=5)
    return parser.parse_args()

def build_vocab(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    min_count = args.min_count
    vocab_path = os.path.join(target_dir, "vocab.json")
    questions_dir = os.path.join(data_dir, "questions")
    annos_dir = os.path.join(data_dir, "annotations")

    vocab_counter = defaultdict(int)
    anno_names = os.listdir(annos_dir)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(anno_names))
    pbar.start()
    for i, anno_name in enumerate(anno_names):
        if os.path.splitext(anno_name)[1] != ".json":
            pbar.update(i)
            continue
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, "rb"))
        for _, d in anno['text'].iteritems():
            text = d['value']
            for word in tokenize(text):
                vocab_counter[word] += 1
        pbar.update(i)
    pbar.finish()

    ques_names = os.listdir(questions_dir)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(ques_names))
    pbar.start()
    for i, ques_name in enumerate(ques_names):
        if os.path.splitext(ques_name)[1] != ".json":
            pbar.update(i)
            continue
        ques_path = os.path.join(questions_dir, ques_name)
        ques = json.load(open(ques_path, "rb"))
        for ques_text, d in ques['questions'].iteritems():
            for word in tokenize(ques_text): vocab_counter[word] += 1
            for choice in d['answerTexts']:
                for word in tokenize(choice): vocab_counter[word] += 1
        pbar.update(i)
    pbar.finish()

    vocab_list = zip(*sorted([pair for pair in vocab_counter.iteritems() if pair[1] > min_count],
                             key=lambda x: -x[1]))[0]

    vocab_dict = {word: idx+1 for idx, word in enumerate(sorted(vocab_list))}
    vocab_dict['UNK'] = 0
    print("vocab size: %d" % len(vocab_dict))
    json.dump(vocab_dict, open(vocab_path, "wb"))

if __name__ == "__main__":
    ARGS = get_args()
    build_vocab(ARGS)