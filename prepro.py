import argparse
import os
import json
from pprint import pprint

import progressbar as pb
import sys
import numpy as np

from utils import tokenize, vlup


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("target_dir")
    return parser.parse_args()


def _get_text(vocab_dict, anno, key):
    if key[0] == 'T':
        value = anno['text'][key]['value']
        return vlup(vocab_dict, tokenize(value))
    elif key[0] == 'O':
        if 'text' in anno['objects'][key]:
            new_key = anno['objects'][key]['text'][0]
            return _get_text(vocab_dict, anno, new_key)
    return []


def _get_center(anno, key):
    type_dict = {'T': 'text', 'A': 'arrows', 'B': 'blobs', 'H': 'arrowHeads', 'R': 'regions', 'O': 'objects'}
    poly_dict = {'T': 'rectangle', 'A': 'polygon', 'B': 'polygon', 'H': 'rectangle', 'R': 'polygon'}
    type_ = type_dict[key[0]]
    if type_ == 'objects':
        if 'blobs' in anno[type_][key] and len(anno[type_][key]['blobs']) > 0:
            new_key = anno[type_][key]['blobs'][0]
        elif 'text' in anno[type_][key]:
            new_key = anno[type_][key]['text'][0]
        else:
            raise Exception("%r" % anno)
        return _get_center(anno, new_key)
    shape = poly_dict[key[0]]
    poly = np.array(anno[type_][key][shape])
    center = map(int, map(round, np.mean(poly, 0)))
    return center


def _get_head_center(anno, arrow_key):
    if len(anno['arrows'][arrow_key]['arrowHeads']) == 0:
        return [0, 0]
    head_key = anno['arrows'][arrow_key]['arrowHeads'][0]
    return _get_center(anno, head_key)


def _get_1hot_vector(dim, idx):
    arr = [0] * dim
    arr[idx] = 1
    return arr


def prepro_annos(args):
    """
    for each annotation file,
    [{'type': one-hot-in-4-vector,
      'r0': rect,
      'r1': rect,
      'rh': rect,
      'ra': rect,
      't0': indexed_words,
      't1': indexed_words}]

    one-hot-in-4-vector: [intraLabel, intraRegionLabel, interLinkage, intraLinkage,] (arrowDescriptor, arrowHeadTail)
    :param args:
    :return:
    """
    data_dir = args.data_dir
    target_dir = args.target_dir
    vocab_path = os.path.join(data_dir, "vocab.json")
    vocab = json.load(open(vocab_path, "rb"))
    relations_path = os.path.join(target_dir, "relations.json")

    relations_dict = {}
    dim = 4
    hot_index_dict = {('intraObject', 'label', 'objectDescription'): 0,
                      ('intraObject', 'label', 'regionDescriptionNoArrow'): 1,
                      ('interObject', 'linkage', 'objectToObject'): 2,
                      ('intraObject', 'linkage', 'regionDescription'): 3,
                      ('intraObject', 'linkage', 'objectDescription'): 3,
                      ('intraObject', 'textLinkage', 'textDescription'): 3}

    annos_dir = os.path.join(data_dir, "annotations")
    anno_names = os.listdir(annos_dir)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(anno_names))
    pbar.start()
    for i, anno_name in enumerate(anno_names):
        image_name = os.path.splitext(anno_name)[0]
        image_id = os.path.splitext(image_name)[0]
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, "rb"))
        relations = []
        if 'relationships' not in anno:
            continue
        for rel_type, d in anno['relationships'].iteritems():
            for rel_subtype, dd in d.iteritems():
                if len(dd) == 0:
                    continue
                for rel_key, ddd in dd.iteritems():
                    category = ddd['category']
                    # FIXME : just choose one for now
                    origin_key = ddd['origin'][0]
                    dest_key = ddd['destination'][0]
                    origin_center = _get_center(anno, origin_key)
                    dest_center = _get_center(anno, dest_key)
                    if 'connector' in ddd:
                        arrow_key = ddd['connector'][0]
                        arrow_center = _get_center(anno, arrow_key)
                        head_center = _get_head_center(anno, arrow_key)
                    else:
                        arrow_center = [0, 0]
                        head_center = [0, 0]
                    idx = hot_index_dict[(rel_type, rel_subtype, category)]
                    # type_ = _get_1hot_vector(dim, idx)
                    type_ = idx
                    origin_text = _get_text(vocab, anno, origin_key)
                    dest_text = _get_text(vocab, anno, dest_key)
                    relation = dict(type=type_, r0=origin_center, r1=dest_center, rh=head_center, ra=arrow_center,
                                    t0=origin_text, t1=dest_text)
                    relations.append(relation)
        # TODO : arrow relations as well?
        relations_dict[image_id] = relations
        pbar.update(i)
    pbar.finish()

    print("dumping json file ... ")
    json.dump(relations_dict, open(relations_path, 'wb'))
    print("done")


def prepro_questions(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    questions_dir = os.path.join(data_dir, "questions")
    questions_path = os.path.join(target_dir, "questions.json")
    vocab_path = os.path.join(data_dir, "vocab.json")
    vocab = json.load(open(vocab_path, "rb"))

    questions_dict = {'sents': {},
                      'answers': {}}

    ques_names = os.listdir(questions_dir)
    question_id = 0
    max_sent_size = 0
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(ques_names))
    pbar.start()
    for i, ques_name in enumerate(ques_names):
        if os.path.splitext(ques_name)[1] != ".json":
            pbar.update(i)
            continue
        ques_path = os.path.join(questions_dir, ques_name)
        ques = json.load(open(ques_path, "rb"))
        for ques_text, d in ques['questions'].iteritems():
            ques_words = tokenize(ques_text)
            choice_wordss = [tokenize(choice) for choice in d['answerTexts']]
            sents = [vlup(vocab, ques_words + choice_words) for choice_words in choice_wordss]
            # TODO : one hot vector or index?
            questions_dict['answers'][str(question_id)] = d['correctAnswer']
            question_id += 1
            max_sent_size = max(max_sent_size, max(len(sent) for sent in sents))
        pbar.update(i)
    pbar.finish()
    questions_dict['max_sent_size'] = max_sent_size

    sys.stdout.write("number of questions: %d\n" % len(questions_dict['answers']))
    sys.stdout.write("max sent size: %d\n" % max_sent_size)
    sys.stdout.write("dumping json file ... ")
    json.dump(questions_dict, open(questions_path, "wb"))
    sys.stdout.write("done\n")

if __name__ == "__main__":
    ARGS = get_args()
    # prepro_questions(ARGS)
    prepro_annos(ARGS)
