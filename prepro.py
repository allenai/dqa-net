import argparse
import os
import json

import progressbar as pb
import sys

from utils import tokenize, vlup


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("target_dir")
    return parser.parse_args()


def _get_text(vocab_dict, anno, key):
    if key[0] != 'T':
        return []
    value = anno['text'][key]['value']
    return vlup(vocab_dict, tokenize(value))


def _get_rect(anno, key):
    type_dict = {'T': 'text', 'A': 'arrows', 'B': 'blobs', 'H': 'arrowHeads'}
    type_ = type_dict[key[0]]
    return anno[type_][key]['rectangle']


def _get_head_rect(anno, arrow_key):
    if anno['arrows'][arrow_key]['headless']:
        return [0, 0, 0, 0]
    head_key = anno['arrow'][arrow_key]['arrowHeads'][0]
    return _get_rect(anno, head_key)


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
                      # FIXME : None will cause error, but to see what is there!
                      ('intraObject', 'linkage', None): 3}

    annos_dir = os.path.join(data_dir, "annotations")
    anno_names = os.listdir(annos_dir)
    for i, anno_name in enumerate(anno_names):
        image_name = os.path.splitext(anno_name)[0]
        image_id = os.path.splitext(image_name)[0]
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, "rb"))
        relations = []
        for rel_type, d in anno['relationships'].iteritems():
            for rel_subtype, dd in d.iteritems():
                category = dd['category']
                # FIXME : just choose one for now
                arrow_key = dd['connector'][0]
                origin_key = dd['origin'][0]
                dest_key = dd['destination'][0]
                arrow_rect = _get_rect(anno, arrow_key)
                head_rect = _get_head_rect(anno, arrow_key)
                origin_rect = _get_rect(anno, origin_key)
                dest_rect = _get_rect(anno, dest_key)
                idx = hot_index_dict[(rel_type, rel_subtype, category)]
                # type_ = _get_1hot_vector(dim, idx)
                type_ = idx
                origin_text = _get_text(vocab, anno, origin_key)
                dest_text = _get_text(vocab, anno, dest_key)
                relation = dict(type=type_, r0=origin_rect, r1=dest_rect, rh=head_rect, ra=arrow_rect,
                                t0=origin_text, t1=dest_text)
                relations.append(relation)
        # TODO : arrow relations as well?
        relations_dict[image_id] = relations

    sys.stdout.write("dumping json file ... ")
    json.dump(relations_dict, open(relations_path, 'wb'))
    sys.stdout.write("done\n")


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
