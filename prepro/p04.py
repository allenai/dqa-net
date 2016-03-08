import argparse
import os
import json
import shutil
from collections import defaultdict
import re
import random

import h5py
import numpy as np

# from qa2hypo import qa2hypo
from utils import get_pbar



def qa2hypo(question, answer):
    return "%s %s" % (question, answer)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("target_dir")
    parser.add_argument("glove_path")
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--vgg_model_path", default="~/caffe-models/vgg-19.caffemodel")
    parser.add_argument("--vgg_proto_path", default="~/caffe-models/vgg-19.prototxt")
    return parser.parse_args()

def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    return tokens


def _vadd(vocab_counter, word):
    word = word.lower()
    vocab_counter[word] += 1


def _vget(vocab_dict, word):
    word = word.lower()
    if word in vocab_dict:
        return vocab_dict[word]
    else:
        keys = list(vocab_dict.keys())
        key = random.choice(keys)
        return vocab_dict[key]


def _vlup(vocab_dict, words):
    return [_vget(vocab_dict, word) for word in words]


def _get_text(vocab_dict, anno, key):
    if key[0] == 'T':
        value = anno['text'][key]['value']
        repText = anno['text'][key]['replacementText']
        return _vlup(vocab_dict, _tokenize(value)), _vlup(vocab_dict, _tokenize(repText))
    elif key[0] == 'O':
        if 'text' in anno['objects'][key]:
            if len(anno['objects'][key]['text']) > 0:
                new_key = anno['objects'][key]['text'][0]
                return _get_text(vocab_dict, anno, new_key)
    return [], []


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
    center = list(map(int, map(round, np.mean(poly, 0))))
    return center


def _get_head_center(anno, arrow_key):
    if 'arrowHeads' not in anno['arrows'][arrow_key] or len(anno['arrows'][arrow_key]['arrowHeads']) == 0:
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
    [{'type': type_num,
      'r0': rect,
      'r1': rect,
      'rh': rect,
      'ra': rect,
      't0': indexed_words,
      't1': indexed_words}]

    type_num: [intraLabel, intraRegionLabel, interLinkage, intraLinkage,] (arrowDescriptor, arrowHeadTail)
    :param args:
    :return:
    """
    data_dir = args.data_dir
    target_dir = args.target_dir
    vocab_path = os.path.join(target_dir, "vocab.json")
    vocab = json.load(open(vocab_path, "r"))
    relations_path = os.path.join(target_dir, "relations.json")
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_path, "r"))

    meta_data['pred_size'] = 2 * 4

    relations_dict = {}
    dim = 4
    hot_index_dict = {('intraObject', 'label', 'objectDescription'): 0,
                      ('intraObject', 'label', 'regionDescriptionNoArrow'): 1,
                      ('interObject', 'linkage', 'objectToObject'): 2,
                      ('intraObject', 'linkage', 'regionDescription'): 3,
                      ('intraObject', 'linkage', 'objectDescription'): 3,
                      ('intraObject', 'textLinkage', 'textDescription'): 3}

    annos_dir = os.path.join(data_dir, "annotations")
    anno_names = [name for name in os.listdir(annos_dir) if name.endswith(".json")]
    max_label_size = 0
    max_num_rels = 0
    pbar = get_pbar(len(anno_names))
    pbar.start()
    for i, anno_name in enumerate(anno_names):
        image_name, _ = os.path.splitext(anno_name)
        image_id, _ = os.path.splitext(image_name)
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, "r", encoding="ISO-8859-1"))
        relations = []
        if 'relationships' not in anno:
            relations_dict[image_id] = relations
            pbar.update(i)
            continue
        for rel_type, d in anno['relationships'].items():
            for rel_subtype, dd in d.items():
                if len(dd) == 0:
                    continue
                for rel_key, ddd in dd.items():
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
                    origin_text, origin_rep = _get_text(vocab, anno, origin_key)
                    dest_text, dest_rep = _get_text(vocab, anno, dest_key)
                    max_label_size = max(max_label_size, len(origin_text), len(dest_text))
                    # relation = dict(type=type_, l0=origin_center, l1=dest_center, lh=head_center, la=arrow_center, t0=origin_text, t1=dest_text)
                    pred = origin_center + dest_center + head_center + arrow_center
                    assert len(pred) == meta_data['pred_size'], "Wrong predicate size: %d" % len(pred)
                    relation = dict(a1=origin_text, pred=pred, a2=dest_text, a1r=origin_rep, a2r=dest_rep)
                    relations.append(relation)
        # TODO : arrow relations as well?
        relations_dict[image_id] = relations
        max_num_rels = max(max_num_rels, len(relations))
        pbar.update(i)
    pbar.finish()
    meta_data['max_label_size'] = max_label_size
    meta_data['max_num_rels'] = max_num_rels

    print("number of relations: %d" % sum(len(relations) for relations in relations_dict))
    print('max label size: %d' % max_label_size)
    print("max num rels: %d" % max_num_rels)
    print("dumping json file ... ", end="", flush=True)
    json.dump(relations_dict, open(relations_path, 'w'))
    json.dump(meta_data, open(meta_data_path, 'w'))
    print("done")


def prepro_questions(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    questions_dir = os.path.join(data_dir, "questions")
    sents_path = os.path.join(target_dir, "sents.json")
    answers_path = os.path.join(target_dir, "answers.json")
    vocab_path = os.path.join(target_dir, "vocab.json")
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    vocab = json.load(open(vocab_path, "r"))
    meta_data = json.load(open(meta_data_path, "r"))

    sentss_dict = {}
    answers_dict = {}

    ques_names = sorted([name for name in os.listdir(questions_dir) if os.path.splitext(name)[1].endswith(".json")],
                        key=lambda x: int(os.path.splitext(os.path.splitext(x)[0])[0]))
    max_sent_size = 0
    num_choices = 0
    num_questions = 0
    pbar = get_pbar(len(ques_names)).start()
    for i, ques_name in enumerate(ques_names):
        image_name, _ = os.path.splitext(ques_name)
        image_id, _ = os.path.splitext(image_name)
        sentss = []
        answers = []
        ques_path = os.path.join(questions_dir, ques_name)
        ques = json.load(open(ques_path, "r"))
        for ques_id, (ques_text, d) in enumerate(ques['questions'].items()):
            if d['abcLabel']:
                continue
            sents = [_vlup(vocab, _tokenize(qa2hypo(ques_text, choice))) for choice in d['answerTexts']]
            assert not num_choices or num_choices == len(sents), "number of choices don't match: %s" % ques_name
            num_choices = len(sents)
            # TODO : one hot vector or index?
            sentss.append(sents)
            answers.append(d['correctAnswer'])
            max_sent_size = max(max_sent_size, max(len(sent) for sent in sents))
            num_questions += 1
        sentss_dict[image_id] = sentss
        answers_dict[image_id] = answers
        pbar.update(i)
    pbar.finish()
    meta_data['max_sent_size'] = max_sent_size
    meta_data['num_choices'] = num_choices

    print("number of questions: %d" % num_questions)
    print("number of choices: %d" % num_choices)
    print("max sent size: %d" % max_sent_size)
    print("dumping json file ... ", end="", flush=True)
    json.dump(sentss_dict, open(sents_path, "w"))
    json.dump(answers_dict, open(answers_path, "w"))
    json.dump(meta_data, open(meta_data_path, "w"))
    print("done")


def build_vocab(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    min_count = args.min_count
    vocab_path = os.path.join(target_dir, "vocab.json")
    emb_mat_path = os.path.join(target_dir, "emb_mat.h5")
    questions_dir = os.path.join(data_dir, "questions")
    annos_dir = os.path.join(data_dir, "annotations")
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_path, 'r'))
    glove_path = args.glove_path

    word_counter = defaultdict(int)
    anno_names = os.listdir(annos_dir)
    pbar = get_pbar(len(anno_names))
    pbar.start()
    for i, anno_name in enumerate(anno_names):
        if os.path.splitext(anno_name)[1] != ".json":
            pbar.update(i)
            continue
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, "r"))
        for _, d in anno['text'].items():
            text = d['value']
            for word in _tokenize(text):
                _vadd(word_counter, word)
        pbar.update(i)
    pbar.finish()

    ques_names = os.listdir(questions_dir)
    pbar = get_pbar(len(ques_names))
    pbar.start()
    for i, ques_name in enumerate(ques_names):
        if os.path.splitext(ques_name)[1] != ".json":
            pbar.update(i)
            continue
        ques_path = os.path.join(questions_dir, ques_name)
        ques = json.load(open(ques_path, "r"))
        for ques_text, d in ques['questions'].items():
            for word in _tokenize(ques_text):
                _vadd(word_counter, word)
            for choice in d['answerTexts']:
                for word in _tokenize(choice):
                    _vadd(word_counter, word)
        pbar.update(i)
    pbar.finish()

    word_list, counts = zip(*sorted([pair for pair in word_counter.items() if pair[1] > min_count],
                             key=lambda x: -x[1]))
    freq = 5
    print("top %d frequent words:" % freq)
    for word, count in zip(word_list[:freq], counts[:freq]):
        print("%r: %d" % (word, count))

    features = {}
    word_size = 0
    print("reading %s ... " % glove_path, end="", flush=True)
    with open(glove_path, 'r') as fp:
        for line in fp:
            array = line.lstrip().rstrip().split(" ")
            word = array[0]
            if word in word_counter:
                vector = list(map(float, array[1:]))
                features[word] = vector
                word_size = len(vector)
    print("done")
    vocab_word_list = [word for word in word_list if word in features]
    vocab_size = len(features)

    f = h5py.File(emb_mat_path, 'w')
    emb_mat = f.create_dataset('data', [vocab_size, word_size], dtype='float')
    vocab = {}
    pbar = get_pbar(len(vocab_word_list)).start()
    for i, word in enumerate(vocab_word_list):
        emb_mat[i, :] = features[word]
        vocab[word] = i
        pbar.update(i)
    pbar.finish()

    meta_data['vocab_size'] = vocab_size
    meta_data['word_size'] = word_size
    print("num of distinct words: %d" % len(word_counter))
    print("vocab size: %d" % vocab_size)
    print("word size: %d" % word_size)

    print("dumping json file ... ", end="", flush=True)
    f.close()
    json.dump(vocab, open(vocab_path, "w"))
    json.dump(meta_data, open(meta_data_path, "w"))
    print("done")


def create_meta_data(args):
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = {'data_dir': args.data_dir}
    json.dump(meta_data, open(meta_data_path, "w"))


def create_image_ids_and_paths(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    images_dir = os.path.join(data_dir, "images")
    image_ids_path = os.path.join(target_dir, "image_ids.json")
    image_paths_path = os.path.join(target_dir, "image_paths.json")
    image_names = [name for name in os.listdir(images_dir) if name.endswith(".png")]
    image_ids = [os.path.splitext(name)[0] for name in image_names]
    ordered_image_ids = sorted(image_ids, key=lambda x: int(x))
    ordered_image_names = ["%s.png" % id_ for id_ in ordered_image_ids]
    print("dumping json files ... ", end="", flush=True)
    image_paths = [os.path.join(images_dir, name) for name in ordered_image_names]
    json.dump(ordered_image_ids, open(image_ids_path, "w"))
    json.dump(image_paths, open(image_paths_path, "w"))
    print("done")


def prepro_images(args):
    model_path = args.vgg_model_path
    proto_path = args.vgg_proto_path
    out_path = os.path.join(args.target_dir, "images.h5")
    image_paths_path = os.path.join(args.target_dir, "image_paths.json")
    os.system("th prepro_images.lua --image_path_json %s --cnn_proto %s --cnn_model %s --out_path %s"
              % (image_paths_path, proto_path, model_path, out_path))


def copy_folds(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    for num in range(1,6):
        from_folds_path = os.path.join(data_dir, "fold%d.json" % num)
        to_folds_path = os.path.join(target_dir, "fold%d.json" % num)
        shutil.copy(from_folds_path, to_folds_path)


if __name__ == "__main__":
    ARGS = get_args()
    create_meta_data(ARGS)
    create_image_ids_and_paths(ARGS)
    # copy_folds(ARGS)
    build_vocab(ARGS)
    prepro_annos(ARGS)
    prepro_questions(ARGS)
    # prepro_images(ARGS)
