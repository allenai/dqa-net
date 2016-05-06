import argparse
import os
import json
import shutil
from collections import defaultdict, namedtuple
import re
import random
from pprint import pprint

import h5py
import numpy as np

from utils import get_pbar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="~/data/shining3")
    parser.add_argument("--target_dir", default="data/s3")
    parser.add_argument("--glove_path", default="~/models/glove/glove.6B.300d.txt")
    parser.add_argument("--min_count", type=int, default=5)
    parser.add_argument("--vgg_model_path", default="~/models/vgg/vgg-19.caffemodel")
    parser.add_argument("--vgg_proto_path", default="~/models/vgg/vgg-19.prototxt")
    parser.add_argument("--debug", default='False')
    parser.add_argument("--qa2hypo", default='True')
    parser.add_argument("--qa2hypo_path", default="../dqa/qa2hypo")
    parser.add_argument("--prepro_images", default='True')
    return parser.parse_args()


def qa2hypo(question, answer, flag, qa2hypo_path):
    if flag == 'True':
        import sys
        # add qa2hypo_path to the Python path at runtime
        sys.path.insert(0, qa2hypo_path)
        from qa2hypo import qa2hypo as f
        return f(question, answer, False, True)
    # attach the answer to the question
    return "%s %s" % (question, answer)


def _tokenize(raw):
    tokens = tuple(re.findall(r"[\w]+", raw))
    return tokens


def _vadd(vocab_counter, word):
    word = word.lower()
    vocab_counter[word] += 1


def _vget(vocab_dict, word):
    word = word.lower()
    if word in vocab_dict:
        return vocab_dict[word]
    else:
        return 0


def _vlup(vocab_dict, words):
    return tuple(_vget(vocab_dict, word) for word in words)


def _get(id_map, key):
    return id_map[key] if key in id_map else None


def rel2text(id_map, rel):
    """
    Obtain text facts from the relation class.
    :param id_map:
    :param rel:
    :return:
    """
    TEMPLATES = ["%s links to %s.",
                 "there is %s.",
                 "the title is %s.",
                 "%s describes region.",
                 "there are %s %s.",
                 "arrows objects regions 0 1 2 3 4 5 6 7 8 9",
                 "% s and %s are related."]
    MAX_LABEL_SIZE = 3
    tup = rel[:3]
    o_keys, d_keys = rel[3:]
    if tup == ('interObject', 'linkage', 'objectToObject'):
        template = TEMPLATES[0]
        o = _get(id_map, o_keys[0]) if len(o_keys) else None
        d = _get(id_map, d_keys[0]) if len(d_keys) else None
        if not (o and d):
            return None
        o_words = _tokenize(o)
        d_words = _tokenize(d)
        if len(o_words) > MAX_LABEL_SIZE:
            o = "object"
        if len(d_words) > MAX_LABEL_SIZE:
            d = "object"
        text = template % (o, d)
        return text

    elif tup == ('intraObject', 'linkage', 'regionDescription'):
        template = TEMPLATES[3]
        o = _get(id_map, o_keys[0]) if len(o_keys) else None
        o = o or "an object"
        o_words = _tokenize(o)
        if len(o_words) > MAX_LABEL_SIZE:
            o = "an object"
        text = template % o
        return text

    elif tup == ('unary', '', 'regionDescriptionNoArrow'):
        template = TEMPLATES[3]
        o = _get(id_map, o_keys[0]) if len(o_keys) else None
        o = o or "an object"
        o_words = _tokenize(o)
        if len(o_words) > MAX_LABEL_SIZE:
            o = "an object"
        text = template % o
        return text

    elif tup[0] == 'unary' and tup[2] in ['objectLabel', 'ownObject']:
        template = TEMPLATES[1]
        val =_get(id_map, o_keys[0])
        if val is not None:
            words = _tokenize(val)
            if len(words) > MAX_LABEL_SIZE:
                return val
            else:
                return template % val

    elif tup == ('unary', '', 'regionLabel'):
        template = TEMPLATES[1]
        val =_get(id_map, o_keys[0])
        if val is not None:
            words = _tokenize(val)
            if len(words) > MAX_LABEL_SIZE:
                return val
            else:
                return template % val

    elif tup == ('unary', '', 'imageTitle'):
        template = TEMPLATES[2]
        val = _get(id_map, o_keys[0])
        return template % val

    elif tup == ('unary', '', 'sectionTitle'):
        template = TEMPLATES[2]
        val = _get(id_map, o_keys[0])
        return template % val

    elif tup[0] == 'count':
        template = TEMPLATES[4]
        category = tup[2]
        num = str(o_keys)
        return template % (num, category)

    elif tup[0] == 'unary':
        val = _get(id_map, o_keys[0])
        return val

    return None


Relation = namedtuple('Relation', 'type subtype category origin destination')
categories = set()


def anno2rels(anno):
    types = set()
    rels = []
    # Unary relations
    for text_id, d in anno['text'].items():
        category = d['category'] if 'category' in d else ''
        categories.add(category)
        rel = Relation('unary', '', category, [text_id], '')
        rels.append(rel)

    # Counting
    if 'arrows' in anno and len(anno['arrows']) > 0:
        rels.append(Relation('count', '', 'stages', len(anno['arrows']), ''))
    if 'objects' in anno and len(anno['objects']) > 0:
        rels.append(Relation('count', '', 'objects', len(anno['objects']), ''))

    if 'relationships' not in anno:
        return rels
    for type_, d in anno['relationships'].items():
        for subtype, dd in d.items():
            for rel_id, ddd in dd.items():
                category = ddd['category']
                origin = ddd['origin'] if 'origin' in ddd else ""
                destination = ddd['destination'] if 'destination' in ddd else ""
                rel = Relation(type_, subtype, category, origin, destination)
                rels.append(rel)
                types.add((type_, subtype, category))
    return rels


def _get_id_map(anno):
    id_map = {}
    if 'text' in anno:
        for key, d in anno['text'].items():
            id_map[key] = d['value']
    if 'objects' in anno:
        for key, d in anno['objects'].items():
            if 'text' in d and len(d['text']) > 0:
                new_key = d['text'][0]
                id_map[key] = id_map[new_key]
    if 'relationships' in anno:
        d = anno['relationships']
        if 'intraOjbect' in d:
            d = d['intraOjbect']
            if 'label' in d:
                d = d['label']
                for _, dd in d.items():
                    category = dd['category']
                    if category in ['arrowHeadTail', 'arrowDescriptor']:
                        continue
                    origin = dd['origin'][0]
                    dest = dd['destination'][0]
                    if origin.startswith("CT") or origin.startswith("T"):
                        id_map[dest] = id_map[origin]
                    elif dest.startswith("CT") or dest.startswith("T"):
                        id_map[origin] = id_map[dest]
    return id_map


def prepro_annos(args):
    """
    Transform DQA annotation.json -> a list of tokenized fact sentences for each image in json file
    The facts are indexed by image id.
    :param args:
    :return:
    """
    data_dir = args.data_dir
    target_dir = args.target_dir

    # For debugging
    if args.debug == 'True':
        sents_path =os.path.join(target_dir, "raw_sents.json")
        answers_path =os.path.join(target_dir, "answers.json")
        sentss_dict = json.load(open(sents_path, 'r'))
        answers_dict = json.load(open(answers_path, 'r'))

    facts_path = os.path.join(target_dir, "raw_facts.json")
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_path, "r"))
    facts_dict = {}
    annos_dir = os.path.join(data_dir, "annotations")
    anno_names = [name for name in os.listdir(annos_dir) if name.endswith(".json")]
    max_num_facts = 0
    max_fact_size = 0
    pbar = get_pbar(len(anno_names)).start()
    for i, anno_name in enumerate(anno_names):
        image_name, _ = os.path.splitext(anno_name)
        image_id, _ = os.path.splitext(image_name)
        anno_path = os.path.join(annos_dir, anno_name)
        anno = json.load(open(anno_path, 'r'))
        rels = anno2rels(anno)
        id_map = _get_id_map(anno)
        text_facts = [rel2text(id_map, rel) for rel in rels]
        text_facts = list(set(_tokenize(fact) for fact in text_facts if fact is not None))
        max_fact_size = max([max_fact_size] + [len(fact) for fact in text_facts])
        # For debugging only
        if args.debug == 'True':
            if image_id in sentss_dict:
                correct_sents = [sents[answer] for sents, answer in zip(sentss_dict[image_id], answers_dict[image_id])]
                # indexed_facts.extend(correct_sents)
                # FIXME : this is very strong prior!
                text_facts = correct_sents
            else:
                text_facts = []
        facts_dict[image_id] = text_facts
        max_num_facts = max(max_num_facts, len(text_facts))
        pbar.update(i)

    pbar.finish()

    meta_data['max_num_facts'] = max_num_facts
    meta_data['max_fact_size'] = max_fact_size
    print("number of facts: %d" % sum(len(facts) for facts in facts_dict.values()))
    print("max num facts per relation: %d" % max_num_facts)
    print("max fact size: %d" % max_fact_size)
    print("dumping json files ... ")
    json.dump(meta_data, open(meta_data_path, 'w'))
    json.dump(facts_dict, open(facts_path, 'w'))
    print("done")


def prepro_questions(args):
    """
    transform DQA questions.json files -> single statements json and single answers json.
    sentences and answers are doubly indexed by image id first and then question number within that image (0 indexed)
    :param args:
    :return:
    """
    data_dir = args.data_dir
    target_dir = args.target_dir
    questions_dir = os.path.join(data_dir, "questions")
    raw_sents_path = os.path.join(target_dir, "raw_sents.json")
    answers_path = os.path.join(target_dir, "answers.json")
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_path, "r"))

    sentss_dict = {}
    answers_dict = {}

    ques_names = sorted([name for name in os.listdir(questions_dir) if os.path.splitext(name)[1].endswith(".json")],
                        key=lambda x: int(os.path.splitext(os.path.splitext(x)[0])[0]))
    num_choices = 0
    num_questions = 0
    max_sent_size = 0
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
            sents = [_tokenize(qa2hypo(ques_text, choice, args.qa2hypo, args.qa2hypo_path)) for choice in d['answerTexts']]
            max_sent_size = max(max_sent_size, max(len(sent) for sent in sents))
            assert not num_choices or num_choices == len(sents), "number of choices don't match: %s" % ques_name
            num_choices = len(sents)
            sentss.append(sents)
            answers.append(d['correctAnswer'])
            num_questions += 1
        sentss_dict[image_id] = sentss
        answers_dict[image_id] = answers
        pbar.update(i)
    pbar.finish()
    meta_data['num_choices'] = num_choices
    meta_data['max_sent_size'] = max_sent_size

    print("number of questions: %d" % num_questions)
    print("number of choices: %d" % num_choices)
    print("max sent size: %d" % max_sent_size)
    print("dumping json file ... ")
    json.dump(sentss_dict, open(raw_sents_path, "w"))
    json.dump(answers_dict, open(answers_path, "w"))
    json.dump(meta_data, open(meta_data_path, "w"))
    print("done")


def build_vocab(args):
    target_dir = args.target_dir
    vocab_path = os.path.join(target_dir, "vocab.json")
    emb_mat_path = os.path.join(target_dir, "init_emb_mat.h5")
    raw_sents_path = os.path.join(target_dir, "raw_sents.json")
    raw_facts_path = os.path.join(target_dir, "raw_facts.json")
    raw_sentss_dict = json.load(open(raw_sents_path, 'r'))
    raw_facts_dict = json.load(open(raw_facts_path, 'r'))

    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_path, 'r'))
    glove_path = args.glove_path

    word_counter = defaultdict(int)

    for image_id, raw_sentss in raw_sentss_dict.items():
        for raw_sents in raw_sentss:
            for raw_sent in raw_sents:
                for word in raw_sent:
                    _vadd(word_counter, word)

    for image_id, raw_facts in raw_facts_dict.items():
        for raw_fact in raw_facts:
            for word in raw_fact:
                _vadd(word_counter, word)

    word_list, counts = zip(*sorted([pair for pair in word_counter.items()], key=lambda x: -x[1]))
    freq = 5
    print("top %d frequent words:" % freq)
    for word, count in zip(word_list[:freq], counts[:freq]):
        print("%r: %d" % (word, count))

    features = {}
    word_size = 0
    print("reading %s ... " % glove_path)
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
    unknown_word_list = [word for word in word_list if word not in features]
    vocab_size = len(features) + 1

    f = h5py.File(emb_mat_path, 'w')
    emb_mat = f.create_dataset('data', [vocab_size, word_size], dtype='float')
    vocab = {}
    pbar = get_pbar(len(vocab_word_list)).start()
    for i, word in enumerate(vocab_word_list):
        emb_mat[i+1, :] = features[word]
        vocab[word] = i + 1
        pbar.update(i)
    pbar.finish()
    vocab['UNK'] = 0

    meta_data['vocab_size'] = vocab_size
    meta_data['word_size'] = word_size
    print("num of distinct words: %d" % len(word_counter))
    print("vocab size: %d" % vocab_size)
    print("word size: %d" % word_size)

    print("dumping json file ... ")
    f.close()
    json.dump(vocab, open(vocab_path, "w"))
    json.dump(meta_data, open(meta_data_path, "w"))
    print("done")


def indexing(args):
    target_dir = args.target_dir
    vocab_path = os.path.join(target_dir, "vocab.json")
    raw_sents_path = os.path.join(target_dir, "raw_sents.json")
    raw_facts_path = os.path.join(target_dir, "raw_facts.json")
    sents_path = os.path.join(target_dir, "sents.json")
    facts_path = os.path.join(target_dir, "facts.json")
    vocab = json.load(open(vocab_path, 'r'))
    raw_sentss_dict = json.load(open(raw_sents_path, 'r'))
    raw_facts_dict = json.load(open(raw_facts_path, 'r'))

    sentss_dict = {image_id: [[_vlup(vocab, sent) for sent in sents] for sents in sentss] for image_id, sentss in raw_sentss_dict.items()}
    facts_dict = {image_id: [_vlup(vocab, fact) for fact in facts] for image_id, facts in raw_facts_dict.items()}

    print("dumping json files ... ")
    json.dump(sentss_dict, open(sents_path, 'w'))
    json.dump(facts_dict, open(facts_path, 'w'))
    print("done")


def create_meta_data(args):
    target_dir = args.target_dir
    if not os.path.exists(target_dir):
        os.mkdir(target_dir)
    meta_data_path = os.path.join(target_dir, "meta_data.json")
    meta_data = {'data_dir': args.data_dir}
    json.dump(meta_data, open(meta_data_path, "w"))


def create_image_ids_and_paths(args):
    if args.prepro_images == 'False':
        print("Skipping image preprocessing.")
        return
    data_dir = args.data_dir
    target_dir = args.target_dir
    images_dir = os.path.join(data_dir, "images")
    image_ids_path = os.path.join(target_dir, "image_ids.json")
    image_paths_path = os.path.join(target_dir, "image_paths.json")
    image_names = [name for name in os.listdir(images_dir) if name.endswith(".png")]
    image_ids = [os.path.splitext(name)[0] for name in image_names]
    ordered_image_ids = sorted(image_ids, key=lambda x: int(x))
    ordered_image_names = ["%s.png" % id_ for id_ in ordered_image_ids]
    print("dumping json files ... ")
    image_paths = [os.path.join(images_dir, name) for name in ordered_image_names]
    json.dump(ordered_image_ids, open(image_ids_path, "w"))
    json.dump(image_paths, open(image_paths_path, "w"))
    print("done")


def prepro_images(args):
    if args.prepro_images == 'False':
        print("Skipping image preprocessing.")
        return
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
    prepro_questions(ARGS)
    prepro_annos(ARGS)
    build_vocab(ARGS)
    indexing(ARGS)
    prepro_images(ARGS)
