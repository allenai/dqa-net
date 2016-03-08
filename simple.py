import argparse
import json
from os import path, listdir
from random import randint

import networkx as nx
import re

from nltk.stem import PorterStemmer

from utils import get_pbar


def _get_args():
    parser =argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("fold_path")
    return parser.parse_args()


def _tokenize(raw):
    tokens = re.findall(r"[\w]+", raw)
    return tokens

stem = True
stemmer = PorterStemmer()
def _normalize(word):
    word = word.lower()
    if stem:
        word = stemmer.stem(word)
    return word

def load_all(data_dir):
    annos_dir = path.join(data_dir, 'annotations')
    images_dir = path.join(data_dir, 'images')
    questions_dir = path.join(data_dir, 'questions')

    anno_dict = {}
    questions_dict = {}
    choicess_dict = {}
    answers_dict = {}

    image_ids = sorted([path.splitext(name)[0] for name in listdir(images_dir) if name.endswith(".png")], key=lambda x: int(x))
    pbar = get_pbar(len(image_ids)).start()
    for i, image_id in enumerate(image_ids):
        json_name = "%s.png.json" % image_id
        anno_path = path.join(annos_dir, json_name)
        ques_path = path.join(questions_dir, json_name)
        if path.exists(anno_path) and path.exists(ques_path):
            anno = json.load(open(anno_path, "r"))
            ques = json.load(open(ques_path, "r"))

            questions = []
            choicess = []
            answers = []
            for question, d in ques['questions'].items():
                if not d['abcLabel']:
                    choices = d['answerTexts']
                    answer = d['correctAnswer']
                    questions.append(question)
                    choicess.append(choices)
                    answers.append(answer)

            questions_dict[image_id] = questions
            choicess_dict[image_id] = choicess
            answers_dict[image_id] = answers
            anno_dict[image_id] = anno
        pbar.update(i)
    pbar.finish()

    return anno_dict, questions_dict, choicess_dict, answers_dict


def _get_val(anno, key):
    first = key[0]
    if first == 'T':
        val = anno['text'][key]['value']
        val = _normalize(val)
        return val
    elif first == 'O':
        d = anno['objects'][key]
        if 'text' in d and len(d['text']) > 0:
            key = d['text'][0]
            return _get_val(anno, key)
        return None
    else:
        raise Exception(key)


def create_graph(anno):
    graph = nx.Graph()
    try:
        d = anno['relationships']['interObject']['linkage']
    except:
        return None
    for dd in d.values():
        if dd['category'] == 'objectToObject':
            dest = _get_val(anno, dd['destination'][0])
            orig = _get_val(anno, dd['origin'][0])
            graph.add_edge(dest, orig)
    print(graph.nodes())
    return graph


def find_node(graph, text):
    words = _tokenize(text)
    words = [_normalize(word) for word in words]
    for word in words:
        if word in graph.nodes():
            return word
    return None


def guess(graph, question, choices):
    MAX = 9999
    ques_node = find_node(graph, question)
    if ques_node is None:
        return None
    dists = []
    for choice in choices:
        choice_node = find_node(graph, choice)
        if nx.has_path(graph, ques_node, choice_node):
            pl = len(nx.shortest_path(graph, ques_node, choice_node))
            dists.append(pl)
        else:
            dists.append(MAX)
    answer, dist = max(enumerate(dists), key=lambda x: x[1])
    if dist == MAX:
        return None
    return answer


def evaluate(anno_dict, questions_dict, choicess_dict, answers_dict):
    total = 0
    correct = 0
    incorrect = 0
    guessed = 0
    pbar = get_pbar(len(anno_dict)).start()
    for i, (image_id, anno) in enumerate(anno_dict.items()):
        graph = create_graph(anno)
        questions = questions_dict[image_id]
        choicess =choicess_dict[image_id]
        answers = answers_dict[image_id]
        for question, choices, answer in zip(questions, choicess, answers):
            total += 1
            a = guess(graph, question, choices)
            if a is None:
                guessed += 1
            elif answer == a:
                correct += 1
            else:
                incorrect += 1
        pbar.update(i)
    pbar.finish()
    print("expected accuracy: (0.25 * %d + %d)/%d = %.4f" % (guessed, correct, total, 0.25*guessed + correct/total))


def select(fold_path, *all_):
    fold = json.load(open(fold_path, 'r'))
    test_ids = fold['test']
    new_all = []
    for each in all_:
        new_each = {id_: each[id_] for id_ in test_ids}
        new_all.append(new_each)
    return new_all


def main():
    args = _get_args()
    all_ = load_all(args.data_dir)
    selected = select(args.fold_path, *all_)
    evaluate(*selected)

if __name__ == "__main__":
    main()



