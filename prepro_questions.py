import argparse
import json
import os

import numpy as np
from PIL import Image
import progressbar as pb

from utils import tokenize, vlup


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    return parser.parse_args()


def prepro_questions(args):
    data_dir = args.data_dir
    questions_dir = os.path.join(data_dir, "questions")
    questions_path = os.path.join(data_dir, "questions.json")
    vocab_path = os.path.join(data_dir, "vocab.json")
    vocab = json.load(open(vocab_path, "rb"))

    questions_dict = {'questions': {},
                      'choices': {},
                      'answers': {}}

    ques_names = os.listdir(questions_dir)
    pbar = pb.ProgressBar(widgets=[pb.Percentage(), pb.Bar(), pb.ETA()], maxval=len(ques_names))
    pbar.start()
    question_id = 0
    for i, ques_name in enumerate(ques_names):
        if os.path.splitext(ques_name)[1] != ".json":
            pbar.update(i)
            continue
        ques_path = os.path.join(questions_dir, ques_name)
        ques = json.load(open(ques_path, "rb"))
        for ques_text, d in ques['questions'].iteritems():
            questions_dict['questions'][str(question_id)] = vlup(vocab, tokenize(ques_text))
            choice_list = [vlup(vocab, tokenize(choice)) for choice in d['answerTexts']]
            questions_dict['choices'][str(question_id)] = choice_list
            questions_dict['answers'][str(question_id)] = choice_list[d['correctAnswer']]
            question_id += 1
        pbar.update(i)
    pbar.finish()

    print("%d questions" % len(questions_dict['questions']))
    print("dumping json file ...")
    json.dump(questions_dict, open(questions_path, "wb"))


if __name__ == "__main__":
    ARGS = get_args()
    prepro_questions(ARGS)
