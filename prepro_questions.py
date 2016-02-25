import argparse
import json
import os

import progressbar as pb

from utils import tokenize, vlup


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("target_dir")
    return parser.parse_args()


def prepro_questions(args):
    data_dir = args.data_dir
    target_dir = args.target_dir
    questions_dir = os.path.join(data_dir, "questions")
    questions_path = os.path.join(target_dir, "questions.json")
    vocab_path = os.path.join(data_dir, "vocab.json")
    vocab = json.load(open(vocab_path, "rb"))

    questions_dict = {'questions': {},
                      'choices': {},
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
            questions_dict['questions'][str(question_id)] = vlup(vocab, tokenize(ques_text))
            choice_words = [tokenize(choice) for choice in d['answerTexts']]
            choice_list = [vlup(vocab, words) for words in choice_words]
            questions_dict['choices'][str(question_id)] = choice_list
            questions_dict['answers'][str(question_id)] = choice_list[d['correctAnswer']]
            question_id += 1
            max_sent_size = max(max_sent_size, len(ques_words), max(len(words) for words in choice_words))
        pbar.update(i)
    pbar.finish()
    questions_dict['max_sent_size'] = max_sent_size

    print("number of questions: %d" % len(questions_dict['questions']))
    print("max sent size: %d" % max_sent_size)
    print("dumping json file ...")
    json.dump(questions_dict, open(questions_path, "wb"))


if __name__ == "__main__":
    ARGS = get_args()
    prepro_questions(ARGS)
