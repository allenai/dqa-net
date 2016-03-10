import argparse
import json
import os
from copy import deepcopy

from jinja2 import Environment, FileSystemLoader

from utils import get_pbar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("prepro_dir")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stop", default=1500, type=int)
    parser.add_argument("--show_im", default='True')
    parser.add_argument("--im_width", type=int, default=200)
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument("--html_path", type=str, default="/tmp/list_relations.html")
    parser.add_argument("--template_name", type=str, default="list_relations.html")

    args = parser.parse_args()
    return args


def _decode_sent(decoder, sent):
    return " ".join(decoder[idx] for idx in sent)


def _decode_relation(decoder, relation):
    new_relation = deepcopy(relation)
    """
    new_relation['a1r'] = _decode_sent(decoder, new_relation['a1r'])
    new_relation['a2r'] = _decode_sent(decoder, new_relation['a2r'])
    """
    new_relation['a1'] = _decode_sent(decoder, new_relation['a1'])
    new_relation['a2'] = _decode_sent(decoder, new_relation['a2'])
    return new_relation


def interpret_relations(args):
    prepro_dir = args.prepro_dir
    meta_data_dir = os.path.join(prepro_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_dir, "r"))
    data_dir = meta_data['data_dir']

    images_dir = os.path.join(data_dir, 'images')
    annos_dir = os.path.join(data_dir, 'annotations')
    html_path = args.html_path

    sents_path = os.path.join(prepro_dir, 'sents.json')
    relations_path = os.path.join(prepro_dir, 'relations.json')
    vocab_path = os.path.join(prepro_dir, 'vocab.json')
    answers_path = os.path.join(prepro_dir, 'answers.json')
    sentss_dict = json.load(open(sents_path, "r"))
    relations_dict = json.load(open(relations_path, "r"))
    vocab = json.load(open(vocab_path, "r"))
    answers_dict = json.load(open(answers_path, "r"))
    decoder = {idx: word for word, idx in vocab.items()}

    headers = ['iid', 'qid', 'image', 'sents', 'answer', 'annotations', 'relations']
    rows = []
    pbar = get_pbar(len(sentss_dict)).start()
    image_ids = sorted(sentss_dict.keys(), key=lambda x: int(x))
    for i, image_id in enumerate(image_ids):
        sentss = sentss_dict[image_id]
        answers = answers_dict[image_id]
        relations = relations_dict[image_id]
        decoded_relations = [_decode_relation(decoder, relation) for relation in relations]
        for question_id, (sents, answer) in enumerate(zip(sentss, answers)):
            image_name = "%s.png" % image_id
            json_name = "%s.json" % image_name
            image_path = os.path.join(images_dir, image_name)
            anno_path = os.path.join(annos_dir, json_name)
            row = {'image_id': image_id,
                   'question_id': question_id,
                   'image_url': image_path,
                   'anno_url': anno_path,
                   'sents': [_decode_sent(decoder, sent) for sent in sents],
                   'answer': answer,
                   'relations': decoded_relations}
            rows.append(row)
        pbar.update(i)
    pbar.finish()
    var_dict = {'title': "Question List: %d - %d" % (args.start, args.stop - 1),
                'image_width': args.im_width,
                'headers': headers,
                'rows': rows,
                'show_im': True if args.show_im == 'True' else False}

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    templates_dir = os.path.join(cur_dir, 'TEMPLATES')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(args.template_name)
    out = template.render(**var_dict)
    with open(html_path, "w") as f:
        f.write(out)

    os.system("open %s" % html_path)


if __name__ == "__main__":
    ARGS = get_args()
    interpret_relations(ARGS)
