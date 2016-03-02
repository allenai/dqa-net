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
    parser.add_argument("--html_path", type=str, default="/tmp/interpret_prepro.html")
    parser.add_argument("--template_name", type=str, default="interpret_prepro.html")

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
    meta_data = json.load(open(meta_data_dir, "rb"))
    data_dir = meta_data['data_dir']

    images_dir = os.path.join(data_dir, 'images')
    annos_dir = os.path.join(data_dir, 'annotations')
    replaced_images_dir = os.path.join(data_dir, "imagesReplacedText")
    html_path = args.html_path

    sents_path = os.path.join(prepro_dir, 'sents.json')
    relations_path = os.path.join(prepro_dir, 'relations.json')
    vocab_path = os.path.join(prepro_dir, 'vocab.json')
    id_map_path = os.path.join(prepro_dir, 'id_map.json')
    answers_path = os.path.join(prepro_dir, 'answers.json')
    replaced_path = os.path.join(prepro_dir, 'replaced.json')
    sents_dict = json.load(open(sents_path, "rb"))
    relations_dict = json.load(open(relations_path, "rb"))
    vocab = json.load(open(vocab_path, "rb"))
    id_map = json.load(open(id_map_path, "rb"))
    answer_dict = json.load(open(answers_path, "rb"))
    replaced = json.load(open(replaced_path, 'rb'))
    decoder = {idx: word for word, idx in vocab.iteritems()}

    headers = ['iid', 'qid', 'image', 'sents', 'answer', 'annotations', 'relations']
    rows = []
    question_ids = sorted(sents_dict.keys(), key=lambda x: int(x))
    question_ids = [id_ for id_ in question_ids if args.start <= int(id_) < args.stop]
    pbar = get_pbar(len(question_ids)).start()
    for i, question_id in enumerate(question_ids):
        sents = sents_dict[question_id]
        answer = answer_dict[question_id]
        image_id = id_map[question_id]
        rep = replaced[question_id]
        image_name = "%s.png" % image_id
        json_name = "%s.json" % image_name
        image_path = os.path.join(images_dir, image_name)
        anno_path = os.path.join(annos_dir, json_name)
        # anno = json.load(open(anno_path, 'rb'))
        replaced_image_path = os.path.join(replaced_images_dir, image_name)
        relations = relations_dict[question_id]
        decoded_relations = [_decode_relation(decoder, relation) for relation in relations]
        row = {'image_id': image_id,
               'question_id': question_id,
               'image_url': image_path,
               'rep_image_url': replaced_image_path,
               'anno_url': anno_path,
               'sents': [_decode_sent(decoder, sent) for sent in sents],
               'answer': answer,
               'relations': decoded_relations,
               'replaced': rep}
        rows.append(row)
        pbar.update(i)
    pbar.finish()
    var_dict = {'title': "Question List: %d - %d" % (args.start, args.stop - 1),
                'image_width': args.im_width,
                'headers': headers,
                'rows': rows,
                'show_im': True if args.show_im == 'True' else False}

    env = Environment(loader=FileSystemLoader('html_templates'))
    template = env.get_template(args.template_name)
    out = template.render(**var_dict)
    with open(html_path, "wb") as f:
        f.write(out.encode('UTF-8'))

    os.system("open %s" % html_path)


if __name__ == "__main__":
    ARGS = get_args()
    interpret_relations(ARGS)
