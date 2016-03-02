import argparse
import json
import os
from copy import deepcopy

from jinja2 import Environment, FileSystemLoader

from utils import get_pbar



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("prepro_dir")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stop", default=1500, type=int)
    parser.add_argument("--show_im", default='True')
    parser.add_argument("--im_width", type=int, default=200)
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument("--html_path", type=str, default="/tmp/interpret_relations.html")
    parser.add_argument("--template_name", type=str, default="interpret_relations.html")

    args = parser.parse_args()
    return args


def _decode_sent(decoder, sent):
    return " ".join(decoder[idx] for idx in sent)


def _decode_relation(decoder, relation):
    new_relation = deepcopy(relation)
    new_relation['a1r'] = _decode_sent(decoder, new_relation['a1r'])
    new_relation['a2r'] = _decode_sent(decoder, new_relation['a2r'])
    new_relation['a1'] = _decode_sent(decoder, new_relation['a1'])
    new_relation['a2'] = _decode_sent(decoder, new_relation['a2'])
    return new_relation


def interpret_relations(args):
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, 'images')
    replaced_images_dir = os.path.join(data_dir, "imagesReplacedText")
    questions_dir = os.path.join(data_dir, "questions")
    annos_dir = os.path.join(data_dir, "annotations")
    html_path = args.html_path

    prepro_dir = args.prepro_dir
    relations_path = os.path.join(prepro_dir, 'relations.json')
    vocab_path = os.path.join(prepro_dir, 'vocab.json')
    relations_dict = json.load(open(relations_path, "rb"))
    vocab = json.load(open(vocab_path, "rb"))
    decoder = {idx: word for word, idx in vocab.iteritems()}

    headers = ['iid', 'qid', 'image', 'question', 'choices', 'answer', 'annotations', 'relations']
    rows = []
    image_names = os.listdir(images_dir)
    image_names = sorted(image_names, key=lambda name: int(os.path.splitext(name)[0]))
    image_names = [name for name in image_names
                   if name.endswith(args.ext) and args.start <= int(os.path.splitext(name)[0]) < args.stop]
    pbar = get_pbar(len(image_names)).start()
    for i, image_name in enumerate(image_names):
        image_id, _ = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        replaced_image_path = os.path.join(replaced_images_dir, image_name)
        relations = relations_dict[image_id]
        decoded_relations = [_decode_relation(decoder, relation) for relation in relations]
        json_name = "%s.json" % image_name
        question_path = os.path.join(questions_dir, json_name)
        anno_path = os.path.join(annos_dir, json_name)
        if not os.path.exists(question_path):
            continue
        question_dict = json.load(open(question_path, "rb"))
        anno_dict = json.load(open(anno_path, "rb"))
        for j, (question, d) in enumerate(question_dict['questions'].iteritems()):
            replaced = d['abcLabel']
            row = {'image_id': image_id,
                   'question_id': str(j),
                   'image_url': image_path,
                   'rep_image_url': replaced_image_path,
                   'anno_url': anno_path,
                   'question': question,
                   'choices': d['answerTexts'],
                   'answer': d['correctAnswer'],
                   'relations': decoded_relations,
                   'replaced': replaced}
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
