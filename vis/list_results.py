import importlib
import shutil

import SimpleHTTPServer
import SocketServer
import argparse
import json
import os
import numpy as np
from copy import deepcopy

from jinja2 import Environment, FileSystemLoader

from utils import get_pbar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_num", type=int)
    parser.add_argument("config_num", type=int)
    parser.add_argument("data_type", type=str)
    parser.add_argument("global_step", type=int)
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stop", default=1500, type=int)
    parser.add_argument("--show_im", default='True')
    parser.add_argument("--im_width", type=int, default=200)
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument("--template_name", type=str, default="list_results.html")
    parser.add_argument("--num_im", type=int, default=50)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--open", type=str, default='False')

    args = parser.parse_args()
    return args


def _decode_sent(decoder, sent):
    return " ".join(decoder[idx] for idx in sent)



def list_results(args):
    model_num = args.model_num
    config_num = args.config_num
    data_type = args.data_type
    global_step =args.global_step
    configs = importlib.import_module("configs.c%s" % str(model_num).zfill(2)).configs
    config = configs[config_num]
    evals_dir = os.path.join("evals", "m%s" % str(model_num).zfill(2), "c%s" % str(config_num).zfill(2))
    evals_name = "%s_%s.json" % (data_type, str(global_step).zfill(8))
    evals_path = os.path.join(evals_dir, evals_name)
    evals = json.load(open(evals_path, 'r'))

    fold_path = config['fold_path']
    fold = json.load(open(fold_path, 'r'))
    fold_data_type = 'test' if data_type == 'val' else data_type
    image_ids = sorted(fold[fold_data_type], key=lambda x: int(x))

    prepro_dir = config['data_dir']
    meta_data_dir = os.path.join(prepro_dir, "meta_data.json")
    meta_data = json.load(open(meta_data_dir, "r"))
    data_dir = meta_data['data_dir']
    _id = 0
    html_dir = "/tmp/list_results%d" % _id
    while os.path.exists(html_dir):
        _id += 1
        html_dir = "/tmp/list_results%d" % _id

    images_dir = os.path.join(data_dir, 'images')
    annos_dir = os.path.join(data_dir, 'annotations')

    sents_path = os.path.join(prepro_dir, 'sents.json')
    facts_path = os.path.join(prepro_dir, 'facts.json')
    vocab_path = os.path.join(prepro_dir, 'vocab.json')
    answers_path = os.path.join(prepro_dir, 'answers.json')
    sentss_dict = json.load(open(sents_path, "r"))
    facts_dict = json.load(open(facts_path, "r"))
    vocab = json.load(open(vocab_path, "r"))
    answers_dict = json.load(open(answers_path, "r"))
    decoder = {idx: word for word, idx in vocab.items()}

    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    os.mkdir(html_dir)

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    templates_dir = os.path.join(cur_dir, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(args.template_name)

    eval_names = list(evals['values'].keys())
    eval_dd = {}
    for idx, id_ in enumerate(evals['ids']):
        eval_d = {}
        for name, d in evals['values'].items():
            eval_d[name] = d[idx]
        eval_dd[tuple(id_)] = eval_d

    # headers = ['iid', 'qid', 'image', 'sents', 'answer', 'annotations', 'relations'] + eval_names
    headers = ['iid', 'qid', 'image', 'sents', 'annotations', 'relations', 'p', 'sig', 'yp']
    rows = []
    pbar = get_pbar(len(sentss_dict)).start()
    for i, image_id in enumerate(image_ids):
        if image_id not in sentss_dict:
            continue
        sentss = sentss_dict[image_id]
        answers = answers_dict[image_id]
        facts = facts_dict[image_id]
        decoded_facts = [_decode_sent(decoder, fact) for fact in facts]
        for question_id, (sents, answer) in enumerate(zip(sentss, answers)):
            eval_id = (image_id, question_id)
            eval_d = eval_dd[eval_id] if eval_id in eval_dd else None

            if eval_d:
                p_all = zip(*eval_d['p:0'])
                p = p_all[:len(decoded_facts)]
                p = [[float("%.3f" % x) for x in y] for y in p]
                yp = [float("%.3f" % x) for x in eval_d['yp:0']]
                sig = [float("%.3f" % x) for x in eval_d['sig:0']]
            else:
                p, yp, sig = [], [], []

            evals = [eval_d[name] if eval_d else "" for name in eval_names]
            image_name = "%s.png" % image_id
            json_name = "%s.json" % image_name
            image_url = os.path.join('images', image_name)
            anno_url = os.path.join('annotations', json_name)
            ap = np.argmax(yp) if len(yp) > 0 else 0
            correct = len(yp) > 0 and ap == answer
            row = {'image_id': image_id,
                   'question_id': question_id,
                   'image_url': image_url,
                   'anno_url': anno_url,
                   'sents': [_decode_sent(decoder, sent) for sent in sents],
                   'answer': answer,
                   'facts': decoded_facts,
                   'p': p,
                   'sig': sig,
                   'yp': yp,
                   'ap': np.argmax(yp) if len(yp) > 0 else 0,
                   'correct': correct,
                   }

            rows.append(row)

        if i % args.num_im == 0:
            html_path = os.path.join(html_dir, "%s.html" % str(image_id).zfill(8))

        if (i + 1) % args.num_im == 0 or (i + 1) == len(image_ids):
            var_dict = {'title': "Question List",
                        'image_width': args.im_width,
                        'headers': headers,
                        'rows': rows,
                        'show_im': True if args.show_im == 'True' else False}
            with open(html_path, "wb") as f:
                f.write(template.render(**var_dict).encode('UTF-8'))
            rows = []
        pbar.update(i)
    pbar.finish()

    os.system("ln -s %s/* %s" % (data_dir, html_dir))
    os.chdir(html_dir)
    port = args.port
    host = args.host
    # Overriding to suppress log message
    class MyHandler(SimpleHTTPServer.SimpleHTTPRequestHandler):
        def log_message(self, format, *args):
            pass
    handler = MyHandler
    httpd = SocketServer.TCPServer((host, port), handler)
    if args.open == 'True':
        os.system("open http://%s:%d" % (args.host, args.port))
    print("serving at %s:%d" % (host, port))
    httpd.serve_forever()


if __name__ == "__main__":
    ARGS = get_args()
    list_results(ARGS)
