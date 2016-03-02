import argparse
import json
import os

from jinja2 import Environment, FileSystemLoader

from utils import get_pbar


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--start", default=0, type=int)
    parser.add_argument("--stop", default=1500, type=int)
    parser.add_argument("--show_im", default='True')
    parser.add_argument("--im_width", type=int, default=200)
    parser.add_argument("--ext", type=str, default=".png")
    parser.add_argument("--html_path", type=str, default="/tmp/list_questions.html")
    parser.add_argument("--template_name", type=str, default="list_dqa_questions.html")

    return parser.parse_args()


def list_dqa_questions(args):
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    questions_dir = os.path.join(data_dir, "questions")
    annos_dir = os.path.join(data_dir, "annotations")
    html_path = args.html_path

    headers = ['image_id', 'question_id', 'image', 'question', 'choices', 'answer', 'annotations']
    rows = []
    image_names = os.listdir(images_dir)
    image_names = sorted(image_names, key=lambda name: int(os.path.splitext(name)[0]))
    image_names = [name for name in image_names
                   if name.endswith(args.ext) and args.start <= int(os.path.splitext(name)[0]) < args.stop]
    pbar = get_pbar(len(image_names)).start()
    for i, image_name in enumerate(image_names):
        image_id, _ = os.path.splitext(image_name)
        image_path = os.path.join(images_dir, image_name)
        json_name = "%s.json" % image_name
        question_path = os.path.join(questions_dir, json_name)
        anno_path = os.path.join(annos_dir, json_name)
        if not os.path.exists(question_path):
            continue
        question_dict = json.load(open(question_path, "rb"))
        anno_dict = json.load(open(anno_path, "rb"))
        for j, (question, d) in enumerate(question_dict['questions'].iteritems()):
            row = {'image_id': image_id,
                   'question_id': str(j),
                   'image_url': image_path,
                   'anno_url': anno_path,
                   'question': question,
                   'choices': d['answerTexts'],
                   'answer': d['correctAnswer']}
            rows.append(row)
        pbar.update(i)
    pbar.finish()
    var_dict = {'title': "Question List: %d - %d" % (args.start, args.stop - 1),
                'image_width': args.im_width,
                'headers': headers,
                'rows': rows,
                'show_im': args.show_im}

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    templates_dir = os.path.join(cur_dir, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(args.template_name)
    out = template.render(**var_dict)
    with open(html_path, "wb") as f:
        f.write(out.encode('UTF-8'))

    os.system("open %s" % html_path)


if __name__ == "__main__":
    ARGS = get_args()
    list_dqa_questions(ARGS)
