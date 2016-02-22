import argparse
import json
import os

from jinja2 import Environment
from jinja2 import FileSystemLoader

parser = argparse.ArgumentParser()
parser.add_argument("images_dir")
parser.add_argument("questions_path")
parser.add_argument("annotations_path")
parser.add_argument("--html_path", default="/tmp/list_questions.html")
parser.add_argument("--ext", default='.png')
parser.add_argument("--prefix", default='')
parser.add_argument("--zfill_width", default=12, type=int)

ARGS = parser.parse_args()

env = Environment(loader=FileSystemLoader('html_templates'))


def list_questions(args):
    images_dir = args.images_dir
    questions_path = args.questions_path
    annotations_path = args.annotations_path
    html_path = args.html_path

    def _get_image_url(image_id):
        return os.path.join(images_dir, "%s%s%s" % (args.prefix, image_id.zfill(args.zfill_width), args.ext))

    questions_dict = json.load(open(questions_path, "rb"))
    annotations_dict = json.load(open(annotations_path, "rb"))

    headers = ['image', 'question', 'choices', 'answer']
    rows = [{'image_url': _get_image_url(question['image_id']),
             'question': question['question'],
             'choices': question['multiple_choices'],
             'answer': question['answer']}
            for question, annotation in zip(questions_dict['questions'], annotations_dict['annotations'])]
    template = env.get_template('%s.html' % list_questions.__name__)
    vars_dict = {'title': "Hello!",
                 'headers': headers,
                 'rows': rows}
    out = template.render(**vars_dict)
    with open(html_path, "wb") as f:
        f.write(out)

    os.system("open %s" % html_path)

if __name__ == "__main__":
    list_questions(ARGS)
