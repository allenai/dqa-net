import argparse
import json
import os

from jinja2 import Environment, FileSystemLoader

parser = argparse.ArgumentParser()
parser.add_argument("root_dir")
parser.add_argument("--images_dir", default='images')
parser.add_argument("--questions_name", default='questions.json')
parser.add_argument("--annotations_name", default="annotations.json")
parser.add_argument("--start", default=0, type=int)
parser.add_argument("--stop", default=-100, type=int)
parser.add_argument("--html_path", default="/tmp/main.html")
parser.add_argument("--image_width", default=200, type=int)
parser.add_argument("--ext", default='.png')
parser.add_argument("--prefix", default='')
parser.add_argument("--zfill_width", default=12, type=int)
parser.add_argument("--template_name", default="list_questions.html")

ARGS = parser.parse_args()

env = Environment(loader=FileSystemLoader('html_templates'))


def main(args):
    root_dir = args.root_dir
    images_dir = os.path.join(root_dir, args.images_dir)
    questions_path = os.path.join(root_dir, args.questions_name)
    annotations_path = os.path.join(root_dir, args.annotations_name)
    html_path = args.html_path

    def _get_image_url(image_id):
        return os.path.join(images_dir, "%s%s%s" % (args.prefix, image_id.zfill(args.zfill_width), args.ext))

    questions_dict = json.load(open(questions_path, "rb"))
    annotations_dict = json.load(open(annotations_path, "rb"))

    headers = ['image_id', 'question_id', 'image', 'question', 'choices', 'answer']
    row_dict = {question['question_id']:
                     {'image_id': question['image_id'],
                      'question_id': question['question_id'],
                      'image_url': _get_image_url(question['image_id']),
                      'question': question['question'],
                      'choices': question['multiple_choices'],
                      'answer': annotation['multiple_choice_answer']}
                 for question, annotation in zip(questions_dict['questions'], annotations_dict['annotations'])}
    idxs = range(args.start, args.stop)
    rows = [row_dict[idx] for idx in idxs]
    template = env.get_template(args.template_name)
    vars_dict = {'title': "Question List: %d - %d" % (args.start, args.stop - 1),
                 'image_width': args.image_width,
                 'headers': headers,
                 'rows': rows[args.start:args.stop]}
    out = template.render(**vars_dict)
    with open(html_path, "wb") as f:
        f.write(out.encode('UTF-8'))

    os.system("open %s" % html_path)

if __name__ == "__main__":
    main(ARGS)
