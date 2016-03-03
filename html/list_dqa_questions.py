import SimpleHTTPServer
import SocketServer
import argparse
import json
import os
import shutil

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
    parser.add_argument("--html_dir", type=str, default="/tmp/list_dqa_questions/")
    parser.add_argument("--template_name", type=str, default="list_dqa_questions.html")
    parser.add_argument("--mode", type=str, default='open')
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--num_im", type=int, default=20)

    return parser.parse_args()


def list_dqa_questions(args):
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    questions_dir = os.path.join(data_dir, "questions")
    html_dir = args.html_dir
    annos_dir = os.path.join(data_dir, "annotations")

    cur_dir = os.path.dirname(os.path.realpath(__file__))
    templates_dir = os.path.join(cur_dir, 'templates')
    env = Environment(loader=FileSystemLoader(templates_dir))
    template = env.get_template(args.template_name)

    if os.path.exists(html_dir):
        shutil.rmtree(html_dir)
    os.mkdir(html_dir)

    headers = ['image_id', 'question_id', 'image', 'question', 'choices', 'answer', 'annotations']
    rows = []
    image_names = os.listdir(images_dir)
    image_names = sorted(image_names, key=lambda name: int(os.path.splitext(name)[0]))
    image_names = [name for name in image_names
                   if name.endswith(args.ext) and args.start <= int(os.path.splitext(name)[0]) < args.stop]
    pbar = get_pbar(len(image_names)).start()
    for i, image_name in enumerate(image_names):
        image_id, _ = os.path.splitext(image_name)
        json_name = "%s.json" % image_name
        anno_path = os.path.join(annos_dir, json_name)
        question_path = os.path.join(questions_dir, json_name)
        if os.path.exists(question_path):
            question_dict = json.load(open(question_path, "rb"))
            anno_dict = json.load(open(anno_path, "rb"))
            for j, (question, d) in enumerate(question_dict['questions'].iteritems()):
                row = {'image_id': image_id,
                       'question_id': str(j),
                       'image_url': os.path.join("images" if d['abcLabel'] else "imagesReplacedText", image_name),
                       'anno_url': os.path.join("annotations", json_name),
                       'question': question,
                       'choices': d['answerTexts'],
                       'answer': d['correctAnswer']}
                rows.append(row)

        if i % args.num_im == 0:
            html_path = os.path.join(html_dir, "%s.html" % str(image_id).zfill(8))

        if (i + 1) % args.num_im == 0 or (i + 1) == len(image_names):
            var_dict = {'title': "Question List",
                        'image_width': args.im_width,
                        'headers': headers,
                        'rows': rows,
                        'show_im': args.show_im}
            with open(html_path, "wb") as f:
                f.write(template.render(**var_dict).encode('UTF-8'))
            rows = []
        pbar.update(i)
    pbar.finish()


    if args.mode == 'open':
        os.system("open %s" % html_path)
    elif args.mode == 'host':
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
        print "serving at %s:%d" % (host, port)
        httpd.serve_forever()


if __name__ == "__main__":
    ARGS = get_args()
    list_dqa_questions(ARGS)
