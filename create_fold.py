import json
import os
import random
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("fold_path")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--shuffle", default="False")
    return parser.parse_args()


def create_fold(args):
    data_dir = args.data_dir
    images_dir = os.path.join(data_dir, "images")
    annotations_dir = os.path.join(data_dir, "annotations")
    ratio = args.ratio
    shuffle = args.shuffle == 'True'
    fold_path = args.fold_path
    annotation_names = set(os.path.splitext(name)[0] for name in os.listdir(annotations_dir) if name.endswith(".json"))
    image_ids = list(sorted([os.path.splitext(name)[0]
                             for name in os.listdir(images_dir) if name.endswith(".png") and name in annotation_names],
                            key=lambda x: int(x)))
    if shuffle:
        random.shuffle(image_ids)

    mid = int(len(image_ids) * (1 - ratio))
    print("train={}, test={}".format(len(image_ids)-mid, mid))
    fold = {'train': image_ids[mid:], 'test': image_ids[:mid]}
    json.dump(fold, open(fold_path, 'w'))

if __name__ == "__main__":
    ARGS = get_args()
    create_fold(ARGS)
