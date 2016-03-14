import json
import os
import random
import argparse
from collections import defaultdict


def create_linear_fold():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("fold_path")
    parser.add_argument("--ratio", type=float, default=0.8)
    parser.add_argument("--shuffle", default="False")

    args = parser.parse_args()

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


def create_randomly_categorized_fold():
    parser = argparse.ArgumentParser()
    parser.add_argument("cat_path")
    parser.add_argument("fold_path")
    parser.add_argument("--test_cats", nargs='*')
    parser.add_argument("--ratio", type=float)
    args = parser.parse_args()
    cats_path = args.cat_path
    test_cats = args.test_cats
    cat_dict = json.load(open(cats_path, 'r'))
    ids_dict = defaultdict(set)
    for image_name, cat in cat_dict.items():
        image_id, _ = os.path.splitext(image_name)
        ids_dict[cat].add(image_id)

    cats = list(ids_dict.keys())
    print(cats)
    if test_cats is None:
        random.shuffle(cats)
        mid = int(args.ratio * len(cats))
        train_cats = cats[:mid]
        test_cats = cats[mid:]
    else:
        for cat in test_cats:
            assert cat in ids_dict, "%d id not a valid category." % cat
        train_cats = [cat for cat in cats if cat not in test_cats]

    print("train categories: %s" % ", ".join(train_cats))
    print("test categories: %s" % ", ".join(test_cats))
    train_ids = sorted(set.union(*[ids_dict[cat] for cat in train_cats]), key=lambda x: int(x))
    test_ids = sorted(set.union(*[ids_dict[cat] for cat in test_cats]), key=lambda x: int(x))
    fold = {'train': train_ids, 'test': test_ids, 'trainCats': train_cats, 'testCats': test_cats}
    json.dump(fold, open(args.fold_path, "w"))


if __name__ == "__main__":
    # create_linear_fold(ARGS)
    create_randomly_categorized_fold()
