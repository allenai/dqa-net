import argparse
import csv
import json
from collections import OrderedDict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("tsv_path")
    parser.add_argument("json_path")
    return parser.parse_args()

def tsv2json(tsv_path, json_path):
    d = tsv2dict(tsv_path)
    json.dump(d, open(json_path, 'w'))


def tsv2dict(tsv_path):
    with open(tsv_path, 'r', newline='') as file:
        reader = csv.reader(file, delimiter='\t')
        fields = next(reader)
        type_names = next(reader)
        casters = list(map(eval, type_names))
        out_dict = {}
        for row in reader:
            cur_dict = OrderedDict(
                (field, None if val == "None" else caster(val))
                for field, caster, val in zip(fields, casters, row))
            id_ = cur_dict['id']
            del cur_dict['id']
            out_dict[id_] = cur_dict
        return out_dict


def main():
    args = get_args()
    tsv2json(args.tsv_path, args.json_path)


if __name__ == "__main__":
    main()
