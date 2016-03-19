from copy import deepcopy


class Config(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def get_config(d0, d1, priority=1):
    """
    d1 replaces d0. If priority = 0, then d0 replaces d1
    :param d0:
    :param d1:
    :param name:
    :param priority:
    :return:
    """
    if priority == 0:
        d0, d1 = d1, d0
    d = deepcopy(d0)
    for key, val in d1.items():
        d[key] = val
    return Config(**d)

def get_config_from_json(d0, json_path, priority=1):
    pass

def get_config_from_tsv(d0, tsv_path, priority=1):
    pass




