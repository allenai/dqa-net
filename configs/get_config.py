from collections import namedtuple
from copy import deepcopy


def _dict_to_nt(name, dict_):
    class_ = namedtuple(name, " ".join(dict_.keys()))
    object_ = class_(**dict_)
    return object_


def get_config(d0, d1, name, priority=1):
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
    return _dict_to_nt(name, d)


def update_config(nt, dict_):
    """
    :param nt:  namedtuple object
    :param dict_:
    :return:
    """
    nt_dict = nt._asdict()
    for key, val in dict_.items():
        nt_dict[key] = val
    name = nt.__class__.__name__
    return _dict_to_nt(name, nt_dict)

