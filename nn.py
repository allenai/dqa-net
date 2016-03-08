from operator import mul
from functools import reduce

import tensorflow as tf

VERY_SMALL_NUMBER = -1e10


def softmax_with_mask(shape, x, mask, name=None):
    if name is None:
        name = softmax_with_mask.__name__
    x_masked = x + VERY_SMALL_NUMBER * (1 - mask)
    x_flat = tf.reshape(x_masked, [reduce(mul, shape[:-1], 1), shape[-1]])
    p_flat = tf.nn.softmax(x_flat)
    p = tf.reshape(p_flat, shape, name=name)
    return p



