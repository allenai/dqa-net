from operator import mul
from functools import reduce

import tensorflow as tf

VERY_SMALL_NUMBER = -1e10


def softmax_with_mask(shape, x, mask, name=None):
    if name is None:
        name = softmax_with_mask.__name__
    x_masked = x # + VERY_SMALL_NUMBER * (1 - mask)
    x_flat = tf.reshape(x_masked, [reduce(mul, shape[:-1], 1), shape[-1]])
    p_flat = tf.nn.softmax(x_flat)
    p = tf.reshape(p_flat, shape, name=name)
    return p


def prod_sum_sim(shape, u, v, name='prod_sum'):
    """
    product-sum similarlity between u and v
    u and v must have [N, C, d] dimension.
    :param shape: [N, C, d]
    :param u:
    :param v:
    :return:
    """
    with tf.variable_scope(name):
        N, C, d = shape
        W_prod = tf.get_variable("W_prod", shape=[d, 1])
        W_sum = tf.get_variable("W_sum", shape=[d, 1])
        b = tf.get_variable("b", shape=[1])
        u_flat = tf.reshape(u, [N*C, d])
        v_flat = tf.reshape(v, [N*C, d])
        logit_flat = tf.matmul(u_flat * v_flat, W_prod) + tf.matmul(u_flat + v_flat, W_sum) + b  # [N*C, 1]
        logit = tf.reshape(logit_flat, [N, C])
        return logit


def linear(batch_size, input_dim, output_dim, input_, name="linear"):
    N, a, b = batch_size, input_dim, output_dim
    with tf.variable_scope(name):
        mat = tf.get_variable("mat", shape=[a, b])
        bias = tf.get_variable("bias", shape=[b])
        out = tf.matmul(input_, mat) + bias
        return out


