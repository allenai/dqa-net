import tensorflow as tf


def softmax_with_mask(shape, x, mask):
    exp = tf.exp(x)  # [N, M]
    masked_exp = exp * mask  # [N, M]
    sum_aug = tf.expand_dims(tf.reduce_sum(masked_exp, len(shape)-1), -1)  # [N, 1]
    p = tf.div(masked_exp, sum_aug, name='p')  # [N, M]
    return p



