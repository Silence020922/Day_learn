"""
初始化变量函数
"""
import numpy as np
import tensorflow as tf


def zero_init(shape, name=None):
    v = tf.zeros(shape,dtype=tf.float32)
    return tf.Variable(v, name=name)


def ones_init(shape, name=None):
    v = tf.ones(shape, dtype=tf.float32)
    return tf.Variable(v, name=name)


def uniform_init(shape, radius=0.05, name=None):
    v = tf.random.uniform(shape, minval=-radius, maxval=radius, dtype=tf.float32)
    return tf.Variable(v, name=name)


# glorot初始化方法
def glorot_init(shape, name=None):
    init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
    initial = tf.random.uniform(
        shape, minval=-init_range, maxval=init_range, dtype=tf.float32
    )
    return tf.Variable(initial, name=name)
