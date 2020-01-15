import tensorflow as tf
from tensorflow.keras.layers import Lambda


def FiLM_simple_layer():
    """multiply scalar to a tensor"""
    def func(args):
        x, gamma, beta = args
        s = list(x.shape)
        s[0] = 1
        # avoid tile with the num of batch -> it is the same for both tensors
        g = tf.tile(tf.expand_dims(tf.expand_dims(gamma, 2), 3), s)
        b = tf.tile(tf.expand_dims(tf.expand_dims(beta, 2), 3), s)
        return tf.add(b, tf.multiply(x, g))
    return Lambda(func)


def FiLM_complex_layer():
    """multiply tensor to tensor"""
    def func(args):
        x, gamma, beta = args
        s = list(x.shape)
        # avoid tile with the num of batch -> same for both tensors
        s[0] = 1
        # avoid tile with the num of channels -> same for both tensors
        s[-1] = 1
        g = tf.tile(tf.expand_dims(gamma, 1), s)
        b = tf.tile(tf.expand_dims(beta, 1), s)
        return tf.add(b, tf.multiply(x, g))
    return Lambda(func)


def slice_tensor(position):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, position]
    return Lambda(func)


def slice_tensor_range(init, end):
    # Crops (or slices) a Tensor
    def func(x):
        return x[:, :, init:end]
    return Lambda(func)
