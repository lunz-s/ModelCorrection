import tensorflow as tf
import elasticdeform as edf
import numpy as np
import random as ran


def ident(batch):
    return batch

def flipud(batch):
    res = np.zeros(shape=batch.shape)
    for k in range(batch.shape[0]):
        res[k,...,0] = np.flipud(batch[k,...,0])
    return res

def fliplr(batch):
    res = np.zeros(shape=batch.shape)
    for k in range(batch.shape[0]):
        res[k,...,0] = np.fliplr(batch[k,...,0])
    return res

def augmentation(batch):
    sigma = ran.uniform(0,5)
    aug = edf.deform_random_grid(batch, axis=(1, 2), sigma=sigma, points=4)
    up = [ident, flipud]
    lr = [ident, fliplr]
    aug = ran.choice(up)(aug)
    aug = ran.choice(lr)(aug)
    return aug


def gradient(x):
    """
    :param x: input image (batch, x, y, channels)
    :return: pointwise squared l2 norm of image gradient
    """
    kernel_1 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([1, -1], dtype=tf.float32), axis=0), axis=-1),
                              axis=-1)
    kernel_2 = tf.expand_dims(tf.expand_dims(tf.expand_dims(tf.constant([1, -1], dtype=tf.float32), axis=1), axis=-1),
                              axis=-1)
    res1 = tf.pad(tf.nn.conv2d(x, kernel_1, strides=[1, 1, 1, 1], padding="VALID"),
                  paddings=[[0, 0], [0, 0], [0, 1], [0, 0]])
    res2 = tf.pad(tf.nn.conv2d(x, kernel_2, strides=[1, 1, 1, 1], padding="VALID"),
                  paddings=[[0, 0], [0, 1], [0, 0], [0, 0]])
    res = tf.square(res1) + tf.square(res2)
    return res


def huber_loss(x, delta=1e-2):
    """
    :param x: squared image to be used for huber norm
    :param delta: the steepness
    :return:
    """
    huber = delta * (tf.sqrt(1 + x / (delta ** 2)) - 1)
    return tf.reduce_sum(huber, axis=(1, 2, 3))


def huber_TV(x, delta=1e-2):
    return huber_loss(gradient(x), delta=delta)