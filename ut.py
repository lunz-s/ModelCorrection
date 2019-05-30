import tensorflow as tf
import elasticdeform as edf
import numpy as np
import random as ran


def ident(batch):
    return batch

def flipud(batch):
    res = np.zeros(shape=batch.shape)
    for k in range(batch.shape[0]):
        res[k,...,0] = np.flipud[k,...,0]
    return res

def fliplr(batch):
    res = np.zeros(shape=batch.shape)
    for k in range(batch.shape[0]):
        res[k,...,0] = np.fliplr[k,...,0]
    return res

def augmentation(batch):
    sigma = ran.uniform(0,4)
    aug = edf.deform_random_grid(batch, axis=(1, 2), sigma=sigma, points=4)
    up = [ident, flipud]
    lr = [ident, fliplr]
    aug = ran.choice(up)(aug)
    aug = ran.choice(lr)(aug)
    return aug
