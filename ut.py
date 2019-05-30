import tensorflow as tf
import elasticdeform as edf
import numpy as np
import random as ran


def augmentation(batch):
    sigma = ran.uniform(0,4)
    aug = edf.deform_random_grid(batch, axis=(1, 2), sigma=sigma, points=4)
    up = [id, np.flipud]
    lr = [id, np.fliplr]
    aug = ran.choice(up)(aug)
    aug = ran.choice(lr)(aug)
    return aug


def id(batch):
    return batch