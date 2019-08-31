import random
import numpy as np


def train_test_divide(data):
    ndim = data['X'].shape
    num_rain = round(0.7*ndim[1])
    num_test = ndim[1] - num_rain

    test = random.sample(range(0, ndim[1]), num_test)
    ttmap = np.zeros([ndim[1], 1])
    ttmap[test] = 1
    train = np.where(ttmap == 0)
    train = {'X': data['X'][:, train[0]], 'Y': data['Y'][:, train[0]]}
    test = {'X': data['X'][:, test], 'Y': data['Y'][:, test]}

    return {'train': train, 'test': test}
