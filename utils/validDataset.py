"""
1. 校验数据集的合法性
2. 将数据集修改成一列一列的形式
3. 将一些一维的数据转化成二维的形式
"""
import numpy as np


def valid_dataset(data, axis=0):
    if ('train' in data) & ('test' in data):
        train = data['train']
        test = data['test']
        [train_flag, train] = valid_data(train, axis)
        if train_flag is False:
            return False, data
        [test_flag, test] = valid_data(test, axis)
        if test_flag is False:
            return False, data
        data['train'] = train
        data['test'] = test
        return True, data
    else:
        return False, data


def valid_data(data, axis=0):
    if ('X' in data) & ('Y' in data):
        x = data['X']
        y = data['Y']
        if axis == 1:
            x = data['X'].T
            y = data['Y'].T
        if x.ndim == 1:
            x = x.reshape(1, x.size)
        if y.ndim == 1:
            y = y.reshape(1, y.size)
        if x.ndim != 2 | y.ndim != 2:
            return False, data
        if x.shape[1] == y.shape[1]:
            data['X'] = x
            data['Y'] = y
            return True, data
        else:
            return False, data
