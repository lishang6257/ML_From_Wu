import numpy as np

'''
1. 校验数据集的合法性
2. 将数据集修改成一列一列的形式
3. 将一些一维的数据转化成二维的形式
'''
def validDataset(data,axis = 0):
    if ('train' in data) & ('test' in data):
        train = data['train']
        test = data['test']
        [trainFlag,train] = validData(train,axis)
        if trainFlag == False:
            return False,data
        [testFlag, test] = validData(test, axis)
        if testFlag == False:
            return False,data
        data['train'] = train
        data['test'] = test
        return True,data
    else :
        return False,data

def validData(data,axis = 0):
    if ('X' in data) & ('Y' in data):
        if axis == 1:
            x = data['X'].T
            y = data['Y'].T
        elif axis == 0:
            x = data['X']
            y = data['Y']
        if x.ndim == 1:
            x = x.reshape(1, x.size)
        if y.ndim == 1:
            y = y.reshape(1, y.size)
        if x.ndim != 2 | y.ndim != 2:
            return False,data
        if x.shape[1] == y.shape[1]:
            data['X'] = x
            data['Y'] = y
            return True,data
        else:
            return False,data
