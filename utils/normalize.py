import numpy as np

def normalize(x,jobs = 1,axis = 0):
    if axis == 0:#列优先,且只考虑了二维向量的情况
        X = x
    else:
        X = x.T
    if jobs == 1:#job1 : mui-sigma 便准化； job2 :极差标准化


