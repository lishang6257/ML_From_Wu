import numpy as np

def normalize(x,jobs = 1,axis = 0):
    if axis == 1:#行优先,且只考虑了二维向量的情况
        X = x
    else:
        X = x.T
    nX = np.zeros(X.shape)
    if jobs == 1:#job1 : mui-sigma 便准化； job2 :极差标准化
        index = -1
        for xx in X:
            index += 1
            std = np.std(xx)
            if std != 0:
                nX[index, :] = (xx - np.mean(xx)) / std
            else:
                nX[index, :] = xx
    elif jobs == 2:
        index = -1
        for xx in X:
            index += 1
            min = np.min(xx)
            max = np.max(xx)
            if max != min:
                nX[index, :] = (xx - min * np.ones(xx.shape)) / (max - min)
            else:
                nX[index, :] = xx
    else:
        return x


    if axis == 1:#行优先,且只考虑了二维向量的情况
        return nX
    else:
        return nX.T

if __name__ == '__main__':
    x = np.array([[1,3,6,5],[1,1,1,1],[1,0,0,-1]])
    nx = normalize(x,2,0)
    print(x)
    print(nx)
