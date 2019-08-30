import numpy as np
'''
[nx,normalPar] = normalize(x,jobs = 1,axis = 1)
正则化 数据
jobs 1:mui-sigama
jobs 2:min-max
否则 返回原数据
nx：正则化后的数据
normalPar:
jobs 1 ：返回［mui, sigma］
jobs 2 : 返回 [min, max - min]

默认对一行一行数据正则化
默认方法为jobs 1 

inverseTransform(x,normalPar,jobs = 1,axis = 0)


'''

def normalize(x,jobs = 1,axis = 1):
    if axis == 1:#行优先,且只考虑了二维向量的情况
        X = x
    else:
        X = x.T
    nX = np.zeros(X.shape)
    normalPar = np.zeros([X.shape[0], 2])
    normalPar[:,1] = np.ones([X.shape[0], ])
    if jobs == 1:#job1 : mui-sigma 便准化； job2 :极差标准化
        index = -1
        for xx in X:
            index += 1
            std = np.std(xx)
            mean = np.mean(xx)
            if std != 0:
                nX[index, :] = (xx - mean) / std
                normalPar[index, :] = [mean,std]
            else:
                nX[index, :] = xx
                normalPar[index, :] = [0, 1]
    elif jobs == 2:
        index = -1
        for xx in X:
            index += 1
            min = np.min(xx)
            max = np.max(xx)
            if max != min:
                nX[index, :] = (xx - min * np.ones(xx.shape)) / (max - min)
                normalPar[index, :] = [min, max - min]
            else:
                nX[index, :] = xx
                normalPar[index, :] = [0, 1]
    else:
        if axis == 1:
            return x, normalPar
        else:
            return x, normalPar.T

    if axis == 1:#行优先,且只考虑了二维向量的情况
        return nX, normalPar
    else:
        return nX.T, normalPar.T

def inverseTransform(x,normalPar,jobs = 1,axis = 1):
    if axis == 1:#行优先,且只考虑了二维向量的情况
        X = x
        NormalPar = normalPar
    else:
        X = x.T
        NormalPar = normalPar.T
    inverse = np.zeros(X.shape)
    index = -1
    for xx in X:
        index += 1
        inverse[index,:] = xx * NormalPar[index,1] + NormalPar[index, 0]
    if axis == 1:#行优先,且只考虑了二维向量的情况
        return inverse
    else:
        return inverse.T




if __name__ == '__main__':
    x = np.array([[1,3,6,5],[1,1,1,1],[1,0,0,-1]])
    job = 10
    axis =0
    [nx,normalPar] = normalize(x,job,axis)
    print(x)
    # print(nx)
    # print(normalPar)
    inX = inverseTransform(nx,normalPar,job,axis)
    print(inX)
