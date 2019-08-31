"""
[nx,normal_par] = normalize(x,jobs = 1,axis = 1)
正则化 数据
jobs 1:mui-sigama
jobs 2:min_par-max_par
否则 返回原数据
nx：标准化后的数据
normal_par:
jobs 1 ：返回［mui, sigma］
jobs 2 : 返回 [min_par, max_par - min_par]

默认对一行一行数据标准化
默认方法为jobs 1

inverse_transform(x,normal_par,jobs = 1,axis = 0)
"""

import numpy as np


def normalize(x, jobs=1, axis=1):
    if axis == 1:  # 行优先,且只考虑了二维向量的情况
        xx = x
    else:
        xx = x.T
    nx = np.zeros(xx.shape)
    if jobs == 1:  # job1 : mui-sigma 便准化； job2 :极差标准化
        index = -1
        for x_line in xx:
            index += 1
            std = np.std(x_line)
            mean = np.mean(x_line)
            if std != 0:
                nx[index, :] = (x_line - mean) / std
            else:
                nx[index, :] = x_line
    elif jobs == 2:
        index = -1
        for x_line in xx:
            index += 1
            min_par = np.min(x_line)
            max_par = np.max(x_line)
            if max_par != min_par:
                nx[index, :] = (x_line - min_par * np.ones(x_line.shape)) / (max_par - min_par)
            else:
                nx[index, :] = x_line
    else:
        return x

    if axis == 1:  # 行优先,且只考虑了二维向量的情况
        return nx
    else:
        return nx.T


# def normalize(x, jobs=1, axis=1, can_inverse=1):
#     if axis == 1:  # 行优先,且只考虑了二维向量的情况
#         xx = x
#     else:
#         xx = x.T
#     nx = np.zeros(xx.shape)
#     normal_par = np.zeros([xx.shape[0], 2])
#     normal_par[:, 1] = np.ones([xx.shape[0], ])
#     if jobs == 1:  # job1 : mui-sigma 便准化； job2 :极差标准化
#         index = -1
#         for x_line in xx:
#             index += 1
#             std = np.std(x_line)
#             mean = np.mean(x_line)
#             if std != 0:
#                 nx[index, :] = (x_line - mean) / std
#                 normal_par[index, :] = [mean, std]
#             else:
#                 nx[index, :] = x_line
#                 normal_par[index, :] = [0, 1]
#     elif jobs == 2:
#         index = -1
#         for x_line in xx:
#             index += 1
#             min_par = np.min(x_line)
#             max_par = np.max(x_line)
#             if max_par != min_par:
#                 nx[index, :] = (x_line - min_par * np.ones(x_line.shape)) / (max_par - min_par)
#                 normal_par[index, :] = [min_par, max_par - min_par]
#             else:
#                 nx[index, :] = x_line
#                 normal_par[index, :] = [0, 1]
#     else:
#         if axis == 1:
#             return x, normal_par
#         else:
#             return x, normal_par.T
#
#     if axis == 1:  # 行优先,且只考虑了二维向量的情况
#         return nx, normal_par
#     else:
#         return nx.T, normal_par.T


def inverse_transform(x, normal_par, jobs=1, axis=1):
    if axis == 1:  # 行优先,且只考虑了二维向量的情况
        xx = x
        normal_par = normal_par
    else:
        xx = x.T
        normal_par = normal_par.T
    inverse = np.zeros(xx.shape)
    index = -1
    for x_line in xx:
        index += 1
        inverse[index, :] = x_line * normal_par[index, 1] + normal_par[index, 0]
    if axis == 1:  # 行优先,且只考虑了二维向量的情况
        return inverse
    else:
        return inverse.T


if __name__ == '__main__':
    X = np.array([[1, 3, 6, 5], [1, 1, 1, 1], [1, 0, 0, -1]])
    job = 10
    Axis = 0
    [nX, normalPar] = normalize(X, job, Axis)
    print(X)
    # print(nx)
    # print(normal_par)
    inx = inverse_transform(nX, normalPar, job, Axis)
    print(inx)
