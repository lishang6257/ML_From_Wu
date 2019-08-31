"""
回归有两种方式实现
第一种使用梯度下降法
第二种使用正规方程

学习私有方法与私有属性的使用

shape[0]返回各个维度元素个数
"""
import numpy as np
import matplotlib.pyplot as plt
from utils.normalize import normalize
from utils.validDataset import validDataset
from dataset.loadhousing import LoadHousing


def hypothesis(x, theta):
    """
    :param x: n*m
    :param theta: n*1
    :return: x的所有函数值 1*m
    """
    return theta.T.dot(x)


def cost(x, theta, y):
    return np.sum((hypothesis(x, theta) - y) ** 2) / 2


class LinearRegression(object):
    def __init__(self, data, axis=0):
        """
        初始化类LR的维度，输入，输出等信息
        """
        self.__maxIter = 200
        self.__Epsilon = 10 ** -3
        self.__LearningRate = 0.005

        self.__regular_term = 0
        self.__normalize_job = 1
        self.__regressJob = 1

        self.testCost = 0

        self.__plot = np.zeros([self.__maxIter, ])

        [flag, data] = validDataset(data, axis)

        if flag is True:
            self.trainData = data['train']
            self.testData = data['test']
            self.nfeature = self.trainData['X'].shape[0]
            self.trainNsample = self.trainData['X'].shape[1]
            self.testNsample = self.testData['X'].shape[1]
            self.__theta = np.zeros([self.nfeature + 1, 1])
        else:
            pass

    def __str__(self):
        print(self.trainData['X'].shape, self.trainData['Y'].shape)
        return 'init completely'

    def set_learning_rate(self, mui):
        self.__LearningRate = mui

    def set_epsilon(self, epsilon):
        self.__Epsilon = epsilon

    def set_max_iteration(self, iteration):
        self.__maxIter = iteration
        self.__plot = np.zeros([iteration, ])

    def train(self, regress_job=1, regular_term=0, normalize_job=1):
        """
        jobs = 1
        梯度下降法
        jobs = 2
        正规方程法
        :return:
        """
        self.__regressJob = regress_job
        self.__normalize_job = normalize_job

        if regress_job == 1:
            xx = normalize(self.trainData['X'], normalize_job)
            xx = np.vstack([np.ones([1, self.trainNsample]), xx])
            yy = self.trainData['Y']

            index = -1
            delta = 10000
            last = 10000 + cost(xx, self.__theta, yy)
            while (index < self.__maxIter - 1) & (delta > self.__Epsilon):
                index += 1
                current_cost = cost(xx, self.__theta, yy)
                self.__theta -= self.__LearningRate / self.trainNsample \
                    * np.sum((hypothesis(xx, self.__theta) - yy) * xx, 1).reshape(self.nfeature + 1, 1)\
                    + self.__regular_term / self.testNsample * self.__theta
                delta = abs(last - current_cost)
                last = current_cost
                self.__plot[index] = current_cost
                # print(delta)
        elif regress_job == 2:
            xx = self.trainData['X'].T
            xx = np.hstack([np.ones([self.trainNsample, 1]), xx])
            yy = self.trainData['Y'].T

            lambda_matrix = np.identity(self.nfeature + 1)
            lambda_matrix[0][0] = 0
            self.__theta = np.dot(np.dot(np.linalg.pinv(np.dot(xx.T, xx) + regular_term * lambda_matrix), xx.T), yy)
        else:
            pass

    def test(self):
        if regressJob == 1:
            xx = normalize(self.testData['X'], self.__normalize_job)
            xx = np.vstack([np.ones([1, self.testNsample]), xx])
            tyy = hypothesis(xx, self.__theta)
            self.testCost = np.sum((tyy - self.testData['Y']) ** 2) / self.testNsample
            return tyy
        elif regressJob == 2:
            xx = self.testData['X']
            xx = np.vstack([np.ones([1, self.testNsample]), xx])
            tyy = hypothesis(xx, self.__theta)
            self.testCost = np.sum((tyy - self.testData['Y']) ** 2) / self.testNsample
            return tyy
        else:
            pass

    def plot(self):
        x = np.arange(0, self.__maxIter)
        plt.plot(x, self.__plot)
        # plt.legend(), ls="-", lw=2, label="plot figure"
        plt.show()


if __name__ == '__main__':
    a = LoadHousing()
    LR = LinearRegression(a.data, 0)
    LR.set_learning_rate(0.1)
    LR.set_max_iteration(1000)
    LR.set_epsilon(10 ** -5)
    regressJob = 1
    regularTerm = 0
    normalizeJob = 2
    LR.train(regressJob, regularTerm, normalizeJob)
    tt = LR.test()
    print(LR.testCost)
    LR.plot()
    print(LR)
