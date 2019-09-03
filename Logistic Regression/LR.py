import numpy as np

import dataset
import utils


def sigmod(x):
    return 1 / (1 + np.exp(-x))


def hypothesis(x, theta):
    """
    :param x: n*m
    :param theta: n*1
    :return: x的所有函数值 1*m
    """
    return sigmod(theta.T.dot(x))


def cost(x, theta, y):
    return np.sum((hypothesis(x, theta) - y) ** 2) / 2


class LogisticRegression(object):
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

        [flag, data] = utils.valid_dataset(data, axis)

        if flag is True:
            self.trainData = data['train']
            self.testData = data['test']
            self.nfeature = self.trainData['X'].shape[0]
            self.trainNsample = self.trainData['X'].shape[1]
            self.testNsample = self.testData['X'].shape[1]
            self.__theta = np.zeros([self.nfeature + 1, 1])
        else:
            pass

    def set_learning_rate(self, mui):
        self.__LearningRate = mui

    def set_epsilon(self, epsilon):
        self.__Epsilon = epsilon

    def set_max_iteration(self, iteration):
        self.__maxIter = iteration
        self.__plot = np.zeros([iteration, ])

    def train(self, regress_job=1, regular_term=1, nomalize_job=1):
        if regress_job == 1: # 梯度下降法
            iteration = -1
            delta = cost(xx, self.__theta, yy) + 1000
            while (iteration < self.__maxIter) & (delta < self.__Epsilon):
                iteration += 1
                current_cost = cost(xx,self.__theta,yy)
                self.__theta
                delta





if __name__ == '__main__':
    LR = dataset.LoadHousing()
    print(LR.attribute)
