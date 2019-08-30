'''
回归有两种方式实现
第一种使用梯度下降法
第二种使用正规方程

学习私有方法与私有属性的使用

shape[0]返回各个维度元素个数
'''

import numpy as np
import matplotlib.pyplot as plt

from utils.normalize import normalize,inverseTransform
from utils.validDataset import validDataset
from dataset.loadhousing import loadhousing



class LinearRegression(object):
    def __init__(self,data,axis = 0):
        '''
        初始化类LR的维度，输入，输出等信息
        :param x: R(N*M). N为维数/特征个数，M为样本个数;若为一维数组按1*m计算
        :param y: R(a*b). 若为一维向量，按1*b计算
        '''
        self.__maxIter = 200
        self.__Epsilon = 10**-3
        self.__LearningRate = 0.005

        self.__normalizeJob = 1
        self.__regressJob = 1

        self.__plot = np.zeros([self.__maxIter,])

        [flag,data] = validDataset(data,axis)

        if flag == True:
            self.trainData = data['train']
            self.testData = data['test']
            self.nfeature = self.trainData['X'].shape[0]
            self.trainNsample = self.trainData['X'].shape[1]
            self.testNsample = self.testData['X'].shape[1]
            self.__theta = np.zeros([self.nfeature + 1, 1])
        else:
            return False

    def __str__(self):
        print(self.trainData['X'].shape,self.trainData['Y'].shape)
        return 'init completely'

    def setLearningRate(self,mui):
        self.__LearningRate = mui

    def setEpsilon(self,epsilon):
        self.__Epsilon = epsilon

    def setMaxIteration(self,iter):
        self.__maxIter = iter
        self.__plot = np.zeros([iter,])

    def hypothesis(self,x,theta):
        '''
        :param x: n*m
        :param theta: n*1
        :return: x的所有函数值 1*m
        '''
        return theta.T.dot(x)

    def cost(self,x,theta,y):
        return np.sum((self.hypothesis(x, theta) - y)**2)/2

    def train(self,regressJobs = 1,normalizeJob = 1):
        '''
        jobs = 1
        梯度下降法
        jobs = 2
        正规方程法
        :return:
        '''
        self.__regressJob = regressJobs
        self.__normalizeJob = normalizeJob

        [XX, self.__XNormalPar] = normalize(self.trainData['X'], normalizeJob)
        XX = np.vstack([np.ones([1,self.trainNsample]),XX])
        YY = self.trainData['Y']
        # [YY, self.__YNormalPar] = normalize(self.trainData['Y'], normalizeJob)


        if regressJobs == 1:
            index = -1
            delta = 10000
            last =  10000 + self.cost(XX,self.__theta,YY)
            while (index < self.__maxIter - 1) & (delta > self.__Epsilon):
                index += 1
                CurrentCost = self.cost(XX,self.__theta,YY)
                self.__theta -= self.__LearningRate/self.trainNsample*np.sum((self.hypothesis(XX, self.__theta) - YY)*XX,1).reshape(self.nfeature+1, 1)
                delta = abs(last - CurrentCost)
                last = CurrentCost
                if CurrentCost > last:
                    print('bigger')
                self.__plot[index] = CurrentCost
                # print(delta)
        elif regressJobs == 2:
            pass
        else :
            pass

    def test(self):
        [XX,XNormalPar] = normalize(self.testData['X'], self.__normalizeJob)
        XX = np.vstack([np.ones([1, self.testNsample]), XX])
        tYY = self.hypothesis(XX,self.__theta)
        # itt = inverseTransform(tYY, self.__YNormalPar, self.__normalizeJob)
        # self.testCost = np.sum((itt - self.testData['Y'])**2)/self.testNsample

        itt = tYY
        self.testCost = np.sum((itt - self.testData['Y']) ** 2) / self.testNsample
        return itt

    # 求解时使用平均绝对误差，并不能用总和，否则容易溢出，且对学习率有不一样的要求


    def plot(self):
        x = np.arange(0,self.__maxIter)
        plt.plot(x,self.__plot)
        # plt.legend(), ls="-", lw=2, label="plot figure"
        plt.show()


if __name__ == '__main__':
    a = loadhousing()
    LR = LinearRegression(a.data,0)
    LR.setLearningRate(0.1)
    LR.setMaxIteration(1000)
    LR.setEpsilon(10**-5)
    LR.train(1,2)
    tt = LR.test()
    print(LR.testCost)
    LR.plot()
    print(LR)





