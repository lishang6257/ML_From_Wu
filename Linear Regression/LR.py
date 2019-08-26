'''
回归有两种方式实现
第一种使用梯度下降法
第二种使用正规方程

学习私有方法与私有属性的使用

shape[0]返回各个维度元素个数
'''

import numpy as np
import sys
import os
from dataset.loadhousing import loadhousing


class LinearRegression(object):
    def __init__(self,x,y):
        '''
        初始化类LR的维度，输入，输出等信息
        :param x: R(N*M). N为维数/特征个数，M为样本个数;若为一维数组按1*m计算
        :param y: R(a*b). 若为一维向量，按1*b计算
        '''
        self.__Epsilon = 10**-3
        self.__LearningRate = 0.1
        self.__Fnormalize = False
        if x.ndim == 1:
            x = x.reshape(1,x.size)
        if y.ndim == 1:
            y = y.reshape(1,y.size)
        if x.ndim != 2 | y.ndim != 2:
            return 'Error input ndim in X | Y，checke them;\n'
        if x.shape[1] == y.shape[1]:
            self.nfeature = x.shape[0]
            self.nsample = x.shape[1]
            self.__X = np.vstack((np.ones([1,self.nsample]),x))
            self.__Y = y
            self.__theta = np.zeros([self.nfeature + 1, 1])
            self.__nX = np.zeros(self.__X.shape)
            self.__nY = np.zeros(self.__Y.shape)
        else:
            return 'Error input nsample in X | Y，check them;\n'

    def __str__(self):
        print(self.__X.shape,self.__Y.shape,self.__theta.shape)
        return 'init completely'

    def setLearningRate(self,mui):
        self.__LearningRate = mui

    def setEpsilon(self,epsilon):
        self.__Epsilon = epsilon

    def normalize(self,jobs = 1):
        '''
        X:由于常数项已经堆叠，因此从第二个特征进行正则化
        y:直接正则化

        由于假设中数据服从正太分布，这里将数据分布修正成标准正态分布
        :return:
        '''
        if jobs == 1:
            self.__Fnormalize = True
            index = -1
            for xx in self.__X:
                index += 1
                std = np.std(xx)
                if std != 0:
                    self.__nX[index,:] = (xx - np.mean(xx)) / std

            index = -1
            for yy in self.__Y:
                index += 1
                std = np.std(yy)
                if std != 0:
                    self.__nY[index, :] = (yy - np.mean(yy)) / std

        if jobs == 2:
            self.__Fnormalize = True
            index = -1
            for xx in self.__X:
                index += 1
                min = np.min(xx)
                max = np.max(xx)
                if max != min:
                    self.__nX[index,:] = (xx - min*np.ones(xx.shape)) / (max - min)

            index = -1
            for yy in self.__Y:
                index += 1
                min = np.min(yy)
                max = np.max(yy)
                if max != min:
                    self.__nY[index, :] = (yy - min * np.ones(yy.shape)) / (max - min)


    def hypothesis(self,x,theta):
        '''
        :param x: n*m
        :param theta: n*1
        :return: x的所有函数值 1*m
        '''
        return theta.T.dot(x)

    # def hypothesis(self,x,n,theta):
    #     '''
    #     :param x: n*m
    #     :param n: 下标；表征求解第n个样本
    #     :param theta: n*1
    #     :return: 第n的样本的函数值
    #     '''
    #     return theta.T.dot(x[:,n])

    def cost(self,x,theta,y):
        return np.sum((self.hypothesis(x, theta) - y)**2)/2

    def fit(self,jobs = 1):
        '''
        jobs = 1
        梯度下降法
        jobs = 2
        正规方程法
        :return:
        '''
        if self.__Fnormalize == True:
            XX = self.__nX
            YY = self.__nY
        else:
            XX = self.__X
            YY = self.__Y

        if jobs == 1:
            delta = 10000
            last =  10000 + self.cost(XX,self.__theta,YY)
            while delta >= self.__Epsilon:
                CurrentCost = self.cost(XX,self.__theta,YY)
                self.__theta -= self.__LearningRate*np.sum((self.hypothesis(XX, self.__theta) - YY)*XX,1).reshape(self.nfeature+1, 1)
                delta = last - CurrentCost


a = loadhousing()


LR = LinearRegression(a.data['X'],a.data['Y'])
LR.normalize(1)
LR.fit(1)
print(LR)




