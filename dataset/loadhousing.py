import numpy as np
from dataset.TrainTestDivide import TTDivide

class loadhousing(object):
    def __init__(self):
        self.ndim = np.array([14,506])
        self.__Xrange = np.arange(0,13)
        self.__Yrange = np.array([13])
        self.__data = r'D:\programer\DataSet\ML\housing\housing.data'
        self.__attribute = r'D:\programer\DataSet\ML\housing\housing.attribute'

    # def load(self):
        __Dfile = open(self.__data,'r')
        Data = np.zeros(self.ndim)
        index = 0
        for line in __Dfile:
            ldata = line.split()
            ldata = [float(x) for x in ldata]
            Data[:,index] = ldata
            index += 1
        __Dfile.close()


        __Afile = open(self.__attribute,'r')
        AttributeX = {}
        AttributeY = {}
        index = -1
        for line in __Afile:
            index += 1
            ldata = line.split()
            describe = ldata[1]
            for i in range(2,len(ldata),1):
                describe += ' ' + ldata[i]

            if index in self.__Xrange:
                AttributeX[ldata[0]] = describe
            elif index in self.__Yrange:
                AttributeY[ldata[0]] = describe
        __Afile.close()

        self.data = {'X':Data[self.__Xrange,:],'Y':Data[self.__Yrange,:]}
        self.attribute = {'X':AttributeX,'Y':AttributeY}

        self.data = TTDivide(self.data)



# a = loadhousing()
# print(a.data['test']['X'])
# print(a.attribute['X'])