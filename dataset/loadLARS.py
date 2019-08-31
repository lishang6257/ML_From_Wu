import numpy as np
from dataset.TrainTestDivide import train_test_divide


class LoadLARS(object):
    def __init__(self):
        self.ndim = np.array([11, 442])
        self.__Xrange = np.arange(0, 10)
        self.__Yrange = np.array([10])
        self.__data = r'D:\programer\DataSet\ML\LARS\LARS.data'
        self.__attribute = r'D:\programer\DataSet\ML\LARS\LARS.attribute'

        dfile = open(self.__data, 'r')
        data = np.zeros(self.ndim)
        index = 0
        for line in dfile:
            ldata = line.split()
            ldata = [float(x) for x in ldata]
            data[:, index] = ldata
            index += 1
        dfile.close()

        afile = open(self.__attribute, 'r')
        attribute_x = {}
        attribute_y = {}
        index = -1
        for line in afile:
            index += 1
            ldata = line.split()
            describe = ldata[1]
            for i in range(2, len(ldata), 1):
                describe += ' ' + ldata[i]

            if index in self.__Xrange:
                attribute_x[ldata[0]] = describe
            elif index in self.__Yrange:
                attribute_y[ldata[0]] = describe
        afile.close()

        self.data = {'X': data[self.__Xrange, :], 'Y': data[self.__Yrange, :]}
        self.attribute = {'X': attribute_x, 'Y': attribute_y}

        self.data = train_test_divide(self.data)


if __name__ == '__main__':
    a = LoadLARS()
    print(a.data['test']['X'])
    print(a.attribute['X'])
