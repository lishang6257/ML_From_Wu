import numpy as np

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
        Attribute = {}
        for line in __Afile:
            ldata = line.split()
            describe = ldata[1]
            for i in range(2,len(ldata),1):
                describe += ' ' + ldata[i]
            Attribute[ldata[0]] = describe
        __Afile.close()

        self.data = {'X':Data[self.__Xrange,:],'Y':Data[self.__Yrange,:]}
        self.attribute = {'X':Attribute[self.__Xrange,:],'Y':Attribute[self.__Yrange,:]}
