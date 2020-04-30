from numpy import *
import operator
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from os import listdir
from mpl_toolkits.mplot3d import Axes3D
import struct


# 读取图片
def ReadImage(fileName):
    # 用二进制方式把文件读进来
    fileHandle = open(fileName, "rb")    # 以二进制打开文件
    fileContent = fileHandle.read()     # 读取到缓冲区中
    offset = 0
    # 取前四个整数，返回一个元组
    head = struct.unpack_from('>IIII', fileContent, offset)
    offset += struct.calcsize('>IIII')
    imageNum = head[1]  # 图片数
    rows = head[2]  # 宽度
    cols = head[3]  # 高度
    # print(imageNum)
    # print(rows)
    # print(cols)

    # 测试读取一个图片是否读取成功
    # im = struct.unpack_from('>748B', fileContent, offset)
    # offset += struct.calcsize('>748B')
    # empty是它所常见的数组内元素均为空，没有实际意义，他是创建数组最快的方法
    images = np.empty((imageNum, 784))
    imageSize = rows * cols   # 单个图片的大小
    fmt = '>' + str(imageSize) + 'B'    # 单个图片的format

    for i in range(imageNum):
        images[i] = np.array(struct.unpack_from(fmt, fileContent, offset))
        # images[i] = np.array(struct.unpack_from(fmt, fileContent, offset)).reshape((rows, cols))
        offset += struct.calcsize(fmt)
    return images


# 读取标签
def ReadLabel(fileName):
    fileHandle = open(fileName, 'rb')   # 以二进制打开文件
    fileContent = fileHandle.read()     # 读取到缓冲区中
    head = struct.unpack_from('>II', fileContent, 0)    # 取前两个整数 ，返回一个元组
    offset = struct.calcsize('>II')
    labelNum = head[1]     # label数
    # print(labelNum)
    bitstring = '>' + str(labelNum) + 'B'   # fmt格式：'>47040000B'
    label = struct.unpack_from(bitstring, fileContent, offset)      # 取data数据，返回一个元组
    return np.array(label)


# KNN算法
def KNN(testData, dataSet, labels, k):
    # dataSet是训练集数据，labels是训练集标签
    dataSetSize = dataSet.shape[0]  # dataSet.shape[0]表示的是读取矩阵第一维度的长度，代表行数
    # 计算欧式距离
    # 即{[(a1-b1)^2]+[(a2-b2)^2]+...}^0.5
    # tile函数在行上重复dataSetSize次，在列上重复1次
    distance1 = tile(testData, dataSetSize).reshape((60000, 784)) - dataSet
    distance2 = distance1**2       # 每个元素平方
    distance3 = distance2.sum(axis=1)   # 矩阵每行相加
    distance4 = distance3**0.5
    # 欧式距离计算结束
    sortedDistIndicies = distance4.argsort()    # 返回从小到大排序的索引
    classCount = np.zeros(10, np.int32)   # 10代表10个类别
    for i in range(k):      # 统计前k个数据类的数量
        voteLabel = labels[sortedDistIndicies[i]]
        classCount[voteLabel] += 1
    max = 0
    id = 0
    print(classCount.shape[0])
    for i in range(classCount.shape[0]):
        if classCount[i] >= max:
            max = classCount[i]
            id = i
    print(id)
    return id


# 文件获取和测试
def TestKNN():
    # 文件读取
    # minst数据集
    # 训练集文件
    train_image = 'MINST_DATA/train-images.idx3-ubyte'
    # 测试集文件
    test_image = 'MINST_DATA/t10k-images.idx3-ubyte'
    # 训练集标签文件
    train_label = 'MINST_DATA/train-labels.idx1-ubyte'
    # 测试集标签文件
    test_label = 'MINST_DATA/t10k-labels.idx1-ubyte'

    # 读取数据
    trainImage = ReadImage(train_image)
    testImage = ReadImage(test_image)
    trainLabel = ReadLabel(train_label)
    testLabel = ReadLabel(test_label)

    testRatio = 0.01       # 取数据集的前0.1为测试数据，比重可以改变
    trainRow = trainImage.shape[0]      # 数据集的行数，即数据集的总的样本数
    testRow = testImage.shape[0]
    testNum = int(testRow * testRatio)
    errorCount = 0      # 判断错误的个数
    for i in range(testNum):
        result = KNN(testImage[i], trainImage, trainLabel, 30)
        # print('返回的结果是：%s, 真实结果是：%s' % (result, trainLabel[i]))

        print(result, testLabel[i])
        if result != testLabel[i]:
            errorCount += 1.0     # 如果minst验证集的标签和本身标签不一样，则出错
    errorRate = errorCount / float(testNum)     # 计算出错率
    acc = 1.0 - errorRate
    print(errorCount)
    print("\nthe total number of errors is :%d" % errorCount)
    print("\nthe total error rate is :%f" % errorRate)
    print("\nthe total accuracy  rate is :%f" % acc)


if __name__ == "__main__":
    TestKNN()

