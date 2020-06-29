#!/usr/bin/env python
# coding: utf-8
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
# get_ipython().run_line_magic('matplotlib', 'inline')
import os


path = 'income.csv'
data = pd.read_csv(path, header = None)


# 因为这个数据中ID没啥用，所以为了方便处理，将ID那一列置为1，作为第一个特征，方便确定$\theta_0$

data[[0]] = data[[0]] / data[[0]]
data.head()

# count_classes = pd.value_counts(data[58], sort = True).sort_index()
# count_classes.plot(kind = 'bar')# kind = 'bar',表示画条形图
# plt.title("income > 50k")
# plt.xlabel("F or T")
# plt.ylabel("Frequency")


Data = data.values 
cols = Data.shape[1]
X = Data[:,0:cols-1] # 特征
y = Data[:,cols-1:cols]  # class

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# print("Number transactions train dataset: ", len(X_train))
# print("Number transactions test dataset: ", len(X_test))
# print("Total number of transactions: ", len(X_train)+len(X_test))
# X_train[:5]
# y_train[:5]

theta = np.zeros([1,58])
# theta.shape


# #### 1.sigmoid函数实现

def sigmoid(z):
    return 1. / (1 + np.exp(-z))



# sns.set_style("whitegrid")
# nums = np.arange(-10, 10, step = 1)
# fig, ax = plt.subplots(figsize=(12, 4))
# ax.plot(nums, sigmoid(nums), 'r')


# #### 2.model(预测函数)实现

def model(X, theta):
    return sigmoid(np.dot(X,theta.T))


# #### 3.损失函数

def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)+0.000000001))
    right = np.multiply(1-y, np.log(1-model(X, theta)+0.000000001))
    return (np.sum(left - right) / len(X))


cost(X_train,y_train,theta)


# #### 4.计算梯度

def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(term) / len(X)
        
    return grad


# #### 几种停止策略

STOP_ITER = 0 # 以迭代次数为准
STOP_COST = 1 # 以损失值为准
STOP_GRAD = 2 # 以梯度为准
# 上面是三种梯度下降停止策略
def stopCriterion(type, value, threshold):
    if type == STOP_ITER:
        return value > threshold
    elif type == STOP_COST:
        return abs(value[-1]-value[-2]) < threshold
    elif type == STOP_GRAD:
        return np.linalg.norm(value) < threshold


# #### 梯度下降求解

import time
# 梯度下降求解
def descent(X, y, theta, batchSize, stopType, thresh, alpha):
    init_time = time.time()
    i = 0  # 迭代次数
    k = 0  # batch
    grad = np.zeros(theta.shape)  # 计算的梯度
    costs = [cost(X, y, theta)]  # 损失值
    while True:
        grad = gradient(X[k:k+batchSize], y[k:k+batchSize], theta)
        k += batchSize  # 取batchSize个数据
        if k >= n:      # n是训练样本个数
            k = 0
        theta = theta - alpha*grad  # 参数更新
        costs.append(cost(X, y, theta))  # 计算新的损失
        i += 1
        
        if stopType == STOP_ITER:
            value = i
        elif stopType == STOP_COST:
            value = costs
        elif stopType == STOP_GRAD:
            value = grad
        if stopCriterion(stopType, value, thresh):
            break
        
    return theta, i-1, costs, grad, time.time() - init_time


def runExpe(X, y, theta, batchSize, stopType, thresh, alpha):
    theta, iter, costs, grad, dur = descent(X, y, theta, batchSize, stopType, thresh, alpha)  
    print("theta:{}".format(theta))
    print("duration:{:03.2f}s".format(dur))
    fig, ax = plt.subplots(figsize=(12,4))
    # print(costs)
    # print(len(costs))
    # print(type(costs))
    ax.plot(np.arange(len(costs)), costs, 'r')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Cost')
    #ax.set_title(name.upper() + ' - Error vs. Iteration')
    ax.set_title('loss')
    return theta


# ### 不同停止策略对结果的影响
# 
# #### 1.设定迭代次数
# 
# #### 2.设定阈值
# 
# 直到损失函数减小的值小于thresh时，停止迭代
# 
# #### 3.根据梯度变化
# 
# 设定阈值thresh，当梯度变化小于这个值的时候停止迭代

# ### 对比不同的梯度下降方法
# 
# #### 1.随机梯度下降
# 
# #### 2.Mini-batch


n = 3000



# runExpe(X_train, y_train, theta, 30, STOP_ITER, thresh=500, alpha=0.01)


# ### 精度预测

def predict(X, theta):
    # 阈值设定为0.5，概率大于0.5预测值为1，小于0.5则为0
    return [1 if x>= 0.5 else 0 for x in model(X, theta)]

from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# theta0 = runExpe(X_train, y_train, theta, 30, STOP_ITER, thresh=50000, alpha=0.0000001)
# y_pred = predict(X_test, theta0)
# accuracy_score(y_test, y_pred)


# ### 实验优化
# 
# ### 用数据预处理之后再次进行实验

# #### 数据预处理
# 
# 利用sklearn的一个函数preprocessing对数据进行预处理，使得生成的数据都在一定范围内波动，可以与没有进行预处理的数据最终的结果进行对比

from sklearn import preprocessing as pp

dealed_data = data.copy()
dealed_data = pp.scale(data, axis=0)
# dealed_data[:5]


# 标准化后数据格式变成了ndarray，不方便观察标准化后结果，给变回来变成dataframe



dealed_data = pd.DataFrame(dealed_data)
dealed_data.head()


# 因为前面用sklearn的一个函数对数据进行标准化的时候把class的值也改变了，所以这里给改回来

dealed_data[58][dealed_data[58]<0]=0
dealed_data[58][dealed_data[58]>0]=1
dealed_data[0] = 1
dealed_data.head()


# 预处理过的数据

dealed_Data = dealed_data.values 
cols2 = dealed_Data.shape[1]
dealed_X = dealed_Data[:,0:cols2-1]
dealed_y = dealed_Data[:,cols2-1:cols2]

from sklearn.model_selection import train_test_split

dealed_X_train, dealed_X_test, dealed_y_train, dealed_y_test = train_test_split(dealed_X, dealed_y, test_size = 0.25, random_state = 0)

# print("Number transactions train dataset: ", len(dealed_X_train))
# print("Number transactions test dataset: ", len(dealed_X_test))
# print("Total number of transactions: ", len(dealed_X_train)+len(dealed_X_test))

# dealed_X_train[:5]

cost(dealed_X_train, dealed_y_train, theta)

theta5 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=50000, alpha=0.1)
y_pred = predict(dealed_X_test, theta5)
accuracy_score(dealed_y_test, y_pred)


# #### 混淆矩阵



def plot_confusion_matrix(cm, classes, title = 'Confusion matrix', cmap = plt.cm.Blues):
    # this fuction is used tp print the confusion matrix
    plt.imshow(cm, interpolation = 'nearest', cmap = cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation = 0)
    plt.yticks(tick_marks, classes)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], 
                 horizontalalignment='center', 
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predict label')



import itertools
from sklearn.metrics import confusion_matrix

y_pred = predict(dealed_X_test, theta5)

cnf_matrix = confusion_matrix(dealed_y_test, y_pred)
np.set_printoptions(precision = 2)

print("recall metric in the test data: ", cnf_matrix[1, 1]/(cnf_matrix[1, 0] + cnf_matrix[1, 1]))
print("accuracy in the test data: ", (cnf_matrix[1, 1] + cnf_matrix[0, 0]) / (cnf_matrix[1,1]+cnf_matrix[1,0]+cnf_matrix[0,1]+cnf_matrix[0,0]))
# plot non-normalized confusion matrix
class_names = [0, 1]
plt.figure()
plot_confusion_matrix(cnf_matrix,
                     classes=class_names,
                     title = 'Confusion matrix')
plt.show()




