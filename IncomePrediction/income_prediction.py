#!/usr/bin/env python
# coding: utf-8

# ### 逻辑回归
# 
# 目标：建立分类器，求解出57个参数
# 
# 设定阈值：根据阈值判断收入是否大于50k
# 
# #### 公式推导：
# 
# sigmoid函数：
# 
# $$g(z)=\frac{1}{1+e^ {-z}} $$
# 
# 根据线性回归中的预测函数$h_\theta(x)$得到一个预测值
# 
# $$h_\theta(x)=\theta_0+\theta_1x_1+\theta_2x_2+...$$
# 
# $$h_\theta(x)=\sum_{i=0}^{m}{\theta_ix_i}=\theta^Tx$$
# 
# 再将$h_\theta(x)$带入sigmoid函数里,就可以完成有预测值到概率的转换，即完成分类任务
# 
# 新的预测函数$h_\theta(x)$为：
# 
# $$h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$$
# 
# 
# 这是一个二分类任务，所以概率可以表示为：
# 
# $$ p(y=1|x;\theta) = h_\theta(x)$$
# 
# $$p(y=0|x;\theta)=1-h_\theta(x)$$ 
# 
# 上面两个式子整合在一起，可以得到
# 
# $$p(y|x;\theta)=(h_\theta(x))^y(1-h_\theta(x))^{1-y}$$
# 
# 得到似然函数
# 
# $$L(\theta)=\prod_{i=1}^{m}{p(y_i|x;\theta)}=\prod_{i=1}^{m}{(h_\theta(x))^{y_i}(1-h_\theta(x))^{1-y_i}}$$
# 
# 再对数似然得到：
# 
# $$l(\theta)=logL(\theta)=\sum_{i=1}^{m}{(y_ilogh_\theta(x_i)+(1-y_i)log(1-h_\theta(x_i)))}$$
# 
# 
# 根据极大似然估计，要求最大值，相当于梯度上升，但是要用梯度下降来解决问题，所以引入$J(\theta)$
# 
# $$J(\theta) = -\frac{1}{m}l(\theta)$$
# 
# 引入正则化参数，使用L2正则化：
# 
# $$J(\theta) = -\frac{1}{m}l(\theta)+\frac{\lambda}{2m}\sum_{j=1}^{n}{\theta_{j}^2}$$
# 
# 即：
# 
# $$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}{(y_ilogh_\theta(x_i)+(1-y_i)log(1-h_\theta(x_i)))}+\frac{\lambda}{2m}\sum_{j=1}^{n}{\theta_{j}^2}$$
# 
# 然后在$J(\theta)$中对$\theta$求偏导，结果为：
# 
# 未引入正则化参数时：
# 
# $$\frac{\partial J(\theta)}{\partial\theta_j} = \frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)x_i^j$$
# 
# 引入正则化参数$\lambda$后：
# 
# $$\frac{\partial J(\theta)}{\partial\theta_j} = (\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)x_i^j)+\frac{\lambda}{m}\theta_j$$
# 
# $x_i^j$  :其中i表示第i个样本，j表示该样本的第j个特征
# 
# 
# 最后就可以进行参数更新了
# 
# 未引入正则化参数时：
# 
# $$\theta_j=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)x_i^j$$
# 
# 引入正则化参数$\lambda$后：
# 
# $$\theta_j=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}((h_\theta(x_i)-y_i)x_i^j+\frac{\lambda}{m}\theta_j)$$
# 
# 式子中$\alpha$表示学习率或步长
# 
# $\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)x_i^j$表示更新的方向
# 
# 步长* 方向=更新的值
# 
# 

# ### 要完成的模块
# 
# 0.数据切分:分为测试集（3000）和验证集（1000）
# 
# 1.sigmoid：映射到概率的函数
# 
# 2.model：返回预测结果
# 
# 3.cost：根据参数计算损失
# 
# 4.gradient：计算每个参数的梯度方向
# 
# 5.descent：进行参数更新
# 
# 6.accuracy：计算精度

# In[12]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
get_ipython().run_line_magic('matplotlib', 'inline')


# In[13]:


import os
path = 'income.csv'
data = pd.read_csv(path, header = None)

data.head()


# In[14]:


data.shape


# 因为这个数据中ID没啥用，所以为了方便处理，将ID那一列置为1，作为第一个特征，方便确定$\theta_0$

# In[15]:


data[[0]] = data[[0]] / data[[0]]
data.head()


# In[16]:


count_classes = pd.value_counts(data[58], sort = True).sort_index()
count_classes.plot(kind = 'bar')# kind = 'bar',表示画条形图
plt.title("income > 50k")
plt.xlabel("F or T")
plt.ylabel("Frequency")


# 可以发现样本中收入大于60k和小于的差距不是特别悬殊，故暂时不对样本进行均衡处理
# 
# 均衡处理方案：下采样（同样少）、过采样（同样多）

# #### 0.数据切分
# 
# 利用sklearn中的train_test_split函数

# In[17]:


Data = data.values 
cols = Data.shape[1]
X = Data[:,0:cols-1] # 特征
y = Data[:,cols-1:cols]  # class

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

print("Number transactions train dataset: ", len(X_train))
print("Number transactions test dataset: ", len(X_test))
print("Total number of transactions: ", len(X_train)+len(X_test))


# In[18]:


X_train[:5]


# In[19]:


y_train[:5]


# 将参数$\theta$初始化

# In[20]:


theta = np.zeros([1,58])
theta.shape


# #### 1.sigmoid函数实现
# 
# $$g(z)=\frac{1}{1+e^ {-z}} $$

# In[21]:


def sigmoid(z):
    return 1. / (1 + np.exp(-z))


# In[22]:


sns.set_style("whitegrid")
nums = np.arange(-10, 10, step = 1)
fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(nums, sigmoid(nums), 'r')


# #### 2.model(预测函数)实现
# 
# $$ h_\theta(x)=g(\theta^Tx)=\frac{1}{1+e^{-\theta^Tx}}$$

# In[23]:


def model(X, theta):
    return sigmoid(np.dot(X,theta.T))


# #### 3.损失函数
# 
# 将对数似然函数去掉负号
# 
# $$Cost(h_\theta(x),y) = -y\log(h_\theta(x))-(1-y)\log(1-h_\theta(x))$$
# 
# 另外引入正则化参数后，损失函数变为，使用L2正则化：
# 
# 求平均损失：
# 
# $$J(\theta)=\frac{1}{m}\sum_{i=1}^{m}{Cost(h_\theta(x_i),y_i)}$$
# 
# 另外引入正则化参数后，损失函数变为，使用L2正则化：
# 
# $$J(\theta) = -\frac{1}{m}\sum_{i=1}^{m}{(y_ilogh_\theta(x_i)+(1-y_i)log(1-h_\theta(x_i)))}+\frac{\lambda}{2m}\sum_{j=1}^{n}{\theta_{j}^2}$$
# 
# 
# 
# $$model(X,theta) = h_\theta(x)$$

# In[58]:


def cost(X, y, theta):
    left = np.multiply(-y, np.log(model(X, theta)+0.000000001))
    right = np.multiply(1-y, np.log(1-model(X, theta)+0.000000001))
    return (np.sum(left - right) / len(X))


# In[59]:


cost(X_train,y_train,theta)


# #### 4.计算梯度
# 
# $$\frac{\partial J}{\partial\theta_j} = -\frac{1}{m}\sum_{i=1}^{m}(y_i-
# h_\theta(x_i))x_i^j$$
# 
# 引入正则化后：
# 
# $$\frac{\partial J(\theta)}{\partial\theta_j} = (\frac{1}{m}\sum_{i=1}^{m}(h_\theta(x_i)-y_i)x_i^j)+\frac{\lambda}{m}\theta_j$$

# In[60]:


def gradient(X, y, theta):
    grad = np.zeros(theta.shape)
    error = (model(X, theta)-y).ravel()
    for j in range(len(theta.ravel())):
        term = np.multiply(error, X[:,j])
        grad[0, j] = np.sum(term) / len(X)
        
    return grad


# #### 几种停止策略

# In[73]:


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

# In[74]:


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


# In[75]:


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

# In[76]:


n = 3000


# In[78]:


runExpe(X_train, y_train, theta, 30, STOP_ITER, thresh=500, alpha=0.01)


# ### 精度预测

# In[83]:


def predict(X, theta):
    # 阈值设定为0.5，概率大于0.5预测值为1，小于0.5则为0
    return [1 if x>= 0.5 else 0 for x in model(X, theta)]


# In[84]:


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


# In[85]:


theta0 = runExpe(X_train, y_train, theta, 30, STOP_ITER, thresh=50000, alpha=0.0000001)
y_pred = predict(X_test, theta0)
accuracy_score(y_test, y_pred)


# ### 实验优化
# 
# ### 用数据预处理之后再次进行实验

# #### 数据预处理
# 
# 利用sklearn的一个函数preprocessing对数据进行预处理，使得生成的数据都在一定范围内波动，可以与没有进行预处理的数据最终的结果进行对比

# In[94]:


from sklearn import preprocessing as pp

dealed_data = data.copy()
dealed_data = pp.scale(data, axis=0)
dealed_data[:5]


# 标准化后数据格式变成了ndarray，不方便观察标准化后结果，给变回来变成dataframe

# In[95]:


dealed_data = pd.DataFrame(dealed_data)
dealed_data.head()


# 因为前面用sklearn的一个函数对数据进行标准化的时候把class的值也改变了，所以这里给改回来

# In[96]:


dealed_data[58][dealed_data[58]<0]=0
dealed_data[58][dealed_data[58]>0]=1
dealed_data[0] = 1
dealed_data.head()


# In[97]:


# 预处理过的数据

dealed_Data = dealed_data.values 
cols2 = dealed_Data.shape[1]
dealed_X = dealed_Data[:,0:cols2-1]
dealed_y = dealed_Data[:,cols2-1:cols2]

from sklearn.model_selection import train_test_split

dealed_X_train, dealed_X_test, dealed_y_train, dealed_y_test = train_test_split(dealed_X, dealed_y, test_size = 0.25, random_state = 0)

print("Number transactions train dataset: ", len(dealed_X_train))
print("Number transactions test dataset: ", len(dealed_X_test))
print("Total number of transactions: ", len(dealed_X_train)+len(dealed_X_test))


# In[98]:


dealed_X_train[:5]


# In[100]:


cost(dealed_X_train, dealed_y_train, theta)


# In[102]:


theta1 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=500, alpha=0.01)
y_pred = predict(dealed_X_test, theta1)
accuracy_score(dealed_y_test, y_pred)


# In[114]:


theta6 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=500, alpha=0.001)
y_pred = predict(dealed_X_test, theta6)
accuracy_score(dealed_y_test, y_pred)


# In[104]:


theta9 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=1)
y_pred = predict(dealed_X_test, theta9)
accuracy_score(dealed_y_test, y_pred)


# In[105]:


theta10 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.1)
y_pred = predict(dealed_X_test, theta10)
accuracy_score(dealed_y_test, y_pred)


# In[106]:


theta11 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.01)
y_pred = predict(dealed_X_test, theta11)
accuracy_score(dealed_y_test, y_pred)


# In[107]:


theta6 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.001)
y_pred = predict(dealed_X_test, theta6)
accuracy_score(dealed_y_test, y_pred)


# In[108]:


theta12 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.0001)
y_pred = predict(dealed_X_test, theta12)
accuracy_score(dealed_y_test, y_pred)


# In[109]:


theta14 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.00001)
y_pred = predict(dealed_X_test, theta14)
accuracy_score(dealed_y_test, y_pred)


# In[115]:


theta2 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.01)
y_pred = predict(dealed_X_test, theta2)
accuracy_score(dealed_y_test, y_pred)


# In[116]:


theta8 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=50000, alpha=0.01)
y_pred = predict(dealed_X_test, theta8)
accuracy_score(dealed_y_test, y_pred)


# In[427]:


theta3 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=5000, alpha=0.1)
y_pred = predict(dealed_X_test, theta3)
accuracy_score(dealed_y_test, y_pred)


# In[110]:


theta4 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=10000, alpha=0.1)
y_pred = predict(dealed_X_test, theta4)
accuracy_score(dealed_y_test, y_pred)


# In[111]:


theta5 = runExpe(dealed_X_train, dealed_y_train, theta, 30, STOP_ITER, thresh=50000, alpha=0.1)
y_pred = predict(dealed_X_test, theta5)
accuracy_score(dealed_y_test, y_pred)


# #### 混淆矩阵

# In[112]:


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


# In[113]:


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


# In[ ]:




