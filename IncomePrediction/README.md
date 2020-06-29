## 小结
班级：CS1807  
学号：U201814752  
姓名：彭良鑫  

本次实验采用Jupyter Notebook平台，因此最终文件是.ipynb格式，.py文件 是直接在Jupyter Notebook中直接导出的。sklearn_LR.ipynb是使用sklearn调库的方式实现的逻辑回归预测模型，目的是与自己写的模型进行对比。

自己的模型最好的效果如下图所示：

精度达到了0.922

![NUZudS.png](https://s1.ax1x.com/2020/06/23/NUZudS.png)

下面是这个结果的混淆矩阵，召回率为0.868

![NUZQiQ.png](https://s1.ax1x.com/2020/06/23/NUZQiQ.png)



使用sklearn训练的模型，效果如下图所示：

![NNZFh9.png](https://s1.ax1x.com/2020/06/23/NNZFh9.png)

经过比较，自己训练的模型和sklearn训练的模型相差不大，说明自己训练的模型效果还不错。

对.py文件 做了点修改现在也可以运行，不过不熟悉python语法，不太规范。

![Nh1RFs.png](https://s1.ax1x.com/2020/06/29/Nh1RFs.png)

可以修改242行红框的内容，第一个是mini-batch算法的batchsize，第二个是迭代次数，第三个是学习率。