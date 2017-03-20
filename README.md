# RNN-Recurrent-Nectural-Network

## 基本RNN 的概述

当前的输出值 s 不仅取决于此刻的输入值，也取决于前一时刻的输出值，在这里引入了一个时刻的变量
具体看下图
![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/1.jpg)

x 是输入层的值，s 是隐藏层的值，o 是输出层的值，如果没有那个W 的回旋箭头，那么这就是一个全连接的神经网络

现在多了一个W的回旋箭头，意味着此时输出的o 值不仅取决于此刻的输入值x 还取决于前一时刻的隐藏层的值s（t-1）

展开上图，我们得到这个图

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/2.jpg)

我们现在盯着中间的x(t),根据上面的分析，我们能得到下面的表达式：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/3.png)

在这个表达式里，U 是输入层到隐藏层的 **权重矩阵**， W是上一个隐藏层的值到这一个隐藏层值得**权重矩阵**
如果我们把这个关于s 的链条扩展，就能得到下面的表达式：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/4.PNG)


## 双向RNN 概述

首先看下图

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/5.png)

由于有正向和反向的存在，我们盯着y2 ,发现它不仅和A2 有关，也和 A'2 有关，具体的表达式变成：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/6.PNG)


其中正向的A2 和 反向的A'2 的表达式分别为

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/7.PNG)

抽象出来，表达式变成如下：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/8.PNG)


这个表达式的重点在于，对于所有x 连接到正向传递的s 链，也就是A2 这条链，它们共享一个权重矩阵W， 而同样的x 连接到反向传递的s 链，也就是A'2这条链，它们共享另一个权重矩阵W'，这个W 和 W' 是不同的。

同理，V 和 V' ,U 和 U' 也是不同的


## 深度RNN 概述

上面是隐藏层个数为1 的情况，隐藏层个数为1 意味着只有一条正向传播的s 链，加上一条反向传播的s链。


![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/9.png)
