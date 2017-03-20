# RNN-Recurrent-Nectural-Network

## RNN 的概述

当前的输出值 s 不仅取决于此刻的输入值，也取决于前一时刻的输出值，在这里引入了一个时刻的变量
具体看下图
![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/1.jpg)

x 是输入层的值，s 是隐藏层的值，o 是输出层的值，如果没有那个W 的回旋箭头，那么这就是一个全连接的神经网络

现在多了一个W的回旋箭头，意味着此时输出的o 值不仅取决于此刻的输入值x 还取决于前一时刻的隐藏层的值s（t-1）

展开上图，我们得到这个图

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/2.jpg)

