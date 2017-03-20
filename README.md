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

上图中，一共有3个隐藏层

## 正向传播

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/10.png)

我们先取只有一个隐藏层来分析
根据下面的公式：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/11.PNG)


注意在这个公式里，s 和 x 是向量，而W 和 U 是矩阵，怎么理解这个s 和 x 呢？
我们假设 x(t) 代表的是中文的“ 我 ”，我们要把个“我” 变成向量（变成向量是因为神经网络是针对向量进行操作的），那我先假设有一个3000维的词库，在这个词库里，第1956位为1 其余为0 代表的是“我”， 那么写成向量的形式就是 {0,0,0·······0，1,0，0·······}，同样我们假设n = 150维
所以把上面的公式写成矩阵的形式，就如下：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/12.PNG)

在这个公式里，输入矩阵是m 维的，输出矩阵是n 维的，那么U 的维度是m x n 而 W的维度就是n x n。 这里的m 维指的是，假设我们希望x3代表的是“我”，那么x3的维度就是m, **而不是说**一条s链有多少个x就是多少维，所以这个词库的维度和正向传播标题下的图来分析，这里的m 取3000 而不是取6

下面来看w 和 u 的下标
u（j,i） 指的是 **输入层**第 i 维神经元（一共3000维），到**循环层** 第j 维（一共150维）神经元的权重


w(j,i) 指的是**循环层** t-1 时刻 第 i 维神经元（一共150维），到**循环层**t 时刻，第j维（一共150维）神经元的权重

### 正向传播代码分析
#### 循环神经网络类Class_RNN的构建

我们观察下图，看看构建RNN 类，需要哪些元素


![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/10.png)

1. 输入的x（t）的维度，在上面的例子中维度为3000

2. 中间每一个状态s（t）的维度，在上面的例子中维度是150

3. W矩阵和U矩阵

4. 记录一条链上的时刻个数的,在上面的例子中时刻个数为6

```

class RNN_layer(object):

    def _init_(self,input_width,state_width,learning_rate,activators):

        self.input_width = input_width # the dimension of  x

        self.state_width = state_width  # the dimension of each state

        self.activators = activators

        self.learning_rate = learning_rate

        self.U = np.randoms.uniform(-1e-4,1e-4,(state_width,input_width))

        self.W = np.zeros(-1e-4,1e-4,(state_width,state_width))

        # we also need to save each state

        self.state_list = []
        self.state_list.append(np.zeros(state_width,1))

        self.times = 0
```
