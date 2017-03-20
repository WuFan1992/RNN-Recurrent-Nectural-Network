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
#### 循环神经网络类RNN_layer的构建

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


#### 正向传播代码


```
def RNN_forward(self,input_array):

    self.times +=1

    state = (np.dot(self.U,input_array) + np.dot(self.W,self.state_list[-1]))

    treat_element(state,self.activator.forward)

    self.state_list.append(state)
```
代码第三行treat_element 实现的是由加权输入，到激励值输出的运算
也就是实现式2中的f 运算，代码的关键在第二行，前一个s 值用state_list[-1]




## 反向传递

对比卷积神经网络和神经网络，误差在传递时，都是横向单方向传递，但是在循环神经网络中，误差有两个传递方向，一个是横向的，沿着时刻的s 链传递，一个是纵向的，沿着隐藏层在层之间传递。
横向传递 ，误差只和W 有关，纵向传递，误差只和U有关

### 横向传递

关于横向传递的表达式如下

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/13.PNG)


![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/14.PNG)

上式左边一项等于

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/15.PNG)

上式右边一项等于

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/16.PNG)

结合在一起等于

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/17.PNG)

下面正式开始求误差，根据定义，误差是误差函数关于加权输入的偏导数，根据链式法则，得到

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/18.PNG)

在我们的代码中，我们先来假设从K+1 时刻反向传递到 K时刻，这个时候根据上面这个式子，公式化为：

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/19.JPG)


我们思考一下卷积神经网络，误差sensitive map是一个m x n的矩阵，而且随着层数的变化，sensitive map 的长和宽也在变化，但是在循环神经网络中，由于一条s链中，所有的state 都是同一维度的，所以对于误差sensitive map 而言，它的sensitive map 的维度就是 state 的维度

我们先考虑从k+1 层传递到k 层的误差sensitive map
```
def calcul_delat_k(self,k,activator):

    state = self.state_list[k+1]

    treat_element(state,activator.backward)

    deltat_k = np.dot((np.dot(self.state_list[k+1].T,self.W)),np.diag(state[:])).T
```
接着我们考虑整一条 s 链

```
def calcul_deltat(self,sensitive_map,activator):

    deltat_list = []

    # make all the deltat to 0
    for i in range(self.times):
        self.deltat_list.append(np.zeros((self.state_width,1)))
        self.deltat_list.append(sensitive_map)

    # backward tranformation
    for j in range(self.times -1,0,-1):
        deltat_k = calcul_deltat_k(j,activator)
        deltat_list.append(deltat_k)
```
这段代码特别注意，因为误差sensitive map 是从后往前计算的，所以需要初始化所有的deltat_list 然后把s链最终的一个sensitive map 压入这个列表中

### 纵向传递

纵向传递实际上和全连接层的完全一致
公式如下

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/19.PNG)

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/20.PNG)

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/21.PNG)



### 权重的传递

最关键的在于权重的传递，也就是要求出误差函数关于权重的偏导数

总的权重传递图如下

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/22.png)

公式如下

![](https://github.com/WuFan1992/RNN-Recurrent-Nectural-Network/blob/master/image/23.PNG)

