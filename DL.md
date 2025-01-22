# Preliminaries



## 安装

### Windows

1. 确定有 Nvidia GPU
2. 安装 CUDA
3. 安装 miniconda (py 3.10)
4. 安装 GPU 版 Pytorch
5. 安装 d2l 和 Jupyter
6. 下载 d2l 记事本运行测试 



## 数据操作



### 张量定义

- 本质是一个数组可能有多个维度 *( 具有一个轴的张量对应数学上的*向量*（vector）； 具有两个轴的张量对应数学上的*矩阵*（matrix）； 具有两个轴以上的张量没有特殊的数学名称。)*
- 深度学习存储和操作数据的主要接口是张量（n维数组）。它提供了各种功能，包括基本数学运算、广 播、索引、切片、内存节省和转换其他Python对象。

### 生成张量

#### `arrange`

我们可以使用 `arange` 创建一个**行向量** `x`。这个行向量包含以0开始的前 n 个整数，它们默认创建为整数。也可指定创建类型为浮点数。张量中的每个值都称为张量的元素（element)

```python
import torch

# 创建一个包含前12个整数的行向量
x = torch.arange(12)

# 创建一个包含前12个浮点数的行向量
x_float = torch.arange(12, dtype=torch.float32)
```



#### `zeros`

使用全 0 来初始化矩阵

```python
import torch

tensor = torch.zeros((2, 4)) 
```



#### `ones`

使用全 1 来初始化矩阵

```python
import torch

tensor = torch.ones((2, 3))
```



#### `randn`

每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样

```python
import torch

tensor = torch.randn((3, 4))
```



#### Python 列表

我们还可以通过提供包含数值的Python列表（或嵌套列表），来为所需张量中的每个元素赋予确定值。 在这里，最外层的列表对应于轴0，内层的列表对应于轴1

```python
pylist = [[[1, 2], [2, 3]], [[3, 3], [1, 4]], [[2, 1], [2, 4]]]
tensor = torch.tensor(pylist)

'''
tensor(	[[1, 2],
				[2, 3]],

				[[3, 3], 
				[1, 4]],
        
				[[2, 1], 
				[2, 4]])
'''

```



### 访问张量

#### `shape`

访问张量的形状

```python
import torch

# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 查看形状
print(tensor.shape)  # 输出: torch.Size([2, 3])
```



#### `numel`

访问张量的元素总数

```python
import torch

# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 计算元素总数
total_elements = tensor.numel()
print( total_elements)  # 输出: 6
```



#### `reshape`

要想**改变一个张量的形状**而不改变元素数量和元素值

```python
import torch

'''创建一个张量'''
tensor = torch.tensor([1, 2, 3, 4, 5, 6])

'''改变张量形状'''
new_tensor = tensor.reshape(2, 3) 

'''利用 -1 自动计算维度'''
new_tensor = tensor.reshape(-1, 3) # 自动算行数
new_tensor = tensor.reshape(2, -1) # 自动算列数
```



### 张量运算



#### 逐元素计算

##### 简单算术运算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])

'''简单算术运算'''
x + y, x - y, x * y, x / y, x ** y, torch.exp(x) # 输入x，输出e**x 

'''
输出结果：
(tensor([ 3., 4., 6., 10.]),
tensor([-1., 0., 2., 6.]),
tensor([ 2., 4., 8., 16.]),
tensor([0.5000, 1.0000, 2.0000, 4.0000]),
tensor([ 1., 4., 16., 64.]), 
tensor([2.7183e+00, 7.3891e+00, 5.4598e+01, 2.9810e+03]))
'''
```



##### 布尔运算

X == Y 是逐元素的比较运算符，用来判断两个张量在每个位置上的值是否相等

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

# 比较运算
X == Y

'''
输出结果：
tensor([[False, True, False, True],
[False, False, False, False],
[False, False, False, False]])
'''
```





#### 张量连接

张量连接的两种维度：

- dim=0：沿 **行** 方向（即第 0 维）连接，行数增加。

- dim=1：沿 **列** 方向（即第 1 维）连接，列数增加。

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

torch.cat((X, Y), dim=0), torch.cat((X, Y), dim=1)

'''
输出结果：

(tensor([[ 0., 1., 2., 3.],
[ 4., 5., 6., 7.],
[ 8., 9., 10., 11.],
[ 2., 1., 4., 3.],
[ 1., 2., 3., 4.],
[ 4., 3., 2., 1.]]),
tensor([[ 0., 1., 2., 3., 2., 1., 4., 3.],
[ 4., 5., 6., 7., 1., 2., 3., 4.],
[ 8., 9., 10., 11., 4., 3., 2., 1.]]))
'''
```



#### 张量求和

将各元素相加

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))

X.sum()

'''
输出结果：
tensor(66.)
'''
```



### 广播机制

形状不同的张量按元素操作

- 适当复制元素来扩展数组，使得两个张量形状相同
- 对复制完成后的数组按元素操作

```python
a = torch.arange(3).reshape((3, 1))
b = torch.arange(2).reshape((1, 2))

a + b

'''
输出结果：

tensor([[0, 1],
[1, 2],
[2, 3]])
'''
```



### 索引和切片

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))

'''读取元素'''
X[-1], X[1:3] # [-1]选择最后一个元素，[1:3]选择第二个和第三个元素

'''改写单个元素'''
X[1, 2] = 9 # 将第一个元素里索引2的元素改为9

'''改写多个元素'''
X[0:2, :] = 12 # 将第一二行的所有元素改为12
```



### 节省内存

1. **默认操作**（如 `Y = X + Y`）会导致新内存分配，增加内存开销。

   ```python
   before = id(Y)
   Y = Y + X
   id(Y) == before
   
   '''
   False
   '''
   ```

2. **优化方法**：

   - **切片赋值**（如 `Z[:] = X + Y`）可以避免新内存分配。

     ```python
     Z = torch.zeros_like(Y)
     print(id(Z))
     Z[:] = X + Y
     print(id(Z))
     
     '''
     id(Z): 140327634811696
     id(Z): 140327634811696
     '''
     ```

   - **原位操作**（如 `X += Y` 或 `X.add_(Y)`）可以直接在原张量上操作，减少内存使用。

     ```python
     before = id(X)
     X += Y
     id(X) == before
     
     '''
     True
     '''
     ```

3. **使用场景**：
   - 如果后续不需要保留原值，可以优先考虑原位操作。
   - 如果需要存储结果且保持原张量不变，可以使用切片赋值。



### 转换为其他Python对象

torch张量和numpy数组将共享它们的底层内存，他们可以随意转换

```python
X = torch.arange(12, dtype=torch.float32).reshape((3,4))

A = X.numpy()
B = torch.tensor(A)
type(A), type(B)

'''
(numpy.ndarray, torch.Tensor)
'''
```



将大小为1的张量转换为Python标量

```python
a = torch.tensor([3.5])
a, a.item(), float(a), int(a)

'''
(tensor([3.5000]), 3.5, 3.5, 3)
'''
```



## 数据预处理

### 读取数据集

利用`pandas.read_csv`读取 csv 文件

```python
import pandas as pd

data = pd.read_csv(data_file) # data_file是已有的 csv 文件
print(data_file)

'''
  NumRooms Alley   Price
0      NaN  Pave  127500
1      2.0   NaN  106000
2      4.0   NaN  178100
3      NaN   NaN  140000
'''
```



### 处理缺失值

- 插入法
- 删除法（略）

*基于上述的例子，我们考虑插入法：*

1. 对于`inputs`中缺少的数值，我们用同一列的**均值**替换“NaN”项

```python
inputs, outputs = data.iloc[:, 0:2], data.iloc[:, 2]
inputs = inputs.fillna(inputs.mean())
print(inputs)

'''
	NumRooms  Alley
0 		 3.0   Pave
1 		 2.0    NaN
2 		 4.0    NaN
3 		 3.0    NaN
'''
```



2. 对于`inputs`中的类别值或离散值，我们将“NaN”视为一个类别。

   由于“巷子类型”（“Alley”）列只接受两 种类型的类别值“Pave”和“NaN”，pandas可以自动将此列转换为两列“Alley_Pave”和“Alley_nan”。巷子类型为“Pave”的行会将“Alley_Pave”的值设置为1，“Alley_nan”的值设置为0。缺少巷子类型的行会 将“Alley_Pave”和“Alley_nan”分别设置为0和1。

```python
import pandas

inputs = pd.get_dummies(inputs, dummy_na=True) # dummy_na=True：当设置为 True 时，会为缺失值（NaN）创建一个额外的列
print(inputs)

'''
	NumRooms Alley_Pave Alley_nan
0 		 3.0 					1 				0
1 		 2.0 					0 				1
2 		 4.0 					0 				1
3 		 3.0 					0 				1
'''
```



### 转换为张量格式

`inputs`和`outputs`中的所有条目都是数值类型，它们可以转换为张量格式。当数据采用张量格式后，可以通过张量函数来进一步操作

```python
import torch

x = torch.tensor(inputs.to_numpy(dtype=float))
y = torch.tensor(output.to_numpy(dtype=float))
x, y

'''
(tensor([[3., 1., 0.],
[2., 0., 1.],
[4., 0., 1.],
[3., 0., 1.]], dtype=torch.float64),
tensor([127500., 106000., 178100., 140000.], dtype=torch.float64))
'''
```



## 线性代数

### 标量

一个元素的张量表示

```python
tensor(5.), tensor(4.)
```



### 向量

一个轴的张量表示

```python
tensor([0, 1, 2])
```

索引访问

```python
x= torch.arange(3)
x[2]

'''
tensor(3)
'''
```

向量长度（向量维度）

```python
len(x)

'''
3
'''
```

向量形状

```python
torch.size([3])
```



### 矩阵

两个轴的张量

矩阵转置 

```python
A.T
```



### 张量

向量是一阶张量，矩阵是二阶张量



### 张量算法的基本性质

对于多个形状相同的张量：

- 按元素相加 `A + B`
- 按元素相乘（Hadamard积）`A * B`

对于标量与张量

- 张量按元素与标量相加
- 张量按元素与标量相乘



### 求和与降维

1. 求和会对张量进行降维

- `sum(axis=0)` ：即行间求和，会由多行变成一行
- ` sum(axis=1)` ：即列间求和，会由多列变成一列
- `sum(axis=[0, 1])` ：即对行列同时求和，会压缩为标量

*与求和相关的平均数同样也会降维*



2. 非降维求和（利于进行广播）

- `sum(axis=0, keepdims=True)` ：维持二维结构避免变成一个行向量
- `sum(axis=1, keepdims=True)` ：维持二维结构避免变成一个列向量



3. `cumsum`函数

- 对行累积 `.cumsum(axis=0)`
- 对列累积 `.cumsum(axis=1)`

```python
'''原始张量 A:'''
tensor([[1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]])

'''沿 axis=0（行方向）计算累积'''
tensor([[ 1,  2,  3],  # 第 1 行，保留原始值
        [ 5,  7,  9],  # 第 1 行 + 第 2 行
        [12, 15, 18]]) # 累加前两行 + 第 3 行

'''沿 axis=1（列方向）计算累积'''
tensor([[ 1,  3,  6],  # 第一行：1, 1+2, 1+2+3
        [ 4,  9, 15],  # 第二行：4, 4+5, 4+5+6
        [ 7, 15, 24]]) # 第三行：7, 7+8, 7+8+9
```



### 点积

两个向量按元素乘法并求和

```python
torch.dot(x, y)
```



### 矩阵向量之积

向量先广播到与矩阵形状一致后，向量与矩阵按元素乘法并求和

```python
 torch.mv(A, x)
```



### 矩阵-矩阵乘法

```python
torch.mm(A, B)
```



### 范数

***目标**，或许是深度学习算法最重要的组成部分（除了数据），通常被表达为**范数***

- $L1$ 范数

 $∥x∥_1 = \sum_{i=1}^n |xi|$ 

```python
torch.abs(u).sum()
```

- $L2$ 范数

$∥x∥_2 =\sqrt{ \sum_{i=1}^n x_i^2}$

```python
torch.norm(u)
```

- $Frobenius$ 范数

$∥X∥_F = \sqrt{\sum_{i=1}^m \sum_{j=1}^n x_{ij}^2}$

```python
torch.norm(torch.ones((4, 9)))
```



## 微积分

### 导数，微分



### 偏导数



### 梯度



### 链式法则



## 自动微分

```python
import torch

x = torch.arange(4.0)
x.requires_grad_(Ture) # 表示将要为 x 计算梯度并储存其梯度值

'''标量函数1的自动微分'''
y = 2 * torch.dot(x, x) # 标量函数1
y.backward() # 后向模式自动微分
x.grad # 查看函数1的梯度值
# 输出：tensor([ 0., 4., 8., 12.])

x.grad.zero_() # 清空pytorch累积的梯度

'''标量函数2的自动微分'''
y = x.sum() # 标量函数2
y.backward()
x.grad # 输出：tensor([1., 1., 1., 1.])

x.grad.zero_() # 清空pytorch累积的梯度

'''非标量函数的自动微分'''
y = x * x # 非标量函数
y.sum().backward() 
x.grad 
# 输出：tensor([0., 2., 4., 6.])

x.grad.zero_() # 清空pytorch累积的梯度

'''分离计算的自动微分'''
y = x * x
u = y.detach() # 分离
z = u * x
z.sum().backward()
x.grad == u
# 输出：tensor([True, True, True, True])

'''python控制流的自动微分'''
def f(a):
  b = a * 2
  while b.norm() < 1000:
    b = b * 2
  if b.sum() > 0:
    c = b
  else:
    c = 100 * b
	return c

a = torch.randn(size=())
a.requires_grad_(True)

d = f(a)
d.backward()
a.grad == d / a
# 输出：tensor(True)
```



 ## 概率



## 查阅文档



# 线性神经网络



## 线性回归

### 线性模型



### 损失函数



### 解析解



### 随机梯度下降



## 线性回归的从零开始实现

```python
'''生成数据集'''
def synthetic_data(w, b, num_examples):
  X = torch.normal(0, 1, (num_examples, len(w)))
  y = torch.matmul(X, w) + b
  y += torch.normal(0, 0.01, y.shape)
  return X, y.reshape((-1, 1))

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)

'''数据集散点图'''
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].detach().numpy(), labels.detach().numpy(), 1);

'''读取数据集'''
def data_iter(batch_size, features, labels):
  num_examples = len(features)
  indices = list(range(num_examples))
  random.shuffle(indices)
  for i in range(0, num_examples, batch_size):
    batch_indices = torch.tensor(indices[i: min(i + batch_size, num_examples)])
    yield features[batch_indices], labels[batch_indices] # 张量索引
    
'''初始化模型参数'''
w = torch.normal(0, 0.01, size=(2,1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

'''定义模型'''
def liner(X, w, b):
  return torch.matmul(X, w) + b

'''定义损失函数'''
def squared_loss(y_hat, y):
  return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2

'''定义优化算法'''
def sgd(params, lr, batch_size):
  with torch.no_grad(): # 上下文管理器，确保在此上下文中进行的操作不计算梯度。这通常用于在模型评估或测试时避免计算梯度，从而节省内存
    for param in params:
      param -= lr * param.grad / batch_size
      param.grad.zero_()
      
'''训练'''
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
  for X, y in data_iter(batch_size, features, labels):
    l = loss(net(X, w, b), y)
    l.sum().backward()
    sgd([w, b], lr, batch_size)
    
  with torch.no_grad():
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
    
print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
    
'''
epoch 1, loss 0.042790
epoch 2, loss 0.000162
epoch 3, loss 0.000051
w的估计误差: tensor([-1.3804e-04, 5.7936e-05], grad_fn=<SubBackward0>)
b的估计误差: tensor([0.0006], grad_fn=<RsubBackward1>)
'''
```



## 线性回归的简洁实现

```python
import numpy as np
import torch
from torch.utils import data
from d2l import torch as d2l

'''生成数据集'''
true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = d2l.synthetic_data(true_w, true_b, 1000)

'''读取数据集'''
def load_array(data_arrays, batch_size, is_train=True):
  dataset = data.TensorDataset(*data_arrays)
  return data.DataLoader(dataset, batch_size, shuffle=is_train)

batch_size = 10
data_iter = load_array((features, labels), batch_size)

next(iter(data_iter))
'''
[tensor([[-1.3116, -0.3062],
[-1.5653, 0.4830],
[-0.8893, -0.9466],
[-1.2417, 1.6891],
[-0.7148, 0.1376],
[-0.2162, -0.6122],
[ 2.4048, -0.3211],
[-0.1516, 0.4997],
[ 1.5298, -0.2291],
[ 1.3895, 1.2602]]),
tensor([[ 2.6073],
[-0.5787],
[ 5.6339],
[-4.0211],
[ 2.3117],
[ 5.8492],
[10.0926],
[ 2.1932],
[ 8.0441],
[ 2.6943]])]
'''

'''定义模型'''
from torch import nn
net = nn.Sequential(nn.Linear(2, 1))

'''初始化模型参数'''
net[0].weight.data.normal_(0, 0.01)
net[0].bias.data.fill_(0)
'''
tensor([0.])
'''

'''定义损失函数'''
loss = nn.MSELoss() # 均方误差

'''定义优化算法'''
trainer = torch.optim.SDG(net.parameters(), lr=0.03) # 梯度下降算法

'''训练'''
num_epochs = 3
for epoch in range(num_epochs):
  for X, y in data_iter:
    l = loss(net(X), y)
    trainer.zero_grad()
    l.backward()
    trainer.step()
    
  l = loss(net(features), labels)
  print(f'epoch{epoch + 1}, loss{l:f}')
'''
epoch 1, loss 0.000248
epoch 2, loss 0.000103
epoch 3, loss 0.000103
'''
  
w = net[0].weight.data
print('w的估计误差：', true_w - w.reshape(true_w.shape))
b = net[0].bias.data
print('b的估计误差：', true_b - b)
'''
w的估计误差： tensor([-0.0010, -0.0003])
b的估计误差： tensor([-0.0003])
'''
```



## Softmax回归

Softmax回归是**分类问题**

- 回归估计一个连续值
- 分类预测一个离散类别

![Screenshot 2025-01-21 at 17.41.42](/Users/aris/Library/Application Support/typora-user-images/Screenshot 2025-01-21 at 17.41.42.png)

尽管 Softmax 函数本身是非线性的，它对输入的作用是将线性变换的结果（如 ${w}^T \mathbf{x} + b$ ）转换为概率，因此 Softmax 回归仍然是一个 **线性模型**，这是因为它的核心决策过程依赖于线性组合



**1. 随机得到一个 w 和 b ，将 x 代进线性组合里算出 z **



**2. 输入 z ，通过Softmax函数算出预测概率向量 $\hat y$**
$$
\hat{y} = softmax(z) \ \ \ 其中\ \ \  \hat{y_j} =
\frac {e^{z_j}}{\sum_k{e^{z_k}}}
$$
![Screenshot 2025-01-22 at 04.16.40](/Users/aris/Library/Application Support/typora-user-images/Screenshot 2025-01-22 at 04.16.40.png)

预测概率向量 $\hat y$ 是一个储存该样本所有类别（即节点）输出值的向量， $\hat y_j$ 则是该样本里某一个类别（节点）的输出值



**2. 得到各样本的条件概率（即预测概率）**

$$P(y^i|x^i)={\hat y_j}^i\ \ 这个条件概率表示在给定第\  i个样本的输入特征 \ {x}^{i}\ 的条件下，模型预测它属于类别\  j\  的概率 \\$$

$$i\ 是指某一个样本\ i\\$$

$$j\ 是指该样本的某一个类别是\ j$$



**3. 计算交叉熵损失**

交叉熵：

 $$H[y,\hat y] = \sum_{j} -\ y_j \log \hat y_j \geq H[y] = \sum_{j} -\ y_j \log y_j$$

损失函数：

 $$l(y, \hat y) = \sum_{j} -\ y_j \ log\ \hat y_j \ \ ～～ \ -log\ \hat y$$

后续不断更新 w 和 b 的值，会使得 $\hat y$ 的值越来越准确。当 $\hat y = y$ 时，交叉熵降为信息熵，且损失函数为0

感性理解：我们预先知道真实的 $y$ ，而 $$l(y, \hat y)$$ 就描述了我们看到 $\hat y$ 时会产生的惊讶。只有当 $\hat y = y$ 时，损失函数为0，我们就不会有惊讶产生了



补充：

1. 为什么交叉熵损失等价于负对数似然；

   原因是one-hot编码使得只有正确类别的项被保留，导致交叉熵退化为单个正确类别概率的负对数。所以交叉熵损失和负对数似然完全等价

2. 交叉熵损失是什么 

   （后面）

3. 为什么不直接进行最大似然估计

   理论上当然可以用，但大多数现代深度学习框架都是基于最小化损失设计的，且"下山"比"上山"更直观



**4. 对损失计算梯度并使用优化算法**

相当于降低 预测是某种类别后结果真的是该类别的惊异 即 随着 w 和 b 的不断优化，预测概率会越来越准，最后达到有较高预测准确率的效果

- 计算梯度：

![Screenshot 2025-01-22 at 21.31.19](/Users/aris/Library/Application Support/typora-user-images/Screenshot 2025-01-22 at 21.31.19.png)

- 使用优化算法（如梯度下降算法）

  得到新的 w和 b 后重新计算预测概率，如此循环直到得到一个预测率高的模型



**补充2：信息论基础**

**信息论**（information theory）涉及编码、解码、发送以及尽可能简洁地处理信息或数据。

**熵**

信息论的核心思想是量化数据中的信息内容。

在信息论中，该数值被称为分布$P$的**熵**（entropy）。可以通过以下方程得到：

$$H[P] = \sum_j - P(j) \log P(j).$$

信息论的基本定理之一指出，为了对从分布$p$中随机抽取的数据进行编码，

我们至少需要$H[P]$“纳特（nat）”对其进行编码。

“纳特”相当于**比特**（bit），但是对数底为$e$而不是2。因此，一个纳特是 $\frac{1}{\log(2)} \approx 1.44$比特。

**信息量**

压缩与预测有什么关系呢？想象一下，我们有一个要压缩的数据流。如果我们很容易预测下一个数据，那么这个数据就很容易压缩。为什么呢？举一个极端的例子，假如数据流中的每个数据完全相同，这会是一个非常无聊的数据流。由于它们总是相同的，我们总是知道下一个数据是什么。所以，为了传递数据流的内容，我们不必传输任何信息。也就是说，“下一个数据是xx”这个事件毫无信息量。

但是，如果我们不能完全预测每一个事件，那么我们有时可能会感到"惊异"。克劳德·香农决定用信息量$\log \frac{1}{P(j)} = -\log P(j)$来量化这种惊异程度。在观察一个事件 $j$ 时，并赋予它（主观）概率 $P(j)$。当我们赋予一个事件较低的概率时，我们的惊异会更大，该事件的信息量也就更大。

所以，**熵 $H[P] = \sum_j - P(j) \log P(j)$ 的定义，就是当分配的概率真正匹配数据生成过程时的信息量的期望**。即**把熵 $H(P)$ 想象为“知道真实概率的人所经历的惊异程度”**

简单总结：

信息量就是用概率来描述信息量，因为一件事概率越小，如果这件事最后真的发生了，那么对这件事的惊讶程度就会越大

信息熵就是一个系统内发生某件事，它信息量的期望



**交叉熵**

如果把熵 $H(P)$ 想象为“知道真实概率的人所经历的惊异程度”，那么什么是交叉熵？

交叉熵从 $P$ 到 $Q$，记为 $H(P, Q)$。

我们可以把交叉熵想象为“主观概率为$Q$的观察者在看到根据概率$P$生成的数据时的预期惊异”。当$P=Q$时，交叉熵达到最低。在这种情况下，从$P$到$Q$的交叉熵是$H(P, P)= H(P)$。**简而言之，我们可以从两方面来考虑交叉熵分类目标：（i）最大化观测数据的似然；（ii）最小化传达标签所需的惊异。**



## softmax回归的从零开始实现

```python
import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l

d2l.use_svg_display()

'''
图像分类数据集的读取：
	将Fashion‐MNIST数据集下载并读取到内存中
'''

def load_data_fashion_mnist(batch_size, resize=None): 
  trans = [transforms.ToTensor()]
  if resize:
    trans.insert(0, transforms.Resize(resize))
  
  trans = transforms.Compose(trans)
	
  mnist_train = torchvision.datasets.FashionMNIST(
root="../data", train=True, transform=trans, download=True)
	
  mnist_test = torchvision.datasets.FashionMNIST(
root="../data", train=False, transform=trans, download=True)
	
  return (data.DataLoader(mnist_train, batch_size, shuffle=True, num_workers=get_dataloader_workers()),
data.DataLoader(mnist_test, batch_size, shuffle=False,
num_workers=get_dataloader_workers()))

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)

'''初始化模型参数'''
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

'''定义softmax操作'''
X = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
X.sum(0, keepdim=True), X.sum(1, keepdim=True)

def softmax(X):
	X_exp = torch.exp(X)
	partition = X_exp.sum(1, keepdim=True)
	return X_exp / partition # 这里应用了广播机制
X = torch.normal(0, 1, (2, 5))
X_prob = softmax(X)
X_prob, X_prob.sum(1)

'''定义模型'''
def net(X):
	return softmax(torch.matmul(X.reshape((-1, W.shape[0])), W) + b)

'''定义损失函数'''
y = torch.tensor([0, 2])
y_hat = torch.tensor([[0.1, 0.3, 0.6], [0.3, 0.2, 0.5]])
y_hat[[0, 1], y]

def cross_entropy(y_hat, y):
	return - torch.log(y_hat[range(len(y_hat)), y])

cross_entropy(y_hat, y)

'''分类精度'''
def accuracy(y_hat, y): 
"""计算预测正确的数量"""
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())

accuracy(y_hat, y) / len(y)

def evaluate_accuracy(net, data_iter): #@save
"""计算在指定数据集上模型的精度"""
	if isinstance(net, torch.nn.Module):
		net.eval() # 将模型设置为评估模式
	metric = Accumulator(2) # 正确预测数、预测总数
	with torch.no_grad():
    for X, y in data_iter:
			metric.add(accuracy(net(X), y), y.numel())
	return metric[0] / metric[1]

class Accumulator: 
"""在n个变量上累加"""
	def __init__(self, n):
		self.data = [0.0] * n
	def add(self, *args):
		self.data = [a + float(b) for a, b in zip(self.data, args)]
	def reset(self):
		self.data = [0.0] * len(self.data)
	def __getitem__(self, idx):
		return self.data[idx]
  evaluate_accuracy(net, test_iter)
  
'''训练'''
def train_epoch_ch3(net, train_iter, loss, updater): 
"""训练模型一个迭代周期（定义见第3章）"""
# 将模型设置为训练模式
	if isinstance(net, torch.nn.Module):
		net.train()
# 训练损失总和、训练准确度总和、样本数
metric = Accumulator(3)
for X, y in train_iter:
	# 计算梯度并更新参数
	y_hat = net(X)
	l = loss(y_hat, y)
	if isinstance(updater, torch.optim.Optimizer):
		# 使用PyTorch内置的优化器和损失函数
		updater.zero_grad()
		l.mean().backward()
		updater.step()
	else:
		# 使用定制的优化器和损失函数
		l.sum().backward()
		updater(X.shape[0])
	metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
# 返回训练损失和训练精度
return metric[0] / metric[2], metric[1] / metric[2]
  
class Animator: #@save
"""在动画中绘制数据"""
	def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None, 
ylim=None, xscale='linear', yscale='linear', 
fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, 
figsize=(3.5, 2.5)):
	# 增量地绘制多条线
	if legend is None:
		legend = []
	d2l.use_svg_display()
	self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
	if nrows * ncols == 1:
		self.axes = [self.axes, ]
	# 使用lambda函数捕获参数
	self.config_axes = lambda: d2l.set_axes(
		self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
	self.X, self.Y, self.fmts = None, None, fmts

def add(self, x, y):
# 向图表中添加多个数据点
	if not hasattr(y, "__len__"):
    y = [y]
	n = len(y)
	if not hasattr(x, "__len__"):
		x = [x] * n
	if not self.X:
		self.X = [[] for _ in range(n)]
	if not self.Y:
		self.Y = [[] for _ in range(n)]
	for i, (a, b) in enumerate(zip(x, y)):
		if a is not None and b is not None:
			self.X[i].append(a)
			self.Y[i].append(b)
	self.axes[0].cla()
	for x, y, fmt in zip(self.X, self.Y, self.fmts):
		self.axes[0].plot(x, y, fmt)
	self.config_axes()
	display.display(self.fig)
	display.clear_output(wait=True)
  
def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater): 
"""训练模型（定义见第3章）"""
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9], 
legend=['train loss', 'train acc', 'test acc'])
	for epoch in range(num_epochs):
		train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
		test_acc = evaluate_accuracy(net, test_iter)
		animator.add(epoch + 1, train_metrics + (test_acc,))
	train_loss, train_acc = train_metrics
	assert train_loss < 0.5, train_loss
	assert train_acc <= 1 and train_acc > 0.7, train_acc
	assert test_acc <= 1 and test_acc > 0.7, test_acc
  
lr = 0.1
def updater(batch_size):
	return d2l.sgd([W, b], lr, batch_size)
num_epochs = 10
train_ch3(net, train_iter, test_iter, cross_entropy, num_epochs, updater)

'''预测'''
def predict_ch3(net, test_iter, n=6): 
	"""预测标签（定义见第3章）"""
	for X, y in test_iter:
		break
	trues = d2l.get_fashion_mnist_labels(y)
	preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
	titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
	d2l.show_images(
		X[0:n].reshape((n, 28, 28)), 1, n, titles=titles[0:n])

predict_ch3(net, test_iter)
```



## softmax回归的简洁实现



 

