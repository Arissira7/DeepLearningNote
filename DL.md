# 深度学习基础



## 安装

### Windows

1. 确定有 Nvidia GPU
2. 安装 CUDA
3. 安装 miniconda (py 3.10)
4. 安装 GPU 版 Pytorch
5. 安装 d2l 和 Jupyter
6. 下载 d2l 记事本运行测试 



## 数据操作，数据预处理



### 张量

#### 定义

- 本质是一个数组可能有多个维度 *( 具有一个轴的张量对应数学上的*向量*（vector）； 具有两个轴的张量对应数学上的*矩阵*（matrix）； 具有两个轴以上的张量没有特殊的数学名称。)*

#### 生成，访问张量

##### `arrange`

我们可以使用 `arange` 创建一个**行向量** `x`。这个行向量包含以0开始的前 n 个整数，它们默认创建为整数。也可指定创建类型为浮点数。张量中的每个值都称为张量的元素（element)

```python
import torch

# 创建一个包含前12个整数的行向量
x = torch.arange(12)

# 创建一个包含前12个浮点数的行向量
x_float = torch.arange(12, dtype=torch.float32)
```



##### `zeros`

使用全 0 来初始化矩阵

```python
import torch

tensor = torch.zeros((2, 4)) 
```



##### `ones`

使用全 1 来初始化矩阵

```python
import torch

tensor = torch.ones((2, 3))
```



##### `randn`

每个元素都从均值为0、标准差为1的标准高斯分布（正态分布）中随机采样

```python
import torch

tensor = torch.randn((3, 4))
```



##### Python 列表

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



##### `shape`

访问张量的形状

```python
import torch

# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 查看形状
print(tensor.shape)  # 输出: torch.Size([2, 3])
```



##### `numel`

访问张量的元素总数

```python
import torch

# 创建一个张量
tensor = torch.tensor([[1, 2, 3], [4, 5, 6]])

# 计算元素总数
total_elements = tensor.numel()
print( total_elements)  # 输出: 6
```



##### `reshape`

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



#### 运算符

##### 逐元素计算

```python
x = torch.tensor([1.0, 2, 4, 8])
y = torch.tensor([2, 2, 2, 2])
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



##### 张量连接



