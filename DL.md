# 深度学习基础



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

### 生成，访问张量

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



### 运算符



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





### 处理缺失值





### 转换为张量格式

