---
title: 子空间学习(4)-LPP&NPE
catalog: true
date: 2022-01-14 12:24:17
subtitle: Subspace Learning-LPP&NPE
top: 11
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Locality Preserving Projections（2004 NIPS）**和**Neighborhood Preserving Embedding（2005 ICCV）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述LPP和NPE的动机，目标函数和限制等。额，然后本系列的解释应该是中英文混合，私以为用简略的句子生动地描述复杂抽象的概念，需要对算法有着极为深刻的理解和一定的勇气。显然我水平不够，如果强制翻译一些原文的描述和概念，难免不太妥当。

# 摘要：

信息处理中的许多问题都涉及某种形式的降维。 因此，人们对降维算法很感兴趣。 与旨在保留全局欧几里德结构的主成分分析，我们将介绍两种旨在保留流形上的局部邻域结构的子空间学习算法：邻域保留嵌入（NPE）是局部线性嵌入（LLE）的线性近似，而LPP是拉普拉斯映射（LE）的线性近似。 在本报告中，我们尝试以数学方式解释 NPE 和 LPP 的主要思想，并讨论 PCA、LLE、LE、NPE 和 LPP 之间的区别。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。



# Method

## 假设

数据点可能位于非线性子流形上，但假设每个局部邻域都是线性的。

## 解决的问题

NPE和LPP中每个词的含义可以分别看作三个算法步骤的，这两个算法的很多步骤都是相似的。 更具体地说，流形的“邻域”（Neighborhood，Locality）使我们能够构建邻接图，Perserving的处理是计算权重 $W$，而“嵌入”（Embedding和Projection）则计算投影。

## LPP和NPE的算法理解

### 构建邻域图

NPE 和LPP中的“邻域”指的是只有邻域对重建有贡献。

由流形的定义可知，流形中的每一点$X_{a}$在欧氏空间中都有一个邻域$X_{U(a)}$同胚的开集$W_{a}$：
$$
\begin{equation}
    \begin{gathered}
        \forall X_{a} \ \exists X_{U(a)} : \ f(X_{U(a)})\rightarrow W_{a} \ (W_{a} \subseteq \mathbb{R}^{n}) \\
        \forall Y_{a} \ \exists Y_{U(a)} : \ f(Y_{U(a)})\rightarrow W_{a} \ (W_{a} \subseteq \mathbb{R}^{n}),
    \end{gathered}
\end{equation}
$$
其中 $W_{a}$ 是不变的，$X$ 是高维数据，$Y$ 是低维数据。 更具体地说，NPE 中的“Neighborhood”使用两种方式构建邻接图$G$：

- $K$ 最近邻（KNN）：如果$x_{j}$ 是$x_{i}$的$k$个最近邻居之一，那么在第$i$个节点和第$j$个节点连一条边。
- 如果 $\|x_{i}-x_{j}\|< \epsilon$，则在节点 $i$ 和 $j$ 之间放置一条边。

### 计算邻域矩阵

这一步中NPE和LPP的做法不同。

#### NPE计算$W$

NPE中的“保留”是指保留流形的局部不变性。 在这一步中，NPE算法尝试计算权重 $W_{ij}$，以每个点的邻域为基础，最好地线性重建每个数据点 $X_{i}$，即最小化线性的重构损失，见公式（1）：
$$
\begin{equation}
    \begin{gathered}
        \arg \min_{W} E(W) = \sum_{i}|X_{i}-\sum_{j}W_{ij}X_{ij}|^2 \\
        \text{ s.t. }\sum_{j}W_{ij}=1,
    \end{gathered}
\end{equation}\tag{1}
$$
其中 $X_{ij}$ 是 $X_{i}$ 的邻居，$\sum_{j}W_{ij}X_{ij}$ 是局部线性重建。

此外，我们解释为什么我们需要约束 $\sum_{j}W_{ij}=1$。 约束是平移不变性、旋转不变性和伸缩不变性的必要条件。 并且可以通过约束$\sum_{j}W_{ij}=1$的拉格朗日乘子法来最小化重构误差。

#### LPP计算$W$

LPP 中的“保留”是指保留数据集的邻域结构。 更具体地说，数据集的邻域结构可以用线性变换$W$来表示。 在这一步中，我们有两种对边进行加权的方法，即计算权重 $W_{ij}$：

- 核函数：如果节点 $i$ 和 $j$ 相连，则放置
  $$
       W_{ij}=e^{-\frac{\|x_{i}-x_{j}\|^{2}}{t}}。
  $$

- 简单连接：$W_{ij}=1$ 当且仅当顶点 $i$ 和 $j$ 由一条边连接。

### 映射

#### NPE计算嵌入

NPE 中的“嵌入”是指将高维数据点 $X_{i}$ 线性映射到低维嵌入坐标，如公式（2）：
$$
\begin{equation}
    \begin{gathered}
        x_{i}\rightarrow y_{i}=A^{T}x_{i}\\
        A=(a_{0}, a_{1}, \cdots, a_{d-1}),
    \end{gathered}
\end{equation}\tag{2}
$$
其中 $y_{i}$ 是一个 d 维向量，而 $A$ 是一个 $n\times d$ 矩阵。 同时，列向量$a_{0}、a_{1}、\cdots、a_{d-1}$是公式(3)中分解问题的解。 根据它们的特征值排序，$\lambda_{0}\leq \cdots \leq \lambda_{d-1}$，见公式(3)：
$$
\begin{equation}
    XMX^{T}a=\lambda XX^{T}a,
\end{equation}\tag{3}
$$
其中 $X=(x_{1},\cdots,x_{m}), M=\left(IW\right)^{T}\left(IW\right), I=diag(1,\cdots,1 )$。

#### LPP计算映射

LPP 中的“投影”是指线性投影，它可以最佳地保留流形上高维数据的邻域结构。

投影与公式(2)类似，但获得投影 $A$ 的方式不同。 列向量 $a_{0}, a_{1}, \cdots, a_{d-1}$ 是公式(4)中特征分解问题的解。根据它们的特征值排序，$\lambda_{0}\leq \cdots \leq \lambda_{d-1}$，见公式(4):
$$
\begin{equation}
        XLX^{T}\mathbf{a}=\lambda XDX^{T}\mathbf{a}
\end{equation}\tag{4}
$$
其中 $D$ ($D_{ii}=\sum_{j}W_{ij}$) 是一个对角矩阵，其条目是 $W$ 的列（或行，因为 $W$ 是对称的）和。 $L=D-W$ 是拉普拉斯矩阵。

## 具体的推导过程

在这一部分，我们展示NPE和LPP实现的数学细节。

### NPE数学推导

在 NPE 中，每个数据点都可以表示为其邻居的线性组合。边权可以通过最小化目标目标函数来计算，即最小化重构误差，见等式：
$$
\begin{equation}
    \phi(W)=\sum_{i}\left\|\mathbf{x}_{i}-\sum_{j} W_{i j} \mathbf{x}_{j}\right\|^{2}.
\end{equation}
$$
设 $\mathbf{y}=\left(y_{1}, y_{2}, \cdots, y_{m}\right)^{T}$ 是这样的映射。选择一个好的映射的合理标准是最小化以下目标函数，见公式(5):
$$
\begin{equation}
    \Phi(\mathbf{y})=\sum_{i}\left(y_{i}-\sum_{j} W_{i j} y_{j}\right)^{2},
\end{equation}\tag{5}
$$
在适当的约束下。该成本函数基于局部线性重构的误差，但在这里我们在优化坐标 $y_{i}$ 的同时固定权重 $W_{ij}$。假设变换是线性的，即$y^{T}=a^{T}X$，其中$X$的第i列向量为$x_{i}$。我们定义：
$$
z_{i}=y_{i}-\sum_{j}W_{ij}y_{ij},
$$
同时$z_{i}$可以写成向量形式：
$$
\begin{equation}\nonumber
    \begin{aligned}
        \mathbf{z}&=\mathbf{y}-w\mathbf{y}\\
        &=(I-W)\mathbf{y}.
    \end{aligned}
\end{equation}
$$
由此目标函数公式(5)可以简化为：
$$
\begin{equation}\nonumber
    \begin{aligned}
        \Phi(\mathbf{y}) &=\sum_{i}\left(y_{i}-\sum_{j} W_{i j} y_{j}\right)^{2} \\
        &=\sum_{i}\left(z_{i}\right)^{2} \\
        &=\mathbf{z}^{T} \mathbf{z} \\
        &=\mathbf{y}^{T}(I-W)^{T}(I-W) \mathbf{y} \\
        &=\mathbf{a}^{T} X(I-W)^{T}(I-W) X^{T} \mathbf{a} \\
        & \doteq \mathbf{a}^{T} X M X^{T} \mathbf{a}
    \end{aligned}
\end{equation}
$$
其中矩阵 $XMX^{T}$ 是对称和半正定的。为了去除投影中的任意缩放因子，NPE 施加如下约束：
$$
\mathbf{y}^{T} \mathbf{y}=1 \Longrightarrow \mathbf{a}^{T} X X^{T} \mathbf{a}=1.
$$
最后，最小化问题简化为求解如下问题，
$$
\begin{equation}
    \underset{\mathbf{a}^{T} X X^{T} \mathbf{a}=1}{\arg \underset{\mathbf{a}}{\min}} \mathbf{a}^{T} X M X^{T} \mathbf{a}.
\end{equation}
$$
然后，我们用拉格朗日乘子法求解上述目标函数，见以下推导：
$$
\begin{equation}
    \begin{aligned}
        f(Y)&=(\textbf{a}^{T}X)M(\textbf{a}^{T}X)^{T} - \lambda(\textbf{a}^{T}X)(\textbf{a}^{T}X)^{T} \\
        \frac{\partial f(Y)}{\partial \textbf{a}}&=2XMX^{T}\textbf{a}-2\lambda XX^{T}\textbf{a}=0 \\
        & \ \ \ \ \ \ X M X^{T} \mathbf{a}=\lambda X X^{T} \mathbf{a}.
    \end{aligned}
\end{equation}
$$
其中$\textbf{a}$的求解可以用特征值分解或者SVD来求解。

### LPP的数学推导

给定一个数据集，我们构建一个加权图 $G = (V, E)$，其中边将附近的点彼此连接起来。假设图是连通的。设 $Y=[\textbf{y}_{1}, \textbf{y}_{2}, \cdots, \textbf{y}_{n}]$ 是这样的映射，并且 $\textbf{y }=(y_{1}, y_{2}, \cdots, y_{n})^{T}$。然后，我们可以将目标函数转换为半正定二次型的迹作为以下推导，
$$
\begin{equation}\nonumber
    \begin{aligned}
        &\ \ \ \ \sum_{i,j}\|y_{i}-y_{j}\|^{2}W_{ij}=\Phi(Y) \\
        &=\sum_{i=1}^{n}\sum_{j=1}^{n}(y_{i}y_{i}-2y_{i}y_{j}+y_{j}y_{j})W_{ij}\\
        &=2 \sum_{i=1}^{n} D_{i i} y_{i}^{2}-2 \sum_{i=1}^{n} \sum_{j=1}^{n} y_{i} y_{j} W_{i j} \\
        &=2 \operatorname{tr}\left[Y^{T}(D-W) Y\right] \\
        &=2 \operatorname{tr}\left(Y^{T} L Y\right)=2\textbf{y}^{T}L\textbf{y}.
    \end{aligned}
\end{equation}
$$
在投影中为消除任意缩放因子添加一个约束。因此，目标函数的优化问题可以转化为如下形式：
$$
\begin{equation}\nonumber
    \begin{gathered}
        \arg \min_{Y^{T}}\Phi(Y)=2\textbf{y}^{T}L\textbf{y} \\
        \text{ s.t. } \textbf{y}^{T}D\textbf{y}=1.
    \end{gathered}
\end{equation}\tag{6}
$$
假设$\textbf{a}$是线性拉普拉斯特征图的变换向量，即$\textbf{y}^{T}=\textbf{a}^{T}X$。公式(6)的最小化问题可以转化为公式：
$$
\begin{equation}
    \underset{(\textbf{a}^{T}X)D(\textbf{a}^{T}X)^{T}=1}{\arg \underset{\textbf{a}}{\min}}=(\textbf{a}^{T}X)L(\textbf{a}^{T}X)^{T}.
\end{equation}
$$
然后，我们用拉格朗日乘子法求解目标函数，
$$
\begin{equation}
    \begin{aligned}
        f(Y)&=(\textbf{a}^{T}X)L(\textbf{a}^{T}X)^{T} - \lambda(\textbf{a}^{T}X)D(\textbf{a}^{T}X)^{T} \\
        \frac{\partial f(Y)}{\partial \textbf{a}}&=2XLX^{T}\textbf{a}-2\lambda XDX^{T}\textbf{a} \\
        &=2(L^{\prime} \mathbf{a}-\lambda D^{\prime}\mathbf{a})=0 \\
        & \ \ \ \ \ \ L^{\prime} \mathbf{a}=\lambda D^{\prime}\mathbf{a},
    \end{aligned}
\end{equation}
$$
其中 $L^{\prime}=X L X^{T}，D^{\prime}=X D X^{T}$。很容易证明矩阵 $D^{\prime}$ 和 $L^{\prime}$ 是对称的和半正定的。求解向量 $\mathbf{a}_{i}(i=1,2, \cdots, k)$ 可以使用特征值分解或SVD进行。

## 讨论：PCA，LLE，LE，NPE和LPP的区别

与旨在保留全局欧几里得结构的 PCA 不同，四种降维方法（LLE、LE、NPE 和 LPP）旨在保留局部流形结构。

LLE 和 NPE 都试图通过计算高维数据的低维、邻域保留嵌入来降低平滑流形的维数。LLE 试图发现流形的非线性结构，而 NPE 是 LLE 的线性近似。 所以，NPE 与 LLE 相比有两个优势：

- 与 LLE 相比，NPE 的定义无处不在，而不仅仅是在训练数据点上。 这意味着 NPE 可以在监督或非监督模式下执行。
- NPE 是线性的。 这使其快速且适用于实际应用。 它可以在原始空间中进行，也可以在数据点映射到的再现核希尔伯特空间（RKHS）中进行，所以有核NPE。

LE 和 LPP 都试图通过使用图拉普拉斯算子之间的对应关系来减少流形的维数，以构建高维空间上的数据在低维流形上的表示。然而，LE 试图发现流形的非线性结构，而 LPP 是非线性 LE 的线性近似。LPP相对于LE的优势类似于NPE相对于LLE的优势。

同时NPE和LPP还是有一些区别的，他们的目标函数完全不同，分别见公式(7)和公式(8):
$$
\begin{equation}
    \begin{aligned}
        \arg \min_{W} E(W) &= \sum_{i}|X_{i}-\sum_{j}W_{ij}X_{ij}|^2 \\
        \arg \min_{Y} \Phi(Y) &= \sum_{i}|Y_{i}-\sum_{j}W_{ij}Y_{ij}|^2 \\
        &=\textbf{y}M\textbf{y}=\textbf{a}^{T}XMX^{T}\textbf{a}\\
    \end{aligned}
\end{equation}\tag{7}
$$

$$
\begin{equation}
    \begin{gathered}
        \arg \min_{Y^{T}}\sum_{i,j}(y_{i}-y_{j})^{2}W_{ij}=2\textbf{y}^{T}L\textbf{y}=2(\textbf{a}^{T}X)L(\textbf{a}^{T}X)^{T} \\
        \text{ s.t. }(\textbf{a}^{T}X)L(\textbf{a}^{T}X)^{T}=1,\ \textbf{y}^{T}=\textbf{a}^{T}X.
    \end{gathered}
\end{equation}\tag{8}
$$

$L$ 提供了流形上拉普拉斯贝尔特拉米算子 $\mathcal{L}$ 的离散近似。 因此，矩阵 $M$ 提供了对 $\mathcal{L}^{2}$ 的离散近似。 这表明 NPE 本质上试图找到迭代拉普拉斯算子 $\mathcal{L}^{2}$ 的特征函数的线性近似。 从这个意义上说，NPE 和 LPP 提供了两种不同的方法来线性逼近 Laplace Beltrami 算子的特征函数。

## 实验效果(MNIST and ORL)

NPE和LPP的代码我自己写的，由粗略实验效果来看，估计有一些问题，也可能是参数没调对。

```python
from sklearn.neighbors import kneighbors_graph
import scipy
import numpy as np
import math

def sparse_max(A, B):
    Ag = (A > B).astype(int)
    return Ag.multiply(A - B) + B

def NPE(data, train_data, component, neighboor):
    W = kneighbors_graph(train_data, neighboor, mode='distance', include_self=False)
    W = W.A
    X = train_data.T
    M = (np.eye(X.shape[1]) - W).dot((np.eye(X.shape[1]) - W).T)
    T1 = X.dot(M).dot(X.T)
    T2 = X.dot(X.T)
    T2 = T2 + np.exp(-10) * np.eye(T2.shape[0])
    T = scipy.linalg.inv(T2).dot(T1)
    eigVals, eigVects = np.linalg.eigh(T)  # 求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[0:component]
    w = eigVects[:, eigValInd]
    npe = data.dot(w)
    return npe

def LPP(data, train_data, component, neighboor, t):
    W = kneighbors_graph(train_data, neighboor, mode='distance', include_self=False)
    W = sparse_max(W, W.T) #让W矩阵对称，不报warning
    W.data[:] = np.exp(-(W.data ** 2 / (t)))
    W = W.A
    D = np.diag(np.sum(W, axis=0))
    L = D - W
    X = train_data.T
    T1 = X.dot(L).dot(X.T)
    T2 = X.dot(D).dot(X.T)
    T2 = T2 + np.exp(-10) * np.eye(T2.shape[0])
    T = scipy.linalg.pinv(T2).dot(T1)
    eigVals, eigVects = np.linalg.eigh(T)  # 求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[0:component]
    w = eigVects[:, eigValInd]
    lpp = data.dot(w)
    return lpp
```
展示一些实验结果，有可视化的降维结果。然后我们在MNIST上，使用PCA，LLE和LE把数据降到2维。

<img src="子空间学习(4)-LPP&NPE\com.png" alt="LLE-Val" style="zoom:72%;" />
<center style="color:#C0C0C0;text-decoration:underline">图1. 六种降维算法在MNIST上的效果。</center>

同时我们在MNIST数据集上探究了NPE和LPP算法(**KNN**的$k=1$时)的参数（NPE：邻居$n$，LPP：邻居$n$和参数$t$）变化对效果的影响：

<img src="子空间学习(4)-LPP&NPE\parameter.png" alt="LLE-Val" style="zoom:72%;" />
<center style="color:#C0C0C0;text-decoration:underline">图2. 参数对NPE和LPP算法的影响。</center>

然后我们展示算法在ORL数据集上的实验效果。

<img src="子空间学习(4)-LPP&NPE\acc.png" alt="LLE-Val" style="zoom:72%;" />
<center style="color:#C0C0C0;text-decoration:underline">图3. ORL数据集上的准确率对比。</center>

<img src="子空间学习(4)-LPP&NPE\face.png" alt="LLE-Val" style="zoom:72%;" />
<center style="color:#C0C0C0;text-decoration:underline">图4. ORL数据集上的可视化降维效果。</center>

<img src="子空间学习(4)-LPP&NPE\eig.png" alt="LLE-Val" style="zoom:72%;" />
<center style="color:#C0C0C0;text-decoration:underline">图5. 算法提取的特征。</center>
# 总结

其实在复现过程中，并没有达到论文中的效果，我怀疑是降维算法和分类算法参数的选择并没有使效果达到最优。





