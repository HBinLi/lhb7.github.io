---
title: 子空间学习(7)-Spectral Clustering and Normalized Cuts
catalog: true
date: 2022-01-17 12:24:17
subtitle: Subspace Learning-Spectral Clustering and Normalized Cuts
top: 14
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Normalized Cuts and Image Segmentation（2000 TPAMI）**和**On Spectral clustering analysis and an algorithm（2001 NIPS）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述N-cut和谱聚类。

# 摘要：

信息处理中的许多问题都涉及使用从数据派生的矩阵的特征向量进行某种形式的聚类点。 其中，N-cut和谱聚类在归一化切割准则的框架内。 N-cut 将聚类问题视为图划分问题，并提出了一种新颖的全局标准，即归一化割，用于对图进行分割。 归一化切割标准测量不同组之间的总差异以及组内的总相似度。 此外，谱聚类通过使用矩阵扰动理论中的工具将 K-means 与 N-cut 相结合，并在许多具有挑战性的聚类问题上取得了良好的实验结果。 在本报告中，我们尝试以数学方式解释 N-cut 和谱聚类的主要思想。 此外，还包括对 DUT-OMRON 数据集的实验，以探索 N-cut 和光谱聚类的有效性。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# 前景知识

## 聚类

聚类是将总体或数据点 $X=[x_{1},x_{2},\cdots,x_{n}]$ 分成若干组 $U=\{U_{1},...,U_{k}\}$ ，使得同一组中的数据点更相似，而不同组之间的数据点不那么相似。 简而言之，目的是分离具有相似特征的组并将它们分配到集群中，请参见以下表示：
$$
{\forall} x_{ji}\in U_{j},\ x_{ji_{1}}\ and\ x_{ji_{2}}\ are\ more\ similar.
$$

## K-means

K-means 聚类是一种矢量量化方法，旨在将 $n$ 个观测值划分为 $k$ 个聚类，其中每个观测值都属于具有最近聚类中心的聚类。 K-means 聚类旨在将 n 个观测值划分为 $k (\leq n)$ 个集合 $S = \{S_{1},S_{2},\cdots,S_{k}\}$ ，目标使最小化簇内平方和，即：
$$
\begin{equation}
    \underset{S}{\arg \min}\sum^{k}_{i=1}\sum_{x\in S_{i}}\|x-\mu_{i}\|^{2}=\underset{S}{\arg \min}\sum_{i=1}^{k}|S_{i}|VarS_{i},
\end{equation}
$$
其中 $\mu_{i}$ 是 $S_{i}$ 中点的平均值。

## 割

一个图 $G=(V,E)$ 可以被分割成两个不相交的集合（A 和 B），$A \cup B = V$, $A \cap B = \emptyset$，主要通过简单地删除连接这两个集合的边部分。这两个部分之间的不同程度可以计算为已删除边缘的总权重。 在图论语言中，它被称为割：
$$
\begin{equation}
    cut(A,B)=\sum_{u\in A,v\in B}w(u,v).
\end{equation}
$$

## 割与聚类

聚类的直觉是根据它们的相似性将不同组中的点分开。 对于以相似度图的形式给出的数据，这个问题可以重述如下：我们希望找到一个图的分区，使得不同组之间的边具有非常低的权重（这意味着不同集群中的点彼此之间是不相似的） ，并且组内的边具有较高的权重（这意味着同一簇内的点彼此相似）。

# Normalized Cuts

图的最优二分法是最小化割值（Min-cut）的二分法，这种划分的数量是指数级的。 此外，Min-cut 在划分小点集时具有不自然的偏差。

为了避免这两个限制，提出了归一化切割（N-cut）作为两组之间分离的度量。 N-cut 不是查看连接两个分区的总边权重的值，而是将切割成本计算为图中所有节点的总边连接的一部分，见公式(1)：
$$
\begin{equation}
    Ncut(A,B)=\frac{cut(A,B)}{assoc(A,V)}+\frac{cut(A,B)}{assoc(B,V)},
\end{equation}\tag{1}
$$
其中 $assoc(A,V)=\sum_{u\in A,t\in V}w(u,t)$ 是从 A 中的节点到图中所有节点的总连接，$assoc(B,V )$ 的定义类似。

## 目标函数

N-cut 问题的目标函数是最小化归一化割，见公式(2)：
$$
\begin{equation}
    \begin{gathered}
    \underset{\textbf{y}}{\arg \min}\frac{\textbf{y}^{T}(D-W)\textbf{y}}{\textbf{y}^{T}D\textbf{y}}\\
    \text{s.t. }\textbf{y}^{T}D\textbf{1}=0.
    \end{gathered}
\end{equation}\tag{2}
$$
我们通过求解广义特征值系统最小化目标函数公式(2)，
$$
\begin{equation}
    (D-W)\textbf{y}=\lambda D\textbf{y}.
\end{equation}\tag{3}
$$
此外，上述公式(2)和公式(3)可以归一化，公式(2)可以被重写为公式(4)，
$$
\begin{equation}
    \begin{gathered}
    \underset{z}{\arg \min}\frac{z^{T}D^{-\frac{1}{2}(D-W)D^{-\frac{1}{2}}}z}{z^{T}z}\\
    \text{s.t. }z^{T}z_{0}=0,
    \end{gathered}
\end{equation}\tag{4}
$$
其中 $z=D^{\frac{1}{2}}\textbf{y}$。 $z_{0}=D^{\frac{1}{2}}\textbf{1}$ 是公式(5)的特征向量，这个特征向量的特征值为 0，即公式(5)的最小特征向量。 此外，$D^{-\frac{1}{2}}(DW)D^{-\frac{1}{2}}$ 是对称正半定的，因为 $DW$，也称为拉普拉斯矩阵，是已知的，为半正定。 因此，我们可以通过求解广义特征值分解来最小化目标函数公式(4)，
$$
\begin{equation}
    D^{-\frac{1}{2}}(D-W)D^{-\frac{1}{2}}z=\lambda z.
\end{equation}\tag{5}
$$
需要额外注意的是我们最后使用的是第二小特征值对应的特征向量来对图进行分割，因为公式(5)中第二小的特征向量 $\textbf{y}$ 仅逼近最优归一化割解，它恰好最小化了以下问题：
$$
\inf _{y^{T} \mathbf{D} 1=0} \frac{\sum_{i} \sum_{j}(y(i)-y(j))^{2} w_{ij} }{\sum_{i} y(i)^{2} \mathbf{d}(i)}
$$
其中 $d(i)=D(i,i)$。 粗略地说，这迫使指示向量 $\textbf{y}$ 对紧密耦合的节点 $i$ 和 $j$采用相似的值。

## 时间复杂度

为所有特征向量求解一个标准特征值问题需要整个算法的主要计算量。 它需要 $0(n^{3})$ 操作，其中 $n$ 是图中的节点数。 但是，$n$ 是图像中的像素数，因此复杂度 $0(n^{3})$ 是不切实际的。 因此，N-cut 通过称为 Lanczos 方法的特征求解器提高了计算效率，其中它的运行时间为 $O(mn)+O(mM(n))$，其中 $m$ 是所需的最大矩阵向量计算次数，$M(n)$ 是 $Ax$ 的矩阵向量计算的成本，其中 $A=D^{-\frac{1}{2}}(DW)D^{-\frac{ 1}{2}}$。 由于$W$ 是稀疏的，$A$ 和矩阵向量也是稀疏的计算量只是 $O(n)$。

一行 $A$ 与向量 $x$ 的内积的成本是 $O(n)$。 将所有内容放在一起，每个矩阵向量计算都需要 $O(n)$ 次操作，且常数因子很小。

## 缺点

当数据不够统一且存在异常值时，N-cut 无法切出图中孤立节点的小集合。

# 谱聚类

## 谱聚类的不同理解

谱是指数据的相似度矩阵的特征值进行降维。在数学中，矩阵的谱是其特征值的集合。更一般地，如果 $T:V\rightarrow v$ 是任何有限维向量空间上的线性算子，则它的谱是标量 $\lambda$ 的集合，使得 $T-\lambda I$ 不可逆。

在降维的观点上，谱聚类技术利用数据相似矩阵的谱（特征值）进行降维，然后再进行较少维度的聚类。

在图分割的观点上，我们希望找到图的一个分区，使得不同组之间的边具有非常低的权重（这意味着不同集群中的点彼此不同）并且组内的边具有较高的权重（这意味着同一簇内的点彼此相似）。

在随机游走的观点上，谱聚类可以解释为试图找到图的一个分区，使得随机游走在同一个集群中停留很长时间，并且很少在集群之间跳跃。

## 算法流程

$X={x_{1},\cdots,x_{n}}$ 在 $\mathcal{R}^{l}$ 中，我们想要聚类成 $k$ 个子集。 谱聚类的算法步骤为算法1。

<img src="子空间学习(7)-Spectral Clustering and Normalized Cuts\algorithm.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图1. 谱聚类算法流程。</center>

### 不同的构造图的方式

A Tutorial on Spectral Clustering（2007）总结了三种图：

- $\epsilon-neighborhood$ 图：所有成对距离小于 $\epsilon$ 的点都已连接。 $\epsilon-neighborhood$ 图被视为未加权图。
- $k$-最近邻图：顶点$v_{i}$和顶点$v_{j}$是连接的$v_{j}$在$v_{i}$的$k$个最近邻居中 . 然而，这个定义导致了有向图，因为邻域关系不是对称的。
- 全连接图：所有点都简单地相互连接，具有正相似性，我们通过 $s_{ij}=s(x_{i},x_{j})$ 对所有边进行加权。

### 不同的图拉普拉斯构造

A Tutorial on Spectral Clustering（2007）总结了三种图拉普拉斯：

- 非归一化图拉普拉斯矩阵定义为：
  $$
  L=D-W
  $$

- 有两个矩阵在文献中被称为归一化图拉普拉斯算子。 我们用 $L_{sym}$ 表示第一个矩阵，因为它是一个对称矩阵，第二个用 $L_{rw}$ 表示，因为它与随机游走密切相关。 两个矩阵彼此密切相关，定义为：
  $$
       L_{sym}:=D^{-\frac{1}{2}}LD^{-\frac{1}{2}}=ID^{-\frac{1}{2}}WD^{- \frac{1}{2}}
  $$
  $$
  L_{rw}:=D^{-1}L=I-D^{-1}W.
  $$

# 实验结果

主要是本节相关算法的代码，贴一下NC的（采用稀疏矩阵计算，会快一点），SC的sklearn有库，然后由于可以自定义相似矩阵$W$也挺方便的：

```python
import numpy as np
import math
from scipy.sparse import csr_matrix
import scipy.sparse as sparse

def sparse_max(A, B):
    Ag = (A > B).astype(int)
    return Ag.multiply(A - B) + B

def construct_graph(img, radius, t1, t2):
    n = img.shape[0] * img.shape[1]
    W = np.zeros((n, n))
    data = []
    row = []
    col = []
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            ori = i * img.shape[0] + j
            for x in range(i - radius, i + radius + 1):
                for y in range(j - radius, j + radius + 1):
                    if x < 0 or x >= img.shape[0] or y < 0 or y >= img.shape[1] or (x == i and y == j):
                        continue
                    else:
                        # if math.fabs(np.sum(np.abs(img[i][j]-img[x][y])))>0.15:
                        #     continue
                        value = math.exp(-((x - i) ** 2 + (y - j) ** 2) / t1) * math.exp(
                            -(np.linalg.norm(img[i][j] - img[x][y]) ** 2) / t2)
                        data.append(value)
                        row.append(ori)
                        col.append(x * img.shape[0] + y)
    return csr_matrix((data, (row, col)), shape=(n, n))

def NC(img, neighboor, t1, t2):
    W = construct_graph(img, neighboor, t1, t2)
    W = sparse_max(W, W.T)
    D = sparse.diags(np.asarray(W.sum(axis=0)).reshape(1, -1), [0])
    L = D - W
    D_1 = sparse.diags(1 / np.sqrt(np.asarray(W.sum(axis=0))).reshape(1, -1), [0])
    T = D_1.dot(L).dot(D_1)
    eigVals, eigVects = sparse.linalg.eigsh(T, k=2, which='SA')
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[1]
    w = eigVects[:, eigValInd]
    w = D_1.dot(w)
    w = np.asarray(w)
    return w.real
```

sklearn调用SC的库：

```python
from sklearn.cluster import SpectralClustering

def SC(img,cluster):
    clustering=SpectralClustering(n_clusters=cluster)
    return clustering.fit_predict(img)
```

展示一些实验结果，主要是使用的数据集是DUT-OMRON数据集，它由5168张自然图像组成。结果是NC的二分类结果和SC的四分类结果，用于图像分割。
<img src="子空间学习(7)-Spectral Clustering and Normalized Cuts\NC.png" alt="LLE-Val" style="zoom:90%;" />
<center style="color:#C0C0C0;text-decoration:underline">图2. N-cut实现二分类图像分割。</center>
<img src="子空间学习(7)-Spectral Clustering and Normalized Cuts\SC.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图3. SC实现四分类分割。</center>


# 总结

效果还不错，实现应该比较到位。



 