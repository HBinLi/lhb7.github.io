---
title: 子空间学习(5)-GE
catalog: true
date: 2022-01-15 12:24:17
subtitle: Subspace Learning-GE
top: 12
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Graph Embedding and Extensions-A General Framework for Dimensionality Reduction（2007 TPAMI）**，Semi-supervised Discriminant Analysis（2005 CVPR），Kernel eigenfaces vs. kernel fisherfaces- Face recognition using kernel methods（2002）。基于此进行概念的梳理和思考，尝试从数学的角度去阐述Eigenfaces，Fisherfaces，Kernel Trick和SDA。

# 摘要：

在过去的几十年里，出现了一大类算法，有监督的或无监督的，源于统计学或几何理论——旨在为降维问题提供不同的解决方案。 尽管子空间学习算法的动机不同，但我们可以引入了一种称为图嵌入的通用公式，以将它们统一在一个通用框架内。 在图嵌入中，每个算法都可以被认为是直接图嵌入或其线性/核/张量扩展的特定内在图，该图描述了数据集的某些所需统计或几何属性，具有来自尺度归一化或惩罚项的约束。 在本报告中，我们将在数学上分析**Graph Embedding**，并在通用框架下比较先前学习的流形学习算法。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# Graph Embedding理解

在本节中，降维算法以统一框架表示。 对于一般分类问题，模型训练的样本集表示为矩阵 $X=[x_{1},x_{2},\cdots,x_{N}]$, $x_{i}\in \mathcal {R}^{m}$，其中 $N$ 是样本数，$m$ 是特征维度。 进一步，样本集的低维表示为 $\textbf{y}=[y_{1},y_{2},\cdots,y_{N}]^{T}$，其中 $y_{i} $ 是顶点 $x_{i}$ 的低维表示。

对于监督学习问题，假设样本$x_{i}$的类标签为$c_{i}=\{1,2,\cdots,N_{c}\}$，其中$N_{c} $ 是类的数量。 我们还让 $\phi_{c}$ 和 $n_{c}$ 分别表示属于 $c_{th}$ 类的索引集和样本数量。

## Graph

令 $G=\{X,W\}$ 是一个无向加权图，其顶点集为 $X$，相似矩阵为 $W\in \mathcal{R}^{N\times N}$。 对于一对顶点，实对称矩阵 $W$ 的每个元素测量其相似度，该相似度可能为负。

在这项工作中，图$G$ 的图嵌入被定义为一种算法，以找到$G$ 顶点之间所需的低维向量关系，同时这种低维向量关系最能表征$G$ 中顶点对之间相似关系。

## Embedding

降维的基本任务是找到一个称为“嵌入”的映射函数 $\phi$，
$$
\begin{equation}
    \Phi:\mathcal{R}^{m}\rightarrow F,\ X \mapsto \textbf{y},
\end{equation}
$$
它将 $x\in \mathcal{R}^{m}$ 转换为所需的低维表示 $\textbf{y}\in R^{\prime}$，通常为 $m\gg m^{\prime }$。 $F$ 指的是特征空间，嵌入$\Phi$ 在不同情况下可以是显式或隐式、线性或非线性的。

## 框架下的目标函数

Graph-preserving criterion的目标函数是公式(1)：
$$
\begin{equation}
    y^{*}=\arg \min _{\textbf{y}^{T} B \textbf{y}=d} \sum_{i \neq j}\left\|y_{i}-y_{j}\right\|^{2} W_{i j}=\arg \min _{\textbf{y}^{T} B \textbf{y}=d} \textbf{y}^{T} L \textbf{y}
\end{equation}\tag{1}
$$
其中 $d$ 是一个常数，$B$ 是为避免目标函数的平凡解而定义的约束矩阵。 $B$ 通常是用于尺度归一化的对角矩阵，也可能是惩罚图的拉普拉斯矩阵。 即$B=L=D-W$，其中$D$为对角矩阵定义为$D_{ii}=\sum_{j}W_{ij}, {\forall} i$。 此外，约束 $\textbf{y}^{T} B \textbf{y}=d$ 用于尺度归一化。

## 为什么LLE，LE，NPE和LPP属于GE框架

流形具有局部结构和局部不变性，局部结构可以用图嵌入来表示（LLE，LE，NPE和LPP都利用了局部不变性）。 更具体地说，根据流形的定义，我们可以知道流形中的每一个点$X_{a}$在欧几里得空间中都有一个邻域$X_{U(a)}$同胚的开集$W_{a}$ ，见公式：
$$
\begin{equation}
    \begin{gathered}
        \forall X_{a} \ \exists X_{U(a)} : \ f(X_{U(a)})\rightarrow W_{a} \ (W_{a} \subseteq \mathbb{R}^{n}) \\
        \forall Y_{a} \ \exists Y_{U(a)} : \ f(Y_{U(a)})\rightarrow W_{a} \ (W_{a} \subseteq \mathbb{R}^{n}),
    \end{gathered}
\end{equation}
$$
其中 $W_{a}$ 是不变的。

## 框架下的分类和总结

| Group           | Algorithm                           |
| --------------- | ----------------------------------- |
| Full supervised | LDA, KLDA, NPE                      |
| Semi-supervised | SDA                                 |
| Un-supervised   | PCA, LLE, LE, LPP, NPE, Isomap, MDS |

<img src="子空间学习(5)-GE\com.png" alt="LLE-Val" style="zoom:72%;" />

# Kernel Eigenfaces and Kernel Fisherfaces

Eigenfaces和Fisherfaces算法见一下篇博客：子空间学习(6)-LDE。

## 动机

Kernel Eigenfaces 旨在找到在非线性映射 $\Phi$ 之后最大化特征空间方差的投影方向，而 Kernel Fisherfaces 通过最小化类内数据点距离和类间数据点距离之间的比率来搜索最有效的区分方向。即Kernel Eigenfaces是在<u>PCA</u>的基础上使用了核函数，Kernel Fisherfaces是在<u>LDA和PCA</u>的基础上使用了核函数。

## 目标函数

Kernel Eigenfaces 和 Kernel Fisherfaces 的目标函数根据它们的动机不同，我们假设每个数据点 $x_{i}$ 从输入空间 $\mathcal{R}^{m}$ 投影到特征空间 $\mathcal {R}^{m^{\prime}}$ ，其中非线性映射函数是：$\Phi: \mathcal{R}^{m}\rightarrow \mathcal{R}^{m^{\prime}}$， $m<m^{\prime}$。Eigenfaces 和 Fisherfaces中的一些计算矩阵将由：$w^{\Phi}$、$C^{\Phi}$、$S^{\Phi}$ 和 $W^{\Phi}$ 表示，它们都与 $\Phi(x_{1} ),\Phi(x_{2}),\cdots,\Phi(x_{m})$相关的。

Kernel Eigenfaces的目标函数是：
$$
\begin{equation}
    \begin{gathered}
    W^{\Phi}=\underset{(w^{\Phi})^{T}w^{\Phi}=1}{\arg \min} (w^{\Phi})^{T}C^{\Phi}w^{\Phi} \ \ with\\
    C^{\Phi}=\frac{1}{N}\sum_{i=1}^{N}(\Phi(x_{i})-\Phi(\bar{x}))(\Phi(x_{i})-\Phi(\bar{x})^{T}.
    \end{gathered}
\end{equation}\tag{2}
$$
其中 $C^{\Phi}$ 是协方差矩阵，$\Phi(\bar{x})$ 是特征空间中所有样本的均值。

Kernel Fisherfaces的目标函数是最大化特征空间中类间方差$S_{B}$和类内方差$S_{W}$的比率：
$$
\begin{equation}
    \begin{aligned}
    W^{\Phi} &=\arg \max _{W^{\Phi}} \frac{\left|\left(W^{\Phi}\right)^{T} S_{B}^{\Phi} W^{\Phi}\right|}{\left|\left(W^{\Phi}\right)^{T} S_{W}^{\Phi} W^{\Phi}\right|}=[w_{1}^{\Phi},\cdots,w_{m}^{\Phi}]\\
    S_{W}&=\sum_{i=1}^{N}(\Phi(x_{i})-\Phi(\bar{x})^{c_{i}})(\Phi(x_{i})-\Phi(\bar{x})^{c_{i}})^{T}\\
    S_{B}&=\sum_{c=1}^{N_{c}}n_{c}(\Phi(\bar{x}^{c})-\Phi(\bar{x}))(\Phi(\bar{x}^{c})-\Phi(\bar{x}))^{T}.
    \end{aligned}
\end{equation}\tag{3}
$$
其中 $\bar{x}^{c}$ 是 $c_{th}$ 类的平均值，$\bar{x}^{c_{i}}$ 是 $c_{th}$ 类第$i_{th}$个样本。

## 监督方式

Kernel Eigenfaces 是一种无监督的降维算法，而 Kernel Fisherfaces 是有监督的。 Kernel Fisherfaces最多只能将维数降到类别数减1。但是Kernel Fisherfaces也可以用于分类，而Kernel Eigenfaces不能。

# 核函数（Kernel Trick）

通过使用核函数，Kernel Eigenfaces是Eigenfaces的非线性推广。 Eigenfaces的基本思想是将数据沿最大方差的方向进行线性投影。 将线性投影方法扩展到非线性情况的一种技术是直接利用核技巧。

## 核函数的优势

- 内核技巧使算法不需要显式计算高维空间中的表示，而只需要在投影到的子空间中计算它。
- 通过使用不同内核的可能性，它包含了可以使用的相当普遍的非线性类别。
- 与Eigenfaces相比，非线性主成分比相应数量的线性主成分提供了更好的识别率，并且可以通过使用比线性情况下更多的成分来提高非线性成分的性能。
- 与其他非线性特征提取技术相比，核方法不需要非线性优化，只需要求解一个特征值问题。

## 核函数的限制

- 与神经方法相比，如果我们需要处理大量的观察，内核技巧可能是不利的，因为这会导致一个大矩阵 $K$。
- 与Eigenfaces相比，内核技巧在输入空间中更难解释。 然而，至少对于多项式核，它在高阶特征方面有非常清晰的解释。

## SDA

SDA 是 LDA 的半监督形式，它额外包含graphical perspective（图嵌入化），linearization（线性化）和kernel trick（核函数）。 这里主要谈一下SDA相对LDA的一些不同。核函数之前在Kernel Fisherfaces已经解释过了，就不提了。

## 监督方式

LDA是一种全监督降维方法，而SDA是一种半监督方法，有标签的数据用于最大化不同类别之间的可分离性，没有标签的数据点用于估计数据的内在几何结构。

## 图嵌入形式

假设映射方向 $w=\sum_{i}\alpha_{i}\Phi(x_{i})$ 和 $K$ 是核 Gram 矩阵，其中 $K_{ij}=\Phi(x_{i} )\Phi(x_{j})$，LDA 的目标函数可以转换为graphical perspective，其中类内和类间方差可以通过矩阵的线性变换来表示，见公式(4)：
$$
\begin{equation}
    \begin{gathered}
    \textbf{a}_{opt}=\underset{\textbf{a}}{\arg \max}\frac{\textbf{a}^{T}S_{b}\textbf{a}}{\textbf{a}^{T}S_{t}\textbf{a}}=  \underset{\textbf{a}}{\arg \max}\frac{\textbf{a}^{T}XW_{l\times l}X^{T}\textbf{a}}{\textbf{a}^{T}XX^{T}\textbf{a}} \\
    S_{b}=\sum^{c}_{k=1}X^{(k)}W^{k}(X^{k})^{T}=XW_{l\times l}X^{T} \\
    S_{t}=\sum_{i=1}^{l}(x_{i}-\mu)(x_{i}-\mu)^{T}=XX^{T}
    \end{gathered}
\end{equation}\tag{4}
$$
其中 $W^{(k)}$ 是一个 $l_{k}\times l_{k}$ 矩阵，其中所有元素都等于 $1/l_{k}$ ，并且 $X^{(k)}=[x_ {1}^{(k)},\cdots,x_{lk}^{(k)}]$表示$k_{th}$类的数据矩阵。 和数据矩阵 $X=[X^{(1)},\cdots,X^{(c)}]$ 并定义 $l\times l$ 矩阵 $W_{l\times l}$ 为：
$$
W_{l \times l}=\left[\begin{array}{cccc}
W^{(1)} & 0 & \cdots & 0 \\
0 & W^{(2)} & \cdots & 0 \\
\vdots & \vdots & \ddots & \vdots \\
0 & 0 & \cdots & W^{(c)}
\end{array}\right].
$$

## 线性化

假设顶点的低维向量表示可以从线性投影中获得，如 $y=\textbf{a}^{T}x$，其中$\textbf{a}$是投影向量，目标函数修改为公式(5)：
$$
\begin{equation}
    \begin{gathered}
        \mathbf{a}_{o p t}=\arg \max _{\mathbf{a}} \frac{\mathbf{a}^{T} S_{b} \mathbf{a}}{\mathbf{a}^{T} S_{w} \mathbf{a}}, \\
        S_{b}=\sum_{k=1}^{c} l_{k}\left(\boldsymbol{\mu}^{(k)}-\boldsymbol{\mu}\right)\left(\boldsymbol{\mu}^{(k)}-\boldsymbol{\mu}\right)^{T}, \\
        S_{w}=\sum_{k=1}^{c}\left(\sum_{i=1}^{l_{k}}\left(\mathbf{x}_{i}^{(k)}-\boldsymbol{\mu}^{(k)}\right)\left(\mathbf{x}_{i}^{(k)}-\boldsymbol{\mu}^{(k)}\right)^{T}\right),
    \end{gathered}
\end{equation}\tag{5}
$$

# 代码

贴一下Fisherfaces的代码，Eigenfaces是PCA的具体实现也就不贴了，同时呢，Sklearn也有自己的库可以直接使用LDA：

```python
from sklearn.decomposition import PCA
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis #sklearn的库，这里未使用

def LDA(data, train_data, train_target, n_dim):
    clusters = np.unique(train_target)
    # within_class scatter matrix
    Sw = np.zeros((train_data.shape[1], train_data.shape[1]))
    for i in clusters:
        train_datai = train_data[train_target[:] == i]
        train_datai = train_datai - train_datai.mean(0)
        Swi = np.mat(train_datai).T * np.mat(train_datai)
        Sw += Swi
    # between_class scatter matrix
    SB = np.zeros((train_data.shape[1], train_data.shape[1]))
    u = train_data.mean(0)  # 所有样本的平均值
    for i in clusters:
        Ni = train_data[train_target[:] == i].shape[0]
        ui = train_data[train_target[:] == i].mean(0)  # 某个类别的平均值
        SBi = Ni * np.mat(ui - u).T * np.mat(ui - u)
        SB += SBi
    Sw = Sw + np.exp(-1) * np.eye(Sw.shape[0])
    S = np.linalg.pinv(Sw).dot(SB)
    eigVals, eigVects = np.linalg.eig(S)  # 求特征值，特征向量
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-n_dim - 1):-1]
    w = eigVects[:, eigValInd]
    data_ndim = np.dot(data, w)
    return data_ndim.real


def Fisherfaces(data, train_data, train_label, component):
    n = train_data.shape[0]
    c = len(np.unique(train_label))
    pca = PCA(n_components=n - c)
    pca.fit(train_data)
    data_train_pca = pca.transform(train_data)
    data_pca = pca.transform(data)
    data_fld = LDA(data, train_data, train_label, component)
    return data_fld
```

# 总结

TPAMI的那篇文章对2007年及之前的降维算法进行了总结，还是比较到位。

