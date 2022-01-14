---
title: 子空间学习(6)-LDE
catalog: true
date: 2022-01-16 12:24:17
subtitle: Subspace Learning-LDE
top: 13
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Local Discriminant Embedding and Its Variants（2005 CVPR）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述LDE，除此之外简要介绍Eigenfaces和Fisherfaces，然后会对LPP和NPE（见-子空间学习(4)-LPP&NPE）进行一些拓展。

# 摘要：

信息处理中的许多问题都涉及某种形式的降维。因此，人们对降维算法很感兴趣。不同于 Eigenface旨在选择最大化所有类的总散射的投影，我们将介绍两种旨在最大化类间距离和类内距离的比率的子空间学习算法：Fisherfaces更关注数据的全局结构，而Local Discriminant Embedding (LDE) 更关注数据的局部关系和类的关系。然而，不关注局部关系使Fisherface从先验上有一定局限性。此外，LDE 通过保留同一类数据点的内在相邻关系来解决该问题，同时使不同类的相邻点不再相互粘连。在本报告中，我们尝试以数学方式解释 Fisherface 和 LDE 的主要思想，并像 LDE 一样扩展邻域保留嵌入 (NPE) 和局部保留投影 (LPP) 。此外，还包括对人脸识别的全面比较和广泛的实验，以探索 Fisherface 和 LDE 的有效性。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# Fisherfaces

## Fisher criterion

Fisher criterion是一个判别准则函数，由类间散点距离与类内散点距离的比率定义。 通过最大化这一标准，可以获得最优的判别投影轴。 在样本被投影到这个投影轴上之后，类内散布被最小化，类间散布被最大化。

## Motivation

Eigenfaces 旨在选择一种降维线性投影，使所有投影样本的分散度最大化。 因此，Eigenfaces 不能很好地处理非线性数据，并且没有考虑类间和类内散点之间的关系。Fisherfaces则考虑了类间和类内散点之间的关系。类间$S_{B}$和类内$S_{W}$散布矩阵是类间散布和类内散布的表示，它们的数学形式如公式(1)：
$$
\begin{equation}
    \begin{gathered}
     S_{W}=\sum_{i=1}^{N}(x^{c_{i}}-\bar{x}^{c})(x^{c_{i}}-\bar{x}^{c})^{T}\\
    S_{B}=\sum_{c=1}^{N_{c}}n_{c}(\bar{x}^{c}-\bar{x})(\bar{x}^{c}-\bar{x})^{T},
    \end{gathered}
\end{equation}\tag{1}
$$

其中 $\bar{x}^{c}$ 是 $c_{th}$ 类的平均值，$\bar{x}$ 是所有样本的平均值，$x^{c_{i}}$ 是 $c_{th}$ 类中的 $i_{th}$ 样本。

## Fisherface vs Eigenface

Eigenface 和 Fisherface 在动机、目标函数、监督方法上是不同的，见下文讨论。

Eigenface 旨在寻找线性投影后数据方差最大的投影方向，而 Fisherface 则通过最大化类间和类内分散之间的比率来寻找一条线，来有效区分不同类。

Eigenface 和 Fisherface 的目标函数根据不同的动机，也有所不同。 Eigenface 的目标函数是公式(2)：
$$
\begin{equation}
    \begin{gathered}
    \underset{W^{T}W=1}{\arg \min}\ \  (W)^{T}CW \ \ with\\
    C=\frac{1}{N}\sum_{i=1}^{N}(x_{i}-\bar{x})(x_{i}-\bar{x})^{T},
    \end{gathered}
\end{equation}\tag{2}
$$
其中 $C$ 是协方差矩阵，$\bar{x}$ 是所有样本的平均值。

Fisherface 的目标函数是最大化类间方差 $S_{B}$ 和类内方差 $S_{W}$ 的比率，见公式(3)：
$$
\begin{equation}
    \begin{gathered}
    W =\arg \max _{W}\ \frac{\left|\left(W\right)^{T}W_{pca}^{T}S_{B}W_{pca} W\right|}{\left|\left(W\right)^{T}W_{pca}^{T}S_{W}W_{pca}W\right|}=[w_{1},\cdots,w_{m}]\\
    \text{s.t. }(W)^{T} S_{W} W=1.
    \end{gathered}
\end{equation}\tag{3

}
$$
Eigenface 是一种无监督的维数算法，而 Fisherface 是有监督的。 Fisherface 与 Eigenface 相比具有以下优势：

- 虽然 Eigenface 实现了更大的总散射，但 FLD 实现了更大的类间散射，因此简化了分类。
- Fisherface 可用于分类，而 Eigenface 不能。

然而，Fisherface 只能将维度减少到类别数减 1，而 Eigenface 则没有这个限制。

## Mathematical Details

拉格朗日方法用于解决方程中的优化问题公式(3)，请参见以下推导：
$$
\begin{equation}\nonumber
    \begin{gathered}
        \mathcal{L}=W^{T}W_{pca}^{T}S_{B}W_{pca}W-\lambda W^{T}W_{pca}^{T}S_{W}W_{pca}W\\
        \frac{\partial \mathcal{L}}{\partial W}=2\left(W_{pca}^{T}S_{B}W_{pca}W-\lambda W_{pca}^{T}S_{W}W_{pca}W\right)=0\\
        S_{W}^{-1}S_{B}W_{pca}W=\lambda W_{pca}W.
    \end{gathered}
\end{equation}
$$
此外，可以应用 SVD 或特征分解来解决上述问题。 选取$m$个最大特征值对应的特征向量形成$W$：
$$
\begin{equation}
    \begin{gathered}
        S_{W}^{-1}S_{B}W_{pca}W_{i}=\lambda_{i} W_{pca}W_{i}\\
        sort(\lambda_{1},\cdots,\lambda_{n})\\
        W=[W_{n-m+1},\cdots,W_{n}].
    \end{gathered}
\end{equation}
$$

# Local Discriminant Embedding

在这里介绍三种LDE，最开始的LDE，2DLDE和Kernel LDE。

## LDE

与 Eigenface 不同，LDE 中使用数据的局部关系和类关系来构建嵌入。 此外，LDE 没有 Eigenface 的限制。 在本节中，我们将详细介绍 LDE。
LDE 旨在保持内在的相邻关系，而不同类别的相邻点不再相互粘连。
在低维嵌入子空间中，如果相邻点具有相同的标签，LDE 希望保持相邻点靠近，而防止其他类的点进入邻域。 考虑到这两个方面，LDE 的目标函数为公式(4)：
$$
\begin{equation}
    \begin{gathered}
        \underset{V}{\arg \max}\ \ J(V)=\sum_{i,j}\|V^{T}x_{i}-V^{T}x_{j}\|^{2}W_{ij}^{\prime}\\
        \text{s.t. }\sum_{i,j}\|V^{T}x_{i}-V^{T}x_{j}\|^{2}w_{ij}=1,
    \end{gathered}
\end{equation}\tag{4}
$$
其中 $V$ 是一个 $n\times l$ 矩阵，$V$ 是线性投影 $\textbf{z}=V^{T}x$。

## 2DLDE

受到少量训练数据的限制，无法准确近似底层流形。 2DLDE 尝试将图像（数据）视为矩阵，并根据矩阵形式解决子空间学习问题。 此外，当已知图像的变化是由translation, pitch或yaw引起时，2DLDE 比基于矢量的公式具有显着的优势。

令 $\{A_{i}|A_{i}\in\mathcal{R}^{n_{1}\times n_{2}}\}$ 为训练数据。 然后，我们修改了矩阵的 LDE，矩阵-向量乘法 $\textbf{z}_{i}=V^{T}x_{i}$ 应改为双边形式 $B_{i}= L^{T}A_{i}R$，其中 $L\in \mathcal{R}^{n_{1}\times l_{1}}$ 和 $R\in \mathcal{R}^{n_{ 2}\times l_{2}}$ 将$A_{i}$ 转换成一个更小的矩阵$B_{i}\in\mathcal{R}^{l_{1}\times l_{2}}$。 因此，目标函数公式(4)可以重写为公式(5)，
$$
\begin{equation}
    \begin{gathered}
        \underset{L,R}{\arg \max}\ \ Q(L, R)=\sum_{i, j}\left\|L^{T} A_{i} R-L^{T} A_{j} R\right\|_{F}^{2} w_{i j}^{\prime} \\
    \text {s.t. } \sum\left\|L^{T} A_{i} R-L^{T} A_{j} R\right\|_{F}^{2} W_{i j}=1,
    \end{gathered}
\end{equation}\tag{5}
$$
其中 Frobenius 矩阵范数为 $\|A\|_{F}=(\sum_{j,k}a_{j,k}^{2})^{\frac{1}{2}}=tr( AA^{T})$。 通过使用拉格朗日乘子法和特征分解，通过迭代直到收敛得到$L$和$R$。 然后，我们通过测试点$\bar{A}$计算$\bar{B}=L^{T}\bar{A}R$，找到它的最近邻$B_{i}$，使得$\ |B_{i}=\bar{B}\|_{F}$，并将标签指定为 $\bar{y}=y_{i}$。

## Kernel LDE

受到线性学习算法有限分类能力的启发。 Kernel LDE 试图通过非线性映射将输入数据转换到更高维空间来提升分类性能。 因此，Kernel LDE 在处理非线性降维问题方面优于 LDE。

假设非线性映射为$\Phi:\mathcal{R}^{n}\rightarrow \mathcal{F}$，$\mathcal{F}$为特征空间。 因此，局部判别嵌入在 $\mathcal{F}$ 中的投影方向 $v$ 是 $v=\sum_{i}^{m}\alpha_{i}\Phi(x_{i})=\sum_{ i}^{m}\alpha_{i}k(x_{i},\bar{x})$, $\boldsymbol{\alpha}_{i}$ 是膨胀系数。 目标函数(5)可以重写为公式(6)，
$$
\begin{equation}
    \begin{gathered}
         \underset{\boldsymbol{\alpha}}{\arg \max}\ \ U(\boldsymbol{\alpha})=\boldsymbol{\alpha}^{T} K\left(D^{\prime}-W^{\prime}\right) K \boldsymbol{\alpha} \\
         \text{s.t. }\boldsymbol{\alpha}^{T} K(D-W) K \boldsymbol{\alpha}=1,
    \end{gathered}
\end{equation}\tag{6}
$$
其中 $K$ 是一个核矩阵，$K_{i,j}=k(x_{i},x_{j})$, 并且 $\boldsymbol{\alpha}=[\alpha_{1},\cdots, \alpha_{n}]^{T}$ 由膨胀系数组成。 $D$ ($D_{ii}=\sum_{j}W_{ij}$) 是一个对角矩阵，其条目是 $W$ 的列（或行，因为 $W$ 是对称的）和。

# 拓展LPP和NPE

我们探讨了 LPP 是否可以扩展到2D和核函数方法。 此外，NPE 是对 Laplace Beltrami 算子的特征函数的另一种线性逼近。 因此，我们探索如何扩展 NPE，就像 LDE 对 LPP 所做的那样。

## 2DLPP

LPP 的目标函数是最小化重建误差，见公式(7)：
$$
\begin{equation}
    \begin{gathered}
        \arg \min_{Y^{T}}\sum_{i,j}(y_{i}-y_{j})^{2}W_{ij}=2(\boldsymbol{a}^{T}X)L(\boldsymbol{a}^{T}X)^{T} \\
        \text{ s.t. }(\boldsymbol{a}^{T}X)D(\boldsymbol{a}^{T}X)^{T}=1,\ \boldsymbol{y}^{T}=\boldsymbol{a}^{T}X.
    \end{gathered}
\end{equation}\tag{7}
$$
根据公式(5)，我们可以把LPP的目标函数公式(7)变成公式(8)，
$$
\begin{equation}
    \begin{gathered}
        \arg \min_{L,R}\ \ \sum_{i,j}\|L^{T}A_{i}R-L^{T}A_{j}R\|^{2}W_{ij} \\
        \text{ s.t. } (L^{T}A_{i}R)D(L^{T}A_{i}R)^{T}=1.
    \end{gathered}
\end{equation}\tag{8}
$$

## Kernel LPP

假设$\mathcal{F}$中的点积可以通过核函数$k(x_{1},x_{2})=\Phi(x_{1})^{T}\Phi(x_{ 2})$。 与公式(6)中的假设，我们可以重写 LPP 的目标函数公式(7)为公式(9)：
$$
\begin{equation}
    \begin{gathered}
        \arg \min_{Y^{T}}\sum_{i,j}(y_{i}-y_{j})^{2}W_{ij}=2(\boldsymbol{a}^{T}K)L(\boldsymbol{a}^{T}K)^{T} \\
        \text{ s.t. }(\boldsymbol{a}^{T}K)L(\boldsymbol{a}^{T}K)^{T}=1,\ \boldsymbol{y}^{T}=\boldsymbol{a}^{T}X,
    \end{gathered}
\end{equation}\tag{9}
$$
其中 $K$ 是一个核矩阵，$K_{ij}=\Phi(x_{i},x_{j})$, 并且 $\boldsymbol{\alpha}=[\alpha_{1},\cdots,\ alpha_{n}]$ 由膨胀系数组成。

## 拓展NPE

LPP 线性逼近 Laplace Beltrami 算子 $\mathcal{L}$ 的特征函数，而 NPE 中的矩阵 $M$ 提供了 $\mathcal{L}^{2}$ 的离散逼近。 NPE 的目标函数是最小化重构误差，见公式(10)：
$$
\begin{equation}
    \begin{aligned}
        \arg \min_{W} E(W) &= \sum_{i}\|x_{i}-\sum_{j}W_{ij}x_{ij}\|^2 \\
        \arg \min_{Y} \Phi(Y) &= \sum_{i}\|y_{i}-\sum_{j}W_{ij}y_{ij}\|^2 \\
        &=\boldsymbol{y}^{T}M\boldsymbol{y}=\boldsymbol{a}^{T}XMX^{T}\boldsymbol{a}
    \end{aligned}
\end{equation}\tag{10}
$$

$$
\text{ s.t. }  \sum_{j}W_{ij}=1,\ \boldsymbol{a}^{T}XX^{T}\boldsymbol{a}=1,\ \boldsymbol{y}=\boldsymbol{a}^{T}X,
$$

其中 $I=diag(1,\cdots,1)$。 此外，我们探讨了 LDE 对 LPP 的作用并扩展了 NPE。 $\mathcal{R}^{n}$中的数据点$\{x_{1}\}_{i=1}^{m}$可以改写为数据矩阵$X=[x_{1}， \cdots,x_{m}]\in\mathcal{R}^{n\times m}$. 然后，我们通过以下步骤扩展 NPE。

- 构建邻域图：对于无向图 $G$ 和 $G^{\prime}$，考虑同一类的每一对点 $x_{i}$ 和 $x_{j}$ ($y_{i}=y_{j}$ ) 和不同的类 ($y_{i}\neq y_{j}$)，如果 $x_{j}$ 是$x_{i}$的$k$个最近邻居之一，那就在$x_{i}$和$x_{j}$之间加一条边。

- 计算权重矩阵：对 $G$ 的矩阵 $W$ 加权
  $$
       w_{ij}=e^{\frac{-\|x_{i}-x_{j}\|^{2}}{t}}。
  $$
  默认情况下，如果 $x_{i}$ 和 $x_{j}$ 没有连接，则 $w_{ij}=0$。 很明显，如此定义的 $W$ 是一个 $m \times m$，稀疏对称矩阵。

- 找到对应于 $l$ 最大特征值的广义特征向量 $v_{1}, v_{2},\cdots, v_{l}$，
  $$
  \begin{equation}
          XM^{\prime}X^{T}v=\lambda XMX^{T}v,
  \end{equation}
  $$
  其中 $M=(I-W)^{T}(I-W)$ 和 $M^{\prime}=(I-W^{\prime})^{T}(I-W^{\prime})$ 是对称矩阵。 $x_{i}$ 的嵌入由 $z_{i}=V^{T}x_{i}$ 完成，其中 $V=[v_{1},\cdots,v_{l}]$。

更进一步，上面的特征分解问题旨在保持相同标签的相邻点靠近，同时防止其他类的点进入邻域。 此外，NPE的目标函数可以变为公式(11)：
$$
\begin{equation}
    \begin{gathered}
        \arg \min_{W} E(W^{\prime}) = \sum_{i,j}\|x_{i}-W_{ij}^{\prime}x_{ij}\|^2 \\
        \arg \min_{Y} \Phi(Y) = \sum_{i,j}\|V^{T}x_{i}-W_{ij}^{\prime}V^{T}x_{ij}\|^2 \\
        \text{s.t. }\sum_{i,j}\|V^{T}x_{i}-W_{ij}V^{T}x_{ij}\|^2=1.
    \end{gathered}
\end{equation}
$$

# 讨论

这里主要讨论一下LDE与之前的算法的区别：

Eigenface 旨在通过将数据在最大方差的方向投影来保持全局欧几里得结构，而 NPE 和 LPP 旨在通过最小化重构误差来保持局部流形结构。 此外，Eigenface、LPP 和 NPE 是无监督方法。 然而，Fisherface 和 LDE 是有监督的方法。 Fisherface 旨在通过最小化类内距离和类间距离之间的比率来寻找最有效的辨别方向。 此外，LDE 受到 Fisherface 的启发，并在 LPP 的基础上进行了改进。 因此，LDE 旨在通过最小化邻域中类内距离和类间距离之间的比率来保持内在的邻域关系。

# 实验结果

主要是本节相关算法的代码，这里只贴LDE有关的，因为太多了。至于拓展LPP和NPE的，我之后应该会上传到github上：

```python
import numpy as np
import scipy
from SubspaceLearning.Kernel import kernel
import math

#构造最经典的权重矩阵
def construct_graph(data, label, k, t):
    clusters = np.unique(label)
    class_i = []
    for i in clusters:
        class_i.append(label[:] == i)
    class_i = np.asarray(class_i)
    n = data.shape[0]
    W_full = np.zeros((n, n))
    SW = np.zeros((n, n))
    SB = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            dist = np.linalg.norm(data[i] - data[j])
            W_full[i][j] = math.exp(-(dist ** 2 / (t)))
            W_full[j][i] = W_full[i][j]
    for i in range(n):
        # G within
        cl = label[i]
        sl = class_i[cl][:]  # same label :true or false
        slindex = np.array(np.where(sl == True))[0, :]  # within
        nslindex = np.array(np.where(sl == False))[0, :]  # between class
        W_sl = W_full[i][sl]
        W_nsl = W_full[i][~sl]
        index = np.argsort(W_sl)[1:k + 1]
        nindex = np.argsort(W_nsl)[0:k]
        SW[i][slindex[index]] = W_full[i][slindex[index]]
        SB[i][nslindex[nindex]] = W_full[i][nslindex[nindex]]
    return SW, SB

#构造01的权重矩阵
def construct_graph_01(data, label, k):
    clusters = np.unique(label)
    class_i = []
    for i in clusters:
        class_i.append(label[:] == i)
    class_i = np.asarray(class_i)
    n = data.shape[0]
    W_full = np.ones((n, n))
    SW = np.zeros((n, n))
    SB = np.zeros((n, n))
    for i in range(n):
        # G within
        cl = label[i]
        sl = class_i[cl][:]  # same label :true or false
        slindex = np.array(np.where(sl == True))[0, :]  # within
        nslindex = np.array(np.where(sl == False))[0, :]  # between class
        W_sl = W_full[i][sl]
        W_nsl = W_full[i][~sl]
        index = np.argsort(W_sl)[1:k + 1]
        nindex = np.argsort(W_nsl)[0:k]
        SW[i][slindex[index]] = W_full[i][slindex[index]]
        SB[i][nslindex[nindex]] = W_full[i][nslindex[nindex]]
    return SW, SB

#计算拉普拉斯矩阵
def Lap(W):
    D = np.diag(np.sum(W, axis=0))
    return D - W


def LDE(data, train_data, train_label, component, k, t):
    SW, SB = construct_graph(train_data, train_label, k, t)
    SW = np.array((SW, SW.T)).max(axis=0)
    LW = Lap(SW)
    SB = np.array((SB, SB.T)).max(axis=0)
    LB = Lap(SB)
    X = train_data.T
    XLWXT = X.dot(LW).dot(X.T)
    XLBXT = X.dot(LB).dot(X.T)
    XLWXT = XLWXT + np.exp(-5) * np.eye(XLWXT.shape[0])
    T = scipy.linalg.inv(XLWXT).dot(XLBXT)
    eigVals, eigVects = np.linalg.eigh(T)
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-component - 1):-1]
    w = eigVects[:, eigValInd]
    data_ndim = np.dot(data, w)
    return data_ndim


def ddLDE(data, train_data, train_label, component1, component2, k, t):
    n = train_data.shape[0]
    n1 = n2 = int(math.sqrt(train_data.shape[1]))
    L = np.ones((n1, component1))
    R = np.ones((n2, component2))
    SW, SB = construct_graph(train_data, train_label, k, t)
    # solve for R
    Rleft = np.zeros((n2, n2))
    Rright = np.zeros((n2, n2))
    Lleft = np.zeros((n2, n2))
    Lright = np.zeros((n2, n2))
    A = np.zeros((n, n, n1, n2))
    for i in range(n):
        for j in range(n):
            A[i][j] = (train_data[i] - train_data[j]).reshape(n1, n2)
    # offset=10
    # while offset>0.1:
    for z in range(2):
        last_R = R.copy()
        last_L = L.copy()
        for i in range(n):
            for j in range(n):
                ATLLTA = ((A[i][j]).T) @ (L) @ (L.T) @ (A[i][j])
                Rleft += SB[i][j] * (ATLLTA)
                Rright += SW[i][j] * (ATLLTA)
        eigVals, eigVects = scipy.linalg.eig(Rleft, Rright)
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:(-component2 - 1):-1]
        R = eigVects[:, eigValInd]

        for i in range(n):
            for j in range(n):
                ATRRTA = (A[i][j]) @ (R) @ (R.T) @ (A[i][j].T)
                Lleft += SB[i][j] * (ATRRTA)
                Lright += SW[i][j] * (ATRRTA)
        eigVals, eigVects = scipy.linalg.eig(Lleft, Lright)
        eigValInd = np.argsort(eigVals)
        eigValInd = eigValInd[:(-component1 - 1):-1]
        L = eigVects[:, eigValInd]
        offset = np.linalg.norm(np.squeeze(abs(last_R) - abs(R))) + np.linalg.norm(np.squeeze(np.abs(last_L) - abs(L)))
        # print(offset)
    B = L.T @ data.reshape(-1, n1, n2) @ R
    return B.reshape(data.shape[0], -1)


def KernelLDE(data, train_data, train_label, ker, component, k,t):
    SW, SB = construct_graph_01(train_data, train_label, k)
    SW = np.array((SW, SW.T)).max(axis=0)
    LW = Lap(SW)
    SB = np.array((SB, SB.T)).max(axis=0)
    LB = Lap(SB)
    K = kernel(ker)(train_data, train_data,t)
    KLWKT = K.T.dot(LW).dot(K)
    KLBKT = K.T.dot(LB).dot(K)
    KLWKT = KLWKT + np.exp(-5) * np.eye(KLWKT.shape[0])
    T = scipy.linalg.inv(KLWKT).dot(KLBKT)
    eigVals, eigVects = np.linalg.eigh(T)
    eigValInd = np.argsort(eigVals)
    eigValInd = eigValInd[:(-component - 1):-1]
    w = eigVects[:, eigValInd]
    data_ndim = (kernel(ker)(data, train_data,t)) @ w
    return data_ndim
```

然后Kernel LDE可以有如下的：

```python
import numpy as np
import math

class kernel:
    def __init__(self, ker):
        self.ker = ker

    # 线性核函数
    def linear(self, x1, x2):
        return x1 @ x2.T

    # 多项式核
    def poly(self, x1, x2):
        return (x1 @ x2.T) ** 2

    # 高斯核
    def RBF(self, x1, x2, t):
        return math.exp(-(np.linalg.norm(x1 - x2)) / (t))

    def ori(self, x1, x2):
        return x1

    def __call__(self, *args, **kwargs):
        if self.ker == 'linear':
            return self.linear(*args, **kwargs)
        if self.ker == 'poly':
            return self.poly(*args, **kwargs)
        if self.ker == 'RBF':
            return self.RBF(*args, **kwargs)
        if self.ker == '':
            return self.ori(*args, **kwargs)
```

展示一些实验结果，主要是使用的数据集还是ORL人脸数据集，它由总共 400 张人脸图像组成，总共 40 人（每人 10 个样本）。

使用九种方法用于对ORL进行降维。 降维后，我们对特征进行KNN（$k=3$）来计算分类精度。 LPP（LPP、2DLPP 和 Kernel LPP）和 LDE（LDE、2DLDE、Kernel LDE）的用户指定参数是分量 $d$、邻居 $k$ 和参数 $t$。

更进一步，我们讨论了降维后维度对准确性的影响，参见图1：

<img src="子空间学习(6)-LDE\com.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图1. 九种算法在ORL数据集上的实验结果。</center>
此外，我们将讨论当 $Component$ 为 $20$ 时参数邻居 $k$ 和 参数 $t$ 的影响。 我们可视化这两个参数对准确性的影响，参见图2。
<img src="子空间学习(6)-LDE\parameter.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图2. 参数变化对LDE和LPP算法的影响。</center>

# 总结

当然实验结果不太对劲，估计还是参数有问题或者写丑了？由于那个年代的代码基本都是matlab的，我这里复现都是使用Python，有些优化应该不太到位，也没有去参考源码。

