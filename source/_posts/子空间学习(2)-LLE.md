---
title: 子空间学习(2)-LLE
catalog: true
date: 2022-01-12 12:24:17
subtitle: Subspace Learning-LLE
top: 9
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Think Globally, Fit Locally- Unsupervised Learning of Low Dimensional Manifolds（2003 JMLR）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述LLE的动机，目标函数和限制等。额，然后本系列的解释应该是中英文混合，私以为用简略的句子生动地描述复杂抽象的概念，需要对算法有着极为深刻的理解和一定的勇气。显然我水平不够，如果强制翻译一些原文的描述和概念，难免不太妥当。

# 摘要：

降维算法出现在信息处理的许多领域，包括机器学习、数据压缩、科学可视化、模式识别和神经计算。 其中，Locally Linear Embedding（局部线性嵌入）是一种无监督学习算法，可计算高维输入的低维、邻域保留嵌入。 我们假设输入是从一个底层流形中采样的，并被映射到一个低维的单一全局坐标系中。 映射源自局部线性重建的对称性，嵌入的实际计算简化为稀疏特征值问题。 在本报告中，我们尝试以数学方式解释 LLE 的主要思想，并讨论 PCA 和 LLE 之间的区别。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# Question&Answer：

## 假设

LLE 假设数据是从平滑的底层流形中采样的，并且有足够的数据（使得流形采样良好）。具体来说，

- 数据是从<u>光滑流形</u>中采样的，即数据是无限可微的。 光滑流形的一阶导数的连续性主要用于 LLE，即
  $$
  \begin{aligned}
          &\lim _{x \rightarrow x_{0}^{-}} f_{(x)}=f_{\left(x_{0}\right)}=\lim _{x \rightarrow x_{0}^{+}} f_{(x)} \\
          &\lim _{x \rightarrow x_{0}^{-}} f_{(x)}^{\prime}=f_{\left(x_{0}\right)}^{\prime}=\lim _{x \rightarrow x_{0}^{+}} f_{(x)}^{\prime}.
      \end{aligned}
  $$

- 数据采样良好意味着采样密度是每个数据点都具有 $2d$ 邻居的数量级，这些邻居在流形上相对于输入空间中的某个度量定义了一个大致线性的补丁。

## 解决的问题

LLE 是一种降维算法，它计算高维数据的低维、邻域保留嵌入，即将从底层流形采样（带有噪声）的高维数据映射到低维的单个全局坐标系中。

## 目标函数

LLE的目标函数是为了最小化重构损失，最大程度的保持数据之间的局部关系（为什么保持数据间的局部关系可以进行更好的降维？下一个回答将解释）：
$$
\begin{aligned}
        \arg \min_{W} E(W) &= \sum_{i}|X_{i}-\sum_{j}W_{ij}X_{ij}|^2 \\
        \arg \min_{Y} \Phi(Y) &= \sum_{i}|Y_{i}-\sum_{j}W_{ij}Y_{ij}|^2 
    \end{aligned}\tag{1}
$$

$$
\text{ s.t. } \ \sum_{i=1}^{N} Y_{i}=0,\ \sum_{i=1}^{N} Y_{i} Y_{i}^{T}=N I_{d \times d},\ \sum_{j}W_{ij}=1
$$



## 逐字理解Locally Linear Embedding的含义

### Locally

"Locally"指的是**流形的局部不变性**。由流形的定义可知，流形中的每一点$X_{a}$都有一个邻域$X_{U(a)}$，这个邻域同胚于欧氏空间中的一个开集$W_{a}$，如下所示：
$$
\begin{gathered}
        \forall X_{a} \ \exists X_{U(a)} : \ f(X_{U(a)})\rightarrow W_{a} \ (W_{a} \subseteq \mathbb{R}^{n}) \\
        \forall Y_{a} \ \exists Y_{U(a)} : \ f(Y_{U(a)})\rightarrow W_{a} \ (W_{a} \subseteq \mathbb{R}^{n})
    \end{gathered}\tag{2}
$$
其中 $W_{a}$ 是不变的，$X$ 是高维数据，$Y$ 是低维数据。更具体地说，LLE 中的“Locally”使用**KNN**为每个重建做出贡献。我们可以通过不同的测距方法获得数据点 $X_{i}$ 的 $k$ 最近邻居，例如由欧几里得距离测量，
$$
\begin{gathered}
        D_{i,j}\!=\!\sqrt{(X_{i,1}\!-\!X_{j,1})^2\!+\!(X_{i,2}\!-\!X_{j,2})^2\!+\!\cdots\!+\!(X_{i,D}\!-\!X_{j,D})^2}\\
        sort(\lbrace D_{i,j}|1\leq i,j\leq D\rbrace),
    \end{gathered}
$$
其中 $D_{i,j}$ 表示 D 维向量 $X_{i}$ 和 $X_{j}$ 的距离。我们对$D_{i,j}$进行排序，得到$X_{i}$的$k$个最近邻，即选择$k$个最小的$D_{i,j}$。更进一步，我们将解释为什么可以使用欧几里得距离来确定不变场。

公式（1）指我们可以用欧几里得空间中的坐标系$\tau$来表示流形中$X_{a}$点的邻域。所以$\tau$中点的欧几里德距离代表了点$X_{a}$与其在流形中的邻域之间的关系，即流形可以局部抽象为欧氏空间。然而，当邻域太大而不能被视为欧几里得空间时，局部不变性将失效。

### Linear

"Linear"指计算权重 $W_{ij}$ 最好从 $X_{i}$的邻居$X_{U(i)}$中线性重建数据点 $X_{i}$，即最小化约束的，线性的，目标函数，见等式：
$$
\begin{gathered}
        \arg \min_{W} E(W) = \sum_{i}|X_{i}-\sum_{j}W_{ij}X_{ij}|^2 \\
        \text{ s.t. }\sum_{j}W_{ij}=1,
    \end{gathered}\tag{3}
$$
其中 $X_{ij}$ 是 $X_{i}$ 的邻居，$\sum_{j}W_{ij}X_{ij}$ 是局部线性重建。 此外，我们解释了为什么我们需要约束 $\sum_{j}W_{ij}=1$。约束是**平移不变性、旋转不变性和伸缩不变性**的必要条件，并且可以通过约束$\sum_{j}W_{ij}=1$来使用<u>拉格朗日乘子法</u>来最小化重构误差。

更进一步，我们给出了平移不变性、旋转不变性和伸缩不变性的证明。现在，我们证明平移不变性，看下面的推导，
$$
\begin{aligned}
        \Phi(Y)&\!=\!\sum_{i=1}^{N}\left\|Y_{i}\!-\!\sum_{j=1}^{k} W_{ij} Y_{ij}\right\|^{2}\!=\!\sum_{i=1}^{N}\left\|\sum_{j=1}^{k}\left(Y_{i}\!-\!Y_{ij}\right) W_{ij}\right\|^{2} \\
        &\!=\!\sum_{i=1}^{N}\left\|\sum_{j=1}^{k}((Y_{i}\!-\!\frac{\sum_{i}Y_{i}}{2})\!-\!(Y_{ij}\!-\!\frac{\sum_{i}Y_{i}}{2}))W_{ij}\right\|^{2}.
    \end{aligned}\tag{4}
$$
其中 $\frac{\sum_{i}Y_{i}}{2}$ 是平移量。然后，我们根据方程证明旋转不变性和伸缩不变性，看下面的推导，
$$
\begin{aligned}
        \Phi(AY)&=\sum_{i}\left\|AY_{i}-\sum_{j}W_{ij}AY_{ij}\right\|^2\\
        &=\sum_{i}\left\|\sum_{j}W_{ij}(AY_{i}-AY_{ij})\right\|^2\\
        &=\sum_{i}(I_{i}-W_{i})^{T}Y^{T}A^{T}AY(I_{i}-W_{i})\\
        &=\sum_{i}(I_{i}-W_{i})^{T}Y^{T}Y(I_{i}-W_{i})=\Phi(Y),
    \end{aligned}\tag{5}
$$
其中 $A$ 是旋转（伸缩）和 $\left|A\right|=0$。因此，第二个约束通过将 $\vec{Y}_{i}$ 约束为具有单位阵来消除旋转自由度。同时，第二个约束将比例固定为 d 维，即降维后的维度。

### Embedding

LLE 中的“Embedding”是指将高维数据点 $X_{i}$ 映射到低维嵌入坐标。 在这一步中，LLE 试图最小化重构损失，见等式，
$$
\begin{gathered}
        \arg \min_{Y} \Phi(Y) = \sum_{i}|Y_{i}-\sum_{j}W_{ij}Y_{ij}|^2 \\
        \text { s.t. } \sum_{i=1}^{N} Y_{i}=0, \sum_{i=1}^{N} Y_{i} Y_{i}^{T}=N I_{d \times d},
    \end{gathered}\tag{6}
$$
其中降维后的数据$Y$ 是由权重 $W$ 重建。 因此，输出 $Y_{i}$ 是高维输入 $X_{i}$ 的低维嵌入。 通过在不影响损失函数的情况下添加约束，目标函数具有唯一的全局最小值。 

## 具体的推导过程

在这一部分，我们展示了 LLE 解决方案的细节。我们可以转化(3)为具有两个变量和一个约束的优化问题，
$$
\begin{aligned}
        \Phi(W)&\!=\!\sum_{i=1}^{N}\left\|X_{i}\!-\!\sum_{j=1}^{k} W_{ij} X_{ij}\right\|^{2}\!=\!\sum_{i=1}^{N}\left\|\sum_{j=1}^{k}\left(X_{i}\!-\!X_{ij}\right) W_{ij}\right\|^{2} \\
        &\!=\!\sum_{i=1}^{N}\left\|\left(X_{i\times k}\!\!-\!\!N_{i}\right) W_{i}\right\|^{2}\!\! \\
        &=\!\!\sum_{i=1}^{N}W_{i}^{T}\!\!\left(X_{i\times k}\!\!-\!\!N_{i}\right)^{T}\!\!\!\left(X_{i\times k}\!\!-\!\!N_{i}\right)W_{i}\\
        &\!=\!\sum_{i=1}^{N} W_{i}^{T} S_{i} w_{i},\ S_{i}\!=\!\left(X_{i\times k}\!-\!N_{i}\right)^{T}\!\!\left(X_{i\times k}\!-\!N_{i}\right)\\
        & \text{ s.t. } W_{i}^{T}1_{k\times 1}=1 ,
    \end{aligned}\tag{7}
$$
其中 $X_{i\times k}=\underbrace{\left[X_{i}, \ldots, X_{i}\right]}_{k}$ 和 $N_{i}=\left[X_{i1 }, \ldots, X_{ik}\right]$。接下来，我们将<u>拉格朗日乘数法</u>应用于方程(7)，然后得到权重 $W=[W_{1}, W_{2}, \cdots, W_{N}]$，见(8)，
$$
\begin{aligned}
        L(W_{i})&=\sum_{i=1}^{N} W_{i}^{T} S_{i} W_{i}+\lambda (W_{i}^{T}1_{k\times 1}-1) \\
        \frac{\partial L}{\partial W_{i}}&=2S_{i}W_{i}+\lambda 1_{k\times 1}\\
        W_{i}&=\frac{S_{i}^{-1}1_{k\times 1}}{1_{k\times 1}^{T}S_{i}^{-1}1_{k\times 1}}.
    \end{aligned}\tag{8}
$$
此外，我们将使用重建权重 $W$ 来获得输出 $Y$。更进一步，我们转换优化公式(6)，
$$
\begin{aligned}
        \Phi(Y)&= \sum_{i}\|Y_{i}-\sum_{j}W_{ij}Y_{ij}\|^2\\
        &=\sum_{i=1}^{N}\|Y(I_{i}-W_{i})\|^2\\
        &=tr(Y(I-W)(I-W)^{T}Y^{T})\\
        &=tr(YMY^{T}),\ M=(I-W)(I-W)^{T}
    \end{aligned}\tag{9}
$$
其中 $Y=[y_{1}, y_{2}, \cdots, y_{N}]$, $\mathbb{X}_{i}$ 是包含 $X_{i}$的$k$ 个最近邻居点的集合。然后利用公式（6）中的约束，我们可以通过拉格朗日乘子法和特征分解求解问题（9），见如下过程，
$$
 \begin{gathered}
        L(Y)=YMY^{T}+\lambda(YY^{T}-NI)\\
        \frac{\partial L}{\partial Y}=2MY^{T}+2\lambda Y^{T}=0\\
        MY^{T}=\lambda^{-1}Y^{T}\\
        sort(\lambda^{-1}), \ if\ j<k,\ then\ \lambda^{-1}_{j}<\lambda^{-1}_{k}\\
        Output=[Y_{1}, Y_{2}, \cdots, Y_{d}].
    \end{gathered}
$$

## 讨论

### PCA和LLE的区别

PCA 和 LLE 都是降维方法。 但它们在动机、要解决的问题和目标函数上是不同的。 我们假设$A$是原始数据，$B$是降维后的数据。 同时，$D$ 是$A$ 的维度，$d$ 是$B$ 的维度。 现在我们从 PCA 和 LLE 的不同动机开始解释。

PCA 试图通过线性变换 $P$ 来降低数据的维度，即找到<u>正交基</u>来表示原始数据 $A$，目的是最大化降维后协方差矩阵的迹。目标函数如下：
$$
\begin{gathered}
        B=PA,\ AE=ED\ (AE_{i}=D_{ii} E_{i})\\
        sort(D_{ii}),\ D_{ii}<=D_{jj}\ when \ i<j\\ 
        P=[E_{1}, E_{2}, \cdots, E_{d}]^{T} \\
        \arg \max_{P} tr(PAA^{T}P^{T}) = tr(\frac{1}{n} \sum_{i=1}^{n}(PA_{i})^{2})\\
        \text { s.t. } PP^{T}=I_{d\times d}.
    \end{gathered}
$$


LLE 通过两个步骤来降低底层流形的维数：首先，计算高维数据的低维、邻域保留嵌入 $W$。其次，通过最小化损失函数方程得到输出 $Y$。目标函数在上面已经提过了。

### LLE的优点

- 保留高维空间中的局部线性关系。
- 可以处理有非线性关系的数据。
- 可以学习任何维度的局部线性低维流形。
- 计算输出时可以进行稀疏矩阵特征分解，计算复杂度比较小。

### LLE的缺点

- LLE 只能用于非封闭流形，样本集需要密集均匀。
- LLE对最近邻样本个数的选择很敏感，不同的近邻个数对最终的降维结果影响较大。

## 实验效果(MNIST)

由于Sklearn已经对LLE做了很多的优化，需要代码的朋友可以之间调库解决：

```python
from sklearn.manifold import LocallyLinearEmbedding
def LLE(test_data, train_data, component, neighbor):
    solver=LocallyLinearEmbedding(n_components = component, n_neighbors = neighbor)
    solver.fit(train_data)
    return solver.transform(data)
```
展示一些实验结果，有可视化的降维结果。然后我们在MNIST数据集上，使用PCA和LLE把数据降到2维，展示了当**KNN**的$k=1$时，当LLE的参数$k$变化的时候，准确率的影响：

<img src="子空间学习(2)-LLE\LLE-Val.png" alt="LLE-Val" style="zoom:72%;" />

<center style="color:#C0C0C0;text-decoration:underline">图1. Performance of LE on MNIST.</center>

<img src="子空间学习(2)-LLE\k.png" alt="k" style="zoom:72%;" />

<center style="color:#C0C0C0;text-decoration:underline">图2. PCA(Baseline)和LE在MNIST数据集上的效果对比。</center>

# 总结

很多地方说的不是很细，本篇博客更重要的是去回答一些看论文中的问题。