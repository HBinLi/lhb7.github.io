---
title: 子空间学习(9)-On Regularization and the Prior
catalog: true
date: 2022-01-19 12:24:17
subtitle: Subspace Learning-On Regularization and the Prior
top: 16
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Sparse Representation For Computer Vision and**
> **Pattern Recognition（2009 CVPR）**，**Robust subspace segmentation by low rank represnetation（2010 ICML）**和**L2graph（2016 TCYB）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述正则化中的一些先验。

# 摘要：

子空间聚类是传统聚类的扩展，旨在在数据集中的不同子空间中找到聚类。 通常在高维数据中，许多维度是不相关的，并且可以掩盖噪声数据中的现有聚类。 特征选择通过分析整个数据集去除不相关和冗余的维度。 在介绍子空间聚类算法之前，我们先探讨 L1-norm、L2-norm 和核范数。 此外，本报告对子空间聚类算法进行了调查，例如稀疏子空间聚类 (SSC)、低秩表示 (LRR) 和 L2graph。 然后，我们使用经验可扩展性和准确性测试比较了子空间聚类的三种主要方法。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# On Regularization and the Prior

## 从概率的角度解释$l_{1}$和$l_{2}$范数

这里主要推导线性回归的概率解释，然后给出$l_{1}$和$l_{2}$范数在线性回归上的概率解释，只给证明：

我们推导出线性回归的概率解释。 有两个假设：$\epsilon$ 的下划线概率分布是一个高斯分布，我们可以在 $\theta$ 上指定一个先验分布 $p(\theta)$。

让我们进一步假设 $y$ 可以分解为两个项，
$$
y=\theta^{T}x+\epsilon
$$
其中$\epsilon$ 是解释噪声或未建模因子的误差项。如果$\epsilon$ 的先验概率分布是一个高斯分布，那么我们将有
$$
p(\varepsilon)=\frac{1}{\sqrt{2 \pi} \sigma} \exp (\frac{-(\varepsilon)^{2}}{2 \sigma^{2}}）。
$$

为了计算 $\theta$ 或 $\hat{\theta}$ 的估计，我们使用最大似然估计，
$$
\hat{\theta}_{M L}=\arg \max _{\theta} L(\theta),
$$
其中 $L(\theta) \equiv p(y \mid x ; \theta)$ 是似然函数。然后我们有，
$$
p(y \mid x ; \theta)=\frac{1}{\sqrt{2 \pi} \sigma} \exp \left(\frac{-\left(y-\theta^{T} x\right )^{2}}{2 \sigma^{2}}\right),
$$
似然函数将是，
$$
L(\theta) \equiv p(y \mid x ; \theta)=\Pi_{i=1}^{N} p(y^{(i)} \mid x^{(i)} ; \theta）
$$
$$
=\Pi_{i=i}^{N} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(\frac{-\left(y^{(i)}-\theta ^{T} x^{(i)}\right)^{2}}{2 \sigma^{2}}\right),
$$
和对数似然$\ell(\theta)$，
$$
\ell(\theta) \equiv \log L(\theta)=N \log \frac{1}{\sqrt{2 \pi} \sigma}-\frac{1}{\sigma^{2}} \ frac{1}{2} \sum_{i=1}^{N}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{2} 。
$$

此外，我们可以很容易地看到最大化 $\ell(\theta)$ 等价于最小化
$$
J(\theta)=\sum_{i=1}^{N}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{2}，
$$
这给了我们熟悉的最小二乘成本函数。
综上所述，在对误差项$\varepsilon$ 的概率假设下，我们发现使用最小二乘回归模型求$\hat{\varepsilon}$ 等价于求$\varepsilon$ 的最大似然估计。

### $l_{2}$范数在线性回归上的概率解释

给定一个数据集 $S=\left\{\left(x^{(i)}, y^{(i)}\right)_{i=1}^{N}\right\}$，我们打算根据当前观察找到最可能的 $\theta$，即
$$
\hat{\theta}_{MA P}=\arg \max _{\theta} p(\theta \mid S),
$$
其中 $p(\theta \mid S)$ 是后验概率分布，这样的估计量 $\hat{\theta}_{M A P}$ 也称为 $\theta$ 的 MAP（最大后验）估计量。
通过应用贝叶斯定理，我们可以得到
$$
p(\theta \mid S) \propto p(S \mid \theta) p(\theta),
$$
所以最大化 $p(\theta \mid S)$ 等价于最大化 $p(S \mid \theta) p(\theta)$。因此，我们会有
$$
\hat{\theta}_{MAP}=\arg \max _{\theta} \prod_{i=1}^{N} p\left(y^{(i)} \mid x^{(i) }, \theta\right) p(\theta) 。
$$

数据是从线性回归模型中提取的，因此 $p(S \mid \theta)$ 等于 $\prod_{i=1}^{N} p\left(y^{(i)} \mid x^{ (i)}\right)$。如果我们假设$p(\theta)$的概率分布是一个多元高斯分布，即$p(\theta) \sim N\left(0, \mathbf{I} \sigma^{2} / \lambda \right)$，我们将有
$$
\hat{\theta}_{MA P}=\arg \max _{\theta} Q(\theta)=\arg \max _{\theta} q(\theta),
$$
$$
\begin{gathered}
Q(\theta) \equiv\left(\prod_{i=1}^{N} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(\frac{-\left(y ^{(i)}-\theta^{T} x^{(i)}\right)^{2}}{2 \sigma^{2}}\right)\right) \sqrt{\frac{\ lambda}{2 \pi}} \frac{1}{\sigma} \exp \left(-\frac{\lambda \theta^{T} \theta}{2 \sigma^{2}}\right) \ \
q(\theta)=\log (Q(\theta))。
\end{gathered}
$$

最大化 $q(\theta)$ 等效于最小化成本函数
$$
J(\theta) \equiv\left[\sum_{i=1}^{N}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{ 2}\right]+\lambda \theta^{T} \theta 。
$$

因此，优化问题变为，
$$
\hat{\theta}_{MAP}=\arg \min _{\theta}\left[\sum_{i=1}^{N}\left(y^{(i)}-\theta^{T } x^{(i)}\right)^{2}\right]+\lambda \theta^{T} \theta,
$$
其中最后一项 $\lambda \theta^{T} \theta=\lambda\|\theta\|^{2}$ 正是 $l_{2}$ 范数正则化项。

### $l_{1}$范数在线性回归上的概率解释

我们直接从拉普拉斯分布的最大后验估计推导出来，
$$
\begin{gathered}
Q(\theta) \equiv\left(\prod_{i=1}^{N} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(\frac{-\left(y ^{(i)}-\theta^{T} x^{(i)}\right)^{2}}{2 \sigma^{2}}\right)\right) \prod_{j=1} ^{d}\frac{1}{2b}exp(-\frac{|\theta_{i}|}{b}) \\
q(\theta)=\log (Q(\theta))。
\end{gathered}
$$

最大化 $q(\theta)$ 等效于最小化成本函数
$$
J(\theta) \equiv\left[\sum_{i=1}^{N}\left(y^{(i)}-\theta^{T} x^{(i)}\right)^{ 2}\right]+\lambda \sum_{j=1}^{d}|\theta_{i}|。
$$

因此，优化问题变为，
$$
\hat{\theta}_{MAP}=\arg \min _{\theta}\left[\sum_{i=1}^{N}\left(y^{(i)}-\theta^{T } x^{(i)}\right)^{2}\right]+\lambda \|\theta\|,
$$
其中最后一项 $\lambda \|\theta\|=\lambda \sum_{j=1}^{d}|\theta_{i}|$ 正是 $l_{1}$ 范数正则化项。

# Sparse Subspace Clustering

## Subspace Clustering

原数据可以由多个子空间中的数据点表示，给定子空间集 $S=[s_{1},\cdots,s_{m}]$ 中的数据点 $X=[x_{1},\cdots,x_{n}]$，任务就是在多个子空间中找到最合适的子空间以及对应的数据点来表示原数据，由此：求子空间的个数$m$，它们的维数$D=[d_{1},\cdot,d_{m}]$，每个子空间的基，以及数据的聚类，见公式(1)：
$$
\begin{equation}
    \begin{gathered}
        F=[f_{1},\cdots,f_{n}]\\
        f_{i}: \ X\rightarrow s_{i} \\ 
        f_{i}(X) \ \in \ \mathcal{R}^{d_{i}\times n}.
    \end{gathered}
\end{equation}\tag{1}
$$

## SSC

稀疏子空间聚类的动机是直接使用位于子空间联合上的向量的稀疏表示将数据聚类到单独的子空间中。SSC的目标函数是：
$$
\begin{equation}
    \begin{gathered}
        min \|s\|_{p}\\
        \text{s.t. } \textbf{y}=As,
    \end{gathered}
\end{equation}\tag{2}
$$
其中 $\|s\|_{p}$ 是 $s$ 的不同范数。 考虑一个向量 $x\in \mathcal{R}^{D}$，它可以表示为 $D$ 个向量 。 如果$\{\phi_{i}\in \mathcal{R}^{D}\}^{D}_{i=1}$我们形成基矩阵 $\Phi= [\phi_{1},\phi_{2},···,\phi_{D}]$，我们可以将 $\textbf{x}$ 写为
$$
\boldsymbol{x}=\sum_{i=1}^{D} s_{i} \boldsymbol{\psi}_{i}=\Psi \boldsymbol{s}，
$$
其中 $\textbf{s}= [s_{1},s_{2}, . . . , s_{D}]$。 我们对 $i\in \{1,2,\cdots ,m\}$。 因此，
$$
\textbf{y}=[y_{1},y_{2},\cdots,y_{m}]^{T}=\Phi \textbf{x}=\Phi \Psi \textbf{s}=A\textbf{s},
$$
其中 $\Phi=[\phi_{1},\phi_{2},\cdots,\phi_{m}]^{T}\in \mathcal{R}^{m\times D}$ 称为测量矩阵。此外，SSC 还考虑了从线性或仿射子空间集合中提取的数据点被噪声污染的情况，带噪声的 SSC 的目标函数为公式(3)：
$$
\begin{equation}
    \underset{Z}{\arg \min}\|\textbf{y}-A\textbf{s}\|^{2}+\lambda \|s\|_{p}.
\end{equation}\tag{3}
$$

# Low Rank Representation

考虑 $\mathcal{R}^{D}$ 中的一组数据向量 $X= [x_{1}, x_{2},\cdots, x_{n}]$（每列是一个样本），每个样本可以用Dictionary中基的线性组合来表示 $A= [\alpha_{1}, \alpha_{2},···, \alpha_{m}]$: $X=AZ$ ，其中 $Z= [z_{1}, z_{2},\cdots, z_{n}]$ 是系数矩阵，每个 $z_{i}$ 是 $x_{i}$ 的表示。

LRR 的目标函数是公式(4)：
$$
\begin{equation}
    \begin{gathered}
        \min_{Z}\|Z\|_{*}\\
        \text{s.t. }X=AZ,
    \end{gathered}
\end{equation}\tag{4}
$$
其中 $\|\cdot\|_{*}$ 表示矩阵的核范数，即矩阵的奇异值之和，用于约束矩阵的低秩。对于稀疏数据，矩阵是低秩的，包含大量冗余信息。此信息可用于恢复数据和提取特征。矩阵 $X$ 的核范数定义为：
$$
\begin{equation}
    \|X\|_{*}=\operatorname{tr}\left(\sqrt{X^{T} X}\right).
\end{equation}\tag{5}
$$
根据上式，可以得出核范数等价于矩阵的特征值之和。考虑 X $X=U \Sigma V^{T}$ 的特征值分解，可以得出以下结论：
$$
\begin{aligned}
    \operatorname{tr}\left(\sqrt{X^{T} X}\right) &=\operatorname{tr}\left(\sqrt{\left(U \Sigma V^{T}\right)^{T} U \Sigma V^{T}}\right) \\
    &=\operatorname{tr}\left(\sqrt{V \Sigma^{T} U^{T} U \Sigma V^{T}}\right) \\
    &=\operatorname{tr}\left(\sqrt{V \Sigma^{2} V^{T}}\right)\left(\Sigma^{T}=\Sigma\right) \\
    &=\operatorname{tr}\left(\sqrt{V^{T} V \Sigma^{2}}\right) \\
    &=\operatorname{tr}(\Sigma).
    \end{aligned}
$$

# L2graph

这篇文章主要是在谱聚类的基础上，使用$l_{2}$范数的特征来构建稀疏的相似矩阵。

<img src="子空间学习(9)-On Regularization and the Prior\algorithm.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图2. L2graph算法流程。</center>
<img src="子空间学习(9)-On Regularization and the Prior\equation.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图2. 相关公式。</center>

# 实验结果

为了更直观地理解这三种算法，展示一些实验结果，在如下两种数据集上做聚类评估。

- MNIST：前 2k 个训练图像和前 2k 个测试图像。
- 耶鲁数据库（$64\times 64$）：包含 15 个人的 165 张 GIF 格式灰度图像，每个对象有 11 张图像。

在DUT-OMRON数据集上做分割和去噪实验。

<img src="子空间学习(9)-On Regularization and the Prior\Acc.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图1. 聚类效果评估。</center>

<img src="子空间学习(9)-On Regularization and the Prior\SSC.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图2. SSC图像分割（二分类）。</center>

<img src="子空间学习(9)-On Regularization and the Prior\L2.png" alt="LLE-Val" style="zoom:90%;" />
<center style="color:#C0C0C0;text-decoration:underline">图3. L2graph图像分割（二分类）。</center>

<img src="子空间学习(9)-On Regularization and the Prior\LRR.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图4. LRR图像重建和图像分割。</center>
代码太多了，我传一下L2graph的Python版代码，SSC和LRR的后续看github放一下吧，这两种算法github也能查到。

```python
import numpy as np
from sklearn.preprocessing import normalize

def Normalization(train_data):
    norm_data = normalize(train_data, norm='l2', axis=1)
    return norm_data

def ClusteringL2Graph(dat, lamda=1e2):
    pos = 0
    dat = dat.T
    tmp = dat.T @ dat
    if lamda == 0:
        Proj_M = np.linalg.pinv(tmp)
    else:
        Proj_M = tmp + lamda * np.eye(tmp.shape[0])
        Proj_M = np.linalg.inv(Proj_M)
    Q = Proj_M @ dat.T
    coef = []
    for ii in range(0, dat.shape[1]):
        stdOrthbasis = np.zeros((dat.shape[1], 1))
        stdOrthbasis[ii] = 1
        tmp1 = stdOrthbasis.T @ Q @ dat[:, ii]
        tmp2 = np.linalg.pinv(stdOrthbasis.T @ Proj_M @ stdOrthbasis).ravel()
        ci = Proj_M @ ((dat.T @ dat[:, ii]).reshape(dat.shape[1], 1) - (tmp1[0] * tmp2[0]) * stdOrthbasis)
        coef.append(ci.tolist())
    coef = np.asarray(coef).reshape(dat.shape[1], dat.shape[1]).T

    # l2-norm
    coef = coef - np.eye(coef.shape[0]) * coef
    coef = Normalization(coef)
    C = np.abs(coef) + np.abs(coef)
    return C
#test
#C = ClusteringL2Graph(data)
#clustering_L2=SpectralClustering(n_clusters=cluster, affinity='precomputed')
#pred_label_L2=clustering_L2.fit_predict(C)
```



# 总结

算法效果符合预期，文本理解可能还是不太深刻。



 