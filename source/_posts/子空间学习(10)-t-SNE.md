---
title: 子空间学习(10)- t-SNE
catalog: true
date: 2022-01-22 12:12:17
subtitle: Subspace Learning- t-SNE
top: 17
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Visualizing Data using t-SNE（2008 JMLR）**，**Learning a Parametric embedding by preserving local structure（2009 AISTATS）**和**Stacked Denoising Autoencoders - Learning Useful Representations in a Deep Network with a Local Denoising Criterion（2010 JMLR）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述数据可视化方法t-SNE。

# 摘要：

无监督降维算法在表示学习中起着重要作用。 其中，t-SNE 旨在更好地创建一个单一的地图，以揭示许多不同尺度的结构。 此外，parametric t-SNE 学习在潜在空间中尽可能好地保留数据的局部结构的参数映射。 在本报告中，我们尝试从数学的角度探索 t-SNE 和parametric t-SNE，并评估 t-SNE 和parametric t-SNE 在 MNIST 上的性能。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# t-SNE

t-SNE 中的 $t$ 是指 Student-t 分布，用于计算低维空间中两点之间的相似度。更具体地说，Student-t 分布在低维空间中采用重尾分布来缓解 SNE 的拥挤问题和优化问题。在 t-SNE 中，Student-t 分布是具有一个自由度的低维映射中的重尾分布，该分布中的联合概率 $q_{ij}$ 为公式(1)：
$$
\begin{equation}
    q_{ij}=\frac{\left(1+\|y_{i}-y_{j}\|^{2}\right)^{-1}}{\sum_{k \neq l}\left(1+\|y_{k}-y_{l}\|^{2}\right)^{-1}}.
\end{equation}\tag{1}
$$
t-SNE 中的 SNE 指的是 Stochastic Neighbor Embedding，旨在找到一种低维数据表示，以最小化 $p_{j|i}$ 和 $q_{j|i}$ 之间的不匹配。更具体地说，SNE 使用梯度下降法最小化所有数据点上的 Kullback-Leibler 散度的总和，见公式(2)：
$$
\begin{equation}
    C=\sum_{i}\operatorname{KL}(P_{i}||Q_{i})=\sum_{i}\sum_{j}p_{j|i}log\frac{p_{j|i}}{q_{j|i}},
\end{equation}\tag{2}
$$
其中 $P_{i}$ 表示给定数据点 $x_{i}$ 的所有其他数据点的条件概率分布，$Q_{i}$ 表示给定数据点 $y_{i}$ 的所有其他数据点的条件概率分布。数据点 $x_{j}$ 与数据点 $x_{i}$ 的相似性是条件概率 $p_{j|i}$（见公式(3)），而 $q_{j|i}$是低维空间中的类似定义，见公式(1)：
$$
\begin{equation}
    p_{i j}=\frac{\exp \left(-\left\|x_{i}-x_{j}\right\|^{2} / 2 \sigma^{2}\right)}{\sum_{k \neq l} \exp \left(-\left\|x_{k}-x_{l}\right\|^{2} / 2 \sigma^{2}\right)}.
\end{equation}\tag{3}
$$

# AutoEncoder

自编码器是一种人工神经网络，用于学习未标记数据的有效编码。 通过尝试从编码重新生成输入来验证和改进编码。 自编码器通过训练网络忽略噪声来学习一组数据的表示（编码），通常用于降维。

自动编码器有两个主要部分：将输入映射到代码的编码器，以及将代码映射到输入重构的解码器。 数学上，我们假设编码器是$\phi$，解码器是$\psi$，自动编码器的目标函数是公式(4)：
$$
\begin{equation}
    \begin{gathered}
        \phi:\ X\rightarrow \mathcal{F}\\
        \psi:\ \mathcal{F}\rightarrow Y \\
        \phi,\psi=\underset{\phi,\psi}{\arg \min}\|X-(\psi \circ \phi) X \|^{2},
    \end{gathered}
\end{equation}\tag{4}
$$
其中 $h=\sigma\left(Wx+b\right)\in \mathcal{R}^{p}=\mathcal{F}$。

# Parametric t-SNE

Parametric t-SNE 旨在保留潜在空间中数据的局部结构，而 t-SNE 也有类似的目标。但是，Parametric t-SNE 具有以下优点：

- 参数映射的缺乏使得非参数降维技术不太适合用于例如分类或回归任务。

- 为了解决反向传播容易陷入局部最小值的问题，t-SNE 使用了一种训练过程，其灵感来自基于受限玻尔兹曼机 (RBM) 的自动编码器训练。更具体地说，所有节点上的联合分布由能量函数 $E(v,h)$ 指定的玻尔兹曼分布给出：
  $$
  E(v,h)=-\sum_{i,j}W_{ij}v_{i}h_{j}-\sum_{i}b_{i}v_{i}-\sum_{j}c_{j}h_{j}.
  $$

Parametric t-SNE 中的参数是指参数映射 $f:X\rightarrow Y$，它通过权重为 $W$ 的前馈神经网络进行参数化。数据空间为$X$，低维潜在空间为$Y$。更具体地说，参数 t-SNE 的参数是自由度的数量，用于计算低维潜在空间中的相似性 $Q$，参见公式(5)：
$$
\begin{equation}
    q_{i j}=\frac{\left(1+\left\|f\left(x_{i} \mid W\right)-f\left(x_{j} \mid W\right)\right\|^{2} / \alpha\right)^{-\frac{\alpha+1}{2}}}{\sum_{k \neq l}\left(1+\left\|f\left(x_{k} \mid W\right)-f\left(x_{l} \mid W\right)\right\|^{2} / \alpha\right)^{-\frac{\alpha+1}{2}}},
\end{equation}\tag{5}
$$
其中 $\alpha$ 表示 Student-t 分布的自由度数。

潜在空间中使用的Student-t分布可能包含分布下的大部分概率质量，因为潜在空间$Y$的体积随其维度呈指数增长。这导致的问题可以通过设置自由度 $\alpha$ 以校正潜在空间体积的指数增长来解决，因为增加自由度 $\alpha$ 会导致分布较轻的尾巴。

# 实验结果

为了更直观地了解 t-SNE 和parametric t-SNE，我们分析了它们在 MNIST 和其他数据集上的性能，分别在如下数据集进行了实验：

- MNIST：前 2k 个训练图像和前 2k 个测试图像。
- COIL-20：数据库包含 20 个对象。 每个物体水平旋转 360°，每 5° 拍摄一张照片。 因此，每个物体有72张像素大小为64X64的图像，总共有360张灰度图像。
- Olivetti Faces：由40个人的400张图片组成，即每个人有10张人脸图片。 每张图片的灰度为8位，每个像素的灰度在0-255之间，每张图片的大小为$64\times 64$。

<img src="子空间学习(10)-t-SNE\MNIST.png" alt="LLE-Val" style="zoom:90%;" />
<center style="color:#C0C0C0;text-decoration:underline">图1. t-SNE在三个数据集上的实验结果。</center>
<img src="子空间学习(10)-t-SNE\MNIST2.png" alt="LLE-Val" style="zoom:90%;" />
<center style="color:#C0C0C0;text-decoration:underline">图2. Parametric 和AutoEncoder在MNIST上的实验结果。</center>
t-SNE在sklearn上有库，然后AutoEncoder网上代码也很多，parametric t-SNE我直接复现但是效果太差，没有找出bug...最后直接参考别人的代码。这里简单放一下t-SNE的代码，另外两个算法的代码，后续github有机会更新。

```python
from sklearn.manifold import TSNE

data_tsne=TSNE(n_components=2,perplexity=5).fit_transform(data) #这里参数可自定义
```



# 总结

简单写的，最后一篇子空间学习笔记了。



 