---
title: 子空间学习(8)-Sparse Representation
catalog: true
date: 2022-01-18 12:24:17
subtitle: Subspace Learning-Sparse Representation
top: 15
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 子空间学习
---

> 子空间学习系列主要讨论从PCA发表开始到2010年中，子空间学习相关论文。本文立足于论文：**Sparse Representation For Computer Vision and**
> **Pattern Recognition（2010 PIEEE）**和**Dictionaries for Sparse Representation Modeling（2010 PIEEE）**。基于此进行概念的梳理和思考，尝试从数学的角度去阐述Sparse Coding（稀疏编码）和Dictionary Learning（字典学习）。

# 摘要：

过去几十年在图像处理方面取得的大部分进展可归因于图像内容的稀疏和冗余表示建模以及这些模型的应用。 数据的稀疏和冗余表示建模假设能够将信号描述为来自预先指定的字典的几个原子的线性组合。 虽然字典学习采取了不同的路线，但将字典附加到它应该提供的一组示例上。 在本报告中，我们尝试探索稀疏编码和字典学习的主要思想，并讨论它们之间的区别。 此外，我们在图像去噪、图像去模糊和图像超分辨率等多个应用中重现了 K-SVD 的性能。本文的解释会比较简略只提重点部分。需要注意的是本文有下划线的部分是一些需要了解的基础概念，由于篇幅，我就不在后面解释，望读者自行查阅。

# 字典学习

字典学习是信号处理和机器学习的一个分支，旨在找到一个框架 $D\in \mathcal{R}^{d\times n}：D=[d_{1},\cdots,d_{n}] $（称为字典）。 在数学上，输入数据可以表示为：
$$
x=D\alpha
$$
其中$\alpha$ 是一个系数向量。 请注意，稀疏编码中的 $\alpha$ 是稀疏的。

# 稀疏编码

稀疏编码是一种表示学习方法，旨在找到输入数据的稀疏表示 $X=[x_{1},\cdots,x_{K}],x_{i}\in \mathcal{R}^{d }$ ，稀疏表示有很多元素，数据可以由元素本身的线性组合表示。 这些元素称为原子，它们组成一个字典$D\in \mathcal{R}^{d\times n}：D=[d_{1},\cdots,d_{n}]$。 在数学上，输入数据可以表示为：
$$
x=D\alpha \tag{1}
$$
其中$\alpha$ 是稀疏系数向量。

## $l_{0}$和$l_{1}$范数的稀疏编码

基于 L0-norm 和 L1-norm 的目标函数是公式(2)和公式(3)：
$$
\begin{equation}
    (\alpha_{0},e_{0})=\arg \min \|\alpha\|_{0}+\|e\|_{0} \ \ subj \ \ x=D\alpha+e,
\end{equation}\tag{2}
$$
其中 $\mathcal{l}^{0}$ 范数 $\|\cdot\|$ 计算向量中非零的数量。 而$\alpha=[0,\cdots,0,\alpha_{i}^{T},0,\cdots,0]^{T}\in \mathcal{R}^{N}$是一个系数向量，除了与第 i 个类相关的条目外，其条目均为零。
$$
\begin{equation}
    \min \|\alpha\|_{1}+\|e\|_{1}\ \ subj \ \ x=D\alpha+e,
\end{equation}\tag{3}
$$
其中$\|\alpha\|_{1}=\sum_{i}|\alpha_{i}|$。

对于数据 $x$，我们可以将 $x$ 表示为字典 $D$ 中原子的线性组合，如 $x=D\alpha$。 当解决像公式(1)这样的稀疏编码问题时，我们将得到向量 $\alpha$ 的许多解。 当我们添加 L0-norm 和 L1-norm 时，$\alpha$ 将成为一个稀疏向量。 因此，我们可以稀疏地表示数据 $x$。

## 不同的稀疏编码的区别

L0-norm 和 L1-norm 的主要区别在于目标函数的不同表示。 更具体地说，L0 范数计算一个向量的非零元素的总数（见公式(4)），而 L1 范数是空间中向量每个元素的大小之和，见公式(5)。
$$
\begin{equation}
    \|e\|_{0}=\sum_{1}^{K}x_{i}^{0}.
\end{equation}\tag{4}
$$

$$
\begin{equation}
    \|e\|_{1}=\sum_{1}^{K}x_{i}.
\end{equation}\tag{5}
$$

# 应用

论文中主要介绍了Image Reconstruction（图像重建），Image Deblurring（图像去噪，可能包含在前面的？），Superresolution（图像超分），Learning to Sense等。每种应用都有一些基本的目标函数和任务描述，这里我不多提了，可以更多地关注原文。

# 实验结果

为了更直观地理解稀疏编码和字典学习，K-SVD 被应用于以下应用：图像去噪、图像去模糊、图像修复和图像超分辨率。

评价标准采用<u>Peak signal-to-noise ratio（PSNR）</u>，在数据集上进行实验后，我们可视化了 K-SVD 在图像去噪、图像去模糊、图像修复和图像超分辨率方面的性能。 其中，图像超分辨率是有监督训练的，图像去噪、图像去模糊和图像修复是无监督训练的。 进一步，结果如图1，图2。

主要是本节相关算法的代码，贴一下KVD做图像去噪的，注意如果是彩色图，KSVD要在每个通道上都使用一次：

```python
import numpy as np
from sklearn import linear_model

def KSVD(Y, dict_size,
         max_iter=20,
         sparse_rate=0.1,
         tolerance=1e-10):
    assert (dict_size <= Y.shape[1])

    def dict_update(y, d, x):
        assert (d.shape[1] == x.shape[0])

        for i in range(x.shape[0]):
            index = np.where(np.abs(x[i, :]) > 1e-7)[0]

            if len(index) == 0:
                continue

            d[:, i] = 0
            r = (y - np.dot(d, x))[:, index]
            u, s, v = np.linalg.svd(r, full_matrices=False)
            d[:, i] = u[:, 0]
            for j, k in enumerate(index):
                x[i, k] = s[0] * v[0, j]
        return d, x

    # initialize dictionary
    if dict_size > Y.shape[0]:
        dic = Y[:, np.random.choice(Y.shape[1], dict_size, replace=False)]
    else:
        u, s, v = np.linalg.svd(Y)
        dic = u[:, :dict_size]

    print('dict shape:', dic.shape)

    n_nonzero_coefs_each_code = int(sparse_rate * dict_size) if int(sparse_rate * dict_size) > 0 else 1
    print(n_nonzero_coefs_each_code)
    for i in range(max_iter):
        x = linear_model.orthogonal_mp(dic, Y, n_nonzero_coefs=n_nonzero_coefs_each_code)
        e = np.linalg.norm(Y - dic @ x)
        if e < tolerance:
            break
        dict_update(Y, dic, x)

    sparse_code = linear_model.orthogonal_mp(dic, Y, n_nonzero_coefs=n_nonzero_coefs_each_code)

    return dic, sparse_code

#读取自己的带噪声的图片
    #imgR = img_with_noise[:,:,0]
    #imgG = img_with_noise[:,:,1]
    #imgB = img_with_noise[:,:,2]
    #dictionaryR, sparsecodeR = KSVD(imgR,size,max_iter=200)
    #dictionaryG, sparsecodeG = KSVD(imgG,size,max_iter=200)
    #dictionaryB, sparsecodeB = KSVD(imgB,size,max_iter=200)
    #img_reconstructedR=dictionaryR @ sparsecodeR
    #img_reconstructedG=dictionaryG @ sparsecodeG
    #img_reconstructedB=dictionaryB @ sparsecodeB
    #img_reconstructed = np.stack((img_reconstructedR, img_reconstructedG, img_reconstructedB), axis=2)#去噪后的图片
```

由于KSVD的四个实验里的三个实验是无监督的，所以效果不太好，PSNR也只有个位数，目测KSVD应该用于有监督会比较好。
<img src="子空间学习(8)-Sparse Representation\Image_debluring.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图1. KSVD图像去模糊。</center>
<img src="子空间学习(8)-Sparse Representation\inpainting_yiding.png" alt="LLE-Val" style="zoom:90%;" />

<center style="color:#C0C0C0;text-decoration:underline">图2. KSVD图片重建。</center>


# 总结

KSVD采用有监督做应该会比较好，本次读的两篇论文偏综述，对算法实现细节的描述不多。



 