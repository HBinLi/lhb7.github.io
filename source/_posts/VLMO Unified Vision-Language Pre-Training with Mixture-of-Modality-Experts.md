---
title: VLMO-使用混合专家模态的视觉-文本预训练模型
catalog: true
date: 2022-02-22 12:12:17
subtitle: VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts
top: 18
header-img: /img/header_img/lml_bg.jpg
mathjax: true
tags:
- Python
categories:
- 跨模态预训练模型
---

> 一篇跨模态的预训练文章，应该是发表在2021的ICML上。论文名字：VLMO: Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts，没有公布代码。

# 摘要：

提出了统一的Vision-Language pretrained Model(VLMo)，联合学习了一个双编码器和一个含有模块化transformer的混合编码器。具体来说，引入了Mixture-of-Modality-Experts (MOME) Transformer，每个块有一个模态特定的专家池，同时有一个共享的注意力层。还提出了分阶段的预训练策略，有效地利用大规模的图像以及图像-文本对之外的纯文本数据。

# 介绍

Vision-Language (VL) pre-training learns generic cross-modal representations from large-scale

image-text pairs. Previous models usually employ image-text matching, image-text contrastive

learning, masked region classifification/feature regression, word-region/patch alignment and masked

language modeling to aggregate and align visual and linguistic information.（从大规模图像-文本对学习通用跨模态表示。以往的模型通常采用图像-文本匹配、图像-文本对比学习、掩蔽区域分类/特征回归、单词区域/补丁对齐和掩蔽语言建模来聚合和对齐视觉和语言信息。） Then the pretrained

models can be directly fifine-tuned on downstream vision-language tasks, such as VL retrieval and

classifification (visual question answering, visual reasoning, etc.).

## 两种主流的框架

- CLIP和ALIGN采用双编码器架构，分别对图像和文本进行编码，并使用余弦相似度或线性投影层来建模图像和文本之间的交互作用。双编码器架构对检索任务是有效的，特别是对大量的图像和文本。可以预先计算和存储图像和文本的特征向量。

  缺点：图像和文本之间的浅层交互并不足以处理需要复杂推理的任务，如余弦相似度的信息量太少，如视觉推理和视觉问题回答(VL分类任务)。ViLT(Kimetal.，2021)发现，CLIP在视觉推理任务上提供的准确性相对较低。

  2021 ICML CLIP

  <img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\1.png" alt="" style="zoom:72%;" />

- 使用具有跨模态的深度融合编码器（deep fusion encoder）来对图像和文本进行交互。多层变压器(Vaswanietal.，2017)网络通常被用于融合图像和文本表示。融合编码器架构在VL分类任务上取得了优越的性能。

  缺点：但它需要联合编码所有可能的图像-文本对，以计算检索任务的相似性得分。O($n^{2}$)时间复杂度导致的推理速度比时间复杂度为线性的双编码器模型要慢得多。同时很大一部分基于融合编码器的模型依赖于一个现成的目标检测器，如更快的R-CNN来获取图像区域特征。生成区域特征会降低推理速度，并使该方法的可伸缩性降低。

  2021 ICML ViLT

  <img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\2.png" alt="" style="zoom:72%;" />

双编码器优于检索任务，融合编码器优于分类任务。          

## 主要的贡献

为了利用这两种类型的架构，提出一个统一的视觉语言预训练模型(VLMO)，可以作为一个双编码器分别编码图像和文本检索任务，或作为一个融合编码器模型的深度交互图像文本对分类任务。是通过引入模态混合专家(MOME)变压器来实现的，MOME采用了一群模态专家来取代标准变压器中的前馈网络。它通过切换到不同的模式专家来捕捉特定模式的模式信息，并使用跨模式共享的自我注意来对齐视觉和语言信息。由三个模态专家组成，分别是图像，文本和图像文本。

利用了三种预训练方法，image-text contrastive learning, image text matching, and masked language modeling。提出了分阶段的预训练策略，

1. 使用BEIT中提出的masked image modeling对MOME Transformer中的图像专家模块和自注意力模块进行预训练，只使用图像数据。
2. 使用masked language modeling对文本专家模块进行预训练，只使用文本数据。
3. 训练的图像和文本模块被用来初始化vision-language预训练模型，对大量的仅图像和仅文本数据进行阶段性的预训练，有助于VLMO学习更一般化的表示。

最后通过在vision-language的检索和分类上fine-tuning来评估VLMo模型。实验结果表明，模型作为双编码器，优于基于融合编码器的模型。同时享受检索任务的推理速度要快得多。此外，用作融合编码器，模型在视觉问题回答(VQA)和视觉推理的自然语言(NLVR2)上取得了最先进的结果。

总结一下贡献，

1. 提出了一个联合的语言文本预训练模型，可作为分类任务一个融合编码器或者作为检索任务的双向编码器进行微调。
2. 对于视觉语言任务，介绍了一个通用的多模态的transformer—MoME Transformer来编码不同的模态，通过不同的模态专家，可以抓住模态特定的信息，然后通过不同模式共享的自注意力模块对齐不同模态的内容。
3. 使用大量的仅图像和仅文本数据的阶段预训练极大地改进了视觉语言预训练模型。

# 模型

输入图像-文本对，VLMo通过MoME Transformer来获取三种表示，即：图像，文本和图像-文本对的表示。然后进行预训练，有三步，基于图像和文本表示的图像-文本对比学习，图像-文本匹配和图像-文本对表示的masked language modeling。

微调的时候，模型可以作为双编码器分别对图像和文本进行编码。它还可以作为融合编码器进行微调，以为分类任务建模更深层次的模态交互。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\3.png" alt="" style="zoom:72%;" />

## Input

输入图像-文本对，生成图像，文本和图像-文本向量。

**Image Representation**：Following的是ViT，2D图像$v\in \mathcal{R}^{H\times W\times C}$转化为$v^{p}\in \mathcal{R}^{N\times (P^{2}C)}$，即把原图像切成$N=HW/P^{2}$块，然后进行通道层上的堆叠，其中$P\times P$是小块的分辨率。然后把image patches压平( flattened)为向量，并进行线性投影以获得patch embeddings，到此是一模一样。

然后是不一样的地方，加入了 a learnable special token [I_CLS]，最终的输入图像表示$H_{0}^{v}\in \mathcal{R}^{(N+1)\times D}$是对patch embeddings，learnable 1D position embeddings $V_{pos}\in \mathcal{R}^{(N+1)\times D}$和 image type embedding $V_{pos}\in \mathcal{R}^{D}$。

$$\boldsymbol{H}_{0}^{v}=\left[\boldsymbol{v}_{\text {[I_CLS] }}, \boldsymbol{V} \boldsymbol{v}_{i}^{p}, \ldots, \boldsymbol{V} \boldsymbol{v}_{N}^{p}\right]+\boldsymbol{V}_{\text {pos }}+\boldsymbol{V}_{\text {type }}$$

其中线性投影$V\in \mathcal{R}^{(P^{2}C)\times D}$。

## **补充知识**

1. [I_CLS]是什么？

   本文写的由来是ViT，但是还可以往前面追溯到Bert。

   [CLS]就是classification的意思，可以理解为用于下游的分类任务。主要应用于单文本分类和语句对分类任务，BERT模型在文本前插入一个[CLS]符号，并将该符号对应的输出向量作为整篇文本的语义表示。优点是这个无明显语义信息的符号会更“公平”地融合文本中各个字/词的语义信息。在语句对分类任务中还有对输入的两句话用一个[SEP]符号作分割，并分别对两句话附加两个不同的文本向量以作区分。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\4.png" alt="" style="zoom:72%;" />



<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\5.png" alt="" style="zoom:72%;" />



2. 位置编码

有绝对位置编码BERT，相对位置编码和三角函数位置编码。

在ViT中有位置编码已经有实验证明优于没有位置编码，其中1D的位置编码效果是最好的。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\6.png" alt="" style="zoom:72%;" />

所采用的方法是三角函数，注意这个PE是二维矩阵，大小和Embedding一样，pos表示词语在句子中的位置，$d_{model}$表示词向量的维度，i表示词向量的位置。
$$
\begin{aligned}
P E_{(p o s, 2 i)} &=\sin \left(p o s / 10000^{2 i / d_{\text {model }}}\right) \\
P E_{(p o s, 2 i+1)} &=\cos \left(p o s / 10000^{2 i / d_{\text {model }}}\right)
\end{aligned}
$$
接着回到我们之前讲的图像表示公式，位置编码我解释了，然后这个$V_{type}$是一个一维向量嘛，文章没有明确提，其他文献我也没找到，我猜想就是每个位置都加一些比较特色的值，最后能分辨是图片表示还是文本表示就行了。

**Text Representations**：Follow的是BERT，把每段文本使用WordPiece的方法打乱，这个WordPiece相当于是把句子和单词使用一定的方法打断，比如BPE（Byte-Pair Encoding）双字节编码，把字变成一个个的字符，然后统计出现最多的字符，然后进行组合。

之后对生成的M个向量，加一个序列开始标记([T_CLS])和一个特殊的边界标记([T_SEP])。最终的文本表示$H_{0}^{w}\in \mathcal{R}^{(M+2)\times D}$是通过对应的word embedding, text position embedding和text type

embedding累加得到：$\boldsymbol{H}_{0}^{w}=\left[\boldsymbol{w}_{\left[\mathrm{T}_{-} \mathrm{CLS}\right]}, \boldsymbol{w}_{i}, \ldots, \boldsymbol{w}_{M}, \boldsymbol{w}_{\left[\mathrm{T}_{\mathrm{SEP}}\right]}\right]+\boldsymbol{T}_{\text {pos }}+\boldsymbol{T}_{\text {type }}$

**Image-Text Representations**：将图像和文本输入向量连接起来，以形成图像-文本输入表示$\boldsymbol{H}_{0}^{vl}=[\boldsymbol{H}_{0}^{w};\boldsymbol{H}_{0}^{v}]$。

## Mixture-of-Modality-Experts Transformer

提出了一种用于Vision-Language任务的通用多模态转换器MOME Transformer来编码不同的模态，用混合模态专家模型替代了transformer的前馈网络，假设前一层的输入为$H_{l-1}, l\in [1,L]$，MOME先通过跨模态共享的multi-head self-attention(MSA)来对齐视觉和语言信息，然后通过不同的专家模块来获得特定的模态信息。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\7.png" alt="" style="zoom:72%;" />

MoME-FFN根据输入向量$H_{l}^{\prime}$和Transformer的不同层来选择专家模块。有三种专家模块，vision expert (V-FFN), language expert (L-FFN) and vision-language expert (VL-FFN)。

输入如果是纯文本和纯图像的向量，分别使用文本专家(L-FFN)和图像专家(V-FFN)来分别对文本和图像向量进行编码。

输入如果是由多模态的向量组成，比如图像-文本对，分两个阶段编码。在底层分别使用文本专家(L-FFN)和图像专家(V-FFN)来分别对文本和图像向量进行编码，然后使用VL-FFN在顶层进行更多的模态交互。

## 训练过程

### **Stagewise Pre-Training**

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\8.png" alt="" style="zoom:72%;" />

**首先预训练V-FFN和self-attention module模块**，使用的是BEIT来训练大尺度的纯图像数据。**然后冻结V-FFN和self-attention module模块的参数，使用masked language modeling 来训练L-FFN**（文本专家）。至于这里为什么冻结，文章没有解释，我理解是将经过MSA后，在同一子空间的图像和文本信息分布尽量相同。

这里简单讲一下BEIT和BERT，以及他们在这里怎么用的。

首先是BERT，BERT是语言的预训练模型，使用的结构是Transformer里面的Encoder模块，然后文本的表示是多了Segment Embeddings。预训练的任务是两个：一个是**Masked LM**，给定一句话，随机抹去几个单词，然后去预测这几个单词，作用是使模型更多地依赖于上下文信息去预测词汇，并且赋予了模型一定的纠错能力；还有一个是**Next Sentence Prediction**，给定一篇文章中的两句话，判断第二句话在文本中是否紧跟在第一句话之后，让模型能够更准确地刻画语句乃至篇章层面的语义信息。

BERT在本文的用法主要是使用了两个预训练任务的**Masked LM**，我推测是目前涉及的跨模态任务还没有需要段落重排序这个任务。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\9.png" alt="" style="zoom:72%;" />

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\10.png" alt="" style="zoom:72%;" />

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\11.png" alt="" style="zoom:72%;" />

然后说一下BEIT，之前的图像取batch的操作就是BEIT的操作，网络结构是Transformer，首先训练一个离散的VAE（自编码器和解码器），达到能够获得Visual Tokens的表示，然后重构图像。之后的预训练任务只有一个，**Masked Image Modeling**，跟之前的BERT的**Masked LM**差不多，这里目标变成了需要尽可能地让这些被mask掉的部分经过网络之后的结果逼近原图片的Visual Tokens。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\12.png" alt="" style="zoom:72%;" />

### Pre-Training Tasks

之前的训练是无监督的，接下来是有监督的训练。然后是对图像-文本对的数据进行三个预训练任务，VLMo使用 image-text contrastive learning on the image and text representations, masked language modeling and image-text matching on the image-text pair representations with shared parameters（在图像和文本表示上的图像-文本对比学习，掩码语言模型和共享参数的图像-文本对图像-文本匹配）。

#### **Image-Text Contrast**

接下来的训练是有监督的，一个batch里面有N个图像-文本对，图像-文本对比学习在$N\times N$个可能的图像-文本对中选择，有$N^{2}-N$和负样本对。之前output vector中的[I-CLS]和[T-CLS] token被用来作为图像和文本的联合表示（我推测是图像和文本的表示）。然后对图像的表示和文本的表示分别经过线性投影和标准化，我们得到图像的表示$\left\{\hat{\boldsymbol{h}}_{i}^{v}\right\}_{i=1}^{N}$和文本的表示$\left\{\hat{\boldsymbol{h}}_{i}^{w}\right\}_{i=1}^{N}$来计算图像到文本和文本到图像的相似性。
$$
\begin{gathered}
s_{i, j}^{i 2 t}=\hat{\boldsymbol{h}}_{i}^{v_{\top}} \hat{\boldsymbol{h}}_{j}^{w}, s_{i, j}^{t 2 i}=\hat{\boldsymbol{h}}_{i}^{w_{\top}} \hat{\boldsymbol{h}}_{j}^{v} \\
p_{i}^{i 2 t}=\frac{\exp \left(s_{i, i}^{i 2 t} / \sigma\right)}{\sum_{j=1}^{N} \exp \left(s_{i, j}^{i 2 t} / \sigma\right)}, p_{i}^{t 2 i}=\frac{\exp \left(s_{i, i}^{t 2 i} / \sigma\right)}{\sum_{j=1}^{N} \exp \left(s_{i, j}^{t 2 i} / \sigma\right)}
\end{gathered}
$$
这里用的应该是余弦相似度。其中$s_{i,j}^{i2t}$代表的是第i个图像和第j个文本的图像-文本相似度，$s_{i,j}^{t2i}$是文本-图像相似度，$\sigma$是可学习的参数，$p_{i}^{i2t}$和$p^{t2i}_{i}$是softmax后的相似性。然后用交叉熵损失函数来衡量两个分布之间的相似性。

$$
H(p^{i2t},p^{t2i})=-\sum_{i}p_{i}^{i2t}log(p_{i}^{t2i})
$$

#### Masked Language Modeling

有监督训练，这部分也是采用BERT的一部分预训练的方法，mask掉15%的token，然后使用没被mask掉的token和视觉的信息取预测mask掉的token。这部分的训练方式，我推测L-F层的输出是作为Q和K，计算加权，然后进行transformer的操作，使用多头注意力机制进行一个集成，防止过拟合和降低计算量。最后是训练一个分类器，把预测的token在整个词汇表进行分类，希望能够分在ground truth的概率逼近1，使用交叉熵损失函数训练这个分类器。

#### Image-Text Matching

有监督训练，使用的是文本专家的最后一层隐藏层里面的[T-CLS]token，来代表图像-文本对的预测标签，然后使用交叉熵来训练一个二分类器。这里涉及到正负样本的构建，采用的是ALBEF构建hard negative的样本，就是大师兄前段时间讲的论文里的方法。从本质上来说，这个方法有点像过去目标检测里面的hard negative mining，主要就是找假阳性样本。A negative image-text pair is hard if they share similar semantics but differ in fine-grained details。利用之前的相似性计算公式去寻找一个batch里面的hard negatives。对于每一个图像找一个负样本文本，对于每一个文本找一个负样本图像。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\13.png" alt="" style="zoom:72%;" />

## Fine-Tuning VLMO on Downstream Tasks

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\14.png" alt="" style="zoom:72%;" />

对于图像-文本分类任务，使用融合编码器来做图像和文本的交互，也是采取[T_CLS]作为表示，送给分类器。

对于图像-文本检索任务，使用双编码器对图像和文本进行编码，计算图像和文本的表示，像之前的公式一样得到相似度得分。

# 实验结果

**Visual Question Answering(VQA)**

Given an image and a question in natural language, it requires reasoning over visual elements of the image and general knowledge to infer the correct answer.

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\15.png" alt="" style="zoom:72%;" />

Other tasks：

Visual Dialogue: 对于一张图片问一连串相关的问题，而不是只问一个问题。

TextCaps 要求模型阅读和推理图像中的文本以生成关于它们的标题。

基于常识的VQA：除了图片和问题，还提供了一段常识问题，辅助推理。

**natural language for visual reasoning（NLVR2）**

需要同时输入两张 Image 和一个描述，输出是描述与 Image 的对应关系是否一致，label 只有两种（true/false）。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\16.png" alt="" style="zoom:72%;" />

## Evaluation on Vision-Language Classification Tasks

主要是两个任务：visual question answering和natural language for visual reasoning。

VQA2.0：

给出一个自然的图像和一个问题，任务是生成/选择正确的答案。

- 204,721 COCO images
  (all of current train/val/test)
- 1,105,904 questions
- 11,059,040 ground truth answers

Natural Language for Visual Reasoning (NLVR2)：要求模型预测一对图像的文本描述是否正确。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\17.png" alt="" style="zoom:72%;" />

VLMO-Base由12层Transformer块组成，具有768个hidden size和12个attention head。VLMO-Large是一个24层的Transformer块组成，具有1024个hidden size和16个attention head。对于base模型和large模型，前馈网络的中间尺寸分别为3072和4096。实验结果里面的test-dev和test-std是VQA2.0两个测试数据集，dev和test-P是NLVR2的两个测试数据集，指标就是Acc。

## Evaluation on Vision-Language Retrieval Tasks

检索任务包括图像到文本检索和文本到图像检索。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\18.png" alt="" style="zoom:72%;" />

是在COCO和Flickr30K数据集上测试，TR是指图像到文本检索，IR是值文本到图像检索，这个R@几应该是前几个检索其中有正确答案的准确率。然后就是这个符号，有一个杠是指ALIGN和VLMo分别对图像和文本进行编码，然后采用浅交互(点积)获得相似度得分，来排序；有两个杠的ALBEF首先分别对图像和文本进行编码，得到最优的候选图像，然后将这些表示形式输入融合编码器，对候选图像进行重新排序。另一种则要求用融合编码器对所有图像-文本组合进行编码。

优于ALBEF这种融合编码器的原因是ALBEF需要联合编码，$O(N^{2})$，速度快。优于ALIGN算法是因为可以在更少的数据集上达到更好的效果。

## 消融实验

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\19.png" alt="" style="zoom:72%;" />

分阶段预训练的消融实验，用的是BEIT-Base。

<img src="VLMO Unified Vision-Language Pre-Training with Mixture-of-Modality-Experts\20.png" alt="" style="zoom:72%;" />

ITC，ITM和MLM分别指的是预训练中的图像-文本对比学习，图像-文本匹配任务和masked language model。Std-TRM指的是标准的transformer，MoME-VLExp指的是缺少语言和图像专家的MoME模型。

# Why it works？

本质上来说两个模态的数据可以看作两个概率分布模型，双编码器结构是在分别解析两个模型后进行交互，在子空间的相似性的计算有待思考，同时只用余弦相似性来表示数据太过简单；混合编码器是在原空间对数据进行交互，然后通过编码器，缺点是复杂度高。而本文提出的双编码器和混合编码器，有点像先PCA再去在子空间进行交互，我认为这样的优点是去除了一些噪声数据，同时让双编码后的子空间数据更加紧凑(当然这只是个描述，让数据更符合之后的后续处理)。

然后聊一下本文对比之前的方法有什么优势。

总结一下，对于图像-文本分类任务，使用的是预训练模型+双编码器和混合编码器融合的结构，而对于图像-文本检索任务使用的是预训练模型+双编码器结构。

首先从双编码器这部分看，为什么从这开始，我们可以发现的是在图片检索这个任务上，VLMo算法主要采用的就是双编码器，这和之前的算法是一致的，那为什么效果好呢？

这里我对比CLIP和ALIGN算法。是这样的，CLIP和VLMO论文中的实验不互通，就是VLMo实验本身没有与CLIP对比，只有ALIGN对比了，但是ALIGN的论文与CLIP对比了，那我就讲讲。论文里给的效果是VLMO大于ALIGN大于CLIP。

CLIP用的是4亿个图像-文本对，使用了图像-文本对比学习+prompt类型的分类器，区别来说主要是CLIP有大数据量和prompt，VLMo多了预训练模型。在检索这个任务上，VLMo就是证明了**预训练有效果**。ALIGN使用了18亿的带noisy图像-文本对，使用了图像-文本对比学习，基本上来说除了数据量和处理noisy的鲁棒性上，VLMo就是ALIGN的改进，也是证明了**预训练有效果**。

这时我们再看消融实验，我认为**MLM**等预训练任务让模型的质量有所提升，在混合模态的交互上(1和2，5和6)有数据，需要探讨的是单模态的MLM是否有显著效果。理论上来说应该是有的，因为模型使用的是[CLS]作为表示。

然后，讨论一下，图像-文本分类任务上的双编码器和混合编码器融合的结构为什么有效？我认为是增加了模态间的交互。

# 总结

未来，文章提出将从以下几个方面对vlmo进行改进:

- 加大模型和数据量。
- 在一个统一的端到端框架下，把目标检测作为一个新的预训练任务，加入VLMo。因为有研究证明，视觉语言模型中的对象信息可以显著提高模型在下游任务上的性能。
- 在图像-文本生成任务中进行fine-tuning。
- 探究图像和文本的预训练在多大程度上可以帮助模型，特别是当模型可以自然地使用文本和图像地预训练模型时。
- 拓展模态。