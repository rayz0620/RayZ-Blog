title: "论文阅读:《Language Models for Image Captioning: The Quirks and What Works》"
date: 2015-05-19 09:35:36
tags:
	- RNN
	- LSTM
	- Image Captioning
categories:
	- academic
description: 直接将CNN Feature输入RNN生成句子，比用CNN预测Bag of Words，再把BOW输入Maximum Entropy模型生成句子更好。RNN倾向于输出和训练样本相似度高的句子，Nearest Neighbor法和RNN的Performance接近。
---
[原文地址](http://arxiv.org/abs/1505.01809)

## 简介
这篇文章是最近发在Arxiv上的一篇实验和综述性质的文。主要对两类主流的Image Captioning方法（Max Entropy 和RNN）进行了对比实验。作者将这两种方法结合，刷出了[MSCOCO Image Captioning Challenge](https://www.codalab.org/competitions/3221)上目前最好的Performance。除了比Performance，这篇文章还分析了生成出来的Caption的特点，引出对MSCOCO数据集内部差异性的讨论。

## 两类主流Image Captioning方法
文章分析了目前主流的两类Image Captioning方法，分别是Detection Conditioned法和Activation Conditioned法。两种方法都使用CNN作为图像特征，区别在于它们如何使用CNN特征。这里只简单介绍一下两类方法，详细的大家可以去看原文，我以后也可能会针对具体文章再写。

### Detection Conditioned
Detection Conditioned方法出自出自[Maximum Entropy](http://arxiv.org/abs/1411.4952)(这篇工作貌似值得一读，下一篇可能写它)。这个方法思路比较特别，我是第一次看到。它把captioning分为两步：
1. 使用CNN预测一个bag of words（BOW）
2. 把BOW输入ME语言模型生成一个句子

第一步可以看做是一个Object Detector，输入图像，对词表中的每个词预测一个概率，选出概率高于阈值$\tau$的词，作为预测出的BOW。每一个词的Detector是单独训练的，比较耗时间，也比较耗数据，算是这个方法的缺点吧。

第二步中用到的ME语言模型和很多语言模型一样，逐个预测下一个词的条件概率。ME模型的特点在于预测下一个词概率时的条件。它不光以之前生成的词为条件，还以第一步中预测的BOW中**尚未出现在生成句子**中的词为条件。写成公式是这样的：
{% math_block %}
	P(w_{next} \mid w_{previous}, w_{unpredicted})
{% endmath_block %}
第一步生成的BOW通过{% math w_{unpredicted} %}影响句子生成的过程。BOW中的一个词如果已经出现在生成的句子中，就不会再出现在{% math w_{unpredicted} %}里。作者认为这样的设计使得模型倾向于覆盖BOW中尽可能多的词，减少了生成句子中的重复。

### Activation Conditioned
使用Activation Conditioned概括来说，就是用CNN生成图像feature(一般是取最后一个全连接层的激活值)，直接作为RNN或者LSTM的初始输入，生成句子，不做额外的object detection.

## 性能实验

### 句子生成：最近邻法大杀四方
作者实现了五种方法进行对比实验：
1. Detection + Maximum Entropy
2. Detection + LSTM
3. Activation + Gated RNN
4. 1-Nearest Neighbor
5. k-Nearest Neighbor

其中Gated RNN是RNN的一个变种，可以参考[这里](http://arxiv.org/abs/1501.00299)。简单来说，Gated RNN就是非线性自连接+逐维系数gating。前者继承自经典RNN，后者貌似是从LSTM吸收来的。总的来说Gated RNN算是介于RNN和LSTM之间的一个东西吧。

Nearest Neighbor(NN)法有两种。其中1-NN是通过比图像feature，从和query图像$t$最相似的图像中随机取一个caption句子作为生成的句子。k-NN要复杂一些。简单来说，是在$k$个最近邻图像的caption中，寻找最通用的一个句子作为输出。具体来说，先找出和$t$最接近的$k$个图像，这$k$个图像的所有caption集合为$C$。对每个{% math c_i \in C %}，计算它和其他所有$c$的相似度，取相似度最大的一个{% math c_i %}作为生成句子。

对比实验结果见表1：
![Generation Performance](/img/lm_for_ic/generation_performance.png)

对比实验用了三个指标：Perplexity(越低越好)，BLEU(越高越好)，METEOR(越高越好)。[METEOR](http://www.cs.cmu.edu/~alavie/METEOR/)是2014年提出的一个新的机器翻译评价指标。METEOR的特点是考虑了语言内部的先验知识，包括同义、词根词缀、释义(paraphasing)，并且能从数据中学出这些知识。经典的评价指标，比如BLEU，只考虑机器输出句子和人工提出的参考翻译之间的*字面*相似性，而忽略了自然语言中常见的同义(不同词同义)、同根(词根相同但词缀不同)、释义(语句/词对语句/词)等现象。

结果中令人惊讶的是，k-Nearest Neighbor法的结果并不差。在强调字面相似性的BLEU分数上，kNN取得了最好的成绩。在综合性较强的METEOR分数上，kNN只比最好的低0.3%，差距非常小。

### Re-Ranking：多模型结合效果好
这篇文章还做了另一类实验，探讨多种模型结合的效果。实验方法是这样的：对每个query，先用一种模型生成500个似然性最高的候选句子，再用另外的模型对500个候选句子重新排序(re-rank)，选出最好的一个作为输出。实际上，作者以Maximum Entropy法作为基本方法，加入其他方法进行比较。结果如下表2：

![Re-ranking Performance](/img/lm_for_ic/re_ranking_table.png)

结论是，结合多种方法做re-ranking，比只用一种方法要好。

## 生成句子的分析
除了使用定量标准分析模型的性能，作者还从语言的角度对生成的句子进行了分析。这是其他工作中很少见的，分析的结果也有一定的启示意义。

### 语言错误和模型表达能力
作者总结了Maximum Entropy(ME)和Gated RNN(GRNN)模型生成句子的一些常见错误：

![Example Errors](/img/lm_for_ic/example_error_table.png)

表中前三行的句子里，ME模型生成的句子存在错误的指代关系(anaphora)，句末的"on top of it"短语中的"it"在前文并没有正确的指代对象。GRNN模型则能够正确地使用"on top of it"这个词组，"it"都有正确的指代对象。作者认为ME模型出现这种错误的原因在于ME模型的上下文窗口太小，没有捕捉到足够远的上文信息。我认为这个结论有些不准确。根据ME法的[原文](http://arxiv.org/abs/1411.4952)，ME模型在预测下一个词的时候是以已经预测出的*所有*词为条件。也就是说，上下文窗口是够大的。ME模型的问题，更准确地说应该是它没有学习"on top of it"背后的"指代关系"。它只学到了"on top of it"在句尾出现的概率很高。

再看表5的后两行，GRNN模型生成的句子存在词语重复的问题。ME模型显式地跟踪哪些词语已经被生成，并且鼓励模型输出未生成词。然而GRNN隐式地用隐含层保存之前生成的词语信息，没有显式的约束。直观地说RNN认为输出一个物体后很可能接着输出"and"，下一个时刻遇到"and"的时候认为应该输出一个物体的词，而并不能区分两个物体其实是一回事。

### 重复的句子与不够多样的数据
作者对各种方法生成句子的重复度进行了分析，并且和人工标注进行了对比：
![Repeated Caption](/img/lm_for_ic/repeated_caption_table.png)

看表格可以看出来，人工标注的句子极少有重复的(只有0.6%)，和训练集的句子重复的也很少。然而在机器生成的句子中，在不同图像之间重复的句子，以及和训练集里一模一样的句子相当多，“原创”的句子比较少。从这个角度看，机器生成的句子似乎并不靠谱。这引出了两个问题：
1. 模型的训练和评价过程是否存在问题？
2. 训练的数据是否欠缺多样性？

## 结论

我个人的见解是，生成的句子趋同，表示训练出的模型可能存在过拟合的问题。过拟合的根本原因，在于训练数据的多样性不够。数据集里的图像内容本身就非常趋同，标注出的句子也就趋同。随着数据收集工作的继续进行，数据的量和多样性上去了，这个问题会缓解。另一个问题在于评测标准。虽然自动化评测的方法也在进步(比如METEOR)，但毕竟还比不上人的判断。既然我们能用各种众包平台(Amazon MTurk之类的)来做数据标注，也就可以用它来做评测。如果能建立起标准化的人工评测方法，结合自动化评测，应该能够更好地对Image Captioning的性能进行评测。