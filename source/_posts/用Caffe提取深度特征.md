title: "用Caffe提取深度特征"
date: 2015-05-28 19:13:08
tags:
categories:
description:
	Caffe提供了extract_feature.bin这个可执行文件用来提取特征。给它网络定义，网络参数，和要提取特征的数据，就能提取出以LMDB存储的特征。再写个简单的脚本读取LMDB数据库，转换成我们需要的格式。
---
最近做对比实验，要比较非深度的方法加上deep feature之后的效果。于是就用Caffe提了一把特征，过程不困难但是有点繁琐，姑且记录下来，留个参考。

#准备工作
用Caffe提取深度特征，需要以下几样东西：
* 能够运行的Caffe环境
* 提取特征的深度网络定义(prototxt)
* 这个网络的参数(caffemodel)
* 需要提取特征的数据

配好环境是必须的，不用多说。网络定义和网络参数，可以用自己学的网络，也可以用公开发表的经典网络，比如AlexNet, VGG之类的。Caffe官网提供了一些经典网络，可以参考[这里]()。被提特征的数据需要处理成能被`DataLayer`或者`ImageDatalayer`读取的格式