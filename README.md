# 对抗训练在NLP的应用

本实验以文本分类中常用的TextCNN<sup>[1](#TextCNN)</sup>模型为基础，实验了对抗训练在NLP中的效果。  
TextCNN的代码来源于：[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch) ，
TextCNN的具体模型结构参考作者的博客：[中文文本分类pytorch实现](https://zhuanlan.zhihu.com/p/73176084) 。
对抗训练的代码参考知乎：[NLP中的对抗训练 + PyTorch实现](https://zhuanlan.zhihu.com/p/91269728) ，同时参考[fast_adversarial](https://github.com/locuslab/fast_adversarial) <sup>[2](#FAST_ADV)</sup> 的实现思路。 
本实验简单的实现了FGSM<sup>[3](#FGSM)</sup>、FGM<sup>[4](#FGM)</sup>、PGD<sup>[5](#PGD)</sup>、Free<sup>[6](#PGD)</sup>，可以通过参数控制。

**本实验仅供个人学习理解对抗训练的基本原理，如果侵权还请联系删除。**

## 对抗训练简介

对抗训练是一种引入噪声的训练方式，可以对参数进行正则化，提升模型的鲁棒性和泛化能力。更详细的介绍可以参考[实验报告](documents/对抗训练的NLP中的应用实验报告.pdf) 。

## 数据集
数据集是[TextCNN作者](https://github.com/649453932/Chinese-Text-Classification-Pytorch) 从[THUCNews](http://thuctc.thunlp.org/) 中抽取了20万条新闻标题，已上传至github，文本长度在20到30之间。一共10个类别，每类2万条。

类别：财经、房产、股票、教育、科技、社会、时政、体育、游戏、娱乐。

数据集划分：

|数据集|数据量|
|:--------:|:--------:|
|训练集|18万|
|验证集|1万|
|测试集|1万|

## 环境
python==3.7.3  
pytorch==1.10.0  
tqdm==4.62.3  
scikit-learn==0.22.1  
tensorboardX==1.0

**执行过程中可能遇到的问题：** 

- 出现got an unexpected keyword argument 'log_dir'报错，这个是因为TextCNN中用到了tensorboard，解决方式是降低tensorflow的版本(亲测1.0.0可以用)
- 出现'function' object has no attribute 'Variable'报错，也和版本有关，需要修改报错文件x2num.py的一行源码，具体参考[链接](https://github.com/lanpa/tensorboardX/commit/c5189bdb019085841dbfeeb457b1f6682c7dbfbf) 



## 效果

|对抗训练方法|acc|micro-precison|micro-recall|micro-f1|训练时间|epoch(20)|Test loss|实验配置|
| :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: | :--------: |
|Base TextCNN|89.91%|0.8993|0.8991|0.8991|23.5分钟|4|0.34|early stop|
|FGSM|91.37%|0.9138|0.9137|<font color="#dd0000">0.9136</font>|107分钟|8|0.29|epsilon=0.1,early stop|
|FGM|89.96%|0.9004|0.8996|0.8997|32分钟|4|0.33|epsilon=0.1,early stop|
|PGD|90.12%|0.9015|0.9012|0.9009|100分钟|4|0.33|epsilon=0.1,K=3,alpha=0.1,early stop|
|Free|89.64%|0.8967|0.8964|0.8964|95分钟|3|0.38|epsilon=0.1,M=3,early stop|

**说明：**     
- 为了比较效果，word embedding不使用预训练的词向量，都使用随机初始化 
- 其他参数常见[models/TextCNN.py](models/TextCNN.py)
- 更详细的指标数据参见训练日志[documents](documents)



## 使用说明
```
# 代码说明：
# 1.训练的入口程序是run.py
# 2.对抗训练的父类是BaseModel，没有实现任何的对抗训练的方法，也就是Baseline TextCNN
# 3.公用的方法均放在了父类BaseModel中，子类实现了各自计算扰动的方法和更新的流程
# 4.各对抗训练的流程对应的Model文件中有完整的注释


# 训练并测试：
# Base TextCnn，at_type不配置的化，默认是Base
python run.py --at_type=Base

# FGSM
python run.py --at_type=FGSM

# FGM
python run.py --at_type=FGM

# PGD
python run.py --at_type=PGD

# PGD
python run.py --at_type=Free

```

### 参数
模型都在[models](models)目录下，超参定义和模型定义在同一文件中。  


## 参考论文
[1] <span id="TextCNN"> [TextCNN:Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)  </span>  
[2] <span id="FAST_ADV">[Fast is better than free: Revisiting adversarial training](https://arxiv.org/abs/2001.03994) </span>  
[3] <span id="FGSM">[FGSM:Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) </span>  
[4] <span id="FGM">[FGM:Adversarial Training Methods for Semi-Supervised Text Classification](https://arxiv.org/abs/1605.07725) </span>  
[5] <span id="PGD">[PGD:Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) </span>   
[6] <span id="Free">[Free:Adversarial Training for Free!](https://arxiv.org/abs/1904.12843) </span>  
