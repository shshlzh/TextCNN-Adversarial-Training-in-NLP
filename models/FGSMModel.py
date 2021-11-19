# coding: UTF-8
import torch.nn.functional as F
from models import BaseModel


class ATModel(BaseModel.ATModel):
    """
    FGSM (Fast Gradient Sign Method) adversarial training class
    reference:
        title: Explaining and Harnessing Adversarial Examples
        link: https://arxiv.org/abs/1412.6572
    """

    def __init__(self, model):
        super(ATModel, self).__init__(model)
        self.epsilon = 0.1
        self.use_sign_grad = True
        self.use_clamp = False

    def train(self, trains, labels, optimizer):
        """
        FGSM/FGM训练流程:(FGSM和FGM流程一致，不同之处在于计算扰动delta)
            (1)对于输入的样本，先前向计算loss，反向传播得到梯度
            (2)根据embedding的梯度计算出扰动delta，并累加到当前的embedding上(embedding + delta)作为对抗样本
            (3)计算对抗样本的loss，反向传播得到对抗样本的梯度，累加到(1)的梯度上
            (4)将embedding恢复为(1)中的原始的值
            (5)根据(3)的梯度对参数进行更新

        Args:
            trains: 2-D int tensor, 训练数据, shape=[batch_size, seq_size]
            labels: 1-D int tensor, 训练标签, shape=[batch_size]
            optimizer:  Adam优化器

        Returns
            outputs : 2-D float tensor, 分类得分, shape=[batch_size, class_num]
            loss : float tensor, 最后的损失, float32
        """

        # 先计算原始样本下的梯度
        outputs = self.model(trains)             # 前向传播，计算分类得分
        self.model.zero_grad()                   # 清除梯度
        loss = F.cross_entropy(outputs, labels)  # 交叉熵loss
        loss.backward()                          # 反向传播，计算当前梯度

        # 生成对抗样本，进行对抗训练
        self.attack_emb(True)                   # 根据梯度，对于embedding增加扰动，生成对抗样本
        outputs = self.model(trains)            # 重新前向计算一次得分，并计算对抗样本的损失
        loss = F.cross_entropy(outputs, labels)
        loss.backward()                         # 重新计算梯度，并累加到原始样本的梯度上
        self.restore_emb()                      # 恢复原始的embedding
        optimizer.step()                        # 根据累加后的梯度更新网络参数

        return outputs, loss

    def calc_delta(self, param, param_name):
        """
        FGSM扰动delta的计算公式：
            delta = epsilon * sign(grad)

        Args:
            param: 参数值
            param_name: 参数名字
        Return:
            delta: 扰动值
        """
        self.sign_grad(param, param_name, self.epsilon)





