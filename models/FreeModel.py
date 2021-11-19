# coding: UTF-8
import torch.nn.functional as F
from models import BaseModel


class ATModel(BaseModel.ATModel):
    """
    Free (Free Adversarial Training) adversarial training class
    reference:
        title: Adversarial Training for Free!
        link: https://arxiv.org/abs/1904.12843
    """

    def __init__(self, model):
        super(ATModel, self).__init__(model)
        self.epsilon = 0.1
        self.M = 3          # 样本扰动的次数，最终的epoch需要*M，才是真实的epoch的次数
        self.use_sign_grad = False
        self.use_project = True

    def train(self, trains, labels, optimizer):
        """
        Free的计算流程:（Free的思想对于每个样本重新M次训练，并在每步计算delta的时候复用上次的梯度，相对于对于PGD在计算速度上进行了优化）
            对于每个step：
                (1)对于输入的样本（第一步是原始样本，其他为扰动样本），先前向计算loss，反向传播计算梯度
                (2)根据梯度更新参数
                (3)利用(1)中计算的梯度计算扰动delta，生成对抗样本

        Args:
            trains: 2-D int tensor, 训练数据, shape=[batch_size, seq_size]
            labels: 1-D int tensor, 训练标签, shape=[batch_size]
            optimizer:  Adam优化器

        Returns
            outputs : 2-D float tensor, 分类得分, shape=[batch_size, class_num]
            loss : float tensor, 最后的损失, float32
        """
        if self.use_project:                         # 投影的时候需要知道当前的embedding和原始的差值
            self.backup_emb()
        for _ in range(self.M):
            outputs = self.model(trains)             # 前向传播，计算分类得分
            self.model.zero_grad()                   # 清除梯度
            loss = F.cross_entropy(outputs, labels)  # 对抗样本的交叉熵loss
            loss.backward()                          # 反向传播，计算当前对抗样本loss的梯度
            optimizer.step()                         # 每个循环都更新网络参数
            self.attack_emb(False)                   # 根据上一次的梯度，计算扰动delta，生成对抗样本

        if self.use_project:                         # 清理一下embedding的备份
            self.clear_emb_back()

        return outputs, loss

    def calc_delta(self, param, param_name):
        """
        Free扰动delta的计算公式：
            delta = alpha * grad/norm(grad)

        Args:
            param: 参数值
            param_name: 参数名字
        Return:
            delta: 扰动值
        """

        if self.use_sign_grad:
            self.sign_grad(param, param_name, self.epsilon)
        else:
            self.norm_grad(param, param_name, self.epsilon)







