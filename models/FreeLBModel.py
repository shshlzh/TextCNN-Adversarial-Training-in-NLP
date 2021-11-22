# coding: UTF-8
import torch.nn.functional as F
from models import BaseModel


class ATModel(BaseModel.ATModel):
    """
    FreeLB (Free Large-Batch) adversarial training class
    reference:
        title: FreeLB: Enhanced Adversarial Training for Natural Language Understanding
        link: https://arxiv.org/abs/1909.11764
    """

    def __init__(self, model):
        super(ATModel, self).__init__(model)
        self.epsilon = 0.1
        self.alpha = 0.1
        self.M = 3          # 样本扰动的次数，最终的epoch需要*M，才是真实的epoch的次数
        self.use_sign_grad = False
        self.use_project = True

    def train(self, trains, labels, optimizer):
        """
        FreeLB的计算流程:
            相对于PGD算法，FreeLB算法保留了每步对抗训练的梯度,并且均参与了网络参数的更新。
            而每一步对抗训练的梯度是由当前步对抗训练样本计算而来，对从模型更新的角度来说，相当于将训练样本扩大了k倍。

        Args:
            trains: 2-D int tensor, 训练数据, shape=[batch_size, seq_size]
            labels: 1-D int tensor, 训练标签, shape=[batch_size]
            optimizer:  Adam优化器

        Returns
            outputs : 2-D float tensor, 分类得分, shape=[batch_size, class_num]
            loss : float tensor, 最后的损失, float32
        """
        self.model.zero_grad()                       # 先清除掉梯度
        self.backup_emb()                            # 备份embedding
        for _ in range(self.M):
            outputs = self.model(trains)             # 前向传播，计算分类得分
            self.model.zero_grad()                   # 清除梯度
            loss = F.cross_entropy(outputs, labels)  # 对抗样本的交叉熵loss
            loss.backward()                          # 反向传播，计算当前对抗样本loss的梯度，梯度会自动的累加

            self.attack_emb(False)                   # 根据上一次的梯度，计算扰动delta，生成对抗样本

        self.restore_emb()                          # 恢复原始的embedding
        self.avg_grad()                             # 计算平均梯度，更新
        optimizer.step()                            # 根据累加后的梯度更新网络参数

        return outputs, loss

    def avg_grad(self):
        """计算梯度的平均"""

        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = param.grad / self.M

    def calc_delta(self, param, param_name):
        """
        PGD扰动delta的计算公式：
            delta_{step+1} = alpha * grad_{step}/norm(grad_{step})

        Args:
            param: 参数值
            param_name: 参数名字
        Return:
            delta: 扰动值
        """

        if self.use_sign_grad:
            self.sign_grad(param, param_name, self.alpha)
        else:
            self.norm_grad(param, param_name, self.alpha)







