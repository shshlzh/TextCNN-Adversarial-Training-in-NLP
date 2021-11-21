# coding: UTF-8
import torch.nn.functional as F
from models import BaseModel


class ATModel(BaseModel.ATModel):
    """
     PGD (Projected Gradient Descent) adversarial training class
     reference:
        title: Towards Deep Learning Models Resistant to Adversarial Attacks
        link: https://arxiv.org/abs/1706.06083
    """

    def __init__(self, model):
        super(ATModel, self).__init__(model)
        self.epsilon = 0.1
        self.K = 3          # delta迭代的次数
        self.alpha = 0.1    # delta每次迭代的scale
        self.use_sign_grad = False
        self.use_project = True

    def train(self, trains, labels, optimizer):
        """
        PGD的计算流程:（和FGSM的本质区别是小步（alpha控制）的迭代，慢慢的找出最优的扰动值，只是用了最后一次扰动生成的对抗样本进行训练）
            (1)对于输入的样本，先前向计算loss，反向传播计算梯度，并备份原始样本的梯度
            (2)对于每个step：
                (2.1)根据embedding的梯度计算出扰动，并累加到当前的embedding上（超出范围投影到epsilon球面范围内）作为对抗样本
                (2.2)如果step不是最后第一步，需要梯度置0，每次重新计算梯度
                (2.3)如果step是最后第一步，需要恢复(1)备份的梯度，并将最后的对抗样本的梯度累加到(1)中原始样本的梯度上
            (3)将embedding恢复为(1)中的原始的值
            (4)根据(2.3)的梯度对参数进行更新

        Args:
            trains: 2-D int tensor, 训练数据, shape=[batch_size, seq_size]
            labels: 1-D int tensor, 训练标签, shape=[batch_size]
            optimizer:  Adam优化器

        Returns
            outputs : 2-D float tensor, 分类得分, shape=[batch_size, class_num]
            loss : float tensor, 最后的损失, float32
        """

        # 先计算原始样本下的梯度
        outputs = self.model(trains)                # 前向传播，计算分类得分
        self.model.zero_grad()                      # 清除梯度
        loss = F.cross_entropy(outputs, labels)     # 交叉熵loss
        loss.backward()                             # 反向传播，计算当前梯度

        # 生成对抗样本，进行对抗训练
        self.backup_grad()
        for step in range(self.K):
            is_first_attack = (step == 0)           # 第一步需要先备份原始的embedding
            self.attack_emb(is_first_attack)        # 根据梯度，对于embedding增加扰动，生成对抗样本
            if step != self.K - 1:
                self.model.zero_grad()              # 不是最后一步的时候，需要每次梯度清零
            else:
                self.restore_grad()                 # 最后一步恢复原始的梯度
            outputs = self.model(trains)            # 重新前向计算一次得分，并计算对抗样本的损失
            loss = F.cross_entropy(outputs, labels)
            loss.backward()                         # 重新计算梯度，并累加到原始样本的梯度上
        self.restore_emb()                          # 恢复原始的embedding
        optimizer.step()                            # 根据累加后的梯度更新网络参数

        return outputs, loss

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






