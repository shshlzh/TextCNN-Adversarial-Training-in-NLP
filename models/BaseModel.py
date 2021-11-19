# coding: UTF-8
import torch
import numpy as np
import torch.nn.functional as F


class ATModel:
    """
    base train class, without adversarial training
    """

    def __init__(self, model, emb_name="embedding"):
        self.model = model
        self.epsilon = 0.1            # 扰动的scale
        self.emb_backup = {}          # 备份原始embedding用
        self.grad_backup = {}         # 备份原始梯度用
        self.emb_name = emb_name      # 模型中定义的embedding参数的名字
        self.use_sign_grad = True     # 采用哪种方式计算delta
        self.use_project = True       # 是否采用投影
        self.use_clamp = False         # 限制delta的范围

    def train(self, trains, labels, optimizer):
        """
        对抗训练过程，如果没有对抗网络，就是普通的训练流程

        Args:
            trains: 2-D int tensor, 训练数据, shape=[batch_size, seq_size]
            labels: 1-D int tensor, 训练标签, shape=[batch_size]
            optimizer:  Adam优化器

        Returns
            outputs : 2-D float tensor, 分类得分, shape=[batch_size, class_num]
            loss : float tensor, 最后的损失, float32
        """

        outputs = self.model(trains)
        self.model.zero_grad()
        loss = F.cross_entropy(outputs, labels)
        loss.backward()
        optimizer.step()

        return outputs, loss

    def sign_grad(self, param, param_name, scale):
        """
        计算sign grad
        Args:
            param: 参数值
            param_name: 参数名字
            scale: 缩放参数
        Return:
            sign_grad: sign_grad

        """
        d_at = scale * np.sign(param.grad)
        if self.use_clamp:
            d_at.clamp_(-self.epsilon, self.epsilon)
        param.data.add_(d_at)

    def norm_grad(self, param, param_name, scale):
        """
        计算norm grad
        Args:
            param: 参数值
            param_name: 参数名字
            scale: 缩放参数
        Return:
            norm_grad: norm_grad
        """
        norm = torch.norm(param.grad)
        if norm != 0 and not torch.isnan(norm):
            d_at = scale * param.grad / norm
        else:
            d_at = scale * torch.zeros_like(param.grad)
        param.data.add_(d_at)
        if self.use_project:
            param.data = self.project(param_name, param.data)

    def calc_delta(self, param, param_name, scale):
        """
        计算扰动值

        Args:
            param: 参数值
            param_name: 参数名字
            scale: 缩放参数
        Return:
            delta: 扰动值
        """
        raise NotImplementedError

    def project(self, param_name, param_data):
        """
        如果delta落在扰动半径为epsilon的球面外，就映射回球面上，以保证扰动不要过大
        即：||delta|| <= epsilon

        Args:
            param_name: 参数名字
            param_data: 参数值
        Return:
            delta: 扰动值
        """
        d_at = param_data - self.emb_backup[param_name]
        if torch.norm(d_at) > self.epsilon:
            d_at = self.epsilon * d_at / torch.norm(d_at)
        return self.emb_backup[param_name] + d_at

    def attack_emb(self, is_backup=True):
        """
        计算在embedding上的扰动值

        Args:
            is_backup: 是否备份，默认True
        Return:

        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                if is_backup:
                    self.emb_backup[name] = param.data.clone()
                self.calc_delta(param, name)

    def backup_emb(self):
        """
        备份embedding
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                self.emb_backup[name] = param.data.clone()

    def clear_emb_back(self):
        """
        清除
        """
        self.emb_backup = {}

    def restore_emb(self):
        """
        恢复embedding
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad and self.emb_name in name:
                assert name in self.emb_backup
                param.data = self.emb_backup[name]

        self.emb_backup = {}

    def backup_grad(self):
        """
        备份梯度
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.grad_backup[name] = param.grad.clone()

    def restore_grad(self):
        """
        恢复梯度值
        """
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.grad = self.grad_backup[name]

        self.grad_backup = {}



