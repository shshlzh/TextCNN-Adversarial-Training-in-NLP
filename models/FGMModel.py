# coding: UTF-8
from models import FGSMModel


class ATModel(FGSMModel.ATModel):
    """
    FGM (Fast Gradient Method) adversarial training class
    reference:
        title: Adversarial Training Methods for Semi-Supervised Text Classification
        link: https://arxiv.org/abs/1605.07725
    """

    def __init__(self, model):
        super(ATModel, self).__init__(model)
        self.epsilon = 0.1
        self.use_sign_grad = False
        self.use_project = False

    def calc_delta(self, param, param_name):
        """
        FGM扰动delta的计算公式：
            delta = epsilon * grad / norm(grad)
        Args:
            param: 参数值
            param_name: 参数名字
        Return:
            delta: 扰动值
        """
        self.norm_grad(param, param_name, self.epsilon)






