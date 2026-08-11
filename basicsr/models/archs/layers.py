import torch.nn as nn


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    定义了一个多层感知机（MLP）类，继承自nn.Module。主要功能包括：
        初始化时设置输入、隐藏和输出特征维度，默认隐藏层和输出层维度与输入层相同。
        使用线性变换、激活函数（默认GELU）和Dropout进行前向传播。
    在Transformer架构中，MLP用于处理自注意力机制后的特征。
    它能够根据输入调整隐藏层大小、选择不同的激活函数以及设置dropout率来防止过拟合
    """

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
