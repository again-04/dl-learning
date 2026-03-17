# 作者: again
# 2026年-03月-14日-12时-47分-24秒
# 3447624182@qq.com

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import GoogLeNet_Weights

# ==================== Inception 模块（核心组件） ====================
class Inception(nn.Module):
    """
    Inception 模块 (v1) —— 四分支并行结构，提取多尺度特征后拼接。

    参数说明:
        in_channels (int): 输入特征图的通道数。
        ch1x1 (int): 分支1（1x1卷积）的输出通道数。
        ch3x3_reduce (int): 分支2中第一个1x1卷积（降维）的输出通道数。
        ch3x3 (int): 分支2中3x3卷积的输出通道数。
        ch5x5_reduce (int): 分支3中第一个1x1卷积（降维）的输出通道数。
        ch5x5 (int): 分支3中5x5卷积的输出通道数。
        pool_proj (int): 分支4中池化后的1x1卷积的输出通道数（降维）。
    """
    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(Inception, self).__init__()
        # 分支1: 1x1卷积 —— 直接进行跨通道信息融合
        # 1x1卷积不会改变特征图的空间尺寸（宽度和高度），但会改变通道数
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),  # 应用1x1卷积，减少或增加通道数
            nn.ReLU(inplace=True),                         # 应用ReLU激活函数
        )

        # 分支2: 1x1降维 + 3x3卷积 —— 先减少通道数再提取3x3特征
        # 通过先应用1x1卷积来降维，然后应用3x3卷积来提取特征，这种设计有助于减少计算量和参数数量
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1),  # 应用1x1卷积进行降维
            nn.ReLU(inplace=True),                               # 应用ReLU激活函数
            nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1),  # 应用3x3卷积提取特征，padding=1保持尺寸不变
            nn.ReLU(inplace=True),                               # 应用ReLU激活函数
        )

        # 分支3: 1x1降维 + 5x5卷积 —— 先减少通道数再提取5x5特征（更大感受野）
        # 通过先应用1x1卷积来降维，然后应用5x5卷积来提取特征，这种设计有助于提取更大感受野的特征
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1),  # 应用1x1卷积进行降维
            nn.ReLU(inplace=True),                               # 应用ReLU激活函数
            nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2),  # 应用5x5卷积提取特征，padding=2保持尺寸不变
            nn.ReLU(inplace=True),                               # 应用ReLU激活函数
        )

        # 分支4: 3x3最大池化 + 1x1降维 —— 提供另一种非线性变换，再通过1x1调整通道数
        # 通过3x3的最大池化来提取特征，然后应用1x1卷积来调整通道数
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),  # 应用3x3的最大池化，stride=1, padding=1保持尺寸不变
            nn.Conv2d(in_channels, pool_proj, kernel_size=1), # 应用1x1卷积进行降维
            nn.ReLU(inplace=True),                             # 应用ReLU激活函数
        )

    def forward(self, x):
        """
        前向传播：四个分支并行计算，然后在通道维度上拼接。
        """
        branch1 = self.branch1(x)  # 计算分支1的输出
        branch2 = self.branch2(x)  # 计算分支2的输出
        branch3 = self.branch3(x)  # 计算分支3的输出
        branch4 = self.branch4(x)  # 计算分支4的输出
        # 在 dim=1 (通道维度) 上拼接四个分支的输出，得到最终的特征图
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
