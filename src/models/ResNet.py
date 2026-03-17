# 作者: again
# 2026年-03月-14日-12时-21分-40秒
# 3447624182@qq.com

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet18_Weights


class BasicBlock(nn.Module):
    """
    ResNet 基本残差块，适用于 ResNet18 和 ResNet34。
    结构：两个 3x3 卷积层，每个卷积后跟批归一化（BatchNorm）和 ReLU 激活。
    包含跳跃连接（shortcut），当输入和输出维度不匹配时，通过 downsample 调整。

    属性:
        expansion (int): 输出通道相对于输入通道的扩展因子，对于 BasicBlock 为 1。
        conv1: 第一个 3x3 卷积，可能改变空间尺寸（由 stride 控制）。
        bn1: 第一个批归一化。
        relu: ReLU 激活函数（inplace=True 节省内存）。
        conv2: 第二个 3x3 卷积，保持空间尺寸和通道数。
        bn2: 第二个批归一化。
        downsample: 可选的下采样模块，用于调整跳跃连接的通道数和尺寸，使相加时维度匹配。
    """
    expansion = 1  # 输出通道数相对于输入通道数的倍数（对于 BasicBlock 是1，因为输出通道=输入通道*expansion）

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        初始化残差块。

        参数:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数（即卷积层的输出通道数，最终输出通道为 out_channels * expansion）。
            stride (int): 第一个卷积层的步幅，用于下采样。
            downsample (nn.Module, optional): 当需要改变维度时，对跳跃连接进行下采样的模块。
        """
        super(BasicBlock, self).__init__()
        # 第一个卷积：可能改变空间尺寸（由 stride 控制），不改变通道数（out_channels）
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # 第二个卷积：保持空间尺寸和通道数
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        """
        前向传播：残差连接 = F(x) + x（或经过下采样的 x）。
        """
        identity = x  # 保存原始输入，用于跳跃连接
        if self.downsample is not None:
            identity = self.downsample(x)  # 如果需要，对输入进行下采样以匹配输出尺寸

        # 第一个卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积块
        out = self.conv2(out)
        out = self.bn2(out)

        # 残差相加
        out += identity
        out = self.relu(out)  # 相加后再经过 ReLU
        return out


class ResNet18(nn.Module):
    """
    ResNet18 实现，包含初始卷积层、四个残差层（每个由多个 BasicBlock 堆叠）、全局平均池化和全连接分类器。
    适用于 ImageNet 等 224x224 输入，但可通过调整第一层适应更小尺寸（如 CIFAR）。
    """

    def __init__(self, num_classes=1000):
        """
        初始化 ResNet18 网络。

        参数:
            num_classes (int): 分类任务的类别数，默认为 1000（ImageNet）。
        """
        super(ResNet18, self).__init__()
        # 初始卷积层：7x7 卷积，步幅 2，填充 3，输出 64 通道，后跟批归一化和 ReLU
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # 最大池化，进一步下采样

        # 四个残差层，每个由多个 BasicBlock 组成
        # layer1: 输入 64，输出 64，步幅 1，2 个块
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        # layer2: 输入 64，输出 128，步幅 2（下采样），2 个块
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        # layer3: 输入 128，输出 256，步幅 2，2 个块
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        # layer4: 输入 256，输出 512，步幅 2，2 个块
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)

        # 全局平均池化，将每个特征图降为 1x1
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接分类器，输入维度 = 512 * expansion
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

        self._initialize_weights()  # 自定义权重初始化

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        """
        构建一个残差层，包含多个 BasicBlock。

        参数:
            in_channels (int): 该层第一个块的输入通道数。
            out_channels (int): 该层每个块的输出通道数（最终输出为 out_channels * expansion）。
            blocks (int): 该层包含的 BasicBlock 数量。
            stride (int): 该层第一个块的步幅，用于下采样；后续块步幅为 1。
        返回:
            nn.Sequential: 包含多个 BasicBlock 的序列。
        """
        downsample = None
        # 如果需要下采样或通道数不匹配，则创建下采样模块（1x1 卷积 + BN）
        if stride != 1 or in_channels != out_channels * BasicBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion),
            )

        layers = []
        # 第一个块可能包含下采样
        layers.append(BasicBlock(in_channels, out_channels, stride, downsample))
        # 后续块输入通道为 out_channels * expansion，步幅为 1
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels * BasicBlock.expansion, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        """
        前向传播流程：
        1. 初始卷积 + BN + ReLU + 最大池化
        2. 四个残差层
        3. 全局平均池化 + 展平
        4. 全连接分类
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 将 (batch, C, 1, 1) 展平为 (batch, C)
        x = self.fc(x)
        return x

    def _initialize_weights(self):
        """
        自定义权重初始化：
        - 卷积层：使用 Kaiming 正态初始化（适用于 ReLU）
        - 批归一化层：权重初始化为 1，偏置初始化为 0
        - 全连接层：权重使用正态分布 N(0,0.01)，偏置初始化为 0
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
    # 测试模型前向传播
    model = ResNet18(num_classes=1000)  # 创建 ResNet18 实例，输出 1000 类
    dummy = torch.randn(1, 3, 224, 224)  # 模拟一个 batch 的输入（ImageNet 标准尺寸）
    out = model(dummy)  # 前向传播
    print(f"ResNet18 output shape: {out.shape}")  # 应输出 torch.Size([1, 1000])

    # 加载 PyTorch 官方预训练权重（ImageNet 上训练）
    # 注意：官方预训练模型的 state_dict 键名与自定义模型完全一致（因为结构相同），可直接加载
    pretrained = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.load_state_dict(pretrained.state_dict())
    print("ResNet18 pretrained weights loaded.")