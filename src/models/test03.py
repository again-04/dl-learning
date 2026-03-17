# 作者: again
# 2026年-03月-15日-09时-20分-27秒
# 3447624182@qq.com
# 作者: again
# 2026年-03月-14日-22时-51分-49秒
# 3447624182@qq.com

"""
AlexNet 模型实现（适配 CIFAR-10 32x32 输入）
"""
import torch.nn as nn
import torch

class AlexNet3(nn.Module):
    """
    AlexNet 卷积神经网络。

    结构包括5个卷积层（部分带批归一化和池化）和3个全连接层。
    为适应 CIFAR-10 的 32x32 输入，对原始 AlexNet 的卷积核步长和填充进行了调整。

    Args:
        num_classes (int): 分类类别数，默认为 10（CIFAR-10）。

    Attributes:
        features (nn.Sequential): 特征提取部分（卷积层）。
        classifier (nn.Sequential): 分类器部分（全连接层）。

    Examples:
        >>> model = AlexNet(num_classes=10)
        >>> x = torch.randn(1, 3, 32, 32)
        >>> output = model(x)  # shape: (1, 10)
    """
    def __init__(self, num_classes=10):
        super(AlexNet3, self).__init__()
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),                 # 16x16

            # Conv2
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # 16x16
            nn.BatchNorm2d(128), # BN
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=1),                  # 15x15

            # Conv3
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),# 15x15
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),# 15x15
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),# 15x15
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 7x7
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        """
        前向传播。

        Args:
            x (torch.Tensor): 输入张量，形状 (batch_size, 3, 32, 32)。

        Returns:
            torch.Tensor: 分类 logits，形状 (batch_size, num_classes)。
        """
        x = self.features(x)
        x = torch.flatten(x, 1)   # 展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        """
        自定义权重初始化。
        卷积层使用 Kaiming 初始化，全连接层使用正态分布初始化，偏置置零。
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)