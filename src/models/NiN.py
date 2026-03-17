# 作者: again
# 2026年-03月-14日-12时-47分-44秒
# 3447624182@qq.com

import torch
import torch.nn as nn

class NiN(nn.Module):
    """
    Network In Network (NiN) 实现
    结构：三个 NiN 块 + 全局平均池化
    """
    def __init__(self, num_classes=10):  # 常用在 CIFAR-10/100，这里设为10便于演示
        super(NiN, self).__init__()
        self.features = nn.Sequential(
            # NiN 块1
            nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, kernel_size=1),   # mlpconv 第二层
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, kernel_size=1),    # mlpconv 第三层
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # NiN 块2
            nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # NiN 块3
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, num_classes, kernel_size=1),  # 输出通道数等于类别数
            nn.ReLU(inplace=True),
        )

        # 全局平均池化
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # 注意：NiN 不使用全连接层，分类由最后的卷积层 + GAP 完成

        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)   # [batch, num_classes]
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

if __name__ == '__main__':
    model = NiN(num_classes=10)
    # NiN 对输入尺寸较灵活，这里使用 32x32（CIFAR 尺寸）
    dummy = torch.randn(1, 3, 32, 32)
    out = model(dummy)
    print(f"NiN output shape: {out.shape}")

    # 注意：torchvision 没有提供 NiN 的预训练权重，需自己训练
    print("No pretrained weights for NiN in torchvision.")