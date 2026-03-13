# 作者: again
# 2026年-03月-13日-21时-15分-39秒
# 3447624182@qq.com

import torch
import torch.nn as nn
import torchvision.models as models

class AlexNet(nn.Module):
    """
    AlexNet 模型实现 (适用于 ImageNet 224x224 输入)
    结构: 5个卷积层 + 3个全连接层
    """
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        # 特征提取部分 (卷积 + 池化)
        self.features = nn.Sequential(
            # 第一层: Conv1 + ReLU + MaxPool + LRN (LRN 可选，现代实现常省略)
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),  # 输入 224x224 -> 55x55
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 55x55 -> 27x27
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),  # LRN (可选)

            # 第二层: Conv2 + ReLU + MaxPool + LRN
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2), # 27x27 -> 27x27
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 27x27 -> 13x13
            # nn.LocalResponseNorm(size=5, alpha=1e-4, beta=0.75, k=2),

            # 第三层: Conv3 + ReLU (无池化)
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),# 13x13 -> 13x13
            nn.ReLU(inplace=True),

            # 第四层: Conv4 + ReLU (无池化)
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),# 13x13 -> 13x13
            nn.ReLU(inplace=True),

            # 第五层: Conv5 + ReLU + MaxPool
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),# 13x13 -> 13x13
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),                  # 13x13 -> 6x6
        )

        # 分类器部分 (全连接层 + Dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

        # 初始化权重 (可选，PyTorch 默认初始化效果也不错)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)        # 展平
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

# ==================== 使用示例 ====================
if __name__ == '__main__':
    # 创建模型
    model = AlexNet(num_classes=1000)

    # 打印模型结构
    print(model)

    # 测试前向传播
    dummy_input = torch.randn(1, 3, 224, 224)  # batch_size=1, 3通道, 224x224
    output = model(dummy_input)
    print(f"输出形状: {output.shape}")  # 应为 torch.Size([1, 1000])

    # 加载预训练权重 (从 torchvision 获取)
    # 如果需要，可以取消下面的注释

    pretrained_model = models.alexnet(weights='DEFAULT')
    model.load_state_dict(pretrained_model.state_dict())
    print("预训练权重加载完成！")