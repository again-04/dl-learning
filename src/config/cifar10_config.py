# 作者: again
# 2026年-03月-14日-19时-29分-41秒
# 3447624182@qq.com

"""
CIFAR-10 数据集的训练配置参数。
"""
class CIFAR10Config:
    """CIFAR-10 训练配置类。"""
    def __init__(self, model_name):
        self.model_name = model_name
        # 路径配置
        self.best_model_path = '../models/'+ model_name +'_model.pth'
        self.last_model_path = '../models/'+ model_name +'_model.pth'
        self.test_result_path = '../results/'+ model_name +'_result.txt'
        self.old_model_path = '../models/'+ model_name +'_model.pth'

    # 数据参数
    batch_size = 64
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    # 训练参数
    epochs = 20
    lr = 1e-3
    num_classes = 10