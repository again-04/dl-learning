"""
主训练脚本，支持通过命令行参数选择不同模型，并内置各模型的最佳实践配置。
用法示例：
    python main.py --model test10               # 使用 test10 及其推荐配置
    python main.py --model test08 --scale 0.8   # 自定义缩放因子
    python main.py --model test03 --lr 0.01     # 手动覆盖学习率
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm

# 导入配置和数据加载函数
from config.cifar10_config import CIFAR10Config
from utils.load_dataset_utils import load_dataset
from utils.train_utils import train_one_epoch
from utils.test_utils import test
from utils.save_utils import save_training_results

# 模型映射字典（模块路径.类名）
MODEL_MAP = {
    'test03': 'models.test03.AlexNet3',
    'test04': 'models.test04.AlexNet4',
    'test05': 'models.test05.HybridNet',
    'test06': 'models.test06.HybridNetWithNiN',
    'test07': 'models.test07.HybridNetWithNiNResNet',
    'test08': 'models.test08.HybridNet',
    'test09': 'models.test09.LightHybridNet',
    'test10': 'models.test10.SimpleHybridNet',
}

# 各模型推荐配置（仅在用户未通过命令行指定时生效）
# 参数说明：epochs, lr, optimizer, momentum, weight_decay, label_smoothing, scale(若支持)
MODEL_RECOMMEND = {
    'test03': {  # 大容量 AlexNet
        'epochs': 200,
        'lr': 0.1,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'label_smoothing': 0.1,
        'scale': None,  # 不支持 scale
    },
    'test04': {  # 改进版 AlexNet (GAP)
        'epochs': 150,
        'lr': 0.1,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'label_smoothing': 0.1,
        'scale': None,
    },
    'test05': {  # Inception 混合
        'epochs': 150,
        'lr': 0.001,
        'optimizer': 'adamw',
        'momentum': 0.9,          # 对 adamw 无效，但保留
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'scale': None,
    },
    'test06': {  # +NiN
        'epochs': 150,
        'lr': 0.001,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'scale': None,
    },
    'test07': {  # +ResNet
        'epochs': 150,
        'lr': 0.1,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'label_smoothing': 0.1,
        'scale': None,
    },
    'test08': {  # 完整混合 (可缩放)
        'epochs': 150,
        'lr': 0.1,
        'optimizer': 'sgd',
        'momentum': 0.9,
        'weight_decay': 5e-4,
        'label_smoothing': 0.1,
        'scale': 0.75,
    },
    'test09': {  # 轻量混合
        'epochs': 150,
        'lr': 0.001,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'scale': 0.5,
    },
    'test10': {  # 简单混合
        'epochs': 150,
        'lr': 0.001,
        'optimizer': 'adamw',
        'momentum': 0.9,
        'weight_decay': 1e-4,
        'label_smoothing': 0.1,
        'scale': 0.6,
    },
}

def import_model(model_name):
    """动态导入模型类"""
    module_path, class_name = model_name.rsplit('.', 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)

def main():
    parser = argparse.ArgumentParser(description='训练不同模型 on CIFAR-10')
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_MAP.keys()),
                        help='选择模型 (test03~test10)')
    parser.add_argument('--scale', type=float, default=None,
                        help='通道缩放因子（仅对混合模型有效）')
    parser.add_argument('--epochs', type=int, default=None,
                        help='训练轮数（默认使用模型推荐值）')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='批次大小（默认128）')
    parser.add_argument('--lr', type=float, default=None,
                        help='学习率（默认使用模型推荐值）')
    parser.add_argument('--optimizer', type=str, default=None,
                        choices=['sgd', 'adam', 'adamw'],
                        help='优化器（默认使用模型推荐值）')
    parser.add_argument('--momentum', type=float, default=None,
                        help='SGD动量（仅当optimizer=sgd时有效，默认使用模型推荐值）')
    parser.add_argument('--weight_decay', type=float, default=None,
                        help='权重衰减（默认使用模型推荐值）')
    parser.add_argument('--label_smoothing', type=float, default=None,
                        help='标签平滑系数（0表示不使用，默认使用模型推荐值）')
    parser.add_argument('--resume', type=str, default=None,
                        help='从检查点恢复训练（传入模型文件路径）')
    parser.add_argument('--save_dir', type=str, default='../models',
                        help='模型保存目录（默认../models）')
    args = parser.parse_args()

    # 获取模型推荐配置
    rec = MODEL_RECOMMEND[args.model]

    # 使用推荐值填充未指定的参数
    if args.epochs is None:
        args.epochs = rec['epochs']
    if args.lr is None:
        args.lr = rec['lr']
    if args.optimizer is None:
        args.optimizer = rec['optimizer']
    if args.momentum is None:
        args.momentum = rec['momentum']
    if args.weight_decay is None:
        args.weight_decay = rec['weight_decay']
    if args.label_smoothing is None:
        args.label_smoothing = rec['label_smoothing']
    if args.scale is None and rec['scale'] is not None:
        args.scale = rec['scale']   # 仅当模型支持且用户未指定时使用推荐scale

    # 设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载配置（使用模型名作为标识）
    config = CIFAR10Config(args.model)
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.epochs = args.epochs

    # 加载数据
    train_loader, test_loader = load_dataset(config.mean, config.std, config.batch_size)

    # 动态创建模型
    model_class = import_model(MODEL_MAP[args.model])
    # 如果模型接受scale参数，则传入；否则忽略
    if 'scale' in model_class.__init__.__code__.co_varnames and args.scale is not None:
        model = model_class(num_classes=config.num_classes, scale=args.scale).to(device)
    else:
        model = model_class(num_classes=config.num_classes).to(device)
    print(f"模型 {args.model} 创建成功，参数量: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")
    if hasattr(args, 'scale') and args.scale is not None:
        print(f"当前 scale = {args.scale}")

    # 损失函数
    if args.label_smoothing > 0:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss()

    # 优化器
    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # 创建保存目录
    os.makedirs(args.save_dir, exist_ok=True)
    last_model_path = os.path.join(args.save_dir, f'{args.model}_last.pth')
    best_model_path = os.path.join(args.save_dir, f'{args.model}_best.pth')
    test_result_path = os.path.join(args.save_dir, f'{args.model}_test_results.txt')

    # 可选恢复训练
    start_epoch = 1
    if args.resume and os.path.exists(args.resume):
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0) + 1
        print(f"从 {args.resume} 恢复训练，起始轮次 {start_epoch}")

    # 训练循环
    print("开始训练...")
    best_acc = 0.0
    for epoch in range(start_epoch, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch=epoch, epochs=args.epochs
        )
        tqdm.write(f'Epoch {epoch:2d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')

        # 每个epoch后测试
        test_loss, test_acc = test(model, test_loader, criterion, device, desc='Testing')
        tqdm.write(f'Epoch {epoch:2d} | Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
                'test_loss': test_loss,
            }, best_model_path)
            tqdm.write(f'最佳模型已更新，准确率: {best_acc:.4f}')

    # 保存最后一个模型
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'test_acc': test_acc,
        'test_loss': test_loss,
    }, last_model_path)
    print(f"最后一次模型已保存至: {last_model_path}")

    # 最终测试结果
    final_loss, final_acc = test_loss, test_acc
    print(f'\n最终模型测试结果 - Loss: {final_loss:.4f}, Acc: {final_acc:.4f}')

    # 保存训练结果
    save_training_results(
        new_model=model,
        best_acc=best_acc,
        final_acc=final_acc,
        final_loss=final_loss,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=device,
        best_model_path=best_model_path,
        last_model_path=last_model_path,
        test_result_path=test_result_path,
        old_acc=None
    )

if __name__ == '__main__':
    main()