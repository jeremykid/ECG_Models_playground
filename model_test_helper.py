import torch
import torch.nn as nn
import numpy as np

def generate_test_data(batch_size, input_dim, num_classes):
    """
    生成测试数据集
    :param batch_size: 样本数量
    :param input_dim: 输入维度 (C, L)
    :param num_classes: 输出类别数量
    :return: 测试输入和标签
    """
    # 生成随机输入数据
    inputs = torch.randn(batch_size, *input_dim)

    # 生成multi-hot 编码的输出标签
    labels = []
    for _ in range(batch_size):
        num_active_classes = np.random.randint(1, num_classes + 1)  # 随机选择激活的类别数量
        active_classes = np.random.choice(num_classes, num_active_classes, replace=False)
        label = np.zeros(num_classes, dtype=np.float32)
        label[active_classes] = 1.0
        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.float32)

    return inputs, labels