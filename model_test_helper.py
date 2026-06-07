import torch
import torch.nn as nn
import numpy as np

def generate_test_data(batch_size, input_dim, num_classes, num_features=2):
    """
    :param batch_size: 
    :param input_dim: input dim (C, L)
    :param num_classes: 
    :return: test inputs and labels
    """
    # Generate Random Input_ECG
    ecg_inputs = torch.randn(batch_size, *input_dim)

    # multi-hot labels
    labels = []
    for _ in range(batch_size):
        num_active_classes = np.random.randint(1, num_classes + 1)  
        active_classes = np.random.choice(num_classes, num_active_classes, replace=False)
        label = np.zeros(num_classes, dtype=np.float32)
        label[active_classes] = 1.0
        labels.append(label)

    labels = torch.tensor(labels, dtype=torch.float32)

    features = torch.randn(batch_size, num_features)
    return ecg_inputs, features, labels