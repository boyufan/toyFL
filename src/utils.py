import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# weight initialization
def init_weights(model, init_type, init_gain):
    """Function for initializing network weights
    
    Args:
        model: A torch.nn instance to be initialized.
        init_type: Name of an initialization method.
        init_gain: Scaling factor for (normal | xavier | orthogonal).

    """
    def init_func(m):
        # 获取类名
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            else:
                raise NotImplementedError(f'[ERROR] initialization method is not implemented!')
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        
        elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    model.apply(init_func)
    

def init_net(model, init_type, init_gain, gpu_ids):
    """Function for initializing network weights
    
    Args:
        model: A torch.nn.Module to be initialized
        init_type: Name of an initialization method (normal | xavier | kaiming | orthogonal)l
        init_gain: Scaling factor for (normal | xavier | orthogonal).
        gpu_ids: List or int indicating which GPU(s) the network runs on. (e.g., [0, 1, 2], 0)
    
    Returns:
        An initialized torch.nn.Module instance.

    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        model.to(gpu_ids[0])
        model = nn.DataParallel(model, gpu_ids)
    init_weights(model, init_type, init_gain)
    return model

class CustomTensorDataset(Dataset):
    """TensorDataset with support of transforms"""
    def __init__(self, tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform
    
    def __getitem__(self, index):
        x = self.tensors[0][index]
        y = self.tensors[1][index]
        if self.transform:
            x = self.transform(x.numpy().astype(np.uint8))
        return x, y
    
    def __len__(self):
        return self.tensors[0].size(0)