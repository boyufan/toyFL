import os
import logging
from random import shuffle

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

from torch.utils.data import Dataset, TensorDataset, ConcatDataset
from torchvision import datasets, transforms
import torchvision

logger = logging.getLogger(__name__)

def launch_tensor_board(log_path, port, host):
    os.system(f'tensorboard --logdir={log_path} --port={port} --host={host}')
    return True

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

def create_datasets(data_path, dataset_name, num_clients, num_shards, iid):
    """split the dataset in iid or non-iid fashion"""
    if hasattr(torchvision.datasets, dataset_name):
        if dataset_name in ['CIFAR10']:
            transform = torchvision.transforms.Compose(
                [transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
            )
        elif dataset_name in ['MNIST']:
            transform = transforms.ToTensor()
        
        training_dataset = datasets.__dict__[dataset_name](
            root=data_path,
            train=True,
            download=True,
            transform=transform
        )

        test_dataset = datasets.__dict__[dataset_name](
            root=data_path,
            train=False,
            download=True,
            transform=transform
        )
    else:
        error_message = f'dataset {dataset_name} is not supported'
        raise AttributeError(error_message)
    
    if training_dataset.data.ndim == 3:
        training_dataset.data.unsqueeze_(3)
    num_categories = np.unique(training_dataset.targets).shape[0]

    if 'ndarray' not in str(type(training_dataset.data)):
        training_dataset.data = np.asarray(training_dataset)
    if 'list' not in str(type(training_dataset.targets)):
        training_dataset.targets = training_dataset.targets.tolist()
    
    if iid:
        # return a random permutation
        shuffle_indices = torch.randperm(len(training_dataset))
        training_inputs = training_dataset.data[shuffle_indices]
        training_labels = torch.Tensor(training_dataset.targets)[shuffle_indices]

        split_size = len(training_dataset) // num_clients
        split_datasets = list(
            zip(
                torch.split(torch.Tensor(training_inputs), split_size),
                torch.split(torch.Tensor(training_labels), split_size)
            )
        )

        local_datasets = [
            CustomTensorDataset(local_dataset, transform=transform)
            for local_dataset in split_datasets
        ]
    else:
        sorted_indices = torch.argsort(torch.Tensor(training_dataset.targets))
        training_inputs = training_dataset[sorted_indices]
        training_labels = torch.Tensor(training_dataset.targets)[sorted_indices]

        # partition data into shards
        shard_size = len(training_dataset) // num_shards
        shard_inputs = list(torch.split(torch.Tensor(training_inputs), shard_size))
        shard_labels = list(torch.split(torch.Tensor(training_labels), shard_size))

        shard_inputs_sorted, shard_labels_sorted = [], []
        # num_shards // num_categories指每一类有多少shards
        # 下面这个循环保证了每个shard取到的都是相邻的两个类
        for i in range(num_shards // num_categories):
            for j in range(0, ((num_shards // num_categories) * num_categories), (num_shards // num_categories)):
                shard_inputs_sorted.append(shard_inputs[i + j])
                shard_labels_sorted.append(shard_labels[i + j])
        
        # how many shards per client
        shards_per_clients = num_shards // num_clients
        local_datasets = [
            CustomTensorDataset(
                (
                    torch.cat(shard_inputs_sorted[i, i + shards_per_clients]),
                    torch.cat(shard_labels_sorted[i:i + shards_per_clients]).long()
                ),
                transform=transform
            )
            for i in range(0, len(shard_inputs_sorted), shards_per_clients)
        ]
    return local_datasets, test_dataset
