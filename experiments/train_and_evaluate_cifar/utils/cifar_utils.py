# code based on https://github.com/visinf/idsds/blob/main/utils/utils.py

import os
import random
from PIL import Image
import json
import numpy as np
from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Union
import argparse

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from torch import nn
from torch.nn import functional as F
import numbers
import math

def str2bool(v):
    #if isinstance(v, bool):
    #    return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

class IndexedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target, index

    def __len__(self):
        return len(self.dataset)

def get_cifar100_loaders(args, shuffle_val=False, train_with_eval_transform=False):
    # Standard CIFAR-100 normalization
    normalize = transforms.Normalize(
        mean=[0.5071, 0.4867, 0.4408],
        std=[0.2675, 0.2565, 0.2761]
    )

    unnormalize = transforms.Compose([
        transforms.Normalize(mean=[0., 0., 0.],
                             std=[1/0.2675, 1/0.2565, 1/0.2761]),
        transforms.Normalize(mean=[-0.5071, -0.4867, -0.4408],
                             std=[1., 1., 1.])
    ])

    if not train_with_eval_transform:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_dataset = IndexedDataset(datasets.CIFAR100(
        root=args.data_dir,
        train=True,
        download=True,
        transform=train_transform
    ))

    val_dataset = IndexedDataset(datasets.CIFAR100(
        root=args.data_dir,
        train=False,
        download=True,
        transform=val_transform
    ))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=shuffle_val,
        num_workers=args.workers, pin_memory=True)

    return train_loader, val_loader

class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel_size, sigma, dim=2, device=None):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            self.kernel_size = kernel_size
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        self.device = device


        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / (2 * std)) ** 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.weight = self.weight.to(self.device)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )
        

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, groups=self.groups, padding=self.kernel_size//2)
