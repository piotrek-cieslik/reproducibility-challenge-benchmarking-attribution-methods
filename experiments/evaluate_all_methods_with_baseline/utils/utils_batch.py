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

class MyImageFolder(ImageFolder):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, index

def get_imagenet_loaders(data_dir, model_string, batch_size, workers, shuffle_val=False, train_with_eval_transform=False):
    
    #classes_to_keep = [11,14,440,740,852,200,201,202,203,234] # 5 random ones and 5 dogs
    """
    if args.number_classes != 1000:
        random.seed(args.seed)
        classes_to_keep = random.sample(range(0, 1000), args.number_classes)
    else:
        classes_to_keep = list(range(1000))
    """

    traindir = os.path.join(data_dir, 'train')
    valdir = os.path.join(data_dir, 'val')

    if model_string == 'vit_base_patch16_224':
        normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                    std=[0.5, 0.5, 0.5])
    elif model_string == 'bcos_resnet50':
        normalize = transforms.Normalize(mean=[0., 0., 0.],
                                    std=[1., 1., 1.])
    else:
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                        std=[0.229, 0.224, 0.225])
    
    unnormalize = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                        std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                        transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                            std = [ 1., 1., 1. ]),
                                    ])
    
    #unnormalize = RemoveInverse()

    if not train_with_eval_transform:
        train_transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ])
    else:
        train_transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])
    
    val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    
    # train
    train_dataset = MyImageFolder(
        traindir,
        train_transform)

    # val
    val_dataset = MyImageFolder(
        valdir,
        val_transform)

    """
    indices_train = [i for i, label in enumerate(train_dataset.targets) if label in classes_to_keep]
    indices_val = [i for i, label in enumerate(val_dataset.targets) if label in classes_to_keep]

    train_dataset = torch.utils.data.Subset(train_dataset, indices_train)
    val_dataset = torch.utils.data.Subset(val_dataset, indices_val)

    # maps 0,1,2,... to the corresponding class labels in [0,999]
    remap = {x:i for i, x in enumerate(classes_to_keep)}
    class_remap = torch.nn.Embedding(1000, 1)
    class_remap.weight.requires_grad = False
    for key in remap:
        class_remap.weight[key] = remap[key]
    class_remap = class_remap.cuda()
    nr_classes = len(classes_to_keep)
    """
    train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            num_workers=workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=shuffle_val,
            num_workers=workers, pin_memory=True)
    
    return train_loader, val_loader
