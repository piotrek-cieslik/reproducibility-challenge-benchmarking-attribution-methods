from torchvision import datasets, transforms
import os
import torch

# code taken from https://github.com/patrickloeber/pytorchTutorial/blob/master/15_transfer_learning.py
def load_data(dataset_dir):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ToTensor()
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(dataset_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}

    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                                  shuffle=True, num_workers=0)
                   for x in ['train', 'val']}

    return image_datasets, dataloaders