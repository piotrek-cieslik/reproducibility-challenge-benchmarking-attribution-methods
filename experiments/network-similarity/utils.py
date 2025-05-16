import torch

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from PIL import Image
import random

class PatchDeletion:
    def __init__(self, patch_size=56):
        self.patch_size = patch_size

    def __call__(self, img):
        if not isinstance(img, Image.Image):
            raise TypeError("Expected PIL Image")

        w, h = img.size
        x = random.randint(0, w - self.patch_size)
        y = random.randint(0, h - self.patch_size)

        patch = Image.new("RGB", (self.patch_size, self.patch_size), (0, 0, 0))

        img.paste(patch, (x, y))
        return img

def get_loader(test_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    dataset = datasets.ImageFolder(test_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

def get_corrupt_loader(test_dir, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        PatchDeletion(patch_size=56),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    dataset = datasets.ImageFolder(test_dir, transform=transform)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)


def load_state_dict(path, model):
    checkpoint = torch.load(path)
    state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    if "module." in list(state_dict.keys())[0]:
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    if "model." in list(state_dict.keys())[0]:
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    return model