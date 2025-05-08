import torch
from torchvision import models
from torchvision.models import ResNet50_Weights
from utils.load_data import load_data
from utils.augment_data import augment_data
from utils.fine_tune import fine_tune

## This file pretrains the ResNet50 model with patches

device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(f"Device used: {device}")

data_dir = 'datasets/imagenet-mini' # choose the path that points to your data
rows_and_cols = 4
sd_baseline = 'zeros'   # available options: zeros, blur, random
batch_size = 64

model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
image_datasets, dataloaders = load_data(data_dir)
fine_tune(model, image_datasets['train'])
augment_data(dataloaders['train'], rows_and_cols, sd_baseline, batch_size, device)
train()

#num_classes = len(image_datasets['train'].classes)
#print(num_classes)



# hey pre-trained a model ResNet50,
# ●​ Fine-tuning: they must have written a short training loop that, on each forward pass,
# applies the “zero-baseline” (black-patch) corruption to half of the samples (using the
# same code from single_deletion.py), computes cross-entropy loss, back-propagates, and
# does an optimizer step.
# ●​ Next they save out that checkpoint (e.g. finetuned_resnet50.pth).
# ●​ And they executed IDSDS pointing to the checkpoint