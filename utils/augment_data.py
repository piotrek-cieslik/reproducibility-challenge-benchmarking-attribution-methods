from tqdm import tqdm
from utils.GaussianSmoothing import GaussianSmoothing
import torch
import random

# this code is mostly based on their single_deletion.py

def add_patches(images, baselines, rows_and_cols=4):

    B,C,H,W = images.shape
    cell_H = int(H/rows_and_cols)
    cell_W = int(W/rows_and_cols)
    assert cell_H == cell_W

    indices_to_modify = random.sample(range(B), k=B // 2)

    for i in indices_to_modify:
        for cell_row in range(rows_and_cols):
            for cell_col in range(rows_and_cols):
                images[i, :,
                cell_row*cell_H:(cell_row+1)*cell_H,
                cell_col*cell_W:(cell_col+1)*cell_W
                ] = baselines[i, :,
                    cell_row*cell_H:(cell_row+1)*cell_H,
                    cell_col*cell_W:(cell_col+1)*cell_W]
    return

def augment_data(dataloader, rows_and_cols, sd_baseline, batch_size, device):
    nr_images = len(dataloader.dataset)

    blur = GaussianSmoothing(3, 51, 41, device=device)

    for batch_idx, (images, targets) in enumerate(tqdm(dataloader, total=int(nr_images/batch_size) - 1)):

        images = images.to(device, non_blocking=True)

        if sd_baseline == 'zeros':
            baselines = torch.zeros_like(images)
        elif sd_baseline == 'random':
            ## This line has a potential of producing some negative values in noise
            # This is exactly the line they used and (if they aren't using clamp() somewhere later)
            # they are producing a patch which is unnatural not only in terms of geometrical shape
            # but also in terms of values the model has never seen, because there are no negative pixel values in any image
            baselines = torch.rand_like(images) * 2. - 1.
        elif sd_baseline == 'blur':
            baselines = blur(images)
        else:
            print('baseline not implemented')

        add_patches(images, baselines, rows_and_cols)
