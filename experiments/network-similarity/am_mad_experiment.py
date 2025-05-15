import models
import utils

import torch
import torch.nn.functional as F

from captum.attr import IntegratedGradients, GuidedGradCam, LayerGradCam, GuidedBackprop, InputXGradient, Saliency, DeepLift, NoiseTunnel
from torchcam.methods import GradCAM, GradCAMpp, SmoothGradCAMpp, XGradCAM, LayerCAM, ScoreCAM, SSCAM, ISCAM

import pandas as pd
import time

import argparse

def compare_mad_attr(model1, model2, attr_fn, loader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model1.to(device).eval()
    model2.to(device).eval()

    total_mad = 0
    num_samples = 0

    cam1 = attr_fn(model1)
    cam2 = attr_fn(model2)

    for images, _ in loader:
        images = images.to(device)

        for i in range(images.size(0)):
            input_tensor = images[i].unsqueeze(0)
    
            output1 = model1(input_tensor)
            pred_class1 = output1.argmax(dim=1)[0].item()
            attribution_map1 = cam1(pred_class1, output1)[0]
        
            output2 = model2(input_tensor)
            pred_class2 = output2.argmax(dim=1)[0].item()
            attribution_map2 = cam2(pred_class2, output2)[0]
        
            if attribution_map1.ndim == 3 and attribution_map1.shape[0] == 1:
                attribution_map1 = attribution_map1.squeeze(0)
            if attribution_map2.ndim == 3 and attribution_map2.shape[0] == 1:
                attribution_map2 = attribution_map2.squeeze(0)
            
            map1_resized = F.interpolate(
                attribution_map1.unsqueeze(0).unsqueeze(0),
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
            map2_resized = F.interpolate(
                attribution_map2.unsqueeze(0).unsqueeze(0), 
                size=(224, 224), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()
        
            total_mad += torch.abs(map1_resized - map2_resized).mean().item()
        num_samples += images.size(0)
        print(f"{num_samples} {total_mad}")

    mad = total_mad / num_samples
    return mad

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Softmax Mean Absolute Difference Experiment")
    parser.add_argument('--models_dir', type=str, required=True, help='Path to the tuned models directory')
    parser.add_argument('--test_dir', type=str, required=True, help='Path to the test set directory')
    args = parser.parse_args()
    
    print("#"*60)
    print("Running Attribution Map Mean Absolute Difference Experiment")
    print("#"*60)
    
    print("Loading Dataset and Models...")
    loader = utils.get_loader(args.test_dir, batch_size=250)
    model_pairs = models.Models(args.models_dir).get_attribution_test_all_pairs()
    print("#"*60)
    
    print("Performing GradCam Comparisons...")
    results = []
    total = len(model_pairs)
    for i, (k, v) in enumerate(model_pairs.items(), 1):
        print(f"Comparing {i}/{total}: {k}")
        start_time = time.time()
        m = compare_mad_attr(*v, GradCAM, loader)
        duration = time.time() - start_time
        print(f"Results: M={m:.6f} in {duration:.0f} sec")
        results.append({"Model": k, "MAD": m})
    print("#"*60)
    
    file_name = "attribution_GradCam_mad_experiment.csv"
    print(f"Finished GradCam Comparisons, Saving to {file_name}...")
    df = pd.DataFrame(results)
    df.to_csv(file_name, index=False)