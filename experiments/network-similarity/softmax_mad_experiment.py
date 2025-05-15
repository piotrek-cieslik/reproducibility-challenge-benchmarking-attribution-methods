import torch
import torch.nn.functional as F

import pandas as pd
import time

def compare_softmax_mad(model1, model2, loader):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    model1.to(device).eval()
    model2.to(device).eval()
    
    num_samples = 0
    logit_mad_total = 0
    softmax_mad_total = 0
    
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device)
    
            # logits
            logits1 = model1(images)
            logits2 = model2(images)

            # logit mad
            logit_mad = torch.abs(logits1 - logits2).mean(dim=1)
            logit_mad_total += logit_mad.sum().item()
    
            # softmax
            probs1 = F.softmax(logits1, dim=1)
            probs2 = F.softmax(logits2, dim=1)
    
            # softmax mad
            softmax_mad = torch.abs(probs1 - probs2).mean(dim=1)
            softmax_mad_total += softmax_mad.sum().item()
            
            num_samples += images.size(0)
            
    logit_mad_score = logit_mad_total / num_samples
    softmax_mad_score = softmax_mad_total / num_samples
    
    return softmax_mad_score, logit_mad_score

print("#"*60)
print("Running Softmax Mean Absolute Difference Experiment")
print("#"*60)

print("Loading Dataset and Models...")
import models
import utils
loader = utils.get_loader()
model_pairs = models.get_softmax_test_pairs()
print("#"*60)

print("Performing Comparisons...")
results = []
total = len(model_pairs)
for i, (k, v) in enumerate(model_pairs.items(), 1):
    print(f"Comparing {i}/{total}: {k}")
    start_time = time.time()
    s, l = compare_softmax_mad(*v, loader)
    duration = time.time() - start_time
    print(f"Results: S={s:.6f}, L={l:.6f} in {duration:.0f} sec")
    results.append({"Model": k, "Softmax MAD": s, "Logit MAD": l})
print("#"*60)

file_name = "softmax_mad_experiment.csv"
print(f"Finished Experiment, Saving to {file_name}...")
df = pd.DataFrame(results)
df.to_csv(file_name, index=False)
    