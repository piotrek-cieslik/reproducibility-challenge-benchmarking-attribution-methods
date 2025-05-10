## Experiment 1: Evaluate all attribution methods on all models
## Experiment 2: Sanity check of the IDSDS protocol, by checking if it always ranks "Dummy" attribution maps lower than the proper attribution methods

Conda can't find the right version of opencv in order to be compatible with already installed packages, so we need to install it manually. Under the conda environment, run:

```
pip install opencv-python
```

In order to run the experiments, copy the code from this directory to idsds. Then, run this command:

```
python idsds/evaluate_batch_edges.py \
    --evaluation_protocol single_deletion \
    --pretrained_ckpt_path 'path to fine-tuned models' \
    --grid_rows_and_cols 4 \
    --data_dir 'path to ImageNet ' \
    --batch_size 1 \
    --nr_images 4 \
    --output_file 'output.log'
```

For path to ImageNet, (see the [main README.md](../../README.md)).
For path to fine-tuned models - see [README.md in idsds directory](../../idsds/README.md)).

Batch size is 1 because for batch size 2 some results in Out of memory exception for some models and attribution methods.
For instance for `resnet152` and `IG-SG-SQ-abs` the 48G or RAM is not sufficient. and we get `CUDA out of memory. Tried to allocate 308.00 MiB. GPU 0 has a total capacity of 44.47 GiB of which 148.38 MiB is free. Including non-PyTorch memory, this process has 44.33 GiB memory in use. Of the allocated memory 43.69 GiB is allocated by PyTorch, and 454.40 MiB is reserved by PyTorch but unallocated.`


Attribution methods evaluated:

| Parameter | Name |
| --- | --- |
| IxG  | InputXGradient |
| IxG-SG | InputXGradient + SmoothGrad |
| IG | Integrated Gradients (zero baseline) |
| IG-U | Integrated Gradients (uniform baseline) |
| IG-SG | Integrated Gradients (zero baseline) + SmoothGrad |
| IxG --attribution_transform abs | InputXGradient (absolute) |
| IxG-SG --attribution_transform abs | InputXGradient + SmoothGrad (absolute) |
| IG --attribution_transform abs | Integrated Gradients (zero baseline) (absolute) |
| IG-U --attribution_transform abs | Integrated Gradients (uniform baseline) (absolute) |
| IG-SG --attribution_transform abs | Integrated Gradients (zero baseline) + SmoothGrad (absolute)|
| IG-SG-SQ --attribution_transform abs | Integrated Gradients (zero baseline) + SmoothGrad (squared) |
| Grad-CAM | Grad-CAM (CNN only) |
| Grad-CAMpp | Grad-CAM++ (CNN only) |
| SG-CAMpp | Grad-CAM++ + SmoothGrad (CNN only) |
| XG-CAM | XGrad-CAM (CNN only) |
| Layer-CAM | Layer-CAM (CNN only) |


Models evaluated:

| Model | Parameter |
| --- | --- | --- |
| ResNet-18 | --model resnet18 |
| ResNet-50 | --model resnet50 |
| ResNet-101 | --model resnet101 |
| ResNet-152 | --model resnet152 |
| Wide ResNet-50 | --model wide_resnet50_2 |
| ResNet-50 w/o BatchNorm | --model fixup_resnet50 |
| ResNet-50 w/o BatchNorm w/o bias | --model x_resnet50 |
| VGG-11 | --model vgg11 |
| VGG-13 | --model vgg13 |
| VGG-16 | --model vgg16 |
| VGG-19 | --model vgg19 |
| VGG-16 w/ BatchNorm | --model vgg16_bn |
| VGG-16 w/o BatchNorm w/o bias | --model x_vgg16 |
| ViT-B-16 | --model vit_base_patch16_224 |
| Bcos-ResNet-50 | --model bcos_resnet50 |
| BagNet-33 | --model bagnet33 |


## Results

the data with the results will be linked here.
