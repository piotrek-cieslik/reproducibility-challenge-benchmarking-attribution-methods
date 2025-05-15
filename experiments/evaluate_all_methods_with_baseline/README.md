## Experiment: Evaluate all attribution methods on all models with baseline comparisons

Apart from evaluation of all the attribution methods on all the models, this experiment aims at sanity checking of the IDSDS protocol, by checking if it always ranks "Dummy" attribution maps lower than the proper attribution methods.

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
    --nr_images -1 \
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
| IxG-abs | InputXGradient (absolute) |
| IxG-SG-abs | InputXGradient + SmoothGrad (absolute) |
| IG | Integrated Gradients (zero baseline) |
| IG-abs | Integrated Gradients (zero baseline) (absolute) |
| IG-U | Integrated Gradients (uniform baseline) |
| IG-U-abs | Integrated Gradients (uniform baseline) (absolute) |
| IG-SG | Integrated Gradients (zero baseline) + SmoothGrad |
| IG-SG-abs | Integrated Gradients (zero baseline) + SmoothGrad (absolute)|
| IG-SG-SQ-abs | Integrated Gradients (zero baseline) + SmoothGrad (squared) |
| Grad-CAM | Grad-CAM |
| Grad-CAMpp | Grad-CAM++ |
| SG-CAMpp | Grad-CAM++ + SmoothGrad |
| XG-CAM | XGrad-CAM |
| Layer-CAM | Layer-CAM |


Models evaluated:

| Model | Parameter |
| --- | --- | --- |
| ResNet-18 | resnet18 |
| ResNet-50 | resnet50 |
| ResNet-101 | resnet101 |
| ResNet-152 | resnet152 |
| Wide ResNet-50 | wide_resnet50_2 |
| ResNet-50 w/o BatchNorm | fixup_resnet50 |
| ResNet-50 w/o BatchNorm w/o bias | x_resnet50 |
| VGG-11 | vgg11 |
| VGG-13 | vgg13 |
| VGG-16 | vgg16 |
| VGG-19 | vgg19 |
| VGG-16 w/ BatchNorm | vgg16_bn |
| VGG-16 w/o BatchNorm w/o bias | x_vgg16 |


## Results

The file with results is available in [here](results.csv).
