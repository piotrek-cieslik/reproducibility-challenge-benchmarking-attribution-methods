## Experiment: Check that the ranking of the models with logits and sofmax is similar


In order to run the experiments, copy the code from this directory to idsds. Then, run this command:

```
python idsds/evaluate_batch_edges.py \
    --evaluation_protocol single_deletion \
    --pretrained_ckpt_path 'path to fine-tuned models' \
    --grid_rows_and_cols 4 \
    --data_dir 'path to ImageNet ' \
    --batch_size 32 \
    --nr_images -1 \
    --use_softmax True \
    --output_file 'output.log'
```

Here, we skipped Integrated Gradients, because it runs for days, even on the university GPU service.
Attribution methods evaluated:

| Parameter | Name |
| --- | --- |
| IxG  | InputXGradient |
| IxG-SG | InputXGradient + SmoothGrad |
| IxG-abs | InputXGradient (absolute) |
| IxG-SG-abs | InputXGradient + SmoothGrad (absolute) |
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

The file with results: [results_logits.csv](results_logits.csv) and [results_softmax.csv](results_softmax.csv).
