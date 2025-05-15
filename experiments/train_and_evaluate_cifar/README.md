## Experiment: Finetune pretrained models on CIFAR-100 dataset and evaluate all attribution methods on all models

We used pretrained models from [https://github.com/chenyaofo/pytorch-cifar-models](https://github.com/chenyaofo/pytorch-cifar-models).

Use `download_cifar_pretrained_models.py` to download pretrained models' weights.

In order to run the experiments, copy the code from this directory to idsds. Then, run these commands:

### For training

```
python idsds/train_cifar.py \
    --pretrained_ckpt_path 'path to pretrained cifar models' \
    --data_dir 'path to cifar-100 dataset' \
    --grid_rows_and_cols 4 \
    --epochs 30 \
    --batch_size 128 \
    --lr 0.005 \
    --momentum 0.9 \
    --step_size 15 \
    --weight-decay 0.0005 \
    --store_path 'path where to store fine-tuned models'

```

### For evaluation

```
python idsds/evaluate_batch_cifar.py \
    --evaluation_protocol single_deletion \
    --pretrained_ckpt_path 'path to fine-tuned models' \
    --grid_rows_and_cols 4 \
    --data_dir 'path to cifar-100 dataset' \
    --batch_size 32 \
    --nr_images -1 \
    --output_file 'output_log_evaluate_cifar.log'
```


### For sanity-check

```
python idsds/sanity_check_accuracy.py \
    --evaluation_protocol single_deletion \
    --ckpt_path_pretrained 'path to pretrained models' \
    --ckpt_path_finetuned 'path to fine-tuned models' \
    --grid_rows_and_cols 4 \
    --data_dir 'path to cifar-100 dataset' \
    --batch_size 128 \
    --use_softmax False \
    --output_file 'output_cifar_sanity_check.log'
```


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
| ResNet-20 | cifar100_resnet20 |
| ResNet-32 | cifar100_resnet32 |
| ResNet-44 | cifar100_resnet44 |
| ResNet-56 | cifar100_resnet56 |
| VGG-11 BatchNorm | cifar100_vgg11_bn |
| VGG-13 BatchNorm | cifar100_vgg13_bn |
| VGG-16 BatchNorm | cifar100_vgg16_bn |
| VGG-19 BatchNorm | cifar100_vgg19_bn |


## Results

The results of the evaluation are available in [results.csv](results.csv) file. The results for sanity check are available in [results_cifar_sanity_check.csv](results_cifar_sanity_check.csv) file.
