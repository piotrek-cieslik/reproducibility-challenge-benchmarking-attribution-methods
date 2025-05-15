# Prerequisites

## Make sure to download this repo with submodules

The [idsds](https://github.com/visinf/idsds) repo is a submodule in this repository. In order to download any repo with submodules, run:

```
git clone --recurse-submodules https://github.com/your/repo.git
```

## Download ImageNet-mini

```
import kagglehub
path = kagglehub.dataset_download("ifigotin/imagenetmini-1000")
print("Path to dataset files:", path)
```

Save the dataset to `datasets/imagenet-mini/`

## Download CIFAR-100

[CIFAR-100 download page](https://www.cs.toronto.edu/~kriz/cifar.html)

## Download fine-tuned models

Download links are in `idsds/README.md`.

## Create an environment

Create conda environment with:

```
conda env create -f idsds/conda_environment_export.yml --name reproducing_idsds
```

To activate this environment, use

```
conda activate reproducing_idsds
```

To deactivate an active environment, use

```
conda deactivate
```

## Running the experiments

For each of our experiments, the files are available in a subdirectory of `experiments/` folder. To run the experiments, follow the instructions in the `README.md` file for chosen experiment.
