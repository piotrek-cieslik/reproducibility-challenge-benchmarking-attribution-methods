# Run that
The below instructions install stuff that are needed for running the code from the paper

```sh
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade virtualenv
python3 -m venv venv
source venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install numpy scipy matplotlib ipympl scikit-learn pandas jupyter jupyterlab drawdata  ucimlrepo typeguard pydantic torch torchvision torchbearer torchbearer cupy-cuda12x
python3 -m jupyterlab build
python3 -m pip install torchvision 
python3 -m pip install captum
python3 -m pip install torchcam
python3 -m pip install einops
python3 -m pip install scikit-image
```

# Download ImageNet-1000 needed 
Maybe there is a version in GitHub/GitLab too, here is from Kaggle: 
https://www.kaggle.com/datasets/ifigotin/imagenetmini-1000?resource=download-directory

# Download the repository 
```sh
git clone git@github.com:visinf/idsds.git
```

After you get the repo, go to [idsds/evaluate.py](idsds/evaluate.py) and delete `BcosIGUExplainer, BcosGCExplainer` from import, because there are no such classes.

# Run the experiment
```shell
source venv/bin/activate
python3 idsds/evaluate.py \
--evaluation_protocol single_deletion \
--grid_rows_and_cols 4 \
--data_dir "datasets/imagenet-mini" \
--model resnet50 \
--explainer IxG \
--batch_size 128
```