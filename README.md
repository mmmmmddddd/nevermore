# Nervermore: multi-task learning benchmark

### Install
Install pytorch/torchvision from [here](https://download.pytorch.org/whl/torch_stable.html) based on your cuda version.

And install others:

```bash
pip install -r requirements.txt
```

### Usage

```bash
# prepare training data
mkdir data
ln -s <paty-to-NYU> data/NYU

# start train using default config, ./configs/baseline.yaml
make train
```
