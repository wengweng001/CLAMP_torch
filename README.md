# CLAMP PyTorch
This is a PyTorch implementation of **[Cross-Domain Continual Learning via CLAMP](https://arxiv.org/abs/2405.07142)**.

## Requirements
The current version of the code has been tested with the following configuration:
- python 3.7
- pytorch 1.8.1
- torchvision 0.9.1

## Usage
All experiments are launched via the command line by running `main.py` with the desired parameters.

**Example: MNIST to USPS**
```bash
python -u main.py --scenario class --source splitMNIST --target splitUSPS --tasks 5 --fc-units 256 --apporach clamp \
  --batch 128 --batch-d 64 --lr-bm 1e-3 --lr-a 1e-3 \
  --pseudo --meta --domain --epoch 5 --epoch-inner 5 --epoch-outer 5 --num-exemplars1 50 --num-exemplars2 50 \
  --runs 5
```
