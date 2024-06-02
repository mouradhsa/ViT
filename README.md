# Vision Transformers (ViT) from Scratch

This repository contains an implementation of Vision Transformers (ViT) from scratch using PyTorch.

## Table of Contents

- [What is Vision Transformers (ViT)?](#what-is-vision-transformers-vit)
- [Getting Started](#getting-started)
  - [1. Install Rye](#1-install-rye)
  - [2. Create Virtual Environment](#2-create-virtual-environment)
  - [3. Activate Virtual Environment](#3-activate-virtual-environment)
- [Prepare Data](#prepare-data)
  - [1. Preprocess Data](#1-preprocess-data)
- [Train Model](#train-model)

## What is Vision Transformers (ViT)?

Vision Transformers (ViT) are a class of neural network architectures introduced in the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al. Unlike convolutional neural networks (CNNs), which rely on convolutions and pooling operations, ViTs employ a transformer architecture, originally designed for natural language processing (NLP), to process image data directly.

## Implementation Details

### 1. install [rye](https://github.com/mitsuhiko/rye)

[install documentation](https://rye-up.com/guide/installation/#installing-rye)

MacOS
```zsh
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.zshrc
source ~/.zshrc
```

Linux
```bash
curl -sSf https://rye-up.com/get | bash
echo 'source "$HOME/.rye/env"' >> ~/.bashrc
source ~/.bashrc
```

### 2. Create virtual environment

```bash
rye sync
```

### 3. Activate virtual environment

```bash
. .venv/bin/activate
```

## Prepare Data

### 1. Preprocess data

```bash
rye run python .\src\ViT\run\prepare_data.py
```

## Train Model
The following commands are for training the model on mnist dataset
```bash
rye run python .\src\ViT\run\train.py dataset.mnist.batch_size=64
```

You can easily perform experiments by changing the parameters since [hydra](https://hydra.cc/docs/intro/) is used.
The following commands perform experiments with batch_size of 64, 128, and 256.

```bash
rye run python .\src\ViT\run\train.py --m dataset.mnist.batch_size=64,128,256
```

