"""
utils.py

This module contains utility functions for various tasks related
to deep learning models.

These utility functions are designed to assist in common tasks encountered
when working with deep learning models, such as generating positional
embeddings for sequences and converting images into patches.

Functions:
- get_positional_embeddings: Generates positional embeddings for sequences,
    which can be used as additional input features in models like transformers.
- patchify: Converts images into patches, which is often required
    in vision tasks when applying techniques like Vision Transformers (ViT).

These utility functions can streamline the preprocessing
and feature engineering steps in your deep learning projects,
making it easier to work with complex data and models.
"""

import numpy as np
import torch


def get_positional_embeddings(sequence_length, d):
    """
    Generate positional embeddings for sequences.

    Args:
        sequence_length (int): The length of the input sequence.
        d (int): The dimensionality of the positional embeddings.

    Returns:
        torch.Tensor: Positional embeddings tensor of shape
            (sequence_length, d).
    """
    result = torch.ones(sequence_length, d)
    for i in range(sequence_length):
        for j in range(d):
            result[i][j] = (
                np.sin(i / (10000 ** (j / d)))
                if j % 2 == 0
                else np.cos(i / (10000 ** ((j - 1) / d)))
            )
    return result


def patchify(images, n_patches: int):
    """
    Convert images into patches.

    Args:
        images (torch.Tensor): Input images tensor of shape (n, c, h, w).
        n_patches (int): Number of patches to divide the images into.

    Returns:
        torch.Tensor: Tensor containing image patches of shape
            (n, n_patches**2, patch_size).
    """
    n, c, h, w = images.shape

    assert h == w, "Patchify method is implemented for square images only"

    patches = torch.zeros(n, n_patches**2, h * w * c // n_patches**2)
    patch_size = h // n_patches

    for idx, image in enumerate(images):
        for i in range(n_patches):
            for j in range(n_patches):
                patch = image[
                    :,
                    i * patch_size : (i + 1) * patch_size,
                    j * patch_size : (j + 1) * patch_size,
                ]
                patches[idx, i * n_patches + j] = patch.flatten()
    return patches
