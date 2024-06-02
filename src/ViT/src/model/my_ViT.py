"""Implementation of Vision Transformer (ViT) model."""

import torch.nn as nn
import torch

from ViT.src.model.encodeur.residual_connection import MyViTBlock
from ViT.src.model.utils import get_positional_embeddings, patchify


class ViT(nn.Module):
    """
    Vision Transformer (ViT) model.

    Args:
        chw (tuple): Tuple containing number of channels,
        height, and width of input images.
        n_patches (int): Number of patches to divide the image into.
        n_blocks (int): Number of transformer blocks.
        hidden_d (int): Dimensionality of the hidden state.
        n_heads (int): Number of attention heads.
        out_d (int): Dimensionality of the output.

    Attributes:
        chw (tuple): Tuple containing number of channels,
        height, and width of input images.
        n_patches (int): Number of patches to divide the image into.
        n_blocks (int): Number of transformer blocks.
        hidden_d (int): Dimensionality of the hidden state.
        patch_size (tuple): Size of each patch.
        input_d (int): Dimensionality of the input after flattening patches.
        linear_mapper (nn.Linear): Linear layer to map input
        patches to hidden dimension.
        class_token (nn.Parameter): Learnable classification token.
        positional_embeddings (torch.Tensor): Positional embeddings
        for each position.
        blocks (nn.ModuleList): List of transformer encoder blocks.
        mlp (nn.Sequential): MLP for final classification.

    Example:
        >>> model = ViT(chw=(3, 224, 224), n_patches=16, n_blocks=4,
        hidden_d=512, n_heads=8, out_d=10)
    """

    def __init__(
        self, chw, n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10
    ):
        # Super constructor
        super(ViT, self).__init__()

        # Attributes
        self.chw = chw  # ( C , H , W )
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d

        # Input and patches sizes
        assert (
            chw[1] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        assert (
            chw[2] % n_patches == 0
        ), "Input shape not entirely divisible by number of patches"
        self.patch_size = (chw[1] / n_patches, chw[2] / n_patches)

        # 1) Linear mapper
        self.input_d = int(chw[0] * self.patch_size[0] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        # 2) Learnable classification token
        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        # 3) Positional embedding
        self.register_buffer(
            "positional_embeddings",
            get_positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )

        # 4) Transformer encoder blocks
        self.blocks = nn.ModuleList(
            [MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)]
        )

        # 5) Classification MLP
        self.mlp = nn.Sequential(
            nn.Linear(self.hidden_d, out_d), nn.Softmax(dim=-1)
        )

    def forward(self, images):
        """
        Forward pass of the ViT model.

        Args:
            images (torch.Tensor): Input images of shape
            (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Output of the model, a tensor containing
            class probabilities.

        Example:
            >>> output = model(torch.randn(1, 3, 224, 224))
        """
        # Dividing images into patches
        n, c, h, w = images.shape
        patches = patchify(images, self.n_patches).to(
            self.positional_embeddings.device
        )

        # Running linear layer tokenization
        # Map the vector corresponding to each patch to the hidden
        # size dimension
        tokens = self.linear_mapper(patches)

        # Adding classification token to the tokens
        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        # Adding positional embedding
        out = tokens + self.positional_embeddings.repeat(n, 1, 1)

        # Transformer Blocks
        for block in self.blocks:
            out = block(out)

        # Getting the classification token only
        out = out[:, 0]

        return self.mlp(
            out
        )  # Map to output dimension, output category distribution
