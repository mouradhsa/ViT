"""
residual_connection.py

This module implements a residual connection mechanism commonly used
in deep learning architectures.

Classes:
- MyViTBlock: Represents a single block of a Vision Transformer (ViT) model
    with residual connections.
"""

import torch
import torch.nn as nn
from ViT.src.model.encodeur.multi_head import MutliHeadSelfAttention


class MyViTBlock(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        """
        Initializes a single block of the Vision Transformer (ViT)
            model with residual connections.

        Args:
            hidden_d (int): The hidden dimensionality for the self-attention
                mechanism and MLP layer.
            n_heads (int): The number of attention heads for the Multi-Head
                Self-Attention layer.
            mlp_ratio (int): The ratio of dimensionality in the MLP layer
                to the hidden dimensionality. Defaults to 4.
        """
        super(MyViTBlock, self).__init__()
        self.hidden_d = hidden_d
        self.n_heads = n_heads

        # Layer normalization before the Multi-Head Self-Attention (MHSA) layer
        self.norm1 = nn.LayerNorm(hidden_d)

        # Multi-Head Self-Attention (MHSA) layer
        self.mhsa = MutliHeadSelfAttention(hidden_d, n_heads)

        # Layer normalization before the Feedforward Neural Network
        # (FFNN) layer
        self.norm2 = nn.LayerNorm(hidden_d)

        # Feedforward Neural Network (FFNN) layer
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),  # Activation function
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        """
        Forward pass of a single ViT block with residual connections.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        # Apply layer normalization before feeding into the
        # Multi-Head Self-Attention (MHSA) layer
        out = x + self.mhsa(self.norm1(x))

        # Apply layer normalization before feeding into the
        # Feedforward Neural Network (FFNN) layer
        out = out + self.mlp(self.norm2(out))

        return out


if __name__ == "__main__":
    model = MyViTBlock(hidden_d=8, n_heads=2)

    x = torch.randn(7, 50, 8)  # Dummy sequences
    print(model(x).shape)  # torch.Size([7, 50, 8])
