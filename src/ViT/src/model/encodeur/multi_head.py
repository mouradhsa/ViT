"""
multi_head.py

This script defines a Multi-Head Self-Attention layer implemented as
a PyTorch module.

In the context of this module, 'multi-head' refers to the use of multiple sets
of parameters for computing attention scores. Instead of a single set
of parameters (query, key, and value), the self-attention mechanism employs
'n_heads' sets of parameters.

Each set of parameters, associated with an attention head, learns
different representations of the input sequence. This enables the model to
attend to different parts of the input sequence simultaneously, allowing it to
capture more complex patterns and dependencies within the data.

This implementation divides the input dimensionality (token_dim) into 'n_heads'
equal parts, and each attention head operates on its respective part of the
input sequence.
After computing attention scores independently for each attention
head, the outputs are concatenated along the token dimension before being
passed to subsequent layers in the neural network.

The 'MutliHeadSelfAttention' class defined in this module encapsulates this
multi-head self-attention mechanism, providing a flexible and efficient way to
incorporate multi-head attention into neural network architectures.
"""

import torch
import torch.nn as nn


class MutliHeadSelfAttention(nn.Module):
    def __init__(self, d, n_heads=2):
        """
        Initializes the Multi-Head Self-Attention layer.

        Args:
            d (int): The input dimensionality (token_dim).
            n_heads (int): The number of attention heads. Default to 2.
        """
        super(MutliHeadSelfAttention, self).__init__()
        self.d = d
        self.n_heads = n_heads

        # Ensure the input dimension can be evenly divided into the specified
        # number of heads
        assert (
            d % n_heads == 0
        ), f"Can't divide dimension {d}\
              into {n_heads} heads"

        # Calculate the dimensionality of each attention head
        d_head = int(d / n_heads)
        self.d_head = d_head

        # Define linear mappings for queries, keys, and values for each
        # attention head
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(self.n_heads)]
        )

        # Softmax operation along the last dimension
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        """
        Forward pass of the Multi-Head Self-Attention layer.

        Args:
            sequences (torch.Tensor): Input sequences with shape
             (N, seq_length, token_dim).

        Returns:
            torch.Tensor: Output sequences with shape
             (N, seq_length, item_dim).
        """
        # List to store results for each sequence in the batch
        result = []

        # Iterate over each sequence in the batch
        for sequence in sequences:
            seq_result = []  # List to store results for each attention head

            # Process each attention head separately
            for head in range(self.n_heads):
                # Retrieve the linear mappings for query, key, and value for
                # the current head
                q_mapping = self.q_mappings[head]
                k_mapping = self.k_mappings[head]
                v_mapping = self.v_mappings[head]

                # Extract the part of the sequence corresponding to the
                # current attention head
                seq = sequence[
                    :, head * self.d_head : (head + 1) * self.d_head
                ]

                # Project the sequence part into query, key, and value spaces
                q, k, v = q_mapping(seq), k_mapping(seq), v_mapping(seq)

                # Calculate the attention scores using dot product and
                # apply softmax
                attention = self.softmax(q @ k.T / (self.d_head**0.5))

                # Compute the weighted sum of values using the attention scores
                seq_result.append(attention @ v)

            # Concatenate the outputs of all attention heads along
            # the token dimension
            result.append(torch.hstack(seq_result))

        # Concatenate results for all sequences in the batch along
        # the batch dimension
        return torch.cat([torch.unsqueeze(r, dim=0) for r in result])
