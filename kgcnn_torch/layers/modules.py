"""Utility module layers for graph neural networks.

Simple wrapper modules around common tensor operations, providing nn.Module
interfaces for operations like embedding lookup, dimension manipulation,
and zero tensor creation.
"""
import torch
import torch.nn as nn


def keras_uniform_init_embedding_(embedding: nn.Embedding) -> nn.Embedding:
    """Initialize nn.Embedding weights with Uniform(-0.05, 0.05) to match Keras default.

    Keras ``Embedding`` uses ``embeddings_initializer='uniform'`` which is
    ``RandomUniform(-0.05, 0.05)``.  PyTorch ``nn.Embedding`` defaults to
    ``Normal(0, 1)`` which produces ~20x larger initial magnitudes.

    Args:
        embedding: The nn.Embedding to initialize in-place.

    Returns:
        The same embedding (for chaining).
    """
    nn.init.uniform_(embedding.weight, -0.05, 0.05)
    return embedding


class Embedding(nn.Module):
    """Embedding layer wrapping nn.Embedding.

    Looks up fixed-dimensional embeddings for integer indices.
    Uses Keras-compatible Uniform(-0.05, 0.05) initialization.
    """

    def __init__(self, input_dim: int, output_dim: int, **kwargs):
        """Initialize embedding.

        Args:
            input_dim: Size of the embedding dictionary (number of distinct tokens).
            output_dim: Size of each embedding vector.
            **kwargs: Additional arguments passed to nn.Embedding (e.g., padding_idx).
        """
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embedding = nn.Embedding(input_dim, output_dim, **kwargs)
        keras_uniform_init_embedding_(self.embedding)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Look up embeddings.

        Args:
            inputs: Integer tensor of indices of arbitrary shape.

        Returns:
            Tensor of shape (*inputs.shape, output_dim).
        """
        return self.embedding(inputs)


class ExpandDims(nn.Module):
    """Add a dimension at the specified axis position."""

    def __init__(self, axis: int):
        """Initialize.

        Args:
            axis: Position at which to insert the new dimension.
        """
        super().__init__()
        self.axis = axis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Unsqueeze input at the specified axis.

        Args:
            inputs: Input tensor of arbitrary shape.

        Returns:
            Tensor with one additional dimension at position axis.
        """
        return inputs.unsqueeze(self.axis)


class SqueezeDims(nn.Module):
    """Remove a dimension at the specified axis position."""

    def __init__(self, axis: int):
        """Initialize.

        Args:
            axis: Position of the dimension to remove (must be size 1).
        """
        super().__init__()
        self.axis = axis

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Squeeze input at the specified axis.

        Args:
            inputs: Input tensor with size 1 at the specified axis.

        Returns:
            Tensor with the dimension at position axis removed.
        """
        return inputs.squeeze(self.axis)


class ZerosLike(nn.Module):
    """Create a zero tensor with the same shape and dtype as input."""

    def __init__(self):
        """Initialize layer."""
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Create zero tensor like input.

        Args:
            inputs: Input tensor of shape ([N], F, ...).

        Returns:
            Zero tensor of same shape, dtype, and device as input.
        """
        return torch.zeros_like(inputs)
