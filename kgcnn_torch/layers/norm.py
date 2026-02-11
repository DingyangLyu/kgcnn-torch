"""Graph-aware normalization layers."""
import torch
import torch.nn as nn
from kgcnn_torch.ops.scatter import scatter_reduce_sum


class GraphBatchNorm(nn.Module):
    """Graph batch normalization for disjoint graph data.

    Wraps standard ``nn.BatchNorm1d`` to match the Keras
    ``GraphBatchNormalization`` behaviour, which computes global statistics
    over all nodes in the disjoint batch (i.e. standard batch normalisation).

    When ``padded_disjoint=True``, a mask (``batch > 0``) is applied so that
    dummy padding nodes are excluded from the batch-norm statistics and their
    output is zeroed, matching the Keras implementation.
    """

    def __init__(self, num_features: int, eps: float = 1e-3,
                 momentum: float = 0.01, affine: bool = True,
                 padded_disjoint: bool = False):
        super().__init__()
        self.padded_disjoint = padded_disjoint
        self.bn = nn.BatchNorm1d(num_features, eps=eps, momentum=momentum,
                                 affine=affine)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, F).
            batch: Batch assignment of shape (N,). Used for padded_disjoint masking.
            batch_size: Number of graphs. Kept for API compatibility.

        Returns:
            Normalized features of shape (N, F).
        """
        if self.padded_disjoint and batch is not None:
            mask = (batch > 0)  # (N,) bool — True for real nodes
            x_valid = x[mask]
            x_valid = self.bn(x_valid)
            out = torch.zeros_like(x)
            out[mask] = x_valid
            return out
        return self.bn(x)


class GraphLayerNorm(nn.Module):
    """Layer normalization wrapper for graph data."""

    def __init__(self, num_features: int, eps: float = 1e-3, **kwargs):
        super().__init__()
        self.ln = nn.LayerNorm(num_features, eps=eps, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ln(x)


class GraphNormalization(nn.Module):
    """Graph normalization (per-graph mean/variance normalization).

    Following 'GraphNorm: A Principled Approach to Accelerating Graph Neural Network Training'.
    Normalizes per-graph instead of per-batch.
    """

    def __init__(self, num_features: int, eps: float = 1e-3,
                 mean_shift: bool = True, center: bool = True, scale: bool = True):
        super().__init__()
        self.eps = eps
        self.mean_shift = mean_shift
        self.center = center
        self.scale_flag = scale

        if scale:
            self.gamma = nn.Parameter(torch.ones(num_features))
        if center:
            self.beta = nn.Parameter(torch.zeros(num_features))
        if mean_shift:
            self.alpha = nn.Parameter(torch.ones(num_features))

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, F).
            batch: Batch assignment of shape (N,).
            batch_size: Number of graphs.

        Returns:
            Normalized features of shape (N, F).
        """
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1

        # Compute per-graph counts
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        counts = scatter_reduce_sum(batch, ones, batch_size)  # (B, 1)

        # Per-graph mean
        mean = scatter_reduce_sum(batch, x, batch_size) / counts  # (B, F)

        # Apply mean shift with learnable alpha BEFORE computing variance,
        # following the GraphNorm paper: sigma^2 = sum((x - alpha*mu)^2) / n
        if self.mean_shift:
            mean = mean * self.alpha
        mean_expanded = mean[batch]  # (N, F)
        diff = x - mean_expanded
        var = scatter_reduce_sum(batch, diff.pow(2), batch_size) / counts  # (B, F)
        std = (var + self.eps).sqrt()
        std_expanded = std[batch]  # (N, F)

        out = diff / std_expanded

        if self.scale_flag:
            out = out * self.gamma
        if self.center:
            out = out + self.beta
        return out


class GraphInstanceNormalization(GraphNormalization):
    """Graph instance normalization (per-graph normalization without mean shift).

    Same as GraphNormalization but with mean_shift=False by default.
    """

    def __init__(self, num_features: int, eps: float = 1e-3,
                 center: bool = True, scale: bool = True):
        super().__init__(num_features, eps=eps, mean_shift=False,
                         center=center, scale=scale)
