"""Scatter reduce operations for graph neural networks.

All functions follow the signature: scatter_reduce_*(indices, values, dim_size)
where indices is 1D (M,), values is (M, ...), and dim_size is the size of
dimension 0 of the output tensor.
"""
import torch


def _expand_indices(indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """Expand 1D indices to match the shape of values for scatter operations."""
    dims_to_add = values.dim() - indices.dim()
    for _ in range(dims_to_add):
        indices = indices.unsqueeze(-1)
    return indices.expand_as(values)


def scatter_reduce_sum(indices: torch.Tensor, values: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter sum: out[indices[i]] += values[i].

    Args:
        indices: 1D indices of shape (M,).
        values: Values of shape (M, ...).
        dim_size: Size of output dimension 0.

    Returns:
        Tensor of shape (dim_size, ...) with scattered sums.
    """
    idx = _expand_indices(indices, values)
    out = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
    return out.scatter_reduce(0, idx, values, reduce='sum')


def scatter_reduce_mean(indices: torch.Tensor, values: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter mean: out[i] = mean of values where indices == i.

    Args:
        indices: 1D indices of shape (M,).
        values: Values of shape (M, ...).
        dim_size: Size of output dimension 0.

    Returns:
        Tensor of shape (dim_size, ...) with scattered means.
    """
    idx = _expand_indices(indices, values)
    out = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
    return out.scatter_reduce(0, idx, values, reduce='mean', include_self=False)


def scatter_reduce_max(indices: torch.Tensor, values: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter max: out[i] = max of values where indices == i.

    Empty groups (no incoming values) return 0, matching the Keras convention
    of ``__safe_scatter_max_min_to_zero__ = True``.

    Args:
        indices: 1D indices of shape (M,).
        values: Values of shape (M, ...).
        dim_size: Size of output dimension 0.

    Returns:
        Tensor of shape (dim_size, ...) with scattered max values.
    """
    idx = _expand_indices(indices, values)
    out = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
    result = out.scatter_reduce(0, idx, values, reduce='amax', include_self=False)
    # Replace empty group results (±inf or undefined) with 0 for safety.
    counts = torch.zeros(dim_size, device=indices.device, dtype=torch.long)
    counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))
    empty_mask = (counts == 0)
    if empty_mask.any():
        # Expand mask to match result shape
        mask = empty_mask.view(-1, *([1] * (result.dim() - 1))).expand_as(result)
        result = torch.where(mask, torch.zeros_like(result), result)
    return result


def scatter_reduce_min(indices: torch.Tensor, values: torch.Tensor, dim_size: int) -> torch.Tensor:
    """Scatter min: out[i] = min of values where indices == i.

    Empty groups (no incoming values) return 0, matching the Keras convention
    of ``__safe_scatter_max_min_to_zero__ = True``.

    Args:
        indices: 1D indices of shape (M,).
        values: Values of shape (M, ...).
        dim_size: Size of output dimension 0.

    Returns:
        Tensor of shape (dim_size, ...) with scattered min values.
    """
    idx = _expand_indices(indices, values)
    out = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
    result = out.scatter_reduce(0, idx, values, reduce='amin', include_self=False)
    # Replace empty group results (±inf or undefined) with 0 for safety.
    counts = torch.zeros(dim_size, device=indices.device, dtype=torch.long)
    counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=torch.long))
    empty_mask = (counts == 0)
    if empty_mask.any():
        mask = empty_mask.view(-1, *([1] * (result.dim() - 1))).expand_as(result)
        result = torch.where(mask, torch.zeros_like(result), result)
    return result


def scatter_reduce_softmax(indices: torch.Tensor, values: torch.Tensor, dim_size: int,
                           normalize: bool = True) -> torch.Tensor:
    """Scatter softmax: compute softmax over groups defined by indices.

    Args:
        indices: 1D indices of shape (M,).
        values: Values of shape (M, ...).
        dim_size: Size of output dimension 0.
        normalize: If True (default), subtract max per group for numerical
            stability before exponentiation.

    Returns:
        Tensor of shape (M, ...) with softmax-normalized values per group.
    """
    idx = _expand_indices(indices, values)

    if normalize:
        out_max = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
        out_max = out_max.scatter_reduce(0, idx, values, reduce='amax', include_self=False)
        values = values - out_max[indices]

    values_exp = torch.exp(values)
    out_sum = torch.zeros(dim_size, *values.shape[1:], dtype=values.dtype, device=values.device)
    out_sum = out_sum.scatter_reduce(0, idx, values_exp, reduce='sum', include_self=False)
    return values_exp / (out_sum[indices] + 1e-8)
