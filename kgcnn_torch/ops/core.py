"""Core tensor operations for graph neural networks using PyTorch."""
import torch
from typing import Optional, Union


def repeat_static_length(x: torch.Tensor, repeats: torch.Tensor,
                         total_repeat_length: int,
                         axis: Optional[int] = None) -> torch.Tensor:
    """Repeat each element of a tensor after themselves, with a known output length.

    Similar to ``torch.repeat_interleave`` but with a statically known output size,
    which can be useful for JIT compilation or pre-allocated buffers.

    Args:
        x (torch.Tensor): Input tensor.
        repeats (torch.Tensor): 1D tensor of repeat counts for each element.
        total_repeat_length (int): Total length of all repeats (sum of repeats).
        axis (int, optional): The axis along which to repeat. If None, the input
            is flattened first.

    Returns:
        torch.Tensor: Repeated tensor with the specified total length along the
            repeat axis.
    """
    if axis is None:
        x = x.flatten()
        axis = 0

    result = torch.repeat_interleave(x, repeats, dim=axis)

    # Verify or truncate/pad to match the expected static length.
    actual_len = result.shape[axis]
    if actual_len == total_repeat_length:
        return result
    elif actual_len > total_repeat_length:
        slices = [slice(None)] * result.dim()
        slices[axis] = slice(0, total_repeat_length)
        return result[tuple(slices)]
    else:
        # Pad with zeros if needed.
        pad_size = total_repeat_length - actual_len
        pad_shape = list(result.shape)
        pad_shape[axis] = pad_size
        padding = torch.zeros(pad_shape, dtype=result.dtype, device=result.device)
        return torch.cat([result, padding], dim=axis)


def norm(x: torch.Tensor, ord: Union[str, int, float] = "fro",
         axis: Optional[Union[int, list]] = None,
         keepdims: bool = False) -> torch.Tensor:
    """Compute tensor norm with optional epsilon for numerical stability.

    Wraps ``torch.linalg.norm`` with a consistent interface.

    Args:
        x (torch.Tensor): Input tensor.
        ord (str, int, float): Order of the norm. Default is 'fro' (Frobenius).
            Common values: 2 (L2 norm), 1 (L1 norm), 'fro', float('inf').
        axis (int or list of int, optional): Dimensions over which to compute.
        keepdims (bool): Whether to keep the reduced dimensions. Default is False.

    Returns:
        torch.Tensor: The computed norm.
    """
    if isinstance(axis, int):
        axis = [axis]
    if axis is not None:
        axis = tuple(axis)

    if ord == "fro" and axis is None:
        return torch.linalg.norm(x, ord="fro", keepdim=keepdims)
    elif axis is not None:
        if len(axis) == 1:
            return torch.linalg.norm(x, ord=None if ord == "fro" else ord,
                                     dim=axis[0], keepdim=keepdims)
        else:
            return torch.linalg.norm(x, ord=ord if ord != "fro" else None,
                                     dim=axis, keepdim=keepdims)
    else:
        return torch.linalg.norm(x, ord=ord, keepdim=keepdims)


def cross(x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
    """Compute the cross product of two (arrays of) vectors.

    Args:
        x1 (torch.Tensor): Components of the first vector(s). The last dimension
            must be 2 or 3.
        x2 (torch.Tensor): Components of the second vector(s). The last dimension
            must be 2 or 3.

    Returns:
        torch.Tensor: The cross product. If both inputs have last dimension 3,
            the output also has last dimension 3. If both have last dimension 2,
            the output is a scalar (per element).
    """
    return torch.linalg.cross(x1, x2, dim=-1)


def safe_norm(x: torch.Tensor, axis: int = -1, keepdims: bool = True,
              eps: float = 1e-8) -> torch.Tensor:
    """Compute a safe (numerically stable) L2 norm with epsilon.

    Useful for normalizing vectors in geometric operations to avoid
    division by zero or gradient issues at zero.

    Args:
        x (torch.Tensor): Input tensor.
        axis (int): Axis along which to compute the norm. Default is -1.
        keepdims (bool): Whether to keep the dimension. Default is True.
        eps (float): Small epsilon for numerical stability. Default is 1e-8.

    Returns:
        torch.Tensor: The L2 norm with epsilon added inside the sqrt.
    """
    return torch.sqrt(torch.sum(x * x, dim=axis, keepdim=keepdims) + eps)


def decompose_ragged_tensor(values: torch.Tensor, row_splits: torch.Tensor):
    """Decompose a flattened ragged tensor into individual arrays.

    Args:
        values (torch.Tensor): Flattened values of shape (total_elements, ...).
        row_splits (torch.Tensor): Row splits of shape (num_rows + 1,).

    Returns:
        list: List of tensors, one per row.
    """
    splits = row_splits.tolist()
    return [values[splits[i]:splits[i + 1]] for i in range(len(splits) - 1)]
