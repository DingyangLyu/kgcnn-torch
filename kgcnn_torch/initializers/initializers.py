"""Custom weight initializers for graph neural networks."""
import math
import torch
import torch.nn as nn


def _compute_fans(shape):
    """Compute the number of input and output units for a weight shape.

    Taken from original TensorFlow/Keras implementation and adapted for PyTorch.

    Args:
        shape: Tuple or list of integers representing the weight shape.

    Returns:
        A tuple of integer scalars (fan_in, fan_out).
    """
    if len(shape) < 1:
        fan_in = fan_out = 1
    elif len(shape) == 1:
        fan_in = fan_out = shape[0]
    elif len(shape) == 2:
        fan_in = shape[0]
        fan_out = shape[1]
    else:
        # Assuming convolution kernels (2D, 3D, or more).
        # kernel shape: (..., input_depth, depth)
        receptive_field_size = 1
        for dim in shape[:-2]:
            receptive_field_size *= dim
        fan_in = shape[-2] * receptive_field_size
        fan_out = shape[-1] * receptive_field_size
    return int(fan_in), int(fan_out)


def glorot_orthogonal_(tensor: torch.Tensor, scale: float = 1.0, mode: str = "fan_avg") -> torch.Tensor:
    """Glorot-orthogonal initialization (used by DimeNetPP).

    Combines Glorot variance scaling with orthogonal initialization.
    Based on a random (semi-)orthogonal matrix, neural networks are expected
    to learn better when features are de-correlated.

    References:
        - "Reducing over-fitting in deep networks by de-correlating representations"
          by M. Cogswell et al. (2016)
        - "Exact solutions to the nonlinear dynamics of learning in deep linear
          neural networks" by A. M. Saxe et al. (2013)

    Note: DimeNetPP uses scale=2.0 explicitly. Default matches Keras GlorotOrthogonal.

    Args:
        tensor: Tensor to initialize in-place.
        scale: Scaling factor (default 1.0, matching Keras).
        mode: One of 'fan_in', 'fan_out', 'fan_avg'. Default is 'fan_avg'.

    Returns:
        The initialized tensor.
    """
    if tensor.dim() < 2:
        raise ValueError("glorot_orthogonal_ requires at least 2D tensor")

    with torch.no_grad():
        nn.init.orthogonal_(tensor)
        fan_in, fan_out = _compute_fans(tensor.shape)
        if mode == "fan_in":
            denom = max(1.0, float(fan_in))
        elif mode == "fan_out":
            denom = max(1.0, float(fan_out))
        else:
            denom = max(1.0, float(fan_in + fan_out) / 2.0)
        target_scale = scale / denom
        s = math.sqrt(target_scale / torch.var(tensor, unbiased=False).item())
        tensor.mul_(s)
    return tensor


def he_orthogonal_(tensor: torch.Tensor, scale: float = 1.0, mode: str = "fan_in") -> torch.Tensor:
    """He-orthogonal initialization (used by GemNet).

    Combines He variance scaling with orthogonal initialization. The kernel
    is first standardized (zero mean, unit variance over fan_in dimensions),
    then scaled according to He initialization.

    Based on a random (semi-)orthogonal matrix, neural networks are expected
    to learn better when features are de-correlated.

    References:
        - "Reducing over-fitting in deep networks by de-correlating representations"
          by M. Cogswell et al. (2016)
        - "Exact solutions to the nonlinear dynamics of learning in deep linear
          neural networks" by A. M. Saxe et al. (2013)
        - GemNet: https://arxiv.org/abs/2106.08903

    Args:
        tensor: Tensor to initialize in-place.
        scale: Scaling factor (default 1.0).
        mode: One of 'fan_in', 'fan_out', 'fan_avg'. Default is 'fan_in'.

    Returns:
        The initialized tensor.
    """
    if tensor.dim() < 2:
        raise ValueError("he_orthogonal_ requires at least 2D tensor")

    with torch.no_grad():
        nn.init.orthogonal_(tensor)

        shape = tensor.shape
        fan_in, fan_out = _compute_fans(shape)
        if mode == "fan_in":
            target_scale = scale / max(1.0, float(fan_in))
        elif mode == "fan_out":
            target_scale = scale / max(1.0, float(fan_out))
        else:
            target_scale = scale / max(1.0, float(fan_in + fan_out) / 2.0)

        # Standardize kernel: zero mean, unit variance over fan_in dimensions.
        tensor = _standardize_kernel(tensor)

        # Scale by sqrt(target_scale).
        tensor.mul_(math.sqrt(target_scale))

    return tensor


def _standardize_kernel(tensor: torch.Tensor) -> torch.Tensor:
    """Standardize kernel so that N*Var(W) = 1 and E[W] = 0 over fan_in dims.

    Args:
        tensor: Weight tensor of at least 2 dimensions.

    Returns:
        The standardized tensor (in-place).
    """
    eps = 1e-7
    ndim = tensor.dim()
    if ndim == 0:
        return tensor

    if ndim >= 3:
        # Reduce over all dims except the last (output) dimension.
        axis = list(range(ndim - 1))
    else:
        # For 2D: reduce over dim 0 (fan_in).
        axis = [0]

    mean = tensor.mean(dim=axis, keepdim=True)
    var = tensor.var(dim=axis, keepdim=True, unbiased=False)
    tensor.sub_(mean).div_(torch.sqrt(var + eps))
    return tensor
