"""Custom activation functions for graph neural networks."""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def shifted_softplus(x: torch.Tensor) -> torch.Tensor:
    r"""Shifted softplus: :math:`\log(e^x + 1) - \log(2)`."""
    return F.softplus(x) - math.log(2.0)


def softplus2(x: torch.Tensor) -> torch.Tensor:
    r"""Numerically stable softplus that is 0 at x=0: :math:`\text{relu}(x) + \log(0.5 e^{-|x|} + 0.5)`."""
    return F.relu(x) + torch.log(0.5 * torch.exp(-torch.abs(x)) + 0.5)


def leaky_softplus(x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    r"""Leaky softplus: :math:`(1 - \alpha) \cdot \text{softplus}(x) + \alpha \cdot x`."""
    return F.softplus(x) * (1.0 - alpha) + alpha * x


def leaky_relu2(x: torch.Tensor, alpha: float = 0.05) -> torch.Tensor:
    """Leaky ReLU with default alpha=0.05."""
    return F.leaky_relu(x, negative_slope=alpha)


def swish(x: torch.Tensor) -> torch.Tensor:
    """Swish / SiLU activation."""
    return F.silu(x)


# Module wrappers

class ShiftedSoftplus(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return shifted_softplus(x)


class Softplus2(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return softplus2(x)


class LeakySoftplus(nn.Module):
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return leaky_softplus(x, self.alpha)


class LeakyReLU2(nn.Module):
    def __init__(self, alpha: float = 0.05):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return leaky_relu2(x, self.alpha)


class Swish(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swish(x)


# Registry for string -> activation lookup
_ACTIVATION_REGISTRY = {
    "shifted_softplus": ShiftedSoftplus,
    "softplus2": Softplus2,
    "leaky_softplus": LeakySoftplus,
    "leaky_relu2": LeakyReLU2,
    "swish": Swish,
    "swish2": Swish,
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "softplus": nn.Softplus,
    "elu": nn.ELU,
    "gelu": nn.GELU,
    "leaky_relu": nn.LeakyReLU,
    "selu": nn.SELU,
    "softmax": lambda: nn.Softmax(dim=-1),
}


def get_activation(name: str) -> nn.Module:
    """Get an activation module by name.

    Args:
        name: Activation function name (e.g. 'relu', 'shifted_softplus').

    Returns:
        An nn.Module activation instance.

    Raises:
        ValueError: If name is not recognized.
    """
    if name is None or name == "linear":
        return nn.Identity()
    key = name.lower().strip()
    if key not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(_ACTIVATION_REGISTRY.keys())}")
    return _ACTIVATION_REGISTRY[key]()
