"""Activation layers with learnable parameters for graph neural networks.

These are nn.Module subclasses that wrap activation functions with optionally
trainable parameters. They complement the functional activations in
kgcnn_torch.ops.activ by providing stateful, learnable versions.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeakySoftplus(nn.Module):
    r"""Leaky softplus activation with optionally learnable leak parameter.

    Computes :math:`(1 - \alpha) \cdot \text{softplus}(x) + \alpha \cdot x`.

    This is a smooth approximation to leaky ReLU, where the leak is controlled
    by alpha. When alpha=0, this reduces to standard softplus.
    """

    def __init__(self, alpha: float = 0.05, trainable: bool = False):
        """Initialize leaky softplus.

        Args:
            alpha: Leak parameter. Default is 0.05.
            trainable: Whether alpha is a learnable parameter. Default is False.
        """
        super().__init__()
        self._alpha_init = float(alpha)
        self._trainable = bool(trainable)
        if trainable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply leaky softplus activation.

        Args:
            inputs: Input tensor of arbitrary shape.

        Returns:
            Activated tensor of same shape.
        """
        return F.softplus(inputs) * (1 - self.alpha) + self.alpha * inputs

    def extra_repr(self) -> str:
        return f"alpha={self._alpha_init}, trainable={self._trainable}"


class LeakyRelu(nn.Module):
    r"""Leaky ReLU activation with optionally learnable negative slope.

    Computes :math:`\text{LeakyReLU}(x, \alpha)`.

    Equivalent to torch.nn.functional.leaky_relu with a potentially trainable
    negative slope parameter.
    """

    def __init__(self, alpha: float = 0.05, trainable: bool = False):
        """Initialize leaky ReLU.

        Args:
            alpha: Negative slope for x < 0. Default is 0.05.
            trainable: Whether alpha is a learnable parameter. Default is False.
        """
        super().__init__()
        self._alpha_init = float(alpha)
        self._trainable = bool(trainable)
        if trainable:
            self.alpha = nn.Parameter(torch.tensor(float(alpha)))
        else:
            self.register_buffer("alpha", torch.tensor(float(alpha)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply leaky ReLU activation.

        Args:
            inputs: Input tensor of arbitrary shape.

        Returns:
            Activated tensor of same shape.
        """
        if self._trainable:
            # Manual implementation to preserve gradient through alpha
            return torch.where(inputs >= 0, inputs, self.alpha * inputs)
        else:
            return F.leaky_relu(inputs, negative_slope=float(self.alpha))

    def extra_repr(self) -> str:
        return f"alpha={self._alpha_init}, trainable={self._trainable}"


class Swish(nn.Module):
    r"""Swish activation with optionally learnable beta parameter.

    Computes :math:`x \cdot \sigma(\beta x)`, where :math:`\sigma` is the sigmoid
    function. When beta=1, this is equivalent to SiLU.
    """

    def __init__(self, beta: float = 1.0, trainable: bool = False):
        """Initialize Swish activation.

        Args:
            beta: Scaling parameter inside the sigmoid. Default is 1.0.
            trainable: Whether beta is a learnable parameter. Default is False.
        """
        super().__init__()
        self._beta_init = float(beta)
        self._trainable = bool(trainable)
        if trainable:
            self.beta = nn.Parameter(torch.tensor(float(beta)))
        else:
            self.register_buffer("beta", torch.tensor(float(beta)))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply Swish activation.

        Args:
            inputs: Input tensor of arbitrary shape.

        Returns:
            Activated tensor of same shape.
        """
        return inputs * torch.sigmoid(self.beta * inputs)

    def extra_repr(self) -> str:
        return f"beta={self._beta_init}, trainable={self._trainable}"
