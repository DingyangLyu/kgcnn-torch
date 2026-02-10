"""Update layers for graph neural networks."""
import torch
import torch.nn as nn


class GRUUpdate(nn.Module):
    """GRU-based node update.

    Uses GRUCell to update node features given new messages.
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)

    def forward(self, message: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Apply GRU update.

        Args:
            message: Input message of shape (N, input_dim).
            hidden: Hidden state of shape (N, hidden_dim).

        Returns:
            Updated hidden state of shape (N, hidden_dim).
        """
        return self.gru_cell(message, hidden)


class ResidualLayer(nn.Module):
    """Residual connection with two dense layers (DimeNetPP).

    output = x + activation(dense_2(activation(dense_1(x))))
    """

    def __init__(self, units: int, activation: str = "swish", use_bias: bool = True,
                 kernel_initializer: str = None):
        super().__init__()
        from kgcnn_torch.ops.activ import get_activation
        self.dense_1 = nn.Linear(units, units, bias=use_bias)
        self.dense_2 = nn.Linear(units, units, bias=use_bias)
        self.activation_1 = get_activation(activation)
        self.activation_2 = get_activation(activation)

        # Apply custom initialization if specified
        if kernel_initializer == "glorot_orthogonal":
            from kgcnn_torch.initializers.initializers import glorot_orthogonal_
            glorot_orthogonal_(self.dense_1.weight)
            glorot_orthogonal_(self.dense_2.weight)
            if self.dense_1.bias is not None:
                nn.init.zeros_(self.dense_1.bias)
            if self.dense_2.bias is not None:
                nn.init.zeros_(self.dense_2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.activation_1(self.dense_1(x))
        out = self.activation_2(self.dense_2(out))
        return x + out
