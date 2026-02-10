"""Relational dense layer for multi-relation graphs."""
import torch
import torch.nn as nn
from kgcnn_torch.ops.activ import get_activation


class RelationalDense(nn.Module):
    """Dense layer with relation-specific weight matrices.

    Each relation has its own weight matrix. Supports optional basis decomposition
    (W_r = sum_b a_rb * V_b) and block-diagonal decomposition for parameter reduction.

    Based on R-GCN (Schlichtkrull et al., 2018).
    """

    def __init__(self, in_features: int, out_features: int, num_relations: int,
                 num_bases: int = None, num_blocks: int = None,
                 activation: str = None, use_bias: bool = True):
        """Initialize RelationalDense.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_relations: Number of relation types.
            num_bases: If set, use basis decomposition with this many basis matrices.
            num_blocks: If set, use block-diagonal decomposition.
            activation: Activation function name.
            use_bias: Whether to use bias.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations
        self.num_bases = num_bases
        self.num_blocks = num_blocks

        if num_bases is not None:
            # Basis decomposition: W_r = sum_b a_rb * V_b
            self.bases = nn.Parameter(torch.Tensor(num_bases, in_features, out_features))
            self.comps = nn.Parameter(torch.Tensor(num_relations, num_bases))
            nn.init.xavier_uniform_(self.bases)
            nn.init.xavier_uniform_(self.comps)
        elif num_blocks is not None:
            assert in_features % num_blocks == 0
            assert out_features % num_blocks == 0
            block_in = in_features // num_blocks
            block_out = out_features // num_blocks
            self.blocks = nn.Parameter(
                torch.Tensor(num_relations, num_blocks, block_in, block_out))
            nn.init.xavier_uniform_(self.blocks.view(num_relations * num_blocks, block_in, block_out))
        else:
            self.weight = nn.Parameter(
                torch.Tensor(num_relations, in_features, out_features))
            nn.init.xavier_uniform_(self.weight)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, relations: torch.Tensor) -> torch.Tensor:
        """Apply relation-specific transformation.

        Args:
            x: Input features of shape (N, in_features).
            relations: Relation type per sample of shape (N,), values in [0, R).

        Returns:
            Transformed features of shape (N, out_features).
        """
        if self.num_bases is not None:
            # W = comps @ bases -> (R, in, out)
            weight = torch.einsum('rb,bio->rio', self.comps, self.bases)
            w = weight[relations]  # (N, in, out)
        elif self.num_blocks is not None:
            w_blocks = self.blocks[relations]  # (N, B, block_in, block_out)
            B = self.num_blocks
            block_in = self.in_features // B
            block_out = self.out_features // B
            x_blocks = x.view(-1, B, block_in)  # (N, B, block_in)
            out_blocks = torch.einsum('nbi,nbio->nbo', x_blocks, w_blocks)  # (N, B, block_out)
            out = out_blocks.reshape(-1, self.out_features)  # (N, out_features)
            if self.bias is not None:
                out = out + self.bias
            return self.activation(out)
        else:
            w = self.weight[relations]  # (N, in, out)

        # Match Keras RelationalDense.batch_dot: sum(expand_dims(x,-1) * k, axis=-2).
        # This avoids small numeric drift compared to GEMM-based bmm.
        out = torch.sum(x.unsqueeze(-1) * w, dim=-2)  # (N, out)
        if self.bias is not None:
            out = out + self.bias
        return self.activation(out)
