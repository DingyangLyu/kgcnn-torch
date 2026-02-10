"""Attention layers for graph neural networks (GAT, GATv2)."""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdgesAttention
from kgcnn_torch.ops.activ import get_activation


class AttentionHeadGAT(nn.Module):
    """GAT attention head (Velickovic et al., 2018).

    Computes: alpha_ij = softmax_j(a^T [W*n_i || W*n_j])
              h_i = sigma(sum_j alpha_ij * W*n_j)
    """

    def __init__(self, in_features: int, units: int,
                 use_edge_features: bool = False,
                 edge_dim: int = 0,
                 activation: str = "leaky_relu2",
                 use_bias: bool = True,
                 use_final_activation: bool = True,
                 normalize_softmax: bool = False):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.units = units

        self.linear_trafo = nn.Linear(in_features, units, bias=use_bias)
        # Attention on TRANSFORMED features: a^T [W*n_i || W*n_j (|| e_ij)]
        concat_dim = 2 * units + (edge_dim if use_edge_features else 0)
        self.linear_alpha = nn.Linear(concat_dim, 1, bias=False)
        self.attention_activation = get_activation(activation)
        self.pool_attention = AggregateLocalEdgesAttention(normalize_softmax=normalize_softmax)

        if use_final_activation:
            self.final_activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F_in).
            edge_index: Edge indices (2, M).
            edge_attr: Optional edge features (M, edge_dim).

        Returns:
            Updated node features (N, units).
        """
        num_nodes = x.size(0)
        w_n = self.linear_trafo(x)  # (N, units)

        # Gather TRANSFORMED node pairs for attention (matches GAT paper)
        wn_i = gather_nodes_ingoing(w_n, edge_index)    # target: (M, units)
        wn_j = gather_nodes_outgoing(w_n, edge_index)   # source: (M, units)

        # Compute attention coefficients on transformed features
        if self.use_edge_features and edge_attr is not None:
            e_ij = torch.cat([wn_i, wn_j, edge_attr], dim=-1)
        else:
            e_ij = torch.cat([wn_i, wn_j], dim=-1)

        a_ij = self.attention_activation(self.linear_alpha(e_ij))  # (M, 1)

        # Pool with attention
        h_i = self.pool_attention(wn_j, a_ij, edge_index, num_nodes)  # (N, units)

        if self.use_final_activation:
            h_i = self.final_activation(h_i)
        return h_i


class AttentionHeadGATV2(nn.Module):
    """GATv2 attention head (Brody et al., 2021).

    Computes: a_ij = a^T sigma(W [n_i || n_j])
              alpha_ij = softmax_j(a_ij)
              h_i = sigma(sum_j alpha_ij * W*n_j)
    """

    def __init__(self, in_features: int, units: int,
                 use_edge_features: bool = False,
                 edge_dim: int = 0,
                 activation: str = "leaky_relu2",
                 use_bias: bool = True,
                 use_final_activation: bool = True,
                 normalize_softmax: bool = False):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.units = units

        self.linear_trafo = nn.Linear(in_features, units, bias=use_bias)
        concat_dim = 2 * in_features + (edge_dim if use_edge_features else 0)
        self.alpha_activation = nn.Sequential(
            nn.Linear(concat_dim, units, bias=use_bias),
            get_activation(activation)
        )
        self.linear_alpha = nn.Linear(units, 1, bias=False)
        self.pool_attention = AggregateLocalEdgesAttention(normalize_softmax=normalize_softmax)

        if use_final_activation:
            self.final_activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F_in).
            edge_index: Edge indices (2, M).
            edge_attr: Optional edge features (M, edge_dim).

        Returns:
            Updated node features (N, units).
        """
        num_nodes = x.size(0)
        w_n = self.linear_trafo(x)  # (N, units)

        n_i = gather_nodes_ingoing(x, edge_index)
        n_j = gather_nodes_outgoing(x, edge_index)
        wn_j = gather_nodes_outgoing(w_n, edge_index)

        if self.use_edge_features and edge_attr is not None:
            e_ij = torch.cat([n_i, n_j, edge_attr], dim=-1)
        else:
            e_ij = torch.cat([n_i, n_j], dim=-1)

        # GATv2: activation BEFORE attention vector
        a_ij = self.linear_alpha(self.alpha_activation(e_ij))  # (M, 1)

        h_i = self.pool_attention(wn_j, a_ij, edge_index, num_nodes)

        if self.use_final_activation:
            h_i = self.final_activation(h_i)
        return h_i


class MultiHeadGATV2Layer(nn.Module):
    """Multi-head GATv2 attention layer.

    Uses multiple attention heads and concatenates or averages them.
    Also returns attention logits (needed for MEGAN).
    """

    def __init__(self, in_features: int, units: int, num_heads: int,
                 activation: str = "leaky_relu2",
                 use_bias: bool = True,
                 concat_heads: bool = True,
                 use_edge_features: bool = False,
                 edge_dim: int = 0,
                 use_final_activation: bool = True,
                 normalize_softmax: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.use_edge_features = use_edge_features
        self.use_final_activation = use_final_activation
        self.units = units

        concat_dim = 2 * in_features + (edge_dim if use_edge_features else 0)

        self.head_linears = nn.ModuleList()
        self.head_alpha_acts = nn.ModuleList()
        self.head_alphas = nn.ModuleList()
        for _ in range(num_heads):
            # Value path: linear + activation, matching Keras MultiHeadGATV2Layer
            # which uses Dense(units, activation=activation).
            self.head_linears.append(nn.Sequential(
                nn.Linear(in_features, units, bias=use_bias),
                get_activation(activation)
            ))
            self.head_alpha_acts.append(nn.Sequential(
                nn.Linear(concat_dim, units, bias=use_bias),
                get_activation(activation)
            ))
            self.head_alphas.append(nn.Linear(units, 1, bias=False))

        self.pool_attention = AggregateLocalEdgesAttention(normalize_softmax=normalize_softmax)

        if use_final_activation:
            self.final_activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None):
        """Forward pass.

        Args:
            x: Node features (N, F_in).
            edge_index: Edge indices (2, M).
            edge_attr: Optional edge features (M, edge_dim).

        Returns:
            Tuple of (h_is, a_ijs):
                h_is: Updated node features (N, num_heads*units) or (N, units).
                a_ijs: Attention logits (M, num_heads, 1).
        """
        num_nodes = x.size(0)
        n_i = gather_nodes_ingoing(x, edge_index)
        n_j = gather_nodes_outgoing(x, edge_index)

        h_list = []
        a_list = []
        for k in range(self.num_heads):
            w_n = self.head_linears[k](x)
            wn_j = gather_nodes_outgoing(w_n, edge_index)

            if self.use_edge_features and edge_attr is not None:
                e_ij = torch.cat([n_i, n_j, edge_attr], dim=-1)
            else:
                e_ij = torch.cat([n_i, n_j], dim=-1)

            a_ij = self.head_alpha_acts[k](e_ij)
            a_ij = self.head_alphas[k](a_ij)  # (M, 1)

            h_i = self.pool_attention(wn_j, a_ij, edge_index, num_nodes)

            if self.use_final_activation:
                h_i = self.final_activation(h_i)

            h_list.append(h_i)
            a_list.append(a_ij.unsqueeze(-2))  # (M, 1, 1)

        a_ijs = torch.cat(a_list, dim=-2)  # (M, num_heads, 1)

        if self.concat_heads:
            h_is = torch.cat(h_list, dim=-1)  # (N, num_heads * units)
        else:
            h_is = torch.stack(h_list, dim=0).mean(dim=0)  # (N, units)

        return h_is, a_ijs


class AttentiveHeadFP(nn.Module):
    """Attentive FP attention head (Xiong et al. 2020).

    a_ij = sigma(W [n_i || n_j])
    alpha_ij = softmax_j(a_ij)
    C_i = sigma_context(sum_j alpha_ij * W_trafo * n_j)
    """

    def __init__(self, in_features: int, units: int,
                 use_edge_features: bool = False,
                 edge_dim: int = 0,
                 activation: str = "leaky_relu2",
                 activation_context: str = "elu",
                 use_bias: bool = True):
        super().__init__()
        self.use_edge_features = use_edge_features
        self.units = units

        if use_edge_features:
            self.fc1 = nn.Sequential(
                nn.Linear(in_features, units, bias=use_bias),
                get_activation(activation)
            )
            self.fc2 = nn.Sequential(
                nn.Linear(in_features + edge_dim, units, bias=use_bias),
                get_activation(activation)
            )
            # After fc1/fc2, n_i and n_j are (M, units), so linear_trafo input is units
            self.linear_trafo = nn.Linear(units, units, bias=use_bias)
            concat_dim = 2 * units
        else:
            self.linear_trafo = nn.Linear(in_features, units, bias=use_bias)
            concat_dim = 2 * in_features

        self.alpha_activation = nn.Sequential(
            nn.Linear(concat_dim, units, bias=use_bias),
            get_activation(activation)
        )
        self.linear_alpha = nn.Linear(units, 1, bias=False)
        self.pool_attention = AggregateLocalEdgesAttention()
        self.final_activation = get_activation(activation_context)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F_in).
            edge_index: Edge indices (2, M).
            edge_attr: Optional edge features (M, edge_dim).

        Returns:
            Updated node features (N, units).
        """
        num_nodes = x.size(0)

        if self.use_edge_features and edge_attr is not None:
            n_i = self.fc1(gather_nodes_ingoing(x, edge_index))
            n_out_raw = gather_nodes_outgoing(x, edge_index)
            n_j = self.fc2(torch.cat([n_out_raw, edge_attr], dim=-1))
        else:
            n_i = gather_nodes_ingoing(x, edge_index)
            n_j = gather_nodes_outgoing(x, edge_index)

        wn_j = self.linear_trafo(n_j)
        e_ij = torch.cat([n_i, n_j], dim=-1)
        e_ij = self.alpha_activation(e_ij)
        a_ij = self.linear_alpha(e_ij)
        h_i = self.pool_attention(wn_j, a_ij, edge_index, num_nodes)
        return self.final_activation(h_i)
