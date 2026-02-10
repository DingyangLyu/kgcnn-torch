"""Graph convolution layers."""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.norm import GraphBatchNorm
from kgcnn_torch.ops.activ import get_activation


class GCNConv(nn.Module):
    """Graph Convolution (Kipf & Welling 2017).

    Computes: sigma(A_s @ (W @ X + b))
    where A_s is pre-scaled adjacency.
    """

    def __init__(self, in_features: int, out_features: int,
                 pooling_method: str = "sum",
                 activation: str = "leaky_relu2",
                 use_bias: bool = True,
                 normalize_by_weights: bool = False):
        super().__init__()
        self.normalize_by_weights = normalize_by_weights
        self.linear = nn.Linear(in_features, out_features, bias=use_bias)
        self.activation = get_activation(activation)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)
        if normalize_by_weights:
            self.aggr_weights = AggregateLocalEdges(pooling_method="sum")

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F_in).
            edge_index: Edge indices (2, M).
            edge_weight: Optional edge weights (M, 1) or (M,). If None,
                all edges are weighted by 1.

        Returns:
            Updated node features (N, F_out).
        """
        num_nodes = x.size(0)
        # Transform nodes
        x_trans = self.linear(x)
        # Gather source node features for each edge
        x_j = gather_nodes_outgoing(x_trans, edge_index)  # (M, F_out)
        # Default to unweighted aggregation if edge weights are not provided.
        if edge_weight is None:
            edge_weight = torch.ones(
                edge_index.size(1), 1, device=x.device, dtype=x_j.dtype
            )
        elif edge_weight.dim() == 1:
            edge_weight = edge_weight.unsqueeze(-1)

        # Weight by edge weights (adjacency entries)
        messages = x_j * edge_weight  # (M, F_out)
        # Aggregate to target nodes
        out = self.aggr(messages, edge_index, num_nodes)  # (N, F_out)

        if self.normalize_by_weights:
            norm = self.aggr_weights(edge_weight, edge_index, num_nodes)
            out = out / (norm + 1e-8)

        return self.activation(out)


class SchNetCFconv(nn.Module):
    """Continuous filter convolution from SchNet.

    Edge features are processed by two Dense layers, then multiplied
    with outgoing node features and aggregated.
    """

    def __init__(self, in_features: int, units: int,
                 activation: str = "shifted_softplus",
                 use_bias: bool = True,
                 pooling_method: str = "sum"):
        super().__init__()
        self.dense1 = nn.Linear(in_features, units, bias=use_bias)
        self.dense2 = nn.Linear(units, units, bias=use_bias)
        self.activation = get_activation(activation)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            edge_attr: Edge features (M, edge_dim).
            edge_index: Edge indices (2, M).

        Returns:
            Updated node features (N, F).
        """
        num_nodes = x.size(0)
        # Process edge features through filter network
        w = self.activation(self.dense1(edge_attr))
        w = self.dense2(w)  # (M, units)
        # Gather source node features
        x_j = gather_nodes_outgoing(x, edge_index)  # (M, F)
        # Element-wise multiply
        messages = x_j * w  # (M, F)
        # Aggregate
        return self.aggr(messages, edge_index, num_nodes)


class SchNetInteraction(nn.Module):
    """SchNet interaction block.

    node -> Dense(linear) -> CFconv -> Dense(activation) -> Dense(linear) -> + residual
    """

    def __init__(self, units: int = 128, edge_dim: int = 20,
                 activation: str = "shifted_softplus",
                 use_bias: bool = True,
                 pooling_method: str = "sum"):
        super().__init__()
        self.dense_in = nn.Linear(units, units, bias=False)
        self.cfconv = SchNetCFconv(edge_dim, units, activation=activation,
                                   use_bias=use_bias, pooling_method=pooling_method)
        self.dense1 = nn.Linear(units, units, bias=use_bias)
        self.dense2 = nn.Linear(units, units, bias=use_bias)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            edge_attr: Edge features (M, edge_dim).
            edge_index: Edge indices (2, M).

        Returns:
            Updated node features (N, F).
        """
        x_proj = self.dense_in(x)
        x_conv = self.cfconv(x_proj, edge_attr, edge_index)
        x_conv = self.activation(self.dense1(x_conv))
        x_conv = self.dense2(x_conv)
        return x + x_conv


class GINConv(nn.Module):
    """Graph Isomorphism Network convolution (Xu et al. 2019).

    h_i' = (1 + eps) * h_i + sum_j h_j

    Note: The non-linear mapping (MLP) is NOT included and should be applied after.
    """

    def __init__(self, pooling_method: str = "sum", epsilon_learnable: bool = False):
        super().__init__()
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable
        eps = torch.zeros(1)
        if epsilon_learnable:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            edge_index: Edge indices (2, M).

        Returns:
            Updated node features (N, F).
        """
        num_nodes = x.size(0)
        x_j = gather_nodes_outgoing(x, edge_index)
        agg = self.aggr(x_j, edge_index, num_nodes)
        return (1.0 + self.eps) * x + agg


class GINEConv(nn.Module):
    """GINE convolution (Hu et al. 2020).

    h_i' = (1 + eps) * h_i + sum_j act(h_j + e_ij)

    Note: The final MLP is NOT included and should be applied after.
    """

    def __init__(self, pooling_method: str = "sum", epsilon_learnable: bool = False,
                 activation: str = "relu"):
        super().__init__()
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable
        eps = torch.zeros(1)
        if epsilon_learnable:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_attr: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            edge_index: Edge indices (2, M).
            edge_attr: Edge features (M, F).

        Returns:
            Updated node features (N, F).
        """
        num_nodes = x.size(0)
        x_j = gather_nodes_outgoing(x, edge_index)
        msg = self.activation(x_j + edge_attr)
        agg = self.aggr(msg, edge_index, num_nodes)
        return (1.0 + self.eps) * x + agg


class CGCNNLayer(nn.Module):
    """Crystal Graph Convolutional Neural Network layer (Xie & Grossman 2018).

    sigma(z_i,j * W_f + b_f) * g(z_i,j * W_s + b_s)
    where z_i,j = [h_i, h_j, e_ij]
    """

    def __init__(self, node_features: int, edge_features: int,
                 activation: str = "softplus", gate_activation: str = "sigmoid",
                 activation_out: str = "softplus",
                 pooling_method: str = "mean",
                 batch_normalization: bool = False):
        super().__init__()
        concat_dim = 2 * node_features + edge_features
        self.linear_filter = nn.Linear(concat_dim, node_features)
        self.linear_gate = nn.Linear(concat_dim, node_features)
        self.activation = get_activation(activation)
        self.gate_activation = get_activation(gate_activation)
        self.activation_out = get_activation(activation_out)
        self.batch_normalization = batch_normalization
        if batch_normalization:
            self.bn_s = GraphBatchNorm(node_features)
            self.bn_f = GraphBatchNorm(node_features)
            self.bn_out = GraphBatchNorm(node_features)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor = None,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            edge_attr: Edge features (M, edge_dim).
            edge_index: Edge indices (2, M).
            batch: Batch assignment (N,) for per-graph batch normalization.
            batch_size: Number of graphs in batch.

        Returns:
            Updated node features (N, F).
        """
        num_nodes = x.size(0)
        x_i = gather_nodes_ingoing(x, edge_index)
        x_j = gather_nodes_outgoing(x, edge_index)
        z = torch.cat([x_i, x_j, edge_attr], dim=-1)
        x_s = self.linear_gate(z)
        x_f = self.linear_filter(z)
        if self.batch_normalization:
            x_s = self.bn_s(x_s, batch, batch_size)
            x_f = self.bn_f(x_f, batch, batch_size)
        x_s = self.gate_activation(x_s)
        x_f = self.activation(x_f)
        msg = x_s * x_f
        agg = self.aggr(msg, edge_index, num_nodes)
        if self.batch_normalization:
            agg = self.bn_out(agg, batch, batch_size)
        return self.activation_out(x + agg)
