"""Aggregation layers for graph neural networks.

All aggregation uses PyG convention: edge_index[0] = source, edge_index[1] = target.
Messages are aggregated at the target node (edge_index[1]).
"""
import torch
import torch.nn as nn
from kgcnn_torch.ops.scatter import (
    scatter_reduce_sum, scatter_reduce_mean, scatter_reduce_max,
    scatter_reduce_min, scatter_reduce_softmax
)


_POOL_FN = {
    "sum": scatter_reduce_sum,
    "mean": scatter_reduce_mean,
    "max": scatter_reduce_max,
    "min": scatter_reduce_min,
    "scatter_sum": scatter_reduce_sum,
    "scatter_mean": scatter_reduce_mean,
    "scatter_max": scatter_reduce_max,
    "scatter_min": scatter_reduce_min,
}


class Aggregate(nn.Module):
    """Aggregate features by indices using scatter operations."""

    def __init__(self, pooling_method: str = "sum"):
        super().__init__()
        self.pooling_method = pooling_method
        if pooling_method not in _POOL_FN:
            raise ValueError(f"Unknown pooling method '{pooling_method}'. Available: {list(_POOL_FN.keys())}")
        self._pool_fn = _POOL_FN[pooling_method]

    def forward(self, values: torch.Tensor, indices: torch.Tensor, dim_size: int) -> torch.Tensor:
        """Aggregate values at indices.

        Args:
            values: Values of shape (M, ...).
            indices: Target indices of shape (M,).
            dim_size: Output dimension 0 size.

        Returns:
            Aggregated tensor of shape (dim_size, ...).
        """
        return self._pool_fn(indices, values, dim_size)


class AggregateLocalEdges(nn.Module):
    """Aggregate edge features per target node.

    In PyG convention, aggregates to edge_index[1] (target).
    """

    def __init__(self, pooling_method: str = "sum"):
        super().__init__()
        self.pooling_method = pooling_method
        self._aggregate = Aggregate(pooling_method=pooling_method)

    def forward(self, edges: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """Aggregate edge features to target nodes.

        Args:
            edges: Edge features of shape (M, F).
            edge_index: Edge indices of shape (2, M).
            num_nodes: Number of nodes N.

        Returns:
            Aggregated features of shape (N, F).
        """
        target_idx = edge_index[1]
        return self._aggregate(edges, target_idx, num_nodes)


class AggregateLocalEdgesAttention(nn.Module):
    """Aggregate local edges with attention weights.

    Computes: n_i = sum_j softmax_j(a_ij) * e_ij
    """

    def __init__(self, normalize_softmax: bool = False):
        super().__init__()
        self.normalize_softmax = normalize_softmax

    def forward(self, edges: torch.Tensor, attention: torch.Tensor,
                edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Forward pass.

        Args:
            edges: Edge/message features of shape (M, F).
            attention: Attention logits of shape (M, 1).
            edge_index: Edge indices of shape (2, M).
            num_nodes: Number of nodes N.

        Returns:
            Aggregated features of shape (N, F).
        """
        target_idx = edge_index[1]
        alpha = scatter_reduce_softmax(target_idx, attention, num_nodes,
                                       normalize=self.normalize_softmax)
        weighted = edges * alpha.expand_as(edges)
        return scatter_reduce_sum(target_idx, weighted, num_nodes)


class AggregateLocalEdgesLSTM(nn.Module):
    """Aggregate local edge features using LSTM.

    Pads edges per node to max_edges_per_node and runs LSTM over them.
    Used by GraphSAGE for sequential aggregation.
    """

    def __init__(self, units: int, input_dim: int = None, max_edges_per_node: int = 10, **lstm_kwargs):
        super().__init__()
        self.units = units
        self.max_edges_per_node = max_edges_per_node
        kwargs = dict(batch_first=True)
        kwargs.update(lstm_kwargs)
        self.input_dim = input_dim
        self._lstm_kwargs = kwargs
        self._lstm_initialized = input_dim is not None
        # If input_dim is not provided, defer LSTM construction to first forward
        # so edge feature dimension can be inferred from data.
        self.lstm = nn.LSTM(input_dim, units, **kwargs) if input_dim is not None else None

    def _lazy_init_lstm(self, input_dim: int, device: torch.device):
        """Initialize LSTM on first forward pass when input_dim was not provided."""
        self.input_dim = input_dim
        self.lstm = nn.LSTM(self.input_dim, self.units, **self._lstm_kwargs)
        # Register as proper submodule so parameters are tracked
        self.add_module('lstm', self.lstm)
        self.lstm = self.lstm.to(device)
        self._lstm_initialized = True

    def forward(self, edges: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> torch.Tensor:
        """Aggregate edges to target nodes via LSTM.

        Args:
            edges: Edge features of shape (M, F).
            edge_index: Edge indices of shape (2, M).
            num_nodes: Number of nodes N.

        Returns:
            Aggregated features of shape (N, units).
        """
        target_idx = edge_index[1]  # (M,)
        M, F = edges.shape
        max_e = self.max_edges_per_node

        if not self._lstm_initialized:
            self._lazy_init_lstm(F, edges.device)

        # Count edges per node
        counts = torch.zeros(num_nodes, dtype=torch.long, device=edges.device)
        counts.scatter_add_(0, target_idx, torch.ones(M, dtype=torch.long, device=edges.device))

        # Vectorized position computation (replaces Python for-loop)
        # Compute cumulative count per target node to get within-node position
        # Sort edges by target node for stable ordering
        sort_idx = torch.argsort(target_idx, stable=True)
        sorted_targets = target_idx[sort_idx]

        # Compute within-group position using cumcount
        group_offsets = torch.zeros(num_nodes + 1, dtype=torch.long, device=edges.device)
        group_offsets[1:] = counts.cumsum(0)
        positions = torch.arange(M, device=edges.device) - group_offsets[sorted_targets]

        # Build padded sequences: (N, max_edges_per_node, F)
        lstm_input = torch.zeros(num_nodes, max_e, F, device=edges.device, dtype=edges.dtype)
        valid_mask = positions < max_e
        valid_sorted_idx = sort_idx[valid_mask]
        valid_targets = sorted_targets[valid_mask]
        valid_positions = positions[valid_mask]
        lstm_input[valid_targets, valid_positions] = edges[valid_sorted_idx]

        # Build mask for LSTM: lengths clamped to max_edges
        lengths = counts.clamp(max=max_e).clamp(min=1)  # (N,)

        # Pack and run LSTM
        packed = nn.utils.rnn.pack_padded_sequence(
            lstm_input, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        return h_n.squeeze(0)  # (N, units)


class RelationalAggregateLocalEdges(nn.Module):
    """Aggregate edges separately per relation type.

    For each relation r, aggregates edge features to target nodes,
    producing per-relation node embeddings.
    """

    def __init__(self, num_relations: int, pooling_method: str = "sum"):
        super().__init__()
        self.num_relations = num_relations
        self.pooling_method = pooling_method
        self._aggregate = Aggregate(pooling_method=pooling_method)

    def forward(self, edges: torch.Tensor, edge_index: torch.Tensor,
                edge_relation: torch.Tensor, num_nodes: int) -> torch.Tensor:
        """Aggregate edges per relation per target node.

        Args:
            edges: Edge features of shape (M, F).
            edge_index: Edge indices of shape (2, M).
            edge_relation: Relation type per edge of shape (M,), values in [0, R).
            num_nodes: Number of nodes N.

        Returns:
            Aggregated features of shape (N, R, F) where R = num_relations.
        """
        target_idx = edge_index[1]  # (M,)
        R = self.num_relations
        # Flatten to (N*R) indexing: target_node * R + relation
        flat_idx = target_idx * R + edge_relation.long()
        out = self._aggregate(edges, flat_idx, num_nodes * R)  # (N*R, F)
        return out.view(num_nodes, R, edges.size(-1))


class AggregateWeightedLocalEdges(nn.Module):
    """Aggregate edge features weighted by edge weights per target node.

    out_i = sum_j(w_ij * e_ij) / sum_j(w_ij)  [if normalize_by_weights]
    out_i = sum_j(w_ij * e_ij)                  [otherwise]
    """

    def __init__(self, pooling_method: str = "sum", normalize_by_weights: bool = False):
        super().__init__()
        self.pooling_method = pooling_method
        self.normalize_by_weights = normalize_by_weights
        self._aggregate = Aggregate(pooling_method=pooling_method)

    def forward(self, edges: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int, weights: torch.Tensor = None) -> torch.Tensor:
        """Aggregate weighted edge features to target nodes.

        Args:
            edges: Edge features of shape (M, F).
            edge_index: Edge indices of shape (2, M).
            num_nodes: Number of nodes N.
            weights: Edge weights of shape (M, 1). If None, treated as 1.

        Returns:
            Aggregated features of shape (N, F).
        """
        target_idx = edge_index[1]
        if weights is not None:
            weighted_edges = edges * weights.expand_as(edges)
        else:
            weighted_edges = edges
        out = self._aggregate(weighted_edges, target_idx, num_nodes)

        if self.normalize_by_weights and weights is not None:
            weight_sum = scatter_reduce_sum(target_idx, weights.expand_as(edges), num_nodes)
            out = out / (weight_sum + 1e-8)

        return out
