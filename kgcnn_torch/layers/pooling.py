"""Graph-level pooling layers."""
import torch
import torch.nn as nn
from kgcnn_torch.ops.scatter import scatter_reduce_sum, scatter_reduce_mean, scatter_reduce_max, scatter_reduce_min
from kgcnn_torch.ops.activ import get_activation


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


class PoolingNodes(nn.Module):
    """Pool node features to graph-level representation."""

    def __init__(self, pooling_method: str = "sum"):
        super().__init__()
        self.pooling_method = pooling_method
        if pooling_method not in _POOL_FN:
            raise ValueError(f"Unknown pooling '{pooling_method}'")
        self._pool_fn = _POOL_FN[pooling_method]

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                batch_size: int = None) -> torch.Tensor:
        """Pool node features to graph level.

        Args:
            x: Node features of shape (N, F).
            batch: Batch assignment of shape (N,).
            batch_size: Number of graphs in batch. If None, inferred from batch.

        Returns:
            Graph-level features of shape (B, F).
        """
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1
        return self._pool_fn(batch, x, batch_size)


class PoolingWeightedNodes(nn.Module):
    """Weighted pooling of node features to graph-level."""

    def __init__(self, pooling_method: str = "sum"):
        super().__init__()
        self.pooling_method = pooling_method
        if pooling_method not in _POOL_FN:
            raise ValueError(f"Unknown pooling '{pooling_method}'")
        self._pool_fn = _POOL_FN[pooling_method]

    def forward(self, x: torch.Tensor, weights: torch.Tensor,
                batch: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """Weighted pool node features.

        Args:
            x: Node features of shape (N, F).
            weights: Node weights of shape (N, 1).
            batch: Batch assignment of shape (N,).
            batch_size: Number of graphs. If None, inferred.

        Returns:
            Weighted graph-level features of shape (B, F).
        """
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1
        xw = x * weights.expand_as(x)
        return self._pool_fn(batch, xw, batch_size)


class PoolingEmbeddingAttention(nn.Module):
    """Attention-based graph pooling.

    s = sum_i softmax_i(a_i) * n_i
    """

    def __init__(self, pooling_method: str = "sum", normalize_softmax: bool = False):
        super().__init__()
        self.pooling_method = pooling_method
        self.normalize_softmax = normalize_softmax
        if pooling_method not in _POOL_FN:
            raise ValueError(f"Unknown pooling '{pooling_method}'")
        self._pool_fn = _POOL_FN[pooling_method]

    def forward(self, x: torch.Tensor, attention: torch.Tensor,
                batch: torch.Tensor, batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            attention: Attention logits (N, 1).
            batch: Batch assignment (N,).
            batch_size: Number of graphs.

        Returns:
            Graph-level features (B, F).
        """
        from kgcnn_torch.ops.scatter import scatter_reduce_softmax
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1
        alpha = scatter_reduce_softmax(batch, attention, batch_size,
                                       normalize=self.normalize_softmax)
        weighted = x * alpha.expand_as(x)
        return self._pool_fn(batch, weighted, batch_size)


# Alias
PoolingNodesAttention = PoolingEmbeddingAttention


class PoolingNodesAttentive(nn.Module):
    """Attentive pooling from AttentiveFP (Xiong et al. 2020).

    Iteratively refines graph embedding via GRU + attention.
    """

    def __init__(self, units: int, depth: int = 3,
                 input_dim: int = None,
                 pooling_method: str = "sum",
                 activation: str = "leaky_relu2",
                 activation_context: str = "elu"):
        super().__init__()
        self.units = units
        self.depth = depth
        in_dim = input_dim if input_dim is not None else units
        self.pool_start = PoolingNodes(pooling_method=pooling_method)
        self.linear_trafo = nn.Linear(in_dim, units)
        self.lay_alpha = nn.Sequential(
            nn.Linear(in_dim + units, 1),
            get_activation(activation)
        )
        self.pool_attention = PoolingEmbeddingAttention()
        self.final_activ = get_activation(activation_context)
        self.gru = nn.GRUCell(units, units)
        self._project_h0 = nn.Linear(in_dim, units) if in_dim != units else None

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F).
            batch: Batch assignment (N,).
            batch_size: Number of graphs.

        Returns:
            Graph-level features (B, F).
        """
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1

        h = self.pool_start(x, batch, batch_size)  # (B, in_dim)
        if self._project_h0 is not None:
            h = self._project_h0(h)  # (B, units)
        wn = self.linear_trafo(x)  # (N, units)

        for _ in range(self.depth):
            hv = h[batch]  # (N, F)
            ev = torch.cat([hv, x], dim=-1)  # (N, 2*F)
            av = self.lay_alpha(ev)  # (N, 1)
            cont = self.pool_attention(wn, av, batch, batch_size)  # (B, F)
            cont = self.final_activ(cont)
            h = self.gru(cont, h)

        return h


class PoolingNodesGRU(nn.Module):
    """GRU-based graph pooling.

    Converts disjoint node features to padded batched format, processes through
    a GRU layer, and returns the last hidden state as the graph embedding.

    Note: This pooling is NOT permutation-invariant since GRU processes nodes
    sequentially. Matches Keras PoolingNodesGRU from CMPNN.
    """

    def __init__(self, units: int):
        """Initialize GRU pooling.

        Args:
            units: GRU hidden dimension.
        """
        super().__init__()
        self.units = units
        self.gru = nn.GRU(input_size=units, hidden_size=units, batch_first=True)

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                batch_size: int = None) -> torch.Tensor:
        """Pool node features using GRU.

        Args:
            x: Node features of shape (N, F) where F == units.
            batch: Batch assignment of shape (N,). Must be sorted (PyG default).
            batch_size: Number of graphs.

        Returns:
            Graph-level features of shape (B, units).
        """
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1

        device = x.device
        F = x.size(-1)

        # Count nodes per graph
        counts = torch.zeros(batch_size, dtype=torch.long, device=device)
        counts.scatter_add_(0, batch, torch.ones(x.size(0), dtype=torch.long, device=device))
        max_nodes = int(counts.max().item())

        # Convert disjoint to padded batched format
        # In PyG, batch is sorted: [0,0,...,1,1,...,2,2,...]
        cum_counts = torch.zeros(batch_size + 1, dtype=torch.long, device=device)
        cum_counts[1:] = counts.cumsum(0)
        positions = torch.arange(x.size(0), device=device) - cum_counts[batch]

        padded = torch.zeros(batch_size, max_nodes, F, device=device, dtype=x.dtype)
        padded[batch, positions] = x

        # Pack sequences for efficient GRU processing
        lengths = counts.clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(
            padded, lengths, batch_first=True, enforce_sorted=False
        )
        _, h_n = self.gru(packed)

        return h_n.squeeze(0)  # (B, units)


class PoolingSet2SetEncoder(nn.Module):
    """Set2Set encoder pooling (Vinyals et al. 2016, used in NMPNN).

    Uses LSTM to iteratively attend to node features.
    """

    def __init__(self, channels: int, T: int = 3, pooling_method: str = "mean",
                 init_qstar: str = "mean"):
        """Initialize Set2Set encoder.

        Args:
            channels: LSTM hidden dimension.
            T: Number of processing steps.
            pooling_method: Reduction method for attention ('mean', 'sum', etc.).
            init_qstar: How to initialize q0 ('mean', 'sum', 'zero').
        """
        super().__init__()
        self.channels = channels
        self.T = T
        self.pooling_method = pooling_method
        self.init_qstar = init_qstar
        self.lstm = nn.LSTMCell(2 * channels, channels)

        self._reduce_map = {
            "sum": lambda x, dim, keepdims: x.sum(dim=dim, keepdim=keepdims),
            "mean": lambda x, dim, keepdims: x.mean(dim=dim, keepdim=keepdims),
            "max": lambda x, dim, keepdims: x.max(dim=dim, keepdim=keepdims).values,
            "min": lambda x, dim, keepdims: x.min(dim=dim, keepdim=keepdims).values,
            "var": lambda x, dim, keepdims: x.var(dim=dim, keepdim=keepdims),
        }
        if pooling_method not in self._reduce_map:
            raise ValueError(f"Unknown pooling_method '{pooling_method}'")
        self._reduce = self._reduce_map[pooling_method]

    def forward(self, x: torch.Tensor, batch: torch.Tensor,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features (N, F) where F == channels.
            batch: Batch assignment (N,).
            batch_size: Number of graphs.

        Returns:
            Pooled features (B, 2*channels).
        """
        from kgcnn_torch.ops.scatter import scatter_reduce_sum, scatter_reduce_max
        if batch_size is None:
            batch_size = int(batch.max().item()) + 1

        # Match Keras kgcnn PoolingSet2SetEncoder:
        # - q_star is the state vector (B, 2*channels)
        # - each iteration computes q = LSTM(q_star) with *fresh zero* initial state
        # - then updates q_star from q via attention over nodes

        def stable_softmax_per_graph(et: torch.Tensor) -> torch.Tensor:
            et_max = scatter_reduce_max(batch, et, batch_size)  # (B, 1)
            et = et - et_max[batch]
            at = torch.exp(et)
            at_sum = scatter_reduce_sum(batch, at, batch_size).clamp(min=1e-8)
            return at / at_sum[batch]

        # Initialize q_star
        if self.init_qstar in ("0", "zero", "zeros"):
            q_star = torch.zeros(batch_size, 2 * self.channels, device=x.device, dtype=x.dtype)
        else:
            # Initialize q via pooling, then compute q_star = [q, r(q)]
            q = scatter_reduce_sum(batch, x, batch_size)
            if self.init_qstar in ("mean",):
                ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
                counts = scatter_reduce_sum(batch, ones, batch_size).clamp(min=1)
                q = q / counts
            qt = q[batch]
            et = self._reduce(x * qt, dim=1, keepdims=True)
            at = stable_softmax_per_graph(et)
            rt = scatter_reduce_sum(batch, x * at, batch_size)
            q_star = torch.cat([q, rt], dim=-1)

        q = torch.zeros(batch_size, self.channels, device=x.device, dtype=x.dtype)
        rt = torch.zeros_like(q)

        for _ in range(self.T):
            h0 = torch.zeros(batch_size, self.channels, device=x.device, dtype=x.dtype)
            c0 = torch.zeros_like(h0)
            q, _c = self.lstm(q_star, (h0, c0))  # (B, channels)

            qt = q[batch]  # (N, channels)
            et = self._reduce(x * qt, dim=1, keepdims=True)  # (N, 1)
            at = stable_softmax_per_graph(et)
            rt = scatter_reduce_sum(batch, x * at, batch_size)  # (B, channels)

            q_star = torch.cat([q, rt], dim=-1)  # (B, 2*channels)

        return q_star  # (B, 2*channels)
