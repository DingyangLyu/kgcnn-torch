"""Multi-layer perceptron for graph neural networks."""
import torch
import torch.nn as nn
from kgcnn_torch.ops.activ import get_activation


def _as_list(val, depth, name="parameter"):
    """Convert a scalar or list to a list of length ``depth``."""
    if isinstance(val, (list, tuple)):
        if len(val) != depth:
            raise ValueError(
                f"{name} list length ({len(val)}) must match "
                f"units length ({depth})")
        return list(val)
    return [val] * depth


def _make_norm(technique, out_dim):
    """Create a normalization module for the given technique and dimension.

    Returns:
        Tuple of (nn.Module, bool) where bool indicates if the norm requires
        (batch, batch_size) arguments in forward.
    """
    if technique in ("batch", "BatchNormalization"):
        return nn.BatchNorm1d(out_dim, momentum=0.01, eps=0.001), False
    elif technique in ("layer", "LayerNormalization"):
        return nn.LayerNorm(out_dim), False
    elif technique in ("graph", "graph_instance", "GraphNormalization", "GraphInstanceNormalization"):
        from kgcnn_torch.layers.norm import GraphNormalization
        mean_shift = technique in ("graph", "GraphNormalization")
        return GraphNormalization(out_dim, mean_shift=mean_shift), True
    elif technique in ("graph_batch", "GraphBatchNormalization"):
        from kgcnn_torch.layers.norm import GraphBatchNorm
        return GraphBatchNorm(out_dim), True
    elif technique in ("graph_layer", "GraphLayerNormalization"):
        from kgcnn_torch.layers.norm import GraphLayerNorm
        return GraphLayerNorm(out_dim), False
    elif technique == "group":
        num_groups = min(32, out_dim)
        while out_dim % num_groups != 0 and num_groups > 1:
            num_groups -= 1
        return nn.GroupNorm(num_groups, out_dim), False
    elif technique == "unit_norm":
        return _UnitNorm(), False
    else:
        raise ValueError(
            f"Unknown normalization_technique '{technique}'. "
            f"Supported: 'batch', 'layer', 'graph', 'graph_instance', "
            f"'graph_batch', 'graph_layer', 'group', 'unit_norm'."
        )


class _UnitNorm(nn.Module):
    """L2 unit normalization along the feature dimension."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x, p=2, dim=-1)


class MLP(nn.Module):
    """Multi-layer perceptron with optional normalization and dropout.

    Supports graph-aware normalization when normalization_technique is set to
    'graph', 'graph_batch', 'graph_layer', or 'graph_instance'. In that case,
    forward() accepts optional `batch` and `batch_size` arguments.

    All constructor parameters (except ``units`` and ``input_dim``) accept
    either a single value (applied to every layer) or a list of per-layer
    values matching the length of ``units``.

    This replaces both MLP and GraphMLP from the Keras version.
    """

    def __init__(self, units: list, input_dim: int,
                 activation="linear",
                 use_bias=True,
                 use_dropout=False,
                 dropout_rate=0.0,
                 use_normalization=False,
                 normalization_technique="batch"):
        """Initialize MLP.

        Args:
            units: List of hidden dimensions for each layer.
            input_dim: Input feature dimension.
            activation: Activation function name (str) or list of per-layer
                activation names. If a single string, the same activation is
                applied to every layer. Use ``"linear"`` for no activation.
            use_bias: Whether to use bias in linear layers. Bool or list of bool.
            use_dropout: Whether to use dropout. Bool or list of bool.
            dropout_rate: Dropout rate. Float or list of float.
            use_normalization: Whether to use normalization layers. Bool or list
                of bool.
            normalization_technique: Normalization type. String or list of
                strings. Supported: 'batch', 'layer', 'graph', 'graph_batch',
                'graph_layer', 'graph_instance'.
        """
        super().__init__()
        if isinstance(units, int):
            units = [units]

        self._depth = len(units)

        # Resolve all per-layer parameters.
        act_list = _as_list(activation, self._depth, "activation")
        bias_list = _as_list(use_bias, self._depth, "use_bias")
        do_list = _as_list(use_dropout, self._depth, "use_dropout")
        rate_list = _as_list(dropout_rate, self._depth, "dropout_rate")
        norm_list = _as_list(use_normalization, self._depth, "use_normalization")
        tech_list = _as_list(normalization_technique, self._depth, "normalization_technique")

        self._needs_batch = any(
            n and t in ("graph", "graph_instance", "graph_batch", "GraphBatchNormalization")
            for n, t in zip(norm_list, tech_list)
        )

        self.linears = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._norm_is_graph = []

        in_dim = input_dim
        for idx, out_dim in enumerate(units):
            self.linears.append(nn.Linear(in_dim, out_dim, bias=bias_list[idx]))

            if do_list[idx] and rate_list[idx] > 0:
                self.dropouts.append(nn.Dropout(p=rate_list[idx]))
            else:
                self.dropouts.append(nn.Identity())

            if norm_list[idx]:
                norm_mod, is_graph = _make_norm(tech_list[idx], out_dim)
                self.norms.append(norm_mod)
                self._norm_is_graph.append(is_graph)
            else:
                self.norms.append(nn.Identity())
                self._norm_is_graph.append(False)

            self.activations.append(get_activation(act_list[idx]))
            in_dim = out_dim

        self.output_dim = units[-1] if units else input_dim

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (..., input_dim).
            batch: Optional batch assignment of shape (N,) for graph normalization.
            batch_size: Optional number of graphs in batch.

        Returns:
            Output tensor of shape (..., units[-1]).
        """
        for i in range(self._depth):
            x = self.linears[i](x)
            x = self.dropouts[i](x)
            if self._norm_is_graph[i]:
                x = self.norms[i](x, batch, batch_size)
            else:
                x = self.norms[i](x)
            x = self.activations[i](x)
        return x


class RelationalMLP(nn.Module):
    """MLP with relation-specific dense layers (RelationalDense).

    Each layer uses a different weight matrix per relation type.
    Used for models like RGCN and HDNNP2nd.

    Supports dropout and normalization matching the Keras RelationalMLP
    (which inherits from MLP).
    """

    def __init__(self, units: list, input_dim: int, num_relations: int,
                 num_bases: int = None, num_blocks: int = None,
                 activation="linear", use_bias=True,
                 use_dropout=False, dropout_rate=0.0,
                 use_normalization=False, normalization_technique="batch"):
        """Initialize RelationalMLP.

        Args:
            units: List of hidden dimensions for each layer.
            input_dim: Input feature dimension.
            num_relations: Number of relation types.
            num_bases: Basis decomposition parameter for RelationalDense.
            num_blocks: Block-diagonal decomposition for RelationalDense.
            activation: Activation function name. Str or list of per-layer str.
            use_bias: Whether to use bias in layers.
            use_dropout: Whether to use dropout. Bool or list of bool.
            dropout_rate: Dropout rate. Float or list of float.
            use_normalization: Whether to use normalization. Bool or list of bool.
            normalization_technique: Normalization type. Str or list of str.
        """
        super().__init__()
        from kgcnn_torch.layers.relational import RelationalDense
        if isinstance(units, int):
            units = [units]

        depth = len(units)
        act_list = _as_list(activation, depth, "activation")
        bias_list = _as_list(use_bias, depth, "use_bias")
        do_list = _as_list(use_dropout, depth, "use_dropout")
        rate_list = _as_list(dropout_rate, depth, "dropout_rate")
        norm_list = _as_list(use_normalization, depth, "use_normalization")
        tech_list = _as_list(normalization_technique, depth, "normalization_technique")

        self._depth = depth
        self._needs_batch = any(
            n and t in ("graph", "graph_instance", "graph_batch", "GraphBatchNormalization")
            for n, t in zip(norm_list, tech_list)
        )

        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.norms = nn.ModuleList()
        self._norm_is_graph = []

        in_dim = input_dim
        for idx, out_dim in enumerate(units):
            self.layers.append(RelationalDense(
                in_dim, out_dim, num_relations,
                num_bases=num_bases, num_blocks=num_blocks,
                activation=None, use_bias=bias_list[idx]
            ))

            if do_list[idx] and rate_list[idx] > 0:
                self.dropouts.append(nn.Dropout(p=rate_list[idx]))
            else:
                self.dropouts.append(nn.Identity())

            if norm_list[idx]:
                norm_mod, is_graph = _make_norm(tech_list[idx], out_dim)
                self.norms.append(norm_mod)
                self._norm_is_graph.append(is_graph)
            else:
                self.norms.append(nn.Identity())
                self._norm_is_graph.append(False)

            self.activations.append(get_activation(act_list[idx]))
            in_dim = out_dim

        self.output_dim = units[-1] if units else input_dim

    def forward(self, x: torch.Tensor, relations: torch.Tensor,
                batch: torch.Tensor = None,
                batch_size: int = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features of shape (N, input_dim).
            relations: Relation type per sample of shape (N,).
            batch: Optional batch assignment for graph normalization.
            batch_size: Optional number of graphs in batch.

        Returns:
            Output features of shape (N, units[-1]).
        """
        for i in range(self._depth):
            x = self.layers[i](x, relations)
            x = self.dropouts[i](x)
            if self._norm_is_graph[i]:
                x = self.norms[i](x, batch, batch_size)
            else:
                x = self.norms[i](x)
            x = self.activations[i](x)
        return x


# Alias for compatibility with Keras version
GraphMLP = MLP
