"""GCN (Graph Convolutional Network) model.

Reference: Kipf & Welling, Semi-Supervised Classification with Graph Convolutional Networks (2017).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.conv import GCNConv
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GCNModel(nn.Module):
    """Graph Convolutional Network.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_weight: Edge weights (M, 1).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 1,
                 depth: int = 3,
                 gcn_units: int = 100,
                 gcn_activation: str = "relu",
                 gcn_pooling: str = "sum",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_final_activation: str = "sigmoid",
                 output_use_bias: list = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize GCN model.

        Args:
            node_dim: Dimension of node features after embedding.
            edge_dim: Dimension of edge features (not used currently, weights are 1D).
            depth: Number of GCN layers.
            gcn_units: Hidden dimension for GCN convolution layers.
            gcn_activation: Activation function for GCN layers.
            gcn_pooling: Pooling method in GCN aggregation.
            node_pooling: Pooling method for graph-level readout.
            output_units: List of hidden dimensions for output MLP. If None, [25, 10].
            output_activation: Activation for output MLP hidden layers.
            output_final_activation: Activation for output MLP final layer.
            output_use_bias: List of bools for per-layer bias in output MLP.
                If None, defaults to [True, ..., True, False] (last layer no bias,
                matching Keras [True, True, False]).
            num_targets: Number of output targets.
            use_node_embedding: Whether to use embedding for integer node features.
            num_embeddings: Vocabulary size for node embedding.
        """
        super().__init__()
        if output_units is None:
            output_units = [25, 10]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        self.dense_in = nn.Linear(node_dim, gcn_units)

        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(GCNConv(
                in_features=gcn_units, out_features=gcn_units,
                pooling_method=gcn_pooling, activation=gcn_activation
            ))

        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=gcn_units,
            activation=out_act,
            use_bias=output_use_bias
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            x = self.node_projection(x)
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        # GCN expects scalar edge weights; only use edge_attr when it is scalar.
        if edge_weight is None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dim() == 1:
                edge_weight = data.edge_attr
            elif data.edge_attr.dim() == 2 and data.edge_attr.size(-1) == 1:
                edge_weight = data.edge_attr
        batch = data.batch

        x = self.dense_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(x, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out


class GCNWeightedModel(nn.Module):
    """GCN model variant that supports node weights for weighted graph pooling.

    This extends the standard GCN architecture by accepting per-node weights
    that are used during the global graph-level readout. This is useful when
    different nodes should contribute differently to the graph-level prediction,
    for example when using importance weighting or attention-based pooling.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_weight: Edge weights (M, 1) - entries of scaled adjacency.
        - data.node_weight: Per-node weights (N, 1) for weighted pooling.
        - data.batch: Batch assignment (N,).

    Args:
        node_dim: Dimension of node features after embedding.
        edge_dim: Dimension of edge features (weights are 1D for GCN).
        depth: Number of GCN layers.
        gcn_units: Hidden dimension for GCN convolution layers.
        gcn_activation: Activation function for GCN layers.
        gcn_pooling: Pooling method in GCN aggregation.
        node_pooling: Pooling method for weighted graph-level readout.
        output_units: List of hidden dimensions for output MLP.
        output_activation: Activation for output MLP.
        num_targets: Number of output targets.
        use_node_embedding: Whether to use embedding for integer node features.
        num_embeddings: Vocabulary size for node embedding.
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 1,
                 depth: int = 3,
                 gcn_units: int = 100,
                 gcn_activation: str = "relu",
                 gcn_pooling: str = "sum",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_final_activation: str = "sigmoid",
                 output_use_bias: list = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        super().__init__()
        if output_units is None:
            output_units = [25, 10]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        self.dense_in = nn.Linear(node_dim, gcn_units)

        self.convs = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(GCNConv(
                in_features=gcn_units, out_features=gcn_units,
                pooling_method=gcn_pooling, activation=gcn_activation
            ))

        self.pooling = PoolingWeightedNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=gcn_units,
            activation=out_act,
            use_bias=output_use_bias
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object. Must have 'node_weight' attribute
                of shape (N, 1) for weighted pooling.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            x = self.node_projection(x)
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') else None
        # GCN expects scalar edge weights; only use edge_attr when it is scalar.
        if edge_weight is None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            if data.edge_attr.dim() == 1:
                edge_weight = data.edge_attr
            elif data.edge_attr.dim() == 2 and data.edge_attr.size(-1) == 1:
                edge_weight = data.edge_attr
        node_weight = data.node_weight
        batch = data.batch

        x = self.dense_in(x)

        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)

        # Weighted graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(x, node_weight, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out
