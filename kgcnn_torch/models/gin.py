"""GIN (Graph Isomorphism Network) model.

Reference: Xu et al., How Powerful are Graph Neural Networks? (2019).
GINE variant: Hu et al., Strategies for Pre-training Graph Neural Networks (2020).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.conv import GINConv, GINEConv
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GINModel(nn.Module):
    """Graph Isomorphism Network.

    Supports both GIN (no edge features) and GINE (with edge features) via
    the ``use_edge_features`` flag.  The architecture follows the original paper:
    embeddings are collected at every layer, each is independently pooled and
    transformed, then summed before a final output MLP.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - data.edge_attr: Edge features (M, F) -- required when use_edge_features=True.
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = 64,
                 gin_mlp_units: list = None,
                 gin_mlp_activation: str = "relu",
                 gin_mlp_use_normalization: bool = True,
                 gin_mlp_normalization_technique: str = "graph_batch",
                 gin_pooling: str = "sum",
                 epsilon_learnable: bool = False,
                 use_edge_features: bool = False,
                 edge_dim: int = 0,
                 gine_activation: str = "relu",
                 node_pooling: str = "sum",
                 last_mlp_units: list = None,
                 last_mlp_activation: str = "relu",
                 dropout_rate: float = 0.0,
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_final_activation: str = "softmax",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize GIN model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of GIN convolution + MLP blocks.
            units: Hidden dimension for GIN layers.
            gin_mlp_units: Hidden dimensions for the per-layer MLP (phi).
                If None, defaults to [units, units].
            gin_mlp_activation: Activation for the per-layer MLP.
            gin_pooling: Aggregation method inside GINConv.
            epsilon_learnable: Whether epsilon in GIN is learnable.
            use_edge_features: If True, use GINEConv instead of GINConv.
            edge_dim: Edge feature dimension (used only when use_edge_features=True).
                Edge features are projected to ``units`` via a linear layer.
            gine_activation: Activation used inside GINEConv.
            node_pooling: Pooling method for graph-level readout.
            last_mlp_units: Hidden dims for the per-layer readout MLP applied
                after pooling each layer embedding.  If None, defaults to [units].
            last_mlp_activation: Activation for the per-layer readout MLP.
            dropout_rate: Dropout rate applied after the per-layer readout MLP.
            output_units: Hidden dims for the final output MLP. If None, defaults
                to [units].
            output_activation: Activation for the final output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for the node embedding.
        """
        super().__init__()
        if gin_mlp_units is None:
            gin_mlp_units = [units, units]
        if last_mlp_units is None:
            last_mlp_units = [64, 64, 64]
        if output_units is None:
            output_units = []

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding
        self.use_edge_features = use_edge_features

        # Node embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Initial projection to hidden dimension
        self.dense_in = nn.Linear(node_dim, units)

        # Optional edge feature projection (for GINE)
        if use_edge_features and edge_dim > 0:
            self.edge_proj = nn.Linear(edge_dim, units)
        else:
            self.edge_proj = None

        # GIN / GINE convolution layers
        self.convs = nn.ModuleList()
        for _ in range(depth):
            if use_edge_features:
                self.convs.append(GINEConv(
                    pooling_method=gin_pooling,
                    epsilon_learnable=epsilon_learnable,
                    activation=gine_activation
                ))
            else:
                self.convs.append(GINConv(
                    pooling_method=gin_pooling,
                    epsilon_learnable=epsilon_learnable
                ))

        # Per-layer MLP (phi) applied after each GINConv
        # Keras default: activation=["relu", "linear"] (last layer has no activation)
        gin_act = [gin_mlp_activation] * max(len(gin_mlp_units) - 1, 0) + ["linear"]
        self.gin_mlps = nn.ModuleList()
        self._gin_mlp_needs_batch = gin_mlp_use_normalization and gin_mlp_normalization_technique in (
            "graph", "graph_instance", "graph_batch", "graph_layer"
        )
        for _ in range(depth):
            self.gin_mlps.append(MLP(
                units=gin_mlp_units,
                input_dim=units,
                activation=gin_act,
                use_normalization=gin_mlp_use_normalization,
                normalization_technique=gin_mlp_normalization_technique
            ))

        # Per-layer readout: Pool -> MLP -> Dropout, then sum
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # (depth + 1) readout heads: one for the initial embedding plus one per layer
        readout_input_dim = gin_mlp_units[-1] if gin_mlp_units else units
        self.readout_mlps = nn.ModuleList()
        self.readout_dropouts = nn.ModuleList()
        # Keras default: activation=["relu", "relu", "linear"] (last layer linear)
        last_act = [last_mlp_activation] * max(len(last_mlp_units) - 1, 0) + ["linear"]
        for i in range(depth + 1):
            in_dim = units if i == 0 else readout_input_dim
            self.readout_mlps.append(MLP(
                units=last_mlp_units,
                input_dim=in_dim,
                activation=last_act
            ))
            self.readout_dropouts.append(
                nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
            )

        # Final output MLP
        readout_out_dim = last_mlp_units[-1] if last_mlp_units else units
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=readout_out_dim,
            activation=out_act
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        edge_index = data.edge_index
        batch = data.batch

        # Edge features (for GINE)
        edge_attr = None
        if self.use_edge_features:
            edge_attr = data.edge_attr
            if self.edge_proj is not None:
                edge_attr = self.edge_proj(edge_attr)

        # Node embedding
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            x = self.node_projection(x)

        # Initial projection
        x = self.dense_in(x)

        # Collect embeddings from each layer (including initial)
        list_embeddings = [x]

        for i in range(self.depth):
            if self.use_edge_features:
                x = self.convs[i](x, edge_index, edge_attr)
            else:
                x = self.convs[i](x, edge_index)
            if self._gin_mlp_needs_batch:
                x = self.gin_mlps[i](x, batch=batch)
            else:
                x = self.gin_mlps[i](x)
            list_embeddings.append(x)

        # Readout: pool each layer, apply MLP + dropout, then sum
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = None
            for i, emb in enumerate(list_embeddings):
                h = self.pooling(emb, batch, batch_size)
                h = self.readout_mlps[i](h)
                h = self.readout_dropouts[i](h)
                if out is None:
                    out = h
                else:
                    out = out + h
        else:
            # Node-level output: use only the final layer embedding (matching Keras)
            out = self.readout_mlps[-1](list_embeddings[-1])
            out = self.readout_dropouts[-1](out)

        # Final output MLP
        out = self.output_mlp(out)
        return out
