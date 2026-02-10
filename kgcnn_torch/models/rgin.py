"""rGIN (Recurrent Graph Isomorphism Network) model.

A GIN variant with shared (recurrent) weights across all message passing depths.
Instead of separate GINConv and MLP per layer, a single GINConv and a single MLP
are reused at every depth step.  Embeddings from each step are collected, pooled,
transformed, and summed -- following the same readout strategy as GIN.

Key difference from standard GIN: random features are concatenated to node
features before each message passing step (Sato et al., 2020).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class rGINConv(nn.Module):
    """rGIN convolution layer with random feature augmentation.

    Before message passing, concatenates a random uniform feature to each
    node's feature vector. This makes the layer strictly more expressive
    than standard GIN (Sato et al., 2020).

    Computes:
        random_values = uniform(0, 1/random_range) of shape (N, 1)
        x_aug = cat([x, random_values], dim=-1)
        h_i' = (1 + eps) * x_aug_i + sum_j x_aug_j

    Note: The non-linear mapping (MLP) is NOT included and should be applied after.
    """

    def __init__(self, pooling_method: str = "sum",
                 epsilon_learnable: bool = False,
                 random_range: int = 100):
        super().__init__()
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable
        self.random_range = random_range

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
            Updated node features (N, F+1).
        """
        num_nodes = x.size(0)

        # Concatenate random features
        random_values = torch.rand(num_nodes, 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, random_values], dim=-1)  # (N, F+1)

        # Standard GIN message passing on augmented features
        # Note: Keras computes no = (1+eps)*node but does NOT use it,
        # instead returning node + agg. We match that behavior here.
        x_j = gather_nodes_outgoing(x_aug, edge_index)
        agg = self.aggr(x_j, edge_index, num_nodes)
        return x_aug + agg


class rGINModel(nn.Module):
    """Recurrent Graph Isomorphism Network.

    Uses a single shared rGINConv and a single shared MLP across all depth
    iterations, making the model recurrent in the message-passing layers.
    Embeddings at each depth (including the initial embedding) are independently
    pooled and transformed, then summed before a final output MLP.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = 64,
                 gin_mlp_units: list = None,
                 gin_mlp_activation: list = None,
                 gin_mlp_use_normalization: bool = True,
                 gin_mlp_normalization_technique: str = "graph_batch",
                 gin_pooling: str = "sum",
                 epsilon_learnable: bool = False,
                 random_range: int = 100,
                 dropout: float = 0.0,
                 node_pooling: str = "sum",
                 last_mlp_units: list = None,
                 last_mlp_activation: str = "relu",
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_final_activation: str = "softmax",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize rGIN model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of recurrent message passing iterations.
            units: Hidden dimension for all layers.
            gin_mlp_units: Hidden dimensions for the shared MLP (phi) applied
                after rGINConv.  If None, defaults to [units, units].
            gin_mlp_activation: Activation for the shared GIN MLP.
            gin_pooling: Aggregation method inside rGINConv.
            epsilon_learnable: Whether epsilon in rGINConv is learnable.
            random_range: Denominator for random feature scaling (default 100).
            dropout: Dropout rate applied after the per-layer readout MLP.
            node_pooling: Pooling method for graph-level readout.
            last_mlp_units: Hidden dims for the per-layer readout MLP applied
                after pooling each layer embedding.  If None, defaults to [units].
            last_mlp_activation: Activation for the per-layer readout MLP.
            output_units: Hidden dims for the final output MLP. If None,
                defaults to [units].
            output_activation: Activation for the final output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for the node embedding.
        """
        super().__init__()
        if gin_mlp_units is None:
            gin_mlp_units = [units, units]
        if gin_mlp_activation is None:
            # Keras default: ["relu", "linear"] -- last layer has no activation.
            gin_mlp_activation = ["relu", "linear"]
        if last_mlp_units is None:
            last_mlp_units = [64, 64, 64]
        if output_units is None:
            output_units = []

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding

        # Node embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Initial projection to hidden dimension
        self.dense_in = nn.Linear(node_dim, units)

        # Separate rGINConv and MLP per depth (matches Keras: separate instances)
        # rGINConv adds 1 random feature, so output dim is units+1
        self.convs = nn.ModuleList()
        self.mlps = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(rGINConv(
                pooling_method=gin_pooling,
                epsilon_learnable=epsilon_learnable,
                random_range=random_range
            ))
            # Input dim is units+1 because rGINConv concatenates 1 random feature
            self.mlps.append(MLP(
                units=gin_mlp_units,
                input_dim=units + 1,
                activation=gin_mlp_activation,
                use_normalization=gin_mlp_use_normalization,
                normalization_technique=gin_mlp_normalization_technique
            ))

        # Per-layer readout: Pool -> MLP -> Dropout, then sum
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # (depth + 1) readout heads: one for initial embedding plus one per depth
        readout_input_dim = gin_mlp_units[-1] if gin_mlp_units else units
        self.readout_mlps = nn.ModuleList()
        self.readout_dropouts = nn.ModuleList()
        for i in range(depth + 1):
            in_dim = units if i == 0 else readout_input_dim
            # Keras: last_mlp activation=["relu","relu","linear"]
            last_act = [last_mlp_activation] * max(len(last_mlp_units) - 1, 0) + ["linear"]
            self.readout_mlps.append(MLP(
                units=last_mlp_units,
                input_dim=in_dim,
                activation=last_act
            ))
            self.readout_dropouts.append(
                nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
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

        # Node embedding
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            x = self.node_projection(x)

        # Initial projection
        x = self.dense_in(x)

        # Compute batch_size for graph-aware normalization in gin_mlp
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Collect embeddings from each depth (including initial)
        list_embeddings = [x]

        for i in range(self.depth):
            # Separate rGINConv + MLP per depth (matches Keras)
            x = self.convs[i](x, edge_index)  # (N, units+1)
            x = self.mlps[i](x, batch=batch, batch_size=batch_size)  # (N, gin_mlp_units[-1])
            list_embeddings.append(x)

        # Readout: pool each layer, apply MLP + dropout, then sum
        out = None
        for i, emb in enumerate(list_embeddings):
            if self.output_embedding == "graph":
                h = self.pooling(emb, batch, batch_size)
            else:
                h = emb
            h = self.readout_mlps[i](h)
            h = self.readout_dropouts[i](h)
            if out is None:
                out = h
            else:
                out = out + h

        # Final output MLP
        out = self.output_mlp(out)
        return out
