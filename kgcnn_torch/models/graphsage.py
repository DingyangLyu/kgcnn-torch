"""GraphSAGE (Graph Sample and Aggregate) model.

Reference: Hamilton et al., Inductive Representation Learning on Large Graphs (2017).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges, AggregateLocalEdgesLSTM
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.norm import GraphLayerNorm
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GraphSAGEModel(nn.Module):
    """GraphSAGE model for graph-level property prediction.

    Implements the SAGE convolution where, at each layer, each node gathers
    messages from its neighbors via an edge MLP, aggregates them, concatenates
    the result with the node's own features, and updates via a node MLP.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Optional edge features (M, edge_dim).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = 64,
                 node_mlp_units: list = None,
                 edge_mlp_units: list = None,
                 edge_dim: int = 0,
                 use_edge_features: bool = True,
                 pooling_method: str = "mean",
                 node_pooling: str = "mean",
                 output_units: list = None,
                 output_use_bias: list = None,
                 num_targets: int = 1,
                 activation: str = "relu",
                 output_final_activation: str = "sigmoid",
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1,
                 use_edge_embedding: bool = False,
                 edge_num_embeddings: int = 0,
                 edge_embedding_dim: int = 0):
        """Initialize GraphSAGE model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of SAGE convolution layers.
            units: Hidden dimension for edge and node MLPs.
            edge_dim: Dimension of edge features (used only if use_edge_features is True).
            use_edge_features: Whether to concatenate edge features with source node
                features before the edge MLP.
            pooling_method: Aggregation method for neighborhood messages ('sum', 'mean', 'max').
            node_pooling: Pooling method for graph-level readout.
            output_units: List of hidden dimensions for the output MLP after graph pooling.
                If None, defaults to [25, 10].
            output_use_bias: List of bools for per-layer bias in output MLP.
                If None, defaults to [True, ..., True, False] (last layer no bias,
                matching Keras [True, True, False]).
            num_targets: Number of output targets.
            activation: Activation function name for all MLPs.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for node embedding.
        """
        super().__init__()
        if output_units is None:
            output_units = [25, 10]
        if node_mlp_units is None:
            node_mlp_units = [100, 50] if units == 64 else [max(2 * units, units), units]
        if edge_mlp_units is None:
            edge_mlp_units = [100, 50] if units == 64 else [max(2 * units, units), units]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.use_edge_features = use_edge_features
        self.use_edge_embedding = use_edge_embedding
        self.depth = depth

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Optional initial projection from node_dim to units (matches Keras dense_in).
        if node_dim != units:
            self.dense_in = nn.Linear(node_dim, units)
        else:
            self.dense_in = None

        # Optional edge embedding (Keras support)
        if use_edge_embedding and edge_num_embeddings > 0:
            self.edge_embedding_layer = nn.Embedding(edge_num_embeddings, edge_embedding_dim)
            keras_uniform_init_embedding_(self.edge_embedding_layer)
            edge_feat_dim = edge_embedding_dim
        else:
            self.edge_embedding_layer = None
            edge_feat_dim = edge_dim

        # Keras GraphSAGE builds per-layer MLPs with input dims that change after the first update.
        # Start with node_dim, then after each layer the node feature dim becomes node_mlp_units[-1].
        self.edge_proj = None
        if self.use_edge_features and (edge_feat_dim is None or edge_feat_dim <= 0):
            # Best-effort: if edge features are requested but no edge dim is provided, project to node feature dim.
            self.edge_proj = nn.LazyLinear(node_dim)
            edge_feat_dim = node_dim

        # Per-layer edge MLPs: transform source node features (optionally with edge features)
        # into messages.
        self.edge_mlps = nn.ModuleList()
        self.node_mlps = nn.ModuleList()
        self.aggregators = nn.ModuleList()
        self.layer_norms = nn.ModuleList()

        cur_node_dim = int(units) if self.dense_in is not None else int(node_dim)
        edge_out_dim = int(edge_mlp_units[-1])
        node_out_dim = int(node_mlp_units[-1])
        for _ in range(depth):
            if use_edge_features:
                edge_input_dim = cur_node_dim + int(edge_feat_dim)
            else:
                edge_input_dim = cur_node_dim
            # Keras default: activation=["relu", "linear"] (last layer linear)
            edge_act = [activation] * max(len(edge_mlp_units) - 1, 0) + ["linear"]
            self.edge_mlps.append(MLP(
                units=edge_mlp_units,
                input_dim=edge_input_dim,
                activation=edge_act
            ))
            # Node MLP takes concatenation of [node_features, aggregated_messages].
            node_act = [activation] * max(len(node_mlp_units) - 1, 0) + ["linear"]
            self.node_mlps.append(MLP(
                units=node_mlp_units,
                input_dim=cur_node_dim + edge_out_dim,
                activation=node_act
            ))
            if pooling_method == "lstm":
                self.aggregators.append(AggregateLocalEdgesLSTM(
                    units=edge_out_dim, input_dim=edge_out_dim))
            else:
                self.aggregators.append(AggregateLocalEdges(pooling_method=pooling_method))
            self.layer_norms.append(GraphLayerNorm(node_out_dim))
            cur_node_dim = node_out_dim

        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [activation] * len(output_units) + [output_final_activation]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=node_out_dim,
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
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        # Optional initial projection to hidden dim
        if self.dense_in is not None:
            x = self.dense_in(x)

        # Optional edge embedding
        if self.use_edge_embedding and self.edge_embedding_layer is not None and edge_attr is not None:
            # If edge_attr already contains embedded edge features (M, F), do NOT re-embed.
            # Only embed integer ids (M,) or (M,1).
            if edge_attr.dtype in (torch.int8, torch.int16, torch.int32, torch.int64) or (
                edge_attr.ndim == 1 and edge_attr.is_floating_point()
            ):
                if edge_attr.ndim > 1 and edge_attr.shape[-1] == 1:
                    edge_attr = edge_attr.reshape(-1)
                edge_attr = self.edge_embedding_layer(edge_attr.long())

        for i in range(self.depth):
            num_nodes = x.size(0)

            # Step a: Gather source node features for each edge.
            x_j = gather_nodes_outgoing(x, edge_index)  # (M, units)

            # Step b: Optionally concatenate with edge features.
            if self.use_edge_features and edge_attr is not None:
                if self.edge_proj is not None:
                    edge_feat = self.edge_proj(edge_attr)
                else:
                    edge_feat = edge_attr
                msg_input = torch.cat([x_j, edge_feat], dim=-1)  # (M, units + edge_dim_eff)
            else:
                msg_input = x_j  # (M, units)

            # Step c: Apply edge MLP to produce messages.
            messages = self.edge_mlps[i](msg_input)  # (M, units)

            # Step d: Aggregate messages to target nodes.
            agg = self.aggregators[i](messages, edge_index, num_nodes)  # (N, units)

            # Step e: Concatenate original node features with aggregated messages.
            x_cat = torch.cat([x, agg], dim=-1)  # (N, 2 * units)

            # Step f: Apply node MLP to update node features.
            x = self.node_mlps[i](x_cat)  # (N, units)

            # Step g: Layer normalization (matches Keras GraphLayerNormalization).
            x = self.layer_norms[i](x)

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(x, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out
