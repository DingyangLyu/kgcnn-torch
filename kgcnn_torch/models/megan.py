"""MEGAN (Multi Explanation Graph Attention Network) model.

Reference: Munch et al., MEGAN: Multi Explanation Graph Attention Network (2023).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.attention import MultiHeadGATV2Layer
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingWeightedNodes
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class MEGANModel(nn.Module):
    """Multi Explanation Graph Attention Network.

    Uses multi-head GATv2 attention layers to produce both node embeddings and
    per-edge attention logits. Edge importance is derived from sigmoid-activated
    attention logits concatenated across layers and summed along the feature dim.
    Node importance is computed via a learned importance sub-network multiplied
    by pooled edge importances. The final readout uses K importance channels for
    weighted graph pooling, concatenates all channels, and feeds through an
    output MLP.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - Optional: data.edge_attr: Edge features (M, edge_dim).
    """

    def __init__(self,
                 node_dim: int = 64,
                 units: list = None,
                 num_heads: int = 2,
                 depth: int = 3,
                 attention_activation: str = "leaky_relu2",
                 use_edge_features: bool = False,
                 edge_dim: int = 0,
                 concat_heads: bool = True,
                 importance_channels: int = 2,
                 importance_units: list = None,
                 importance_activation: str = "relu",
                 final_units: list = None,
                 final_activation: str = "linear",
                 use_bias: bool = True,
                 final_pooling: str = "sum",
                 dropout_rate: float = 0.0,
                 final_dropout_rate: float = 0.0,
                 regression_reference: float = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize MEGAN model.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            units: Per-layer attention hidden units. If None, [node_dim] * depth.
            num_heads: Number of attention heads per layer.
            depth: Number of multi-head GATv2 layers.
            attention_activation: Activation for attention layers.
            use_edge_features: Whether to use edge features in attention.
            edge_dim: Edge feature dimension.
            concat_heads: Whether to concatenate heads (True) or average (False).
            importance_channels: Number of explanation/importance channels (K).
            importance_units: Hidden dims for importance MLP. If None, [] (empty).
            importance_activation: Activation for intermediate importance layers.
            final_units: Complete list of units for output MLP including the final
                output layer. If None, defaults to [num_targets]. The last element
                determines the output dimension.
            final_activation: Activation for the last layer of the output MLP.
            use_bias: Whether to use bias in layers. Keras MEGAN applies this
                to ALL final Dense layers (including the last one).
            final_pooling: Pooling method for weighted node aggregation.
            dropout_rate: Dropout rate applied after each attention layer.
            final_dropout_rate: Dropout rate applied before the output MLP.
            regression_reference: Optional reference value added to model output.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
        """
        super().__init__()
        if units is None:
            units = [node_dim] * depth
        if importance_units is None:
            importance_units = []
        if final_units is None:
            final_units = [num_targets]

        self.output_embedding = output_embedding
        self.use_node_embedding = use_node_embedding
        self.use_edge_features = use_edge_features
        self.depth = depth
        self.num_heads = num_heads
        self.concat_heads = concat_heads
        self.importance_channels = importance_channels
        self.regression_reference = regression_reference

        if importance_channels != num_heads:
            raise ValueError(
                f"importance_channels ({importance_channels}) must equal num_heads "
                f"({num_heads}) because node_importances is computed as the element-wise "
                f"product of importance MLP output and pooled edge importances."
            )

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            # Keras MEGAN directly consumes float node_attributes without an explicit projection
            # when the feature dimension already matches `node_dim`.
            self.node_projection = nn.Linear(node_input_dim, node_dim) if node_input_dim != node_dim else None

        # Build multi-head GATv2 layers
        self.attention_layers = nn.ModuleList()
        in_dim = node_dim
        for i in range(depth):
            self.attention_layers.append(MultiHeadGATV2Layer(
                in_features=in_dim,
                units=units[i],
                num_heads=num_heads,
                activation=attention_activation,
                use_edge_features=use_edge_features,
                edge_dim=edge_dim,
                concat_heads=concat_heads,
                use_final_activation=True,
                normalize_softmax=False
            ))
            if concat_heads:
                in_dim = units[i] * num_heads
            else:
                in_dim = units[i]

        # Inter-layer dropout (matching Keras use_dropout/dropout_rate)
        self.layer_dropouts = nn.ModuleList([
            nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
            for _ in range(depth)
        ])
        # Final dropout before output MLP
        self.final_dropout = nn.Dropout(p=final_dropout_rate) if final_dropout_rate > 0 else nn.Identity()

        # Node feature dimension after all attention layers
        final_node_dim = in_dim

        # Edge importance: each layer produces (M, num_heads, 1) attention logits.
        # We concatenate along last axis across layers -> (M, depth * num_heads)
        # then sum along last axis -> (M,) -> sigmoid -> (M, 1) edge importances.
        # (This matches Keras: lay_concat_alphas on axis=-1, then ops.sum on axis=-1.)

        # Pool edge importances to nodes (both directions), average
        self.pool_edges_in = AggregateLocalEdges(pooling_method='mean')
        self.pool_edges_out = AggregateLocalEdges(pooling_method='mean')

        # Node importance sub-network: maps node features to K importance channels
        # Keras: final layer has "linear" activation, sigmoid applied externally
        imp_units = importance_units + [importance_channels]
        imp_act = [importance_activation] * len(importance_units) + ["linear"]
        self.importance_mlp = MLP(
            units=imp_units,
            input_dim=final_node_dim,
            activation=imp_act
        )

        # Weighted pooling per importance channel
        self.pooling = PoolingWeightedNodes(pooling_method=final_pooling)

        # Output MLP: takes concatenated pooled features (B, final_node_dim * K) -> (B, num_targets)
        # Keras uses "relu" for all intermediate layers, only the last layer uses final_activation.
        # Keras also sets use_bias=False on the last layer of the output MLP.
        output_mlp_input_dim = final_node_dim * importance_channels if output_embedding == "graph" else final_node_dim
        output_acts = ["relu"] * len(final_units)
        if output_acts:
            output_acts[-1] = final_activation
        # Keras MEGAN uses `use_bias` (constructor param) for ALL final Dense layers,
        # ignoring the prepared per-layer bias list. Match that behavior.
        output_biases = [use_bias] * len(final_units)
        self.output_mlp = MLP(
            units=final_units,
            input_dim=output_mlp_input_dim,
            activation=output_acts,
            use_bias=output_biases
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            if self.node_projection is not None:
                x = self.node_projection(x)

        # Multi-head GATv2 attention layers
        # Collect attention logits per layer
        attention_logits_list = []
        for i, layer in enumerate(self.attention_layers):
            x, a_ijs = layer(x, edge_index, edge_attr)
            x = self.layer_dropouts[i](x)
            # a_ijs: (M, num_heads, 1) -> squeeze last dim -> (M, num_heads)
            attention_logits_list.append(a_ijs.squeeze(-1))

        # Edge importance per explanation channel:
        # Keras concatenates alpha tensors on last axis and sums across layers,
        # preserving the head/channel axis.
        # Here each element in attention_logits_list is (M, num_heads), so stack
        # to (M, num_heads, depth), sum over depth -> (M, num_heads), then sigmoid.
        all_attn = torch.stack(attention_logits_list, dim=-1)  # (M, num_heads, depth)
        edge_importance = torch.sigmoid(all_attn.sum(dim=-1))  # (M, num_heads)

        # Pool edge importances to nodes (both directions), average
        # Keras: pooling_index=0 pools to edge_index[0] (source), pooling_index=1 pools to edge_index[1] (target)
        # In our PyG convention, AggregateLocalEdges pools to edge_index[1] (target).
        # To pool to source (index 0), we swap edge_index rows.
        num_nodes = x.size(0)
        edge_index_reversed = torch.stack([edge_index[1], edge_index[0]], dim=0)

        # Pool to target nodes (index 1) - this is "in" direction
        pooled_edges_in = self.pool_edges_in(edge_importance, edge_index, num_nodes)
        # Pool to source nodes (index 0) - this is "out" direction
        pooled_edges_out = self.pool_edges_out(edge_importance, edge_index_reversed, num_nodes)
        # Average both directions
        pooled_edges = (pooled_edges_in + pooled_edges_out) / 2.0  # (N, num_heads)

        # Node importance: sigmoid(MLP(x)) * pooled_edges
        node_importances_tilde = torch.sigmoid(self.importance_mlp(x))  # (N, K)
        node_importances = node_importances_tilde * pooled_edges  # (N, K)

        if self.output_embedding == "graph":
            # Weighted graph pooling per importance channel, then concatenate
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            channel_outputs = []
            for k in range(self.importance_channels):
                # Weight nodes by k-th importance channel
                w_k = node_importances[:, k:k + 1]  # (N, 1)
                pooled = self.pooling(x, w_k, batch, batch_size)  # (B, final_node_dim)
                channel_outputs.append(pooled)

            # Concatenate all channels: (B, final_node_dim * K)
            out = torch.cat(channel_outputs, dim=-1)
        else:
            # Node-level output: use final node embeddings directly
            out = x

        # Final dropout and output MLP
        out = self.final_dropout(out)
        out = self.output_mlp(out)
        if self.regression_reference is not None:
            out = out + self.regression_reference
        return out

    def forward_explanations(self, data) -> dict:
        """Forward pass returning explanations alongside predictions.

        Args:
            data: PyG Data batch object.

        Returns:
            Dictionary with keys:
                - 'output': Graph-level predictions (B, num_targets).
                - 'edge_importance': Per-edge importance (M, 1).
                - 'node_importance': Per-node importance channels (N, K).
        """
        edge_index = data.edge_index
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') else None
        batch = data.batch

        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            x = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            if self.node_projection is not None:
                x = self.node_projection(x)

        attention_logits_list = []
        for i, layer in enumerate(self.attention_layers):
            x, a_ijs = layer(x, edge_index, edge_attr)
            x = self.layer_dropouts[i](x)
            attention_logits_list.append(a_ijs.squeeze(-1))

        all_attn = torch.stack(attention_logits_list, dim=-1)  # (M, num_heads, depth)
        edge_importance = torch.sigmoid(all_attn.sum(dim=-1))  # (M, num_heads)

        num_nodes = x.size(0)
        edge_index_reversed = torch.stack([edge_index[1], edge_index[0]], dim=0)
        pooled_edges_in = self.pool_edges_in(edge_importance, edge_index, num_nodes)
        pooled_edges_out = self.pool_edges_out(edge_importance, edge_index_reversed, num_nodes)
        pooled_edges = (pooled_edges_in + pooled_edges_out) / 2.0  # (N, num_heads)

        node_importances_tilde = torch.sigmoid(self.importance_mlp(x))
        node_importances = node_importances_tilde * pooled_edges

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        channel_outputs = []
        for k in range(self.importance_channels):
            w_k = node_importances[:, k:k + 1]
            pooled = self.pooling(x, w_k, batch, batch_size)
            channel_outputs.append(pooled)

        out = torch.cat(channel_outputs, dim=-1)
        out = self.final_dropout(out)
        out = self.output_mlp(out)
        if self.regression_reference is not None:
            out = out + self.regression_reference

        return {
            'output': out,
            'edge_importance': edge_importance,
            'node_importance': node_importances
        }
