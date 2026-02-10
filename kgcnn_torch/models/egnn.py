"""EGNN (E(n) Equivariant Graph Neural Network) model.

Reference: Satorras et al., E(n) Equivariant Graph Neural Networks (2021).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.norm import GraphLayerNorm
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class EGNNLayer(nn.Module):
    """Single E(n) equivariant message passing layer.

    Updates node features h and coordinates x simultaneously while maintaining
    E(n) equivariance of the coordinate updates.

    Implements:
        1. Edge model:  m_ij = phi_e(h_i, h_j, ||x_i - x_j||^2, [a_ij])
        2. Coord model: x_i += C * sum_j (x_i - x_j) * phi_x(m_ij)
        3. Node model:  h_i = h_i + phi_h(h_i, sum_j m_ij)
    """

    def __init__(self,
                 units: int = 64,
                 edge_mlp_units: list = None,
                 edge_mlp_activation: str = "swish",
                 coord_mlp_units: list = None,
                 coord_mlp_activation: str = "swish",
                 node_mlp_units: list = None,
                 node_mlp_activation: str = "swish",
                 use_edge_attr: bool = False,
                 edge_attr_dim: int = 0,
                 use_attention: bool = False,
                 use_normalize: bool = False,
                 use_skip: bool = True,
                 use_node_attributes: bool = False,
                 pooling_method: str = "sum",
                 coord_pooling_method: str = "mean"):
        """Initialize EGNN layer.

        Args:
            units: Hidden dimension for node features.
            edge_mlp_units: Units for the edge model MLP. If None, [units, units].
            edge_mlp_activation: Activation for edge MLP.
            coord_mlp_units: Units for the coordinate weight MLP. If None, [units, 1].
            coord_mlp_activation: Activation for coordinate weight MLP.
            node_mlp_units: Units for the node update MLP. If None, [units, units].
            node_mlp_activation: Activation for node update MLP.
            use_edge_attr: Whether to use edge attributes in the edge model.
            edge_attr_dim: Dimension of edge attributes.
            use_attention: Whether to apply attention gating on messages.
            use_normalize: Whether to normalize coordinate differences.
            use_skip: Whether to use residual connection in node update.
            use_node_attributes: Whether to concatenate original node features h0
                into the node update input.
            pooling_method: Aggregation method for messages.
        """
        super().__init__()
        if edge_mlp_units is None:
            edge_mlp_units = [units, units]
        if coord_mlp_units is None:
            coord_mlp_units = [units, 1]
        if node_mlp_units is None:
            node_mlp_units = [units, units]

        self.use_edge_attr = use_edge_attr
        self.use_attention = use_attention
        self.use_normalize = use_normalize
        self.use_skip = use_skip
        self.use_node_attributes = use_node_attributes
        self.units = units

        # Edge model input: h_i + h_j + radial (1) + optional edge_attr
        edge_input_dim = 2 * units + 1
        if use_edge_attr and edge_attr_dim > 0:
            edge_input_dim += edge_attr_dim

        # Keras default: per-layer activation with "linear" final
        edge_act = [edge_mlp_activation] * max(len(edge_mlp_units) - 1, 0) + ["linear"]
        self.edge_mlp = MLP(
            units=edge_mlp_units,
            input_dim=edge_input_dim,
            activation=edge_act
        )

        # Coordinate weight model: m_ij -> scalar weight
        coord_act = [coord_mlp_activation] * max(len(coord_mlp_units) - 1, 0) + ["linear"]
        self.coord_mlp = MLP(
            units=coord_mlp_units,
            input_dim=edge_mlp_units[-1],
            activation=coord_act
        )

        # Node update model: h_i + agg(m_ij) [+ h0] -> h_i'
        node_input_dim = units + edge_mlp_units[-1]
        if use_node_attributes:
            node_input_dim += units
        node_act = [node_mlp_activation] * max(len(node_mlp_units) - 1, 0) + ["linear"]
        self.node_mlp = MLP(
            units=node_mlp_units,
            input_dim=node_input_dim,
            activation=node_act
        )

        # Optional attention gate (matching Keras GraphMLP(**edge_attention_kwargs))
        if use_attention:
            self.att_mlp = MLP(
                units=[1],
                input_dim=edge_mlp_units[-1],
                activation=["sigmoid"]
            )

        self.aggr_msg = AggregateLocalEdges(pooling_method=pooling_method)
        self.aggr_coord = AggregateLocalEdges(pooling_method=coord_pooling_method)

    def forward(self, h: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor,
                edge_attr: torch.Tensor = None,
                h0: torch.Tensor = None,
                pre_residual_norm=None,
                batch: torch.Tensor = None,
                batch_size: int = None) -> tuple:
        """Forward pass.

        Args:
            h: Node features (N, units).
            pos: Node positions (N, 3).
            edge_index: Edge indices (2, M).
            edge_attr: Optional edge features (M, edge_attr_dim).
            h0: Original node features for use_node_attributes (N, units).
            pre_residual_norm: Optional normalization module to apply to h
                before the residual addition (matching Keras placement).
            batch: Batch assignment (N,), required if pre_residual_norm is set.
            batch_size: Number of graphs, required if pre_residual_norm is set.

        Returns:
            Tuple of (h_updated, pos_updated):
                - h_updated: Updated node features (N, units).
                - pos_updated: Updated node positions (N, 3).
        """
        num_nodes = h.size(0)

        # Gather node features for each edge
        h_i = gather_nodes_ingoing(h, edge_index)    # (M, units)
        h_j = gather_nodes_outgoing(h, edge_index)   # (M, units)

        # Compute coordinate differences and Euclidean distances (matching Keras EuclideanNorm)
        pos_i = pos[edge_index[1]]  # target (M, 3)
        pos_j = pos[edge_index[0]]  # source (M, 3)
        diff_x = pos_i - pos_j      # (M, 3)
        norm_x = torch.sqrt((diff_x * diff_x).sum(dim=-1, keepdim=True) + 1e-8)  # (M, 1)

        # Normalize differences if requested
        if self.use_normalize:
            diff_x_normed = diff_x / norm_x
        else:
            diff_x_normed = diff_x

        # Edge model
        edge_input = [h_i, h_j, norm_x]
        if self.use_edge_attr and edge_attr is not None:
            edge_input.append(edge_attr)
        edge_input = torch.cat(edge_input, dim=-1)  # (M, edge_input_dim)
        m_ij = self.edge_mlp(edge_input)             # (M, edge_mlp_units[-1])

        # Optional attention
        if self.use_attention:
            att = self.att_mlp(m_ij)   # (M, 1)
            m_ij = m_ij * att          # (M, edge_mlp_units[-1])

        # Coordinate model: compute per-edge weight and scale difference
        coord_weight = self.coord_mlp(m_ij)                    # (M, 1)
        coord_msg = diff_x_normed * coord_weight               # (M, 3)
        coord_agg = self.aggr_coord(coord_msg, edge_index, num_nodes)  # (N, 3)
        pos_updated = pos + coord_agg                          # (N, 3)

        # Node model: aggregate messages, concatenate with node features, update
        m_agg = self.aggr_msg(m_ij, edge_index, num_nodes)     # (N, edge_mlp_units[-1])
        node_input_parts = [h, m_agg]
        if self.use_node_attributes and h0 is not None:
            node_input_parts.append(h0)
        node_input = torch.cat(node_input_parts, dim=-1)
        node_update = self.node_mlp(node_input)

        # Apply normalization to h BEFORE residual addition (matching Keras GraphLayerNormalization)
        if pre_residual_norm is not None:
            h = pre_residual_norm(h)

        h_updated = (h + node_update) if self.use_skip else node_update

        return h_updated, pos_updated


class EGNNModel(nn.Module):
    """E(n) Equivariant Graph Neural Network model.

    Iteratively updates node features and coordinates through equivariant
    message passing layers, then pools to graph-level predictions.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - data.edge_attr: Optional edge features (M, edge_attr_dim).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 4,
                 units: int = 64,
                 edge_mlp_units: list = None,
                 edge_mlp_activation: str = "swish",
                 coord_mlp_units: list = None,
                 coord_mlp_activation: str = "swish",
                 node_mlp_units: list = None,
                 node_mlp_activation: str = "swish",
                 use_edge_attr: bool = True,
                 edge_attr_dim: int = 0,
                 use_attention: bool = False,
                 use_normalize: bool = False,
                 use_skip: bool = True,
                 use_node_attributes: bool = False,
                 use_node_normalization: bool = False,
                 layer_pooling: str = "sum",
                 coord_pooling: str = "mean",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "swish",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_mlp_initialize: list = None,
                 node_decoder_units: list = None):
        """Initialize EGNN model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of EGNN layers.
            units: Hidden dimension for EGNN layers.
            edge_mlp_units: Hidden dims for edge model MLP per layer.
            edge_mlp_activation: Activation for edge model MLP.
            coord_mlp_units: Hidden dims for coordinate weight MLP per layer.
            coord_mlp_activation: Activation for coordinate weight MLP.
            node_mlp_units: Hidden dims for node update MLP per layer.
            node_mlp_activation: Activation for node update MLP.
            use_edge_attr: Whether to use edge attributes.
            edge_attr_dim: Dimension of edge attributes.
            use_attention: Whether to use attention gating on messages.
            use_normalize: Whether to normalize coordinate differences.
            use_skip: Whether to use residual connection in node update.
            use_node_attributes: Whether to concatenate original node features
                into the node update.
            use_node_normalization: Whether to apply graph normalization after
                each layer.
            layer_pooling: Aggregation method inside EGNN layers.
            node_pooling: Pooling method for graph-level readout.
            output_units: Hidden dims for the output MLP. If None, [units, units].
            output_activation: Activation for the output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for the node embedding.
            node_decoder_units: Hidden dims for optional node decoder MLP applied
                after the layer loop. If None, no decoder is applied.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if output_units is None:
            output_units = [64]

        self.use_node_embedding = use_node_embedding
        self.use_edge_attr = use_edge_attr
        self.use_node_attributes = use_node_attributes
        self.use_node_normalization = use_node_normalization
        self.depth = depth

        self.node_mlp_initialize = node_mlp_initialize

        # Node embedding
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)

        # Optional initial projection to hidden dimension (matching Keras node_mlp_initialize)
        if node_mlp_initialize is not None:
            init_act = [output_activation] * max(len(node_mlp_initialize) - 1, 0) + ["linear"]
            self.dense_in = MLP(
                units=node_mlp_initialize, input_dim=node_dim, activation=init_act
            )
        else:
            self.dense_in = None

        # Determine effective input dim for layers
        if node_mlp_initialize is not None:
            effective_units = node_mlp_initialize[-1]
        else:
            effective_units = node_dim  # No projection, use raw dim (matching Keras)

        # EGNN layers
        self.layers = nn.ModuleList()
        for _ in range(depth):
            self.layers.append(EGNNLayer(
                units=units,
                edge_mlp_units=edge_mlp_units,
                edge_mlp_activation=edge_mlp_activation,
                coord_mlp_units=coord_mlp_units,
                coord_mlp_activation=coord_mlp_activation,
                node_mlp_units=node_mlp_units,
                node_mlp_activation=node_mlp_activation,
                use_edge_attr=use_edge_attr,
                edge_attr_dim=edge_attr_dim,
                use_attention=use_attention,
                use_normalize=use_normalize,
                use_skip=use_skip,
                use_node_attributes=use_node_attributes,
                pooling_method=layer_pooling,
                coord_pooling_method=coord_pooling
            ))

        # Optional per-layer graph normalization
        if use_node_normalization:
            self.node_norms = nn.ModuleList([GraphLayerNorm(units) for _ in range(depth)])
        else:
            self.node_norms = None

        # Optional node decoder MLP
        if node_decoder_units is not None:
            dec_act = [output_activation] * max(len(node_decoder_units) - 1, 0) + ["linear"]
            self.node_decoder = MLP(
                units=node_decoder_units, input_dim=units, activation=dec_act
            )
        else:
            self.node_decoder = None

        # Graph-level pooling and output
        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        output_in_dim = node_decoder_units[-1] if node_decoder_units else units
        self.output_mlp = MLP(
            units=out_units,
            input_dim=output_in_dim,
            activation=out_act
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        x = data.z if hasattr(data, 'z') else data.x
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        edge_attr = data.edge_attr if (hasattr(data, 'edge_attr') and self.use_edge_attr) else None

        # Node embedding
        if self.use_node_embedding:
            h = self.node_embedding(x.long())
        else:
            h = x

        # Optional initial projection
        if self.dense_in is not None:
            h = self.dense_in(h)

        # Save original node features for use_node_attributes
        h0 = h if self.use_node_attributes else None

        # Message passing layers (update both h and pos)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
        for i, layer in enumerate(self.layers):
            norm = self.node_norms[i] if self.use_node_normalization else None
            h, pos = layer(h, pos, edge_index, edge_attr, h0=h0,
                           pre_residual_norm=norm, batch=batch,
                           batch_size=batch_size)

        # Optional node decoder
        if self.node_decoder is not None:
            h = self.node_decoder(h)

        # Graph-level pooling on node features
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(h, batch, batch_size)
        else:
            out = h
        out = self.output_mlp(out)
        return out
