"""GNN-FiLM (Graph Neural Network with Feature-wise Linear Modulation) model.

Reference: Brockschmidt, GNN-FiLM: Graph Neural Networks with Feature-wise Linear Modulation (2020).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.relational import RelationalDense
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GNNFilmLayer(nn.Module):
    """GNN-FiLM message passing layer.

    Feature-wise Linear Modulation (FiLM) conditioned on (node_features, relation).
    For each edge (j -> i) with relation type r:
        gamma = RelationalDense(n_i, r)
        beta = RelationalDense(n_i, r)
        h_j = RelationalDense(n_j, r)
        m_ij = h_j * gamma + beta
    Aggregated modulated neighbor features are used to update each node.

    Computes:
        n_i' = activation( sum_{j in N(i)} (RelationalDense(n_j, r) * RelationalDense(n_i, r) + RelationalDense(n_i, r)) )
    """

    def __init__(self, units: int, num_relations: int,
                 pooling_method: str = "sum",
                 activation: str = "relu",
                 modulation_activation: str = "sigmoid",
                 use_bias: bool = True):
        """Initialize GNNFilmLayer.

        Args:
            units: Node feature dimension (both input and output).
            num_relations: Number of relation types.
            pooling_method: Aggregation method for neighbor messages.
            activation: Activation function applied after aggregation.
            modulation_activation: Activation for modulation (gamma/beta) layers.
            use_bias: Whether to use bias in dense layers.
        """
        super().__init__()
        self.units = units
        self.num_relations = num_relations

        # RelationalDense for gamma (modulation scale) - depends on target node + relation
        self.rel_dense_gamma = RelationalDense(
            units, units, num_relations,
            activation=modulation_activation, use_bias=use_bias
        )
        # RelationalDense for beta (modulation shift) - depends on target node + relation
        self.rel_dense_beta = RelationalDense(
            units, units, num_relations,
            activation=modulation_activation, use_bias=use_bias
        )
        # RelationalDense for source node transformation - depends on source node + relation
        self.rel_dense_hj = RelationalDense(
            units, units, num_relations,
            activation=None, use_bias=use_bias
        )

        self.activation = get_activation(activation)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, units).
            edge_index: Edge indices of shape (2, M).
            edge_type: Relation type per edge of shape (M,), integer values in [0, num_relations).

        Returns:
            Updated node features of shape (N, units).
        """
        num_nodes = x.size(0)

        # Gather target (n_i) and source (n_j) node features per edge
        n_i = gather_nodes_ingoing(x, edge_index)   # (M, units) - target nodes
        n_j = gather_nodes_outgoing(x, edge_index)   # (M, units) - source nodes

        # Relation-conditioned FiLM parameters from target node features
        gamma = self.rel_dense_gamma(n_i, edge_type.long())  # (M, units)
        beta = self.rel_dense_beta(n_i, edge_type.long())     # (M, units)

        # Relation-conditioned transformation of source node features
        h_j = self.rel_dense_hj(n_j, edge_type.long())  # (M, units)

        # Apply feature-wise linear modulation: (M, units)
        m = h_j * gamma + beta

        # Aggregate modulated messages to target nodes: (N, units)
        h = self.aggr(m, edge_index, num_nodes)

        # Apply activation (no residual, matching Keras)
        out = self.activation(h)

        return out


class GNNFilmModel(nn.Module):
    """Graph Neural Network with Feature-wise Linear Modulation.

    Uses relation-type-conditioned FiLM layers for message passing on
    multi-relational graphs. Each layer modulates neighbor features with
    learned gamma and beta parameters that depend on both the target node
    features and the edge/relation type via RelationalDense layers.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_type: Relation type per edge (M,), integer.
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = None,
                 num_relations: int = 20,
                 activation: str = "swish",
                 modulation_activation: str = "sigmoid",
                 film_pooling: str = "sum",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize GNN-FiLM model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of GNN-FiLM message passing layers.
            units: Hidden dimension for GNN-FiLM layers.
            num_relations: Number of relation types in the graph.
            activation: Activation function for GNN-FiLM layers.
            modulation_activation: Activation for modulation (gamma/beta) RelationalDense layers.
            film_pooling: Aggregation method inside FiLM layers.
            node_pooling: Pooling method for graph-level readout.
            output_units: Hidden dims for the output MLP. If None, defaults to [units, units].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for the node embedding.
        """
        super().__init__()
        # In Keras, embedding output_dim feeds directly into FiLM layers (no projection).
        # So units defaults to node_dim.
        if units is None:
            units = node_dim
        if output_units is None:
            output_units = []

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding
        self.units = units

        # Node embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Projection only if node_dim != units (Keras has no projection layer)
        if node_dim != units:
            self.dense_in = nn.Linear(node_dim, units)
        else:
            self.dense_in = None

        # GNN-FiLM layers (no residual connections, matching Keras)
        self.film_layers = nn.ModuleList()
        for _ in range(depth):
            self.film_layers.append(GNNFilmLayer(
                units=units, num_relations=num_relations,
                pooling_method=film_pooling, activation=activation,
                modulation_activation=modulation_activation
            ))

        # Graph-level pooling
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # Output MLP
        out_units = output_units + [num_targets]
        # Regression heads must stay linear to match kgcnn keras defaults.
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=units,
            activation=out_act
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        # Prefer dense features when not embedding; otherwise use integer atomic numbers.
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z
        edge_index = data.edge_index
        edge_type = data.edge_type
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            if x.dim() > 1:
                x = x.squeeze(-1)
            x = self.node_embedding(x.long())
        else:
            if x.dim() == 1:
                x = x.unsqueeze(-1).float()
            else:
                x = x.float()
            x = self.node_projection(x)

        # Projection only if dimensions differ (Keras has no projection)
        if self.dense_in is not None:
            x = self.dense_in(x)

        # FiLM message passing layers (no residual, matching Keras)
        for film_layer in self.film_layers:
            x = film_layer(x, edge_index, edge_type)

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(x, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out
