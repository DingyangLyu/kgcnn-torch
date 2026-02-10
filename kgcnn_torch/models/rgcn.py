"""RGCN (Relational Graph Convolutional Network) model.

Reference: Schlichtkrull et al., Modeling Relational Data with Graph Convolutional Networks (2018).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.gather import gather_nodes_outgoing
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class RGCNConv(nn.Module):
    """Relational Graph Convolution layer.

    For each relation type r, applies a separate linear transformation W_r to source
    node features, optionally weights by edge attributes, and aggregates to target
    nodes.  A self-loop transformation W_0 is also applied.

    Computes:
        h_i' = activation( W_0 * h_i + sum_r sum_{j in N_r(i)} (edge_attr * (W_r @ h_j)) )
    """

    def __init__(self, in_features: int, out_features: int, num_relations: int,
                 pooling_method: str = "sum", activation: str = "swish",
                 use_bias: bool = True):
        """Initialize RGCNConv.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            num_relations: Number of relation types.
            pooling_method: Aggregation method for messages.
            activation: Activation function name.
            use_bias: Whether to use bias in the self-loop and relational transforms.
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_relations = num_relations

        # Relation-specific weight matrices: (num_relations, in_features, out_features)
        self.weight = nn.Parameter(torch.Tensor(num_relations, in_features, out_features))

        # Bias for the relational dense path (matches Keras RelationalDense use_bias=True)
        if use_bias:
            self.rel_bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('rel_bias', None)

        # Self-loop transformation
        self.self_loop = nn.Linear(in_features, out_features, bias=use_bias)

        self.activation = get_activation(activation)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

        # Initialize parameters
        nn.init.xavier_uniform_(self.weight)
        nn.init.xavier_uniform_(self.self_loop.weight)
        if use_bias:
            nn.init.zeros_(self.self_loop.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                edge_attr: torch.Tensor = None) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, in_features).
            edge_index: Edge indices of shape (2, M).
            edge_type: Relation type per edge of shape (M,), integer values in [0, num_relations).
            edge_attr: Optional edge weights of shape (M, 1). If None, treated as 1.

        Returns:
            Updated node features of shape (N, out_features).
        """
        num_nodes = x.size(0)

        # Gather source node features: (M, in_features)
        x_j = gather_nodes_outgoing(x, edge_index)

        # Gather relation-specific weight for each edge: (M, in_features, out_features)
        w = self.weight[edge_type.long()]

        # Apply relation-specific transformation: (M, out_features)
        messages = torch.bmm(x_j.unsqueeze(1), w).squeeze(1)

        # Add relational bias (matches Keras RelationalDense use_bias=True)
        if self.rel_bias is not None:
            messages = messages + self.rel_bias

        # Optionally weight by edge attributes
        if edge_attr is not None:
            messages = messages * edge_attr

        # Aggregate messages to target nodes: (N, out_features)
        agg = self.aggr(messages, edge_index, num_nodes)

        # Self-loop: (N, out_features)
        out = self.self_loop(x) + agg

        return self.activation(out)


class RGCNModel(nn.Module):
    """Relational Graph Convolutional Network.

    Applies relation-specific transformations for multi-relational graphs, with
    residual connections and graph-level readout.

    Expects PyG Data batch with:
        - data.x or data.z: Node features (N,) int or (N, F) float.
        - data.edge_index: Edge indices (2, M).
        - data.edge_type: Relation type per edge (M,), integer.
        - data.edge_attr: Optional edge weights (M, 1).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = 64,
                 num_relations: int = 20,
                 rgcn_activation: str = "swish",
                 rgcn_pooling: str = "sum",
                 use_residual: bool = False,
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize RGCN model.

        Args:
            node_dim: Dimension of node features after embedding.
            depth: Number of RGCN convolution layers.
            units: Hidden dimension for RGCN layers.
            num_relations: Number of relation types in the graph.
            rgcn_activation: Activation function for RGCN layers.
            rgcn_pooling: Aggregation method inside RGCNConv.
            use_residual: Whether to add residual connections between layers.
            node_pooling: Pooling method for graph-level readout.
            output_units: Hidden dims for the output MLP. If None, defaults to [units, units].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for the node embedding.
        """
        super().__init__()
        if output_units is None:
            output_units = []

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding
        self.use_residual = use_residual

        # Node embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # RGCN convolution layers
        # First layer takes node_dim as input; subsequent layers take units.
        self.convs = nn.ModuleList()
        for i in range(depth):
            in_dim = node_dim if i == 0 else units
            self.convs.append(RGCNConv(
                in_features=in_dim, out_features=units,
                num_relations=num_relations,
                pooling_method=rgcn_pooling,
                activation=rgcn_activation
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
        edge_attr = data.edge_attr if hasattr(data, 'edge_attr') and data.edge_attr is not None else None
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

        # RGCN edge weighting expects scalar edge weights.
        if edge_attr is not None:
            if edge_attr.dim() == 1:
                edge_attr = edge_attr.unsqueeze(-1)
            elif edge_attr.dim() > 1 and edge_attr.size(-1) != 1:
                edge_attr = None

        # Message passing layers with optional residual
        for conv in self.convs:
            x_new = conv(x, edge_index, edge_type, edge_attr)
            if self.use_residual:
                x = x + x_new
            else:
                x = x_new

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(x, batch, batch_size)
        else:
            out = x
        out = self.output_mlp(out)
        return out
