"""HamNet (Hamiltonian Neural Network) model.

Reference: Li et al., HamNet: Conformation-Guided Molecular Representation
with Hamiltonian Neural Networks (2021).

Uses Hamiltonian-inspired message passing with momentum (p) and position (q)
coordinate differences, combined with attention-based aggregation and an
iterative fingerprint readout.
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdgesAttention
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingEmbeddingAttention
from kgcnn_torch.layers.update import GRUUpdate
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.ops.scatter import scatter_reduce_sum, scatter_reduce_softmax
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class HamNaiveDynMessage(nn.Module):
    """Hamiltonian-inspired dynamic message passing layer.

    Computes node and edge messages using coordinate differences (p_ij, q_ij)
    and edge features with attention-based aggregation.

    For each edge (i -> j):
        p_ij = p_i - p_j  (momentum difference, shape (M, 3))
        q_ij = q_i - q_j  (position difference, shape (M, 3))
        align = w^T [p_ij || q_ij || e_ij]  (attention logit, shape (M, 1))
        attend = Dense(h_j)  (attended node feature, shape (M, F))
        node_msg = softmax(align) * attend, pooled to target nodes
        edge_msg = Dense([h_i || p_ij || q_ij || h_j])
    """

    def __init__(self, units: int, edge_dim: int,
                 activation: str = "leaky_relu2", activation_last: str = "elu",
                 use_dropout: bool = False, dropout_rate: float = 0.5):
        """Initialize HamNaiveDynMessage.

        Args:
            units: Hidden dimension for node features.
            edge_dim: Dimension of edge features.
            activation: Activation function for attend and edge transforms.
            activation_last: Final activation for node messages (Keras default: elu).
            use_dropout: Whether to apply dropout on node and edge messages.
            dropout_rate: Dropout rate (only used when use_dropout=True).
        """
        super().__init__()
        self.use_dropout = use_dropout
        # Attention alignment: projects [p_ij(3) || q_ij(3) || e_ij(edge_dim)] -> 1
        self.align_dense = nn.Linear(3 + 3 + edge_dim, 1)
        # Attended node transform with activation (Keras: Dense(units, activation=activation))
        self.attend_dense = nn.Sequential(
            nn.Linear(units, units),
            get_activation(activation)
        )
        # Edge message: [h_i(units) || p_ij(3) || q_ij(3) || h_j(units)] -> edge_dim
        self.edge_dense = nn.Linear(units + 3 + 3 + units, edge_dim)
        self.activation = get_activation(activation)
        self.final_activ = get_activation(activation_last)
        # Keras kgcnn.layers.aggr.AggregateLocalEdgesAttention defaults normalize_softmax=False.
        self.aggr_attention = AggregateLocalEdgesAttention(normalize_softmax=False)
        if use_dropout:
            self.dropout_layer = nn.Dropout(p=dropout_rate)

    def forward(self, h: torch.Tensor, p: torch.Tensor, q: torch.Tensor,
                e: torch.Tensor, edge_index: torch.Tensor,
                num_nodes: int) -> tuple:
        """Forward pass.

        Args:
            h: Node features of shape (N, units).
            p: Momentum coordinates of shape (N, 3).
            q: Position coordinates of shape (N, 3).
            e: Edge features of shape (M, edge_dim).
            edge_index: Edge indices of shape (2, M).
            num_nodes: Total number of nodes N.

        Returns:
            Tuple of (node_message, edge_message):
                - node_message: (N, units)
                - edge_message: (M, edge_dim)
        """
        # Gather source (i) and target (j) features
        # PyG convention: edge_index[0] = source, edge_index[1] = target
        # Messages aggregate at target, so h_j = source features for attention
        h_i = gather_nodes_ingoing(h, edge_index)   # (M, units) - target node
        h_j = gather_nodes_outgoing(h, edge_index)   # (M, units) - source node
        p_i = gather_nodes_ingoing(p, edge_index)     # (M, 3)
        p_j = gather_nodes_outgoing(p, edge_index)    # (M, 3)
        q_i = gather_nodes_ingoing(q, edge_index)     # (M, 3)
        q_j = gather_nodes_outgoing(q, edge_index)    # (M, 3)

        # Coordinate differences
        # Match Keras HamNet: p_uv = p_source - p_target, q_uv = q_source - q_target.
        p_ij = p_j - p_i  # (M, 3)
        q_ij = q_j - q_i  # (M, 3)

        # Attention alignment: w^T [p_ij || q_ij || e_ij]
        align_input = torch.cat([p_ij, q_ij, e], dim=-1)  # (M, 6 + edge_dim)
        align = self.align_dense(align_input)  # (M, 1)

        # Attended source node features
        attend = self.attend_dense(h_j)  # (M, units)

        # Aggregate with attention to target nodes
        node_msg = self.aggr_attention(attend, align, edge_index, num_nodes)  # (N, units)
        # Apply final activation to node message (Keras: activation_last="elu")
        node_msg = self.final_activ(node_msg)

        # Edge message
        edge_input = torch.cat([h_i, p_ij, q_ij, h_j], dim=-1)  # (M, units+6+units)
        edge_msg = self.activation(self.edge_dense(edge_input))  # (M, edge_dim)

        # Apply dropout if enabled (matching Keras HamNaiveDynMessage use_dropout)
        if self.use_dropout:
            node_msg = self.dropout_layer(node_msg)
            edge_msg = self.dropout_layer(edge_msg)

        return node_msg, edge_msg


class HamNetNaiveUnion(nn.Module):
    """Simple concatenation-based update for HamNet.

    Computes: x' = activation(Dense([x || x_update]))
    """

    def __init__(self, units: int, activation: str = "leaky_relu2"):
        """Initialize HamNetNaiveUnion.

        Args:
            units: Output dimension.
            activation: Activation function name.
        """
        super().__init__()
        self.dense = nn.Linear(2 * units, units)
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor, x_update: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Current features of shape (*, units).
            x_update: Update features of shape (*, units).

        Returns:
            Updated features of shape (*, units).
        """
        return self.activation(self.dense(torch.cat([x, x_update], dim=-1)))


class HamNetFingerprintGenerator(nn.Module):
    """Iterative fingerprint readout for HamNet.

    Generates a graph-level fingerprint by iteratively refining a graph
    embedding through attention over node features and GRU updates.

    Procedure:
        1. s = mean_pool(Dense(h_v))
        2. For each depth step:
            attend = Dense(h_v)
            align = Dense([s_broadcast || h_v])
            m = final_activ(softmax_pool(attend * softmax(align)))
            s = final_activ(GRUCell(m, s))
    """

    def __init__(self, units: int, fingerprint_dim: int, depth: int = 2,
                 activation: str = "leaky_relu2", activation_context: str = "leaky_relu2",
                 activation_last: str = "elu",
                 pooling_method: str = "mean",
                 use_dropout: bool = False, dropout_rate: float = 0.5):
        """Initialize HamNetFingerprintGenerator.

        Args:
            units: Input node feature dimension.
            fingerprint_dim: Output fingerprint dimension (units_attend in Keras).
            depth: Number of iterative refinement steps.
            activation: Activation for attend transform (Keras: leaky_relu2).
            activation_context: Activation after GRU update (Keras: leaky_relu2).
            activation_last: Activation for attention output (Keras: elu).
            pooling_method: Pooling method for initial state.
            use_dropout: Whether to apply dropout on attend and align features.
            dropout_rate: Dropout rate (only used when use_dropout=True).
        """
        super().__init__()
        self.depth = depth
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout_layer = nn.Dropout(p=dropout_rate)

        # Initial state transform (Keras: vertex2mol = Dense(units, activation=activation))
        self.init_dense = nn.Sequential(
            nn.Linear(units, fingerprint_dim),
            get_activation(activation)
        )
        self.pool_start = PoolingNodes(pooling_method=pooling_method)

        # Separate readout layers per depth (Keras: separate HamNetGlobalReadoutAttend per depth)
        self.attend_denses = nn.ModuleList()
        self.align_denses = nn.ModuleList()
        self.pool_attentions = nn.ModuleList()
        self.grus = nn.ModuleList()
        for _ in range(depth):
            # Keras: dense_attend = Dense(units_attend, activation=activation)
            self.attend_denses.append(nn.Sequential(
                nn.Linear(units, fingerprint_dim),
                get_activation(activation)
            ))
            # Keras: dense_align = Dense(1, activation="linear")
            # Input is [state_broadcast || node_features] = fingerprint_dim + units
            self.align_denses.append(nn.Linear(fingerprint_dim + units, 1))
            self.pool_attentions.append(PoolingEmbeddingAttention())
            # Keras: separate GRUCell per depth
            self.grus.append(nn.GRUCell(fingerprint_dim, fingerprint_dim))

        self.final_activ_last = get_activation(activation_last)  # elu for attention output
        self.final_activ_context = get_activation(activation_context)  # leaky_relu2 after GRU

    def forward(self, h: torch.Tensor, batch: torch.Tensor,
                batch_size: int) -> torch.Tensor:
        """Forward pass.

        Args:
            h: Node features of shape (N, units).
            batch: Batch assignment of shape (N,).
            batch_size: Number of graphs in the batch.

        Returns:
            Graph-level fingerprint of shape (B, fingerprint_dim).
        """
        # Initial state: s = pool(activation(Dense(h_v)))
        h_transformed = self.init_dense(h)  # (N, fingerprint_dim)
        s = self.pool_start(h_transformed, batch, batch_size)  # (B, fingerprint_dim)

        for i in range(self.depth):
            # Per-depth attend transform (Keras: separate per depth)
            attend = self.attend_denses[i](h)  # (N, fingerprint_dim)
            if self.use_dropout:
                attend = self.dropout_layer(attend)

            # Broadcast graph state to nodes
            s_broadcast = s[batch]  # (N, fingerprint_dim)

            # Alignment: Dense([s_broadcast || h_v]) -> 1 (Keras: linear activation)
            align_input = torch.cat([s_broadcast, h], dim=-1)  # (N, fingerprint_dim + units)
            align = self.align_denses[i](align_input)  # (N, 1)
            if self.use_dropout:
                align = self.dropout_layer(align)

            # Attention-weighted pooling
            m = self.pool_attentions[i](attend, align, batch, batch_size)  # (B, fingerprint_dim)
            # Keras: activation_last="elu" applied to attention output
            m = self.final_activ_last(m)

            # GRU state update (Keras: separate GRUCell per depth)
            s = self.grus[i](m, s)  # (B, fingerprint_dim)
            # Keras: final_activ (activation=leaky_relu2) applied after GRU
            s = self.final_activ_context(s)

        return s


class HamNetModel(nn.Module):
    """HamNet (Hamiltonian Neural Network) model for graph-level prediction.

    Implements the HamNet architecture which uses Hamiltonian-inspired message
    passing with momentum and position coordinate differences, combined with
    attention-based aggregation and an iterative fingerprint readout.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Node coordinates (N, 3), used as position (q) coordinates.
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Edge features (M, E).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 64,
                 depth: int = 1,
                 units: int = 128,
                 fingerprint_dim: int = 128,
                 fingerprint_depth: int = 2,
                 activation: str = "leaky_relu2",
                 activation_last: str = "elu",
                 fingerprint_activation: str = "leaky_relu2",
                 fingerprint_activation_context: str = "leaky_relu2",
                 use_gru_update: bool = True,
                 use_gru_update_edge: bool = False,
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_use_bias: list | bool | None = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize HamNet model.

        Args:
            node_dim: Dimension of initial node features after embedding.
            edge_dim: Dimension of edge features.
            depth: Number of message passing iterations.
            units: Hidden dimension for node features.
            fingerprint_dim: Dimension of the graph-level fingerprint.
            fingerprint_depth: Number of iterative readout steps.
            activation: Activation function for message passing layers.
            fingerprint_activation: Activation for fingerprint alignment.
            fingerprint_activation_context: Final activation in fingerprint.
            use_gru_update: If True, use GRU-based updates; otherwise naive union.
            output_units: Hidden dimensions for the output MLP. If None, [units, units // 2].
            output_activation: Activation for the output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed integer node features via Embedding.
            num_embeddings: Vocabulary size for node embedding table.
            node_input_dim: Input feature dimension when use_node_embedding=False.
        """
        super().__init__()
        if output_units is None:
            output_units = [25, 10]

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding
        self.use_gru_update = use_gru_update
        self.use_gru_update_edge = use_gru_update_edge

        # Node embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Initial dense transforms (Keras: Dense(gru_kwargs["units"], activation="tanh"))
        self.node_init = nn.Sequential(
            nn.Linear(node_dim, units),
            get_activation("tanh")
        )
        # Edge init projects to units dimension (matching Keras: Dense(gru_kwargs["units"]))
        self.edge_init = nn.Sequential(
            nn.Linear(edge_dim, units),
            get_activation("tanh")
        )

        # After edge_init, internal edge dim becomes units
        internal_edge_dim = units

        # Message passing layers
        self.message_layers = nn.ModuleList()
        self.node_update_layers = nn.ModuleList()
        self.edge_update_layers = nn.ModuleList()

        for _ in range(depth):
            self.message_layers.append(
                HamNaiveDynMessage(units=units, edge_dim=internal_edge_dim,
                                   activation=activation, activation_last=activation_last)
            )
            if use_gru_update:
                self.node_update_layers.append(
                    GRUUpdate(input_dim=units, hidden_dim=units)
                )
            else:
                self.node_update_layers.append(
                    HamNetNaiveUnion(units=units, activation=activation)
                )
            if use_gru_update_edge:
                self.edge_update_layers.append(
                    GRUUpdate(input_dim=internal_edge_dim, hidden_dim=internal_edge_dim)
                )
            else:
                # Keras default: union_type_edge="None" means no update, just use message
                self.edge_update_layers.append(None)

        # Fingerprint readout
        self.fingerprint = HamNetFingerprintGenerator(
            units=units,
            fingerprint_dim=fingerprint_dim,
            depth=fingerprint_depth,
            activation=fingerprint_activation,
            activation_context=fingerprint_activation_context,
            activation_last="elu"
        )

        # Output MLP (Keras: ["relu", "relu", "linear"])
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        if output_use_bias is None:
            # Match HamNet literature defaults: no bias on final regression head.
            output_use_bias = [True] * len(output_units) + [False]
        output_mlp_input_dim = fingerprint_dim if output_embedding == "graph" else units
        self.output_mlp = MLP(
            units=out_units,
            input_dim=output_mlp_input_dim,
            activation=out_act,
            use_bias=output_use_bias,
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object with attributes z, pos, edge_index,
                  edge_attr, and batch.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        pos = data.pos
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            h = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            h = self.node_projection(x)

        num_nodes = h.size(0)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Initial transforms
        h = self.node_init(h)       # (N, units)
        e = self.edge_init(edge_attr)  # (M, edge_dim)

        # Initialize Hamiltonian coordinates
        q = pos.float()                          # (N, 3) - position
        p = torch.zeros_like(q)                  # (N, 3) - momentum, initialized to zero

        # Message passing loop
        for i in range(self.depth):
            # Compute node and edge messages
            node_msg, edge_msg = self.message_layers[i](h, p, q, e, edge_index, num_nodes)

            # Update node features
            if self.use_gru_update:
                h = self.node_update_layers[i](node_msg, h)
            else:
                h = self.node_update_layers[i](h, node_msg)

            # Update edge features
            if self.edge_update_layers[i] is not None:
                if self.use_gru_update_edge:
                    e = self.edge_update_layers[i](edge_msg, e)
                else:
                    e = self.edge_update_layers[i](e, edge_msg)
            else:
                # Keras default union_type_edge="None": just use the message directly
                e = edge_msg

        # Graph readout via fingerprint generator
        if self.output_embedding == "graph":
            out = self.fingerprint(h, batch, batch_size)  # (B, fingerprint_dim)
        else:
            out = h  # (N, units)

        # Output MLP
        out = self.output_mlp(out)
        return out
