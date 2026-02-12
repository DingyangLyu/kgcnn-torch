"""NMPN (Neural Message Passing Network) model.

Reference: Gilmer et al., Neural Message Passing for Quantum Chemistry (2017).
http://arxiv.org/abs/1704.01212

Follows the Keras kgcnn implementation: edge networks produce transformation
matrices (node_dim x node_dim) for each edge, which are multiplied with gathered
node features (MatMulMessages). Messages from both incoming and outgoing
directions are concatenated before aggregation, and initial node embeddings
(n0) are concatenated with final embeddings before the readout MLP.
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.message import MatMulMessages
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingSet2SetEncoder
from kgcnn_torch.layers.update import GRUUpdate
from kgcnn_torch.layers.geom import GaussBasisLayer, shift_periodic_lattice
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


def _apply_keras_init(model: nn.Module):
    """Apply Keras-compatible weight initialization to all layers in a model.

    Keras defaults that differ from PyTorch:
    - Dense: glorot_uniform weights, zeros biases
    - GRUCell: glorot_uniform input weights, orthogonal recurrent weights, zeros biases
    - LSTMCell: glorot_uniform input weights, orthogonal recurrent weights,
                zeros biases + unit_forget_bias (forget gate bias = 1)

    The orthogonal recurrent initialization is critical for RNN training stability.
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.GRUCell):
            nn.init.xavier_uniform_(module.weight_ih)
            nn.init.orthogonal_(module.weight_hh)
            nn.init.zeros_(module.bias_ih)
            nn.init.zeros_(module.bias_hh)
        elif isinstance(module, nn.LSTMCell):
            nn.init.xavier_uniform_(module.weight_ih)
            nn.init.orthogonal_(module.weight_hh)
            nn.init.zeros_(module.bias_ih)
            nn.init.zeros_(module.bias_hh)
            # unit_forget_bias: set forget gate bias to 1.0
            # Gate order in PyTorch: input, forget, cell, output
            hidden_size = module.hidden_size
            module.bias_ih.data[hidden_size:2 * hidden_size].fill_(1.0)


class TrafoEdgeNetMessages(nn.Module):
    """Transform edge network output into message matrices.

    Takes the output of an edge MLP and reshapes it into (M, node_dim, node_dim)
    matrices for matrix-multiplication-based message passing. Includes a Dense
    (Linear) layer that maps from the MLP output dimension to node_dim * node_dim.

    This mirrors the Keras TrafoEdgeNetMessages layer.
    """

    def __init__(self, input_dim: int, target_shape: tuple):
        """Initialize layer.

        Args:
            input_dim: Input feature dimension from the edge MLP.
            target_shape: Target shape (node_dim, node_dim) for the message matrix.
        """
        super().__init__()
        self.target_shape = target_shape
        self._units_out = target_shape[0]
        self._units_in = target_shape[1]
        self.dense = nn.Linear(input_dim, self._units_out * self._units_in)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Edge MLP output of shape (M, input_dim).

        Returns:
            Message matrices of shape (M, units_out, units_in).
        """
        up = self.dense(x)  # (M, units_out * units_in)
        return up.view(up.size(0), self._units_out, self._units_in)


class NMPNModel(nn.Module):
    """Neural Message Passing Network.

    Implements the MPNN framework with edge-network-based matrix multiplication
    message functions and GRU node updates. Two separate edge networks produce
    transformation matrices for incoming and outgoing messages. Messages from
    both directions are concatenated before aggregation. Initial node embeddings
    are concatenated with final embeddings before readout.

    Supports both simple graph pooling and Set2Set pooling for the readout phase.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.edge_index: Edge indices (2, M), PyG convention [source, target].
        - data.edge_attr: Edge features (M, edge_dim).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = 64,
                 edge_dim: int = 20,
                 edge_mlp_units: list = None,
                 edge_mlp_activation: str = "swish",
                 message_pooling: str = "sum",
                 use_set2set: bool = True,
                 set2set_T: int = 3,
                 set2set_channels: int = 32,
                 set2set_pooling_method: str = "sum",
                 set2set_init_qstar: str = "0",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "selu",
                 output_final_activation: str = "sigmoid",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        """Initialize NMPN model.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            depth: Number of message passing steps.
            units: Hidden dimension for node features (corresponds to node_dim
                in the Keras version).
            edge_dim: Input edge feature dimension.
            edge_mlp_units: Hidden dims for the edge network MLP. If None,
                defaults to [64, 64, 64] (matching Keras).
            edge_mlp_activation: Activation for the edge network.
            message_pooling: Aggregation method for messages ('sum', 'mean', etc.).
            use_set2set: Whether to use Set2Set pooling for graph readout.
            set2set_T: Number of Set2Set processing steps.
            set2set_channels: Hidden dim for Set2Set LSTM. Default 32 (matching Keras).
            node_pooling: Pooling method for graph readout (used if use_set2set=False).
            output_units: Hidden dims for output MLP. If None, [25, 10].
            output_activation: Activation for output MLP hidden layers.
            output_final_activation: Activation for output MLP final layer.
            num_targets: Number of output targets.
            output_embedding: Output embedding mode, "graph" for graph-level or "node" for node-level.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            node_input_dim: Input feature dimension when use_node_embedding=False.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if output_units is None:
            output_units = [25, 10]
        if edge_mlp_units is None:
            edge_mlp_units = [64, 64, 64]

        self.use_node_embedding = use_node_embedding
        self.use_set2set = use_set2set
        self.depth = depth
        self.units = units
        self.node_dim = node_dim

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
            # n0 will be node_dim; dense_in maps node_dim -> units
            dense_in_dim = node_dim
            self._n0_dim = node_dim
        else:
            # Keras: n0 stays as raw input (no projection); Dense(units) maps directly.
            dense_in_dim = node_input_dim
            self._n0_dim = node_input_dim

        # Project input node features to hidden dimension
        self.dense_in = nn.Linear(dense_in_dim, units)

        # Two edge networks (incoming and outgoing), each maps edge features
        # to transformation matrices of shape (units, units) via MLP + reshape.
        # These are computed once and reused across all message passing steps.
        self.edge_mlp_in = MLP(
            units=edge_mlp_units,
            input_dim=edge_dim,
            activation=edge_mlp_activation
        )
        self.edge_trafo_in = TrafoEdgeNetMessages(
            input_dim=edge_mlp_units[-1],
            target_shape=(units, units)
        )
        self.edge_mlp_out = MLP(
            units=edge_mlp_units,
            input_dim=edge_dim,
            activation=edge_mlp_activation
        )
        self.edge_trafo_out = TrafoEdgeNetMessages(
            input_dim=edge_mlp_units[-1],
            target_shape=(units, units)
        )

        # Matrix multiplication for messages
        self.matmul_msg = MatMulMessages()

        # Message aggregation
        self.aggr = AggregateLocalEdges(pooling_method=message_pooling)

        # GRU update: concatenated messages (2*units) are input,
        # node features (units) are hidden state
        self.gru_update = GRUUpdate(input_dim=2 * units, hidden_dim=units)

        # After message passing, concatenate initial embeddings (n0) with final
        # embeddings (n, dim=units). n0 dim is node_dim when using embedding,
        # or node_input_dim when using raw features (matching Keras).
        concat_dim = self._n0_dim + units
        if use_set2set:
            self.dense_set2set_in = nn.Linear(concat_dim, set2set_channels)
            self.pooling = PoolingSet2SetEncoder(
                channels=set2set_channels,
                T=set2set_T,
                pooling_method=set2set_pooling_method,
                init_qstar=set2set_init_qstar,
            )
            readout_dim = 2 * set2set_channels  # Set2Set output is 2*channels
        else:
            self.pooling = PoolingNodes(pooling_method=node_pooling)
            readout_dim = concat_dim  # From n0 || n concatenation

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        # Keras default: use_bias=[True, True, False] -- no bias on last layer.
        out_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=readout_dim,
            activation=out_act,
            use_bias=out_use_bias
        )

        # Apply Keras-compatible initialization (glorot_uniform for Dense,
        # orthogonal recurrent weights for GRU/LSTM, unit_forget_bias for LSTM).
        _apply_keras_init(self)

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            n0 = self.node_embedding(z.long())
        else:
            # Keras: n0 stays as raw input; no projection.
            n0 = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)

        # Project to hidden dimension; keep n0 as original embedding for skip connection
        n = self.dense_in(n0)
        num_nodes = n.size(0)

        # Compute edge network transformation matrices (once, reused each step)
        # edge_net_in: for messages from source (outgoing) nodes
        edge_net_in = self.edge_trafo_in(self.edge_mlp_in(edge_attr))
        # edge_net_out: for messages from target (ingoing) nodes
        edge_net_out = self.edge_trafo_out(self.edge_mlp_out(edge_attr))

        # Message passing with GRU update
        for i in range(self.depth):
            # Gather source and target node features for each edge
            n_in = gather_nodes_outgoing(n, edge_index)   # source features (M, units)
            n_out = gather_nodes_ingoing(n, edge_index)    # target features (M, units)

            # Matrix multiply: A_edge @ node_features for each direction
            m_in = self.matmul_msg(edge_net_in, n_in)      # (M, units)
            m_out = self.matmul_msg(edge_net_out, n_out)    # (M, units)

            # Concatenate messages from both directions
            eu = torch.cat([m_in, m_out], dim=-1)           # (M, 2*units)

            # Aggregate messages to target nodes
            agg = self.aggr(eu, edge_index, num_nodes)      # (N, 2*units)

            # GRU update: aggregated message is input, current node is hidden
            n = self.gru_update(agg, n)                     # (N, units)

        # Concatenate initial node embeddings with final
        n = torch.cat([n0, n], dim=-1)                      # (N, node_dim+units)

        # Graph-level pooling or node-level output
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            if self.use_set2set:
                n = self.dense_set2set_in(n)  # (N, units) for Set2Set
                out = self.pooling(n, batch, batch_size).squeeze(1)  # (B, 2*units)
            else:
                out = self.pooling(n, batch, batch_size)  # (B, node_dim+units)
        else:
            if self.use_set2set:
                out = self.dense_set2set_in(n)
            else:
                out = n

        out = self.output_mlp(out)
        return out


class NMPNCrystalModel(nn.Module):
    """Neural Message Passing Network for crystalline materials with periodic boundaries.

    Unlike the molecular NMPNModel which expects pre-computed edge features
    (data.edge_attr), this crystal variant computes edge distances from atomic
    positions using periodic lattice shifts, then expands them with a Gaussian
    basis to produce edge features. The rest of the architecture (edge networks
    with matrix multiplication messages, bidirectional message passing, GRU update,
    n0 concatenation, pooling, output MLP) is identical to the molecular version.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M), PyG convention [source, target].
        - data.batch: Batch assignment (N,).
        - data.lattice: Lattice matrix per graph (B, 3, 3).
        - data.edge_image: Periodic image shift vectors per edge (M, 3).
    """

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 units: int = 64,
                 edge_mlp_units: list = None,
                 edge_mlp_activation: str = "swish",
                 message_pooling: str = "sum",
                 use_set2set: bool = True,
                 set2set_T: int = 3,
                 set2set_channels: int = 32,
                 set2set_pooling_method: str = "sum",
                 set2set_init_qstar: str = "0",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "selu",
                 output_final_activation: str = "sigmoid",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1,
                 gauss_bins: int = 20,
                 gauss_distance: float = 4.0,
                 gauss_sigma: float = 0.4,
                 gauss_offset: float = 0.0,
                 expand_distance: bool = True):
        """Initialize NMPNCrystalModel.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            depth: Number of message passing steps.
            units: Hidden dimension for node features.
            edge_mlp_units: Hidden dims for the edge network. If None,
                defaults to [64, 64, 64] (matching Keras).
            edge_mlp_activation: Activation for the edge network.
            message_pooling: Aggregation method for messages ('sum', 'mean', etc.).
            use_set2set: Whether to use Set2Set pooling for graph readout.
            set2set_T: Number of Set2Set processing steps.
            set2set_channels: Hidden dim for Set2Set LSTM. Default 32 (matching Keras).
            node_pooling: Pooling method for graph readout (used if use_set2set=False).
            output_units: Hidden dims for output MLP. If None, [25, 10].
            output_activation: Activation for output MLP hidden layers.
            output_final_activation: Activation for output MLP final layer.
            num_targets: Number of output targets.
            output_embedding: Output embedding mode, "graph" for graph-level or "node" for node-level.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            gauss_bins: Number of Gaussian basis functions for distance expansion.
            gauss_distance: Maximum distance for Gaussian expansion.
            gauss_sigma: Width of Gaussian basis functions.
            gauss_offset: Offset for Gaussian basis functions.
            expand_distance: Whether to expand distances with Gaussian basis.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if output_units is None:
            output_units = [25, 10]

        # Determine edge feature dimension from Gaussian expansion
        if expand_distance:
            edge_dim = gauss_bins
        else:
            edge_dim = 1

        if edge_mlp_units is None:
            edge_mlp_units = [64, 64, 64]

        self.use_node_embedding = use_node_embedding
        self.use_set2set = use_set2set
        self.depth = depth
        self.units = units
        self.node_dim = node_dim
        self.expand_distance = expand_distance

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Gaussian basis layer for distance expansion
        if expand_distance:
            self.gauss_basis = GaussBasisLayer(
                bins=gauss_bins, distance=gauss_distance,
                sigma=gauss_sigma, offset=gauss_offset
            )

        # Project input node features to hidden dimension
        self.dense_in = nn.Linear(node_dim, units)

        # Two edge networks (incoming and outgoing), each maps edge features
        # to transformation matrices of shape (units, units) via MLP + reshape.
        self.edge_mlp_in = MLP(
            units=edge_mlp_units,
            input_dim=edge_dim,
            activation=edge_mlp_activation
        )
        self.edge_trafo_in = TrafoEdgeNetMessages(
            input_dim=edge_mlp_units[-1],
            target_shape=(units, units)
        )
        self.edge_mlp_out = MLP(
            units=edge_mlp_units,
            input_dim=edge_dim,
            activation=edge_mlp_activation
        )
        self.edge_trafo_out = TrafoEdgeNetMessages(
            input_dim=edge_mlp_units[-1],
            target_shape=(units, units)
        )

        # Matrix multiplication for messages
        self.matmul_msg = MatMulMessages()

        # Message aggregation
        self.aggr = AggregateLocalEdges(pooling_method=message_pooling)

        # GRU update: concatenated messages (2*units) are input,
        # node features (units) are hidden state
        self.gru_update = GRUUpdate(input_dim=2 * units, hidden_dim=units)

        # After message passing, concatenate initial embeddings (n0, dim=node_dim)
        # with final embeddings (n, dim=units), giving node_dim+units.
        concat_dim = node_dim + units
        if use_set2set:
            self.dense_set2set_in = nn.Linear(concat_dim, set2set_channels)
            self.pooling = PoolingSet2SetEncoder(
                channels=set2set_channels,
                T=set2set_T,
                pooling_method=set2set_pooling_method,
                init_qstar=set2set_init_qstar,
            )
            readout_dim = 2 * set2set_channels  # Set2Set output is 2*channels
        else:
            self.pooling = PoolingNodes(pooling_method=node_pooling)
            readout_dim = concat_dim  # From n0 || n concatenation

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        # Keras default: use_bias=[True, True, False] -- no bias on last layer.
        out_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=readout_dim,
            activation=out_act,
            use_bias=out_use_bias
        )

        # Apply Keras-compatible initialization.
        _apply_keras_init(self)

    def forward(self, data) -> torch.Tensor:
        """Forward pass for periodic crystal systems.

        Args:
            data: PyG Data batch object with lattice and edge_image attributes.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        edge_image = data.edge_image
        lattice = data.lattice

        # Node embedding
        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            n0 = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            n0 = self.node_projection(x)

        # Project to hidden dimension; keep n0 as original embedding for skip connection
        n = self.dense_in(n0)
        num_nodes = n.size(0)

        # Compute edge distances with periodic boundary conditions
        batch_edge = batch[edge_index[0]]
        pos_j = pos[edge_index[0]]
        pos_j = shift_periodic_lattice(pos_j, edge_image, lattice, batch_edge)
        pos_i = pos[edge_index[1]]
        diff = pos_j - pos_i
        ed = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)

        # Expand distances with Gaussian basis
        if self.expand_distance:
            edge_attr = self.gauss_basis(ed)
        else:
            edge_attr = ed

        # Compute edge network transformation matrices (once, reused each step)
        edge_net_in = self.edge_trafo_in(self.edge_mlp_in(edge_attr))
        edge_net_out = self.edge_trafo_out(self.edge_mlp_out(edge_attr))

        # Message passing with GRU update
        for i in range(self.depth):
            # Gather source and target node features for each edge
            n_in = gather_nodes_outgoing(n, edge_index)   # source features (M, units)
            n_out = gather_nodes_ingoing(n, edge_index)    # target features (M, units)

            # Matrix multiply: A_edge @ node_features for each direction
            m_in = self.matmul_msg(edge_net_in, n_in)      # (M, units)
            m_out = self.matmul_msg(edge_net_out, n_out)    # (M, units)

            # Concatenate messages from both directions
            eu = torch.cat([m_in, m_out], dim=-1)           # (M, 2*units)

            # Aggregate messages to target nodes
            agg = self.aggr(eu, edge_index, num_nodes)      # (N, 2*units)

            # GRU update: aggregated message is input, current node is hidden
            n = self.gru_update(agg, n)                     # (N, units)

        # Concatenate initial node embeddings with final
        n = torch.cat([n0, n], dim=-1)                      # (N, node_dim+units)

        # Graph-level pooling or node-level output
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            if self.use_set2set:
                n = self.dense_set2set_in(n)  # (N, units) for Set2Set
                out = self.pooling(n, batch, batch_size).squeeze(1)  # (B, 2*units)
            else:
                out = self.pooling(n, batch, batch_size)  # (B, node_dim+units)
        else:
            if self.use_set2set:
                out = self.dense_set2set_in(n)
            else:
                out = n

        out = self.output_mlp(out)
        return out
