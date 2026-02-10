"""DMPNN (Directed Message Passing Neural Network) model.

Reference: Yang et al., Analyzing Learned Molecular Representations for Property
Prediction (2019).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import (
    gather_nodes_outgoing, gather_edges_pairs, gather_state
)
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class DMPNNModel(nn.Module):
    """Directed Message Passing Neural Network.

    Key idea: messages live on *directed edges* rather than nodes.  At each
    step the message on edge (i->j) is updated using information aggregated
    from all incoming edges to node i *except* the reverse edge (j->i).  The
    reverse-edge exclusion is handled via ``data.edge_pair_index`` which maps
    each directed edge to its reverse counterpart.

    Matches Keras implementation:
      - Uses a SINGLE shared Dense layer across all message passing iterations
      - Adds initial edge embedding h0 directly as residual: h = Dense(agg) + h0
      - Then applies activation after the addition

    Expects PyG Data batch with:
        - data.z or data.x: Node features (N,) int or (N, F) float.
        - data.edge_index: Directed edge indices (2, M).
        - data.edge_attr: Edge features (M, edge_dim).
        - data.edge_pair_index: Index of the reverse edge for each edge (M,).
        - data.batch: Batch assignment (N,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 16,
                 depth: int = 5,
                 units: int = 128,
                 message_activation: str = "relu",
                 init_activation: str = None,
                 node_activation: str = None,
                 message_pooling: str = "sum",
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1,
                 use_edge_embedding: bool = False,
                 num_edge_embeddings: int = 0,
                 edge_embedding_dim: int = 0,
                 dropout_rate: float = 0.1,
                 use_graph_state: bool = False,
                 use_graph_embedding: bool = False,
                 graph_num_embeddings: int = 100,
                 graph_state_dim: int = None):
        """Initialize DMPNN model.

        Args:
            node_dim: Dimension of node features after embedding.
            edge_dim: Dimension of input edge features (or after edge embedding).
            depth: Number of directed message passing iterations.
            units: Hidden dimension for edge messages.
            message_activation: Activation function for message updates.
            node_activation: Activation for the node readout Dense layer. If None,
                defaults to message_activation (matching Keras node_dense activation).
            message_pooling: Aggregation method for incoming messages at nodes.
            node_pooling: Pooling method for graph-level readout.
            output_units: Hidden dims for the output MLP.  If None, [units, units].
            output_activation: Activation for the output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to use nn.Embedding for integer node features.
            num_embeddings: Vocabulary size for the node embedding.
            use_edge_embedding: Whether to embed integer edge features.
            num_edge_embeddings: Vocabulary size for optional edge embedding.
            edge_embedding_dim: Embedding dimension for edges (replaces edge_dim
                when use_edge_embedding is True).
            dropout_rate: Dropout rate applied after each message update.
        """
        super().__init__()
        if output_units is None:
            output_units = [64, 32]

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding
        self.use_edge_embedding = use_edge_embedding
        self.use_graph_state = use_graph_state
        self.use_graph_embedding = use_graph_embedding
        self.units = units

        # Node embedding
        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Optional edge embedding
        if use_edge_embedding and num_edge_embeddings > 0:
            self.edge_embedding = nn.Embedding(num_edge_embeddings, edge_embedding_dim)
            keras_uniform_init_embedding_(self.edge_embedding)
            effective_edge_dim = edge_embedding_dim
        else:
            self.edge_embedding = None
            effective_edge_dim = edge_dim

        if use_graph_state:
            if use_graph_embedding:
                graph_embed_dim = units if graph_state_dim is None else graph_state_dim
                self.graph_embedding = nn.Embedding(graph_num_embeddings, graph_embed_dim)
                keras_uniform_init_embedding_(self.graph_embedding)
                self.graph_state_project = None
                self.graph_state_out_dim = graph_embed_dim
            else:
                self.graph_embedding = None
                if graph_state_dim is None or graph_state_dim <= 0:
                    self.graph_state_project = nn.LazyLinear(units)
                    self.graph_state_out_dim = units
                else:
                    self.graph_state_project = nn.Linear(graph_state_dim, units)
                    self.graph_state_out_dim = units
        else:
            self.graph_embedding = None
            self.graph_state_project = None
            self.graph_state_out_dim = 0

        # Initial message: h0 = Dense(cat(n_j, e_ij)), then activation applied outside
        # Keras: h0 = Dense(**edge_initialize)(Concatenate([h_n0, ed]))
        # The activation is applied after: h0 is the output of Dense (linear)
        # Then in the loop, activation is applied after Add([h, h0])
        # But actually looking at Keras DMPNN more carefully:
        # h0 = Dense(**edge_initialize)(h0) -- edge_initialize has its own activation
        # So h0 = tau(W * [n_j, e_ij])
        self.message_init = nn.Linear(node_dim + effective_edge_dim, units)
        self.activation = get_activation(message_activation)
        _init_act = init_activation if init_activation is not None else message_activation
        self.init_act = get_activation(_init_act)

        # Separate activation for node readout (Keras node_dense has its own activation)
        if node_activation is None:
            self.node_activation = get_activation(message_activation)
        else:
            self.node_activation = get_activation(node_activation)

        # Single shared Dense layer for all message steps (matches Keras)
        # Keras: edge_dense_all = Dense(**edge_dense) used for ALL iterations
        # edge_dense should be linear (no activation), activation applied after Add
        self.W_h = nn.Linear(units, units)

        # Dropout for message steps
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

        self.aggr = AggregateLocalEdges(pooling_method=message_pooling)

        # Node readout: h_i = tau( W_o * cat(n_i, sum_j m_ji^T) )
        # Keras: mv = Concatenate([mv, n]) then Dense(**node_dense)(mv)
        self.node_readout = nn.Linear(node_dim + units, units)

        # Graph-level pooling and output
        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        output_in_dim = units + self.graph_state_out_dim
        # Keras default: use_bias=[True, True, False] -- no bias on last layer.
        out_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=output_in_dim,
            activation=out_act,
            use_bias=out_use_bias
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.  Must contain ``edge_pair_index`` of
                shape (M,) mapping each directed edge to its reverse edge index.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        edge_pair_index = data.edge_pair_index
        batch = data.batch
        graph_state = getattr(data, 'graph_state', None)

        # Node embedding
        if self.use_node_embedding:
            x = data.z if hasattr(data, 'z') and data.z is not None else data.x
            n = self.node_embedding(x.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            n = self.node_projection(x)

        num_nodes = n.size(0)

        # Edge embedding
        if self.use_edge_embedding and self.edge_embedding is not None:
            e = self.edge_embedding(edge_attr.long())
        else:
            e = edge_attr

        # Initial edge messages: h0 = tau( W_input * [n_j || e_ij] )
        # Keras: h_n0 = GatherNodesOutgoing()([n, edi])
        #        h0 = Concatenate()([h_n0, ed])
        #        h0 = Dense(**edge_initialize)(h0)
        n_j = gather_nodes_outgoing(n, edge_index)     # (M, node_dim)
        h0 = torch.cat([n_j, e], dim=-1)                # (M, node_dim + edge_dim)
        h0 = self.init_act(self.message_init(h0))           # (M, units)

        h = h0

        # Directed message passing iterations
        # Keras pattern:
        #   for i in range(depth):
        #       m_vw = DMPNNPPoolingEdgesDirected()([n, h, edi, ed_pairs])
        #       h = edge_dense_all(m_vw)  # Single shared Dense (linear)
        #       h = Add()([h, h0])         # Add initial embedding directly
        #       h = Activation(**edge_activation)(h)
        #       if dropout: h = Dropout(h)
        for t in range(self.depth):
            # DMPNNPPoolingEdgesDirected: aggregate messages at source node,
            # then subtract reverse edge message
            a = self.aggr(h, edge_index, num_nodes)  # (N, units)
            a_source = gather_nodes_outgoing(a, edge_index)  # (M, units)
            m_reverse = gather_edges_pairs(h, edge_pair_index)  # (M, units)
            a_corrected = a_source - m_reverse  # (M, units)

            # Single shared Dense + h0 residual + activation
            h = self.W_h(a_corrected)  # Linear transform (no activation)
            h = h + h0                 # Add initial embedding directly (not learned transform)
            h = self.activation(h)     # Activation after addition
            h = self.dropout(h)

        # Node readout: aggregate final messages to target nodes
        # Keras: mv = AggregateLocalEdges()([n, h, edi])
        #        mv = Concatenate()([mv, n])
        #        hv = Dense(**node_dense)(mv)
        m_agg = self.aggr(h, edge_index, num_nodes)  # (N, units)

        # Keras concatenation order: [mv, n] = [aggregated_messages, node_features]
        # Uses node_activation (matching Keras node_dense activation), not message_activation
        h_v = self.node_activation(self.node_readout(
            torch.cat([m_agg, n], dim=-1)
        ))  # (N, units)

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(h_v, batch, batch_size)
            if self.use_graph_state:
                if graph_state is not None:
                    if self.graph_embedding is not None:
                        gs = self.graph_embedding(graph_state.long())
                        if gs.dim() > 2:
                            gs = gs.squeeze(-2)
                    else:
                        gs = graph_state
                        if gs.dim() == 1:
                            gs = gs.unsqueeze(-1)
                        gs = self.graph_state_project(gs)
                else:
                    gs = torch.zeros(
                        out.size(0), self.graph_state_out_dim,
                        device=out.device, dtype=out.dtype
                    )
                out = torch.cat([gs, out], dim=-1)
        else:
            out = h_v
            if self.use_graph_state:
                if graph_state is not None:
                    if self.graph_embedding is not None:
                        gs = self.graph_embedding(graph_state.long())
                        if gs.dim() > 2:
                            gs = gs.squeeze(-2)
                    else:
                        gs = graph_state
                        if gs.dim() == 1:
                            gs = gs.unsqueeze(-1)
                        gs = self.graph_state_project(gs)
                else:
                    batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
                    gs = torch.zeros(
                        batch_size, self.graph_state_out_dim,
                        device=out.device, dtype=out.dtype
                    )
                gs_node = gather_state(gs, batch)
                out = torch.cat([out, gs_node], dim=-1)
        out = self.output_mlp(out)
        return out
