"""CMPNN (Communicative Message Passing Neural Network) model.

Reference: Song et al., Communicative Representation Learning on Attributed
Molecular Graphs (2020).

Faithfully implements the CMPNN algorithm as described in the Keras kgcnn
reference implementation. The key difference from DMPNN is the "communicate"
operation that uses the element-wise product of sum-aggregated and
max-aggregated edge messages to update node features.
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_edges_pairs
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingNodesGRU
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class CMPNNModel(nn.Module):
    """CMPNN model for molecular property prediction.

    Implements the communicative message passing algorithm:
    1. Initialize node features h0 and edge features he0.
    2. For each step:
       a) Node update: m = sum_agg(he) * max_agg(he), h = h + m
       b) Edge update: h_out = gather(h, source), e_rev = gather(he, pairs),
          he = activation(Dense(h_out - e_rev) + he0)
    3. Final: m_final = sum_agg(he) * max_agg(he),
       h_final = Dense(concat(m_final, h, h0))

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Edge features (M, E).
        - data.batch: Batch assignment (N,).
        - data.edge_pair_index: Reverse edge mapping (M,).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 14,
                 depth: int = 5,
                 units: int = 300,
                 dropout: float = 0.1,
                 activation: str = "relu",
                 node_dense_activation: str = "linear",
                 use_final_gru: bool = True,
                 gru_units: int = None,
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_use_bias: list = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        super().__init__()
        if output_units is None:
            output_units = [300, 100]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]

        if gru_units is None:
            gru_units = units

        self.output_embedding = output_embedding
        self.depth = depth
        self.use_node_embedding = use_node_embedding
        self.use_final_gru = use_final_gru

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Initialize node features: Dense(n, activation) -> h0
        # Keras: node_initialize has {"units": 300, "activation": "relu"}
        self.node_init = nn.Linear(node_dim, units)
        self.node_init_act = get_activation(activation)

        # Initialize edge features: Dense(ed, activation) -> he0
        # Keras: edge_initialize has {"units": 300, "activation": "relu"}
        self.edge_init = nn.Linear(edge_dim, units)
        self.edge_init_act = get_activation(activation)

        self.activation = get_activation(activation)

        # Edge update Dense layers (one per message passing step, matching Keras).
        self.edge_denses = nn.ModuleList([
            nn.Linear(units, units) for _ in range(max(depth - 1, 1))
        ])

        # Dropout for edge features.
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

        # Aggregation layers: sum and max for communicative operation.
        self.aggr_sum = AggregateLocalEdges(pooling_method="sum")
        self.aggr_max = AggregateLocalEdges(pooling_method="max")

        # Final node Dense: concat(m, h, h0) -> h_final.
        # Keras: Dense(**node_dense) which includes activation from config.
        self.node_dense = nn.Linear(units * 3, units)
        self.node_dense_act = get_activation(node_dense_activation)

        # Graph-level readout: GRU-based (Keras default) or simple pooling.
        if use_final_gru:
            self.pooling = PoolingNodesGRU(units=gru_units)
        else:
            self.pooling = PoolingNodes(pooling_method=node_pooling)

        pool_out_dim = gru_units if use_final_gru else units
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=pool_out_dim,
            activation=out_act,
            use_bias=output_use_bias
        )

    def forward(self, data) -> torch.Tensor:
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        edge_pair_index = data.edge_pair_index  # reverse edge mapping

        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            n = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            n = self.node_projection(x)

        num_nodes = n.size(0)

        # Initialize node and edge features (with activation, matching Keras).
        h0 = self.node_init_act(self.node_init(n))  # (N, units)
        he0 = self.edge_init_act(self.edge_init(edge_attr))  # (M, units)

        h = h0
        he = he0

        # Message passing loop (depth - 1 steps, matching Keras).
        for i in range(self.depth - 1):
            # Node update via communicative aggregation.
            # m_pool = sum aggregation of edge features to target nodes.
            m_pool = self.aggr_sum(he, edge_index, num_nodes)  # (N, units)
            # m_max = max aggregation of edge features to target nodes.
            m_max = self.aggr_max(he, edge_index, num_nodes)  # (N, units)
            # CMPNN innovation: element-wise product of sum and max.
            m = m_pool * m_max  # (N, units)
            # Node update: h = h + m.
            h = h + m

            # Edge update.
            # Gather outgoing node features for each edge (source node).
            h_out = gather_nodes_outgoing(h, edge_index)  # (M, units)
            # Gather reverse edge features.
            e_rev = gather_edges_pairs(he, edge_pair_index)  # (M, units)
            # Compute new edge features: activation(Dense(h_out - e_rev) + he0).
            he = h_out - e_rev
            he = self.edge_denses[i](he)
            he = he + he0
            he = self.activation(he)
            he = self.dropout(he)

        # Final step: one more communicative aggregation.
        m_pool = self.aggr_sum(he, edge_index, num_nodes)  # (N, units)
        m_max = self.aggr_max(he, edge_index, num_nodes)  # (N, units)
        m = m_pool * m_max  # (N, units)

        # Final node features: concat(m, h, h0) -> Dense.
        h_final = torch.cat([m, h, h0], dim=-1)  # (N, 3*units)
        h_final = self.node_dense_act(self.node_dense(h_final))  # (N, units)

        # Pooling.
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(h_final, batch, batch_size)
        else:
            out = h_final
        out = self.output_mlp(out)
        return out
