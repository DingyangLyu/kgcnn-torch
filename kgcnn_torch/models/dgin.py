"""DGIN (Deeply Initialized Graph Information Network) model.

Combines DMPNN-style directed edge message passing with GIN-style node updates.
Matches Keras implementation with:
  - h0 residual in DMPNN stage: h = activation(Dense(agg) + h0)
  - GIN_D uses initial node embedding (node_0) for self-loop, not current embedding
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_edges_pairs
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class GIN_D(nn.Module):
    """GIN convolution modified to use initial node embeddings (node_0) for self-loop.

    Matches Keras GIN_D layer:
        h_v^(k) = (1 + eps) * h_v^0 + sum_{u in N(v)} h_u^{k-1}

    Note: The non-linear mapping (MLP) is NOT included and should be applied after.
    """

    def __init__(self, pooling_method: str = "sum", epsilon_learnable: bool = False):
        super().__init__()
        self.pooling_method = pooling_method
        self.epsilon_learnable = epsilon_learnable
        eps = torch.zeros(1)
        if epsilon_learnable:
            self.eps = nn.Parameter(eps)
        else:
            self.register_buffer("eps", eps)
        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                x_0: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Current node features (N, F).
            edge_index: Edge indices (2, M).
            x_0: Initial node features (N, F) - used for self-loop term.

        Returns:
            Updated node features (N, F).
        """
        num_nodes = x.size(0)
        x_j = gather_nodes_outgoing(x, edge_index)
        agg = self.aggr(x_j, edge_index, num_nodes)
        # Modified to use x_0 instead of x for self-loop (matches Keras GIN_D)
        return (1.0 + self.eps) * x_0 + agg


class DGINModel(nn.Module):
    """DGIN model combining DMPNN edge message passing with GIN.

    Matches Keras implementation:
      - DMPNN stage uses single shared Dense + h0 residual + activation
      - GIN_D uses initial node embedding for self-loop

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
                 depth_dmpnn: int = 4,
                 depth_gin: int = 4,
                 units: int = 128,
                 dropout_dmpnn: float = 0.15,
                 dropout_gin: float = 0.15,
                 activation: str = "relu",
                 gin_mlp_units: list = None,
                 gin_mlp_activation: list = None,
                 gin_mlp_use_normalization: bool = True,
                 gin_mlp_normalization_technique: str = "graph_batch",
                 last_mlp_units: list = None,
                 node_pooling: str = "mean",
                 output_units: list = None,
                 output_activation: str = "relu",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1):
        super().__init__()
        if output_units is None:
            output_units = []
        if gin_mlp_units is None:
            gin_mlp_units = [64, 64]
        if gin_mlp_activation is None:
            gin_mlp_activation = ["relu", "linear"]
        if last_mlp_units is None:
            last_mlp_units = [64, 64]

        self.output_embedding = output_embedding
        self.depth_dmpnn = depth_dmpnn
        self.depth_gin = depth_gin
        self.use_node_embedding = use_node_embedding

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # DMPNN stage: edge message passing
        # Keras: h0 = Dense(**edge_initialize)(Concatenate([h_n0, ed]))
        #        h0 = Activation(**edge_activation)(h0)
        self.edge_init = nn.Linear(node_dim + edge_dim, units)
        self.activation = get_activation(activation)

        # Single shared Dense for all DMPNN message steps (matches Keras)
        # Keras: edge_dense_all = Dense(**edge_dense) -- one Dense for all iterations
        self.dmpnn_dense = nn.Linear(units, units)

        self.dmpnn_dropout = nn.Dropout(p=dropout_dmpnn) if dropout_dmpnn > 0 else nn.Identity()
        self.aggr = AggregateLocalEdges(pooling_method="sum")

        # Transition from DMPNN to GIN
        # Keras: n_units = gin_mlp["units"][-1]; h_v = Dense(n_units, activation='linear')(m_v)
        gin_out_dim = gin_mlp_units[-1] if len(gin_mlp_units) > 0 else units
        self.node_dense = nn.Linear(node_dim + units, gin_out_dim)

        # GIN_D stage (uses initial node embedding for self-loop)
        self.gin_convs = nn.ModuleList()
        self.gin_mlps = nn.ModuleList()
        for i in range(depth_gin):
            self.gin_convs.append(GIN_D(pooling_method="sum", epsilon_learnable=False))
            self.gin_mlps.append(MLP(
                units=gin_mlp_units, input_dim=gin_out_dim,
                activation=gin_mlp_activation,
                use_normalization=gin_mlp_use_normalization,
                normalization_technique=gin_mlp_normalization_technique
            ))

        self.gin_dropout = nn.Dropout(p=dropout_gin) if dropout_gin > 0 else nn.Identity()
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # Collect embeddings from each GIN layer
        # Keras: last_mlp has units=[64, 64], activation=["relu", "relu"]
        self.last_mlps = nn.ModuleList()
        for i in range(depth_gin + 1):
            self.last_mlps.append(MLP(
                units=last_mlp_units, input_dim=gin_out_dim, activation=output_activation
            ))

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=last_mlp_units[-1],
            activation=out_act
        )

    def forward(self, data) -> torch.Tensor:
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch
        edge_pair_index = data.edge_pair_index

        if self.use_node_embedding:
            z = data.z if hasattr(data, 'z') and data.z is not None else data.x
            n = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, 'x') and data.x is not None else data.z.float().unsqueeze(-1)
            n = self.node_projection(x)

        num_nodes = n.size(0)

        # --- DMPNN Stage ---
        # Keras: h0 = Dense(**edge_initialize)(Concatenate([h_n0, ed]))
        #        h0 = Activation(**edge_activation)(h0)
        n_j = gather_nodes_outgoing(n, edge_index)
        h0 = self.activation(self.edge_init(torch.cat([n_j, edge_attr], dim=-1)))

        # Keras pattern with h0 residual:
        #   h = h0
        #   for i in range(depthDMPNN):
        #       m_vw = DMPNNPPoolingEdgesDirected()([n, h, edi, ed_pairs])
        #       h = edge_dense_all(m_vw)
        #       h = Add()([h, h0])          # h0 residual
        #       h = Activation()(h)
        #       if dropout: h = Dropout(h)
        h = h0
        for _ in range(self.depth_dmpnn):
            agg = self.aggr(h, edge_index, num_nodes)
            agg_j = gather_nodes_outgoing(agg, edge_index)
            m_rev = gather_edges_pairs(h, edge_pair_index)
            agg_j = agg_j - m_rev

            # Shared Dense + h0 residual + activation (matches Keras DGIN)
            h = self.dmpnn_dense(agg_j)
            h = h + h0                     # Add h0 residual
            h = self.activation(h)
            h = self.dmpnn_dropout(h)

        # Aggregate final edge messages to nodes
        # Keras: m_v = AggregateLocalEdges()([n, h, edi])
        #        m_v = Concatenate()([n, m_v])
        a_i = self.aggr(h, edge_index, num_nodes)
        # Keras concatenation order: [n, m_v]
        m_v = torch.cat([n, a_i], dim=-1)

        # Keras: h_v = Dense(n_units, activation='linear')(m_v)
        h_v = self.node_dense(m_v)  # Linear projection (no activation)

        # Store initial node embedding for GIN_D self-loop
        h_v_0 = h_v

        # --- GIN_D Stage ---
        # GIN_D uses h_v_0 (initial node embedding) for self-loop term
        list_embeddings = [h_v_0]
        for i in range(self.depth_gin):
            # GIN_D: (1+eps)*h_v_0 + sum_neighbors(h_v)
            h_v = self.gin_convs[i](h_v, edge_index, h_v_0)
            h_v = self.gin_mlps[i](h_v, batch)
            list_embeddings.append(h_v)

        # Pool each layer's embedding, apply MLP, sum
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        pooled = []
        for i, emb in enumerate(list_embeddings):
            if self.output_embedding == "graph":
                p = self.pooling(emb, batch, batch_size)
            else:
                p = emb
            p = self.last_mlps[i](p)
            p = self.gin_dropout(p)
            pooled.append(p)

        out = sum(pooled)
        out = self.output_mlp(out)
        return out
