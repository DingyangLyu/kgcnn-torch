"""INorp (Interaction Network with Pooling) model.

This torch implementation is aligned with the kgcnn Keras literature version
(`kgcnn.literature.INorp`) used in this repo's parity harness:
- categorical node + edge embeddings
- per-graph `graph_attributes` gathered to nodes (no projection)
- per-layer MLPs whose *input dims* differ for layer 0 vs later layers
  because node features change from `node_dim` to `node_mlp_units[-1]`
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing, gather_state
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingSet2SetEncoder
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class INorpModel(nn.Module):
    """Interaction Network with Pooling (INorp)."""

    def __init__(self,
                 node_dim: int = 64,
                 depth: int = 3,
                 edge_dim: int = 1,
                 edge_mlp_units: list | None = None,
                 edge_mlp_activation: str | list = "relu",
                 node_mlp_units: list | None = None,
                 node_mlp_activation: str | list = "relu",
                 message_pooling: str = "mean",
                 use_set2set: bool = False,
                 set2set_T: int = 3,
                 set2set_channels: int = 32,
                 use_graph_state: bool = True,
                 graph_state_dim: int = 64,
                 node_pooling: str = "mean",
                 output_units: list = None,
                 output_activation: str = "relu",
                 output_final_activation: str = "sigmoid",
                 output_use_bias: list = None,
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_input_dim: int = 1,
                 use_edge_embedding: bool = True,
                 num_edge_embeddings: int = 5,
                 edge_embedding_dim: int = 64):
        """Initialize INorp model.

        Args:
            depth: Number of interaction blocks.
            message_pooling: Aggregation method for edge-to-node messages.
            use_set2set: Whether to use Set2Set pooling for graph readout.
            set2set_T: Number of Set2Set processing steps.
            set2set_channels: Number of channels for Set2Set encoder. Keras default is 32.
            use_graph_state: Whether to use a global graph state.
            graph_state_dim: Dimension of the graph state feature.
            node_pooling: Pooling method for graph readout (used if use_set2set=False).
            output_units: Hidden dims for output MLP. If None, [25, 10].
            output_activation: Activation for hidden layers of output MLP.
            output_final_activation: Activation for the final layer of output MLP.
                Keras default is 'sigmoid'.
            output_use_bias: Per-layer use_bias for output MLP. If None,
                defaults to [True, True, False] matching Keras.
            num_targets: Number of output targets.
            output_embedding: Output embedding mode, "graph" for graph-level or "node" for node-level.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            node_input_dim: Input feature dimension when use_node_embedding=False.
            use_edge_embedding: Whether to embed integer edge types from `data.edge_type`.
            num_edge_embeddings: Edge vocabulary size for embedding.
            edge_embedding_dim: Output dim for edge embedding.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if output_units is None:
            output_units = [25, 10]

        # Keras literature defaults.
        if edge_mlp_units is None:
            edge_mlp_units = [100, 100, 100, 100, 50]
        if node_mlp_units is None:
            node_mlp_units = [100, 50]

        def _norm_act(units_list: list[int], act: str | list) -> list[str]:
            if isinstance(act, (list, tuple)):
                if len(act) != len(units_list):
                    raise ValueError("activation list length must match units length")
                return [str(a) for a in act]
            if not units_list:
                return []
            return [str(act)] * (len(units_list) - 1) + ["linear"]

        edge_mlp_activation = _norm_act(edge_mlp_units, edge_mlp_activation)
        node_mlp_activation = _norm_act(node_mlp_units, node_mlp_activation)

        self.depth = int(depth)

        self.use_node_embedding = use_node_embedding
        self.use_graph_state = use_graph_state
        self.graph_state_dim = int(graph_state_dim) if (use_graph_state and graph_state_dim is not None) else 0

        self.use_edge_embedding = use_edge_embedding
        self.num_edge_embeddings = int(num_edge_embeddings)
        self.edge_embedding_dim = int(edge_embedding_dim)

        self.node_input_dim = node_input_dim
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        else:
            self.node_projection = nn.Linear(node_input_dim, node_dim)

        # Optional edge embedding (Keras uses an Embedding for `edge_number`).
        if use_edge_embedding:
            self.edge_embedding = nn.Embedding(self.num_edge_embeddings, self.edge_embedding_dim)
            keras_uniform_init_embedding_(self.edge_embedding)
            effective_edge_dim = self.edge_embedding_dim
        else:
            self.edge_embedding = None
            effective_edge_dim = int(edge_dim)

        # Per-layer dims follow Keras behavior:
        # layer 0: node_dim -> node_mlp_units[-1]; subsequent layers keep node_mlp_units[-1].
        self.node_dim0 = int(node_dim)
        self.node_dim = int(node_mlp_units[-1]) if node_mlp_units else int(node_dim)
        self.edge_out_dim = int(edge_mlp_units[-1]) if edge_mlp_units else self.node_dim
        graph_extra_dim = self.graph_state_dim if self.use_graph_state else 0

        # Build per-depth blocks in Keras' ordering: edge_mlp_i then node_mlp_i.
        # This also makes shape-based weight porting deterministic for ambiguous shapes.
        self.blocks = nn.ModuleList()
        for i in range(self.depth):
            nd = self.node_dim0 if i == 0 else self.node_dim
            edge_input_dim = 2 * nd + effective_edge_dim
            node_input_dim = nd + self.edge_out_dim + graph_extra_dim
            self.blocks.append(nn.ModuleDict({
                "edge_mlp": MLP(
                    units=edge_mlp_units,
                    input_dim=edge_input_dim,
                    activation=edge_mlp_activation
                ),
                "node_mlp": MLP(
                    units=node_mlp_units,
                    input_dim=node_input_dim,
                    activation=node_mlp_activation
                ),
            }))

        # Message aggregation
        self.aggr = AggregateLocalEdges(pooling_method=message_pooling)

        # Graph-level pooling
        self.use_set2set = use_set2set
        if use_set2set:
            # Keras applies Dense(set2set_channels, "linear") before Set2Set
            self.set2set_dense = nn.Linear(self.node_dim, set2set_channels)
            self.pooling = PoolingSet2SetEncoder(channels=set2set_channels, T=set2set_T)
            readout_dim = 2 * set2set_channels
        else:
            self.pooling = PoolingNodes(pooling_method=node_pooling)
            readout_dim = self.node_dim

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + [output_final_activation]
        if output_use_bias is None:
            output_use_bias = [True] * len(output_units) + [False]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=readout_dim,
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
        edge_index = data.edge_index
        batch = data.batch

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Node embedding
        if self.use_node_embedding:
            z = data.z if hasattr(data, "z") and data.z is not None else data.x
            n = self.node_embedding(z.long())
        else:
            x = data.x if hasattr(data, "x") and data.x is not None else data.z.float().unsqueeze(-1)
            n = self.node_projection(x)
        num_nodes = int(n.size(0))

        # Edge features
        if self.use_edge_embedding and self.edge_embedding is not None:
            et = getattr(data, "edge_type", None)
            if et is None and hasattr(data, "edge_attr") and data.edge_attr is not None and data.edge_attr.ndim == 1:
                et = data.edge_attr.to(dtype=torch.int64)
            if et is None:
                et = torch.ones((edge_index.size(1),), dtype=torch.int64, device=n.device)
            ed = self.edge_embedding(et.to(dtype=torch.int64).reshape(-1))
        else:
            edge_attr = getattr(data, "edge_attr", None)
            if edge_attr is None:
                edge_attr = torch.zeros((edge_index.size(1), 1), dtype=torch.float32, device=n.device)
            ed = edge_attr.to(dtype=torch.float32)
            if ed.ndim == 1:
                ed = ed.unsqueeze(-1)
            if ed.size(-1) != self.edge_embedding_dim and self.use_edge_embedding:
                ed = ed[:, : self.edge_embedding_dim]

        # Graph state (graph_attributes) gathered to nodes, matching Keras GatherState.
        ev = None
        if self.use_graph_state and self.graph_state_dim > 0:
            gs = getattr(data, "graph_state", None)
            if gs is None:
                gs = torch.zeros((batch_size, self.graph_state_dim), dtype=torch.float32, device=n.device)
            else:
                gs = gs.to(dtype=torch.float32).reshape(batch_size, -1)
                if gs.size(-1) != self.graph_state_dim:
                    if gs.size(-1) > self.graph_state_dim:
                        gs = gs[:, : self.graph_state_dim]
                    else:
                        pad = torch.zeros((batch_size, self.graph_state_dim - gs.size(-1)), dtype=gs.dtype, device=gs.device)
                        gs = torch.cat([gs, pad], dim=-1)
            ev = gather_state(gs, batch)

        # Interaction blocks
        for i in range(self.depth):
            # Gather node pairs for edges
            n_j = gather_nodes_outgoing(n, edge_index)
            n_i = gather_nodes_ingoing(n, edge_index)

            edge_input = torch.cat([n_j, n_i, ed], dim=-1)
            eu = self.blocks[i]["edge_mlp"](edge_input)

            # Aggregate edge messages to nodes
            agg = self.aggr(eu, edge_index, num_nodes)

            if ev is not None:
                node_input = torch.cat([n, agg, ev], dim=-1)
            else:
                node_input = torch.cat([n, agg], dim=-1)
            n = self.blocks[i]["node_mlp"](node_input)

        # Graph-level pooling or node-level output
        if self.output_embedding == "graph":
            if self.use_set2set:
                n_pool = self.set2set_dense(n)
                out = self.pooling(n_pool, batch, batch_size)
            else:
                out = self.pooling(n, batch, batch_size)
        else:
            out = n

        out = self.output_mlp(out)
        return out
