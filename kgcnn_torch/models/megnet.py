"""MEGNet (MatErials Graph Network) model.

Reference: Chen et al., Graph Networks as a Universal Machine Learning Framework
for Molecules and Crystals (2019).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing, gather_state
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.pooling import PoolingNodes, PoolingSet2SetEncoder
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.geom import GaussBasisLayer, shift_periodic_lattice
from kgcnn_torch.ops.activ import get_activation
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class MEGNetBlock(nn.Module):
    """Single MEGNet interaction block.

    Performs edge update, node update, and graph state update.
    The update order follows the original MEGNet paper: edges -> nodes -> graph state.

    Matches Keras MEGnetBlock: each sub-MLP is 3 Dense layers where the last
    Dense uses activation='linear' (no activation).

    Concatenation order matches Keras:
      - Edge update: [target || source (from GatherNodes), edge, state]
      - Node update: [aggregated_edges, node, state]
      - State update: [mean_edges, mean_nodes, state]
    """

    def __init__(self,
                 edge_dim: int,
                 node_dim: int,
                 state_dim: int,
                 units_edge: list = None,
                 units_node: list = None,
                 units_state: list = None,
                 activation: str = "softplus2"):
        """Initialize MEGNet block.

        Args:
            edge_dim: Edge feature dimension.
            node_dim: Node feature dimension.
            state_dim: Graph state feature dimension.
            units_edge: Hidden dims for edge update MLP (3 layers). Defaults to [edge_dim]*3.
            units_node: Hidden dims for node update MLP (3 layers). Defaults to [node_dim]*3.
            units_state: Hidden dims for state update MLP (3 layers). Defaults to [state_dim]*3.
            activation: Activation function name (applied to all but last Dense layer).
        """
        super().__init__()
        if units_edge is None:
            units_edge = [edge_dim, edge_dim, edge_dim]
        if units_node is None:
            units_node = [node_dim, node_dim, node_dim]
        if units_state is None:
            units_state = [state_dim, state_dim, state_dim]

        # Keras concatenation order for edge update: [GatherNodes(target,source), edge, state]
        # GatherNodes default: split_indices=(0,1) with index_receive=0, index_send=1
        # So it gathers [target, source] concatenated.
        edge_input_dim = 2 * node_dim + edge_dim + state_dim
        self.edge_dense_layers = nn.ModuleList()
        in_d = edge_input_dim
        for i, out_d in enumerate(units_edge):
            self.edge_dense_layers.append(nn.Linear(in_d, out_d))
            in_d = out_d

        # Node update: [agg_edges, node, state]
        node_input_dim = units_edge[-1] + node_dim + state_dim
        self.node_dense_layers = nn.ModuleList()
        in_d = node_input_dim
        for i, out_d in enumerate(units_node):
            self.node_dense_layers.append(nn.Linear(in_d, out_d))
            in_d = out_d

        # State update: [mean_edges, mean_nodes, state]
        state_input_dim = units_edge[-1] + units_node[-1] + state_dim
        self.state_dense_layers = nn.ModuleList()
        in_d = state_input_dim
        for i, out_d in enumerate(units_state):
            self.state_dense_layers.append(nn.Linear(in_d, out_d))
            in_d = out_d

        self.activation = get_activation(activation)

        # Aggregation for edges -> nodes (mean, as in original MEGNet)
        self.edge_aggr = AggregateLocalEdges(pooling_method="mean")

        # Pooling for node/edge mean to update graph state
        self.pool_nodes = PoolingNodes(pooling_method="mean")
        self.pool_edges = PoolingNodes(pooling_method="mean")

    def _apply_mlp(self, x, layers):
        """Apply MLP with activation on all but last layer (linear last layer)."""
        for i, layer in enumerate(layers):
            x = layer(x)
            if i < len(layers) - 1:
                x = self.activation(x)
        return x

    def forward(self,
                n: torch.Tensor,
                e: torch.Tensor,
                state: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor,
                batch_edge: torch.Tensor,
                batch_size: int) -> tuple:
        """Forward pass for one MEGNet block.

        Args:
            n: Node features (N, node_dim).
            e: Edge features (M, edge_dim).
            state: Graph state features (B, state_dim).
            edge_index: Edge indices (2, M).
            batch: Node batch assignment (N,).
            batch_edge: Edge batch assignment (M,).
            batch_size: Number of graphs in batch.

        Returns:
            Tuple of updated (n, e, state).
        """
        num_nodes = n.size(0)

        # --- Edge update ---
        # Keras GatherNodes default: split_indices=(0,1) where index 0=receive=target, 1=send=source
        # So e_n = [target, source] concatenated
        n_tgt = gather_nodes_ingoing(n, edge_index)    # (M, node_dim) - target (receive)
        n_src = gather_nodes_outgoing(n, edge_index)   # (M, node_dim) - source (send)
        state_e = gather_state(state, batch_edge)      # (M, state_dim)
        # Keras: ec = Concatenate([e_n, edge_input, e_u]) = [target||source, edge, state]
        e_input = torch.cat([n_tgt, n_src, e, state_e], dim=-1)
        e_new = self._apply_mlp(e_input, self.edge_dense_layers)  # (M, edge_dim)

        # --- Node update ---
        # Keras: vc = Concatenate([vb, node_input, v_u]) = [agg_edges, node, state]
        agg_e = self.edge_aggr(e_new, edge_index, num_nodes)  # (N, edge_dim)
        state_n = gather_state(state, batch)                    # (N, state_dim)
        n_input = torch.cat([agg_e, n, state_n], dim=-1)
        n_new = self._apply_mlp(n_input, self.node_dense_layers)  # (N, node_dim)

        # --- Graph state update ---
        # Keras: ub = Concatenate([es, vs, env_input]) = [mean_edges, mean_nodes, state]
        mean_e = self.pool_edges(e_new, batch_edge, batch_size)  # (B, edge_dim)
        mean_n = self.pool_nodes(n_new, batch, batch_size)      # (B, node_dim)
        s_input = torch.cat([mean_e, mean_n, state], dim=-1)
        state_new = self._apply_mlp(s_input, self.state_dense_layers)  # (B, state_dim)

        return n_new, e_new, state_new


class MEGNetModel(nn.Module):
    """MEGNet model for graph-level property prediction.

    Implements the Materials Graph Network with external accumulator skip
    connections matching the Keras implementation:
      1. Apply initial FFN to node/edge/state features
      2. vp2=vp, ep2=ep, up2=up (accumulators)
      3. For each block i:
         - if has_ff and i>0: apply FFN to accumulated vp/ep/up -> vp2/ep2/up2
         - MEGNetBlock(vp2, ep2, up2) -> vp2, ep2, up2
         - Add to accumulators: vp += vp2, ep += ep2, up += up2

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,) int.
        - data.edge_index: Edge indices (2, M).
        - data.edge_attr: Edge features (M, edge_input_dim).
        - data.batch: Node batch assignment (N,).
        - data.graph_state: Optional graph-level state features (B, state_input_dim).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 64,
                 state_dim: int = 32,
                 edge_input_dim: int = 1,
                 state_input_dim: int = 0,
                 depth: int = 3,
                 block_units_edge: list = None,
                 block_units_node: list = None,
                 block_units_state: list = None,
                 node_ff_units: list = None,
                 edge_ff_units: list = None,
                 state_ff_units: list = None,
                 activation: str = "softplus2",
                 has_ff: bool = True,
                 dropout: float = None,
                 node_pooling: str = "sum",
                 use_set2set: bool = True,
                 set2set_channels: int = 16,
                 set2set_T: int = 3,
                 output_units: list = None,
                 output_activation: str = "softplus2",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 node_embedding_dim: int = None,
                 use_graph_embedding: bool = False,
                 graph_num_embeddings: int = 100,
                 graph_embedding_dim: int = None):
        """Initialize MEGNet model.

        Args:
            node_dim: Internal node feature dimension.
            edge_dim: Internal edge feature dimension.
            state_dim: Internal graph state dimension.
            edge_input_dim: Dimension of input edge features (data.edge_attr).
            state_input_dim: Dimension of input graph state. If 0, graph state is
                initialized to zeros of size state_dim.
            depth: Number of MEGNet interaction blocks.
            block_units_edge: Hidden dims for edge MLP in each block (3 layers).
            block_units_node: Hidden dims for node MLP in each block (3 layers).
            block_units_state: Hidden dims for state MLP in each block (3 layers).
            activation: Activation function name.
            has_ff: Whether to apply pre-block feedforward (Keras has_ff pattern).
            dropout: Dropout rate for skip connections. None means no dropout.
            node_pooling: Pooling method for final graph-level readout (used when
                use_set2set=False). Default 'sum' matches Keras PoolingNodes().
            use_set2set: Whether to use Set2Set encoder for final readout (default
                True, matching Keras).
            set2set_channels: LSTM hidden dim for Set2Set encoder.
            set2set_T: Number of processing steps for Set2Set encoder.
            output_units: Hidden dims for output MLP. If None, [node_dim, node_dim].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            output_embedding: Output embedding mode, "graph" for graph-level or "node" for node-level.
            use_node_embedding: Whether to embed integer node features.
            num_embeddings: Vocabulary size for node embedding.
            node_embedding_dim: Output dim for node embedding (Keras input_node_embedding.output_dim).
                If None, defaults to node_dim.
            use_graph_embedding: Whether to embed graph state as integer.
            graph_num_embeddings: Vocabulary size for graph embedding.
            graph_embedding_dim: Output dim for graph embedding (Keras input_graph_embedding.output_dim).
                If None, defaults to 64 (matching Keras input_graph_embedding.output_dim=64).
        """
        super().__init__()
        self.output_embedding = output_embedding
        if node_embedding_dim is None:
            node_embedding_dim = node_dim
        if graph_embedding_dim is None:
            graph_embedding_dim = 64
        if output_units is None:
            output_units = [32, 16]
        if node_ff_units is None:
            node_ff_units = [64, 32]
        if edge_ff_units is None:
            edge_ff_units = [64, 32]
        if state_ff_units is None:
            state_ff_units = [64, 32]

        # Effective internal dimensions are determined by FF output (no back-projection).
        # Keras FFN output dimension is simply units[-1].
        eff_node_dim = node_ff_units[-1] if node_ff_units else node_dim
        eff_edge_dim = edge_ff_units[-1] if edge_ff_units else edge_dim
        eff_state_dim = state_ff_units[-1] if state_ff_units else state_dim

        # Keras default meg_block_args: {node_embed: [64, 32, 32], edge_embed: [64, 32, 32], ...}
        # The first layer is 64, remaining match the effective dimension.
        if block_units_edge is None:
            block_units_edge = [64, eff_edge_dim, eff_edge_dim]
        if block_units_node is None:
            block_units_node = [64, eff_node_dim, eff_node_dim]
        if block_units_state is None:
            block_units_state = [64, eff_state_dim, eff_state_dim]

        self.use_node_embedding = use_node_embedding
        self.use_graph_embedding = use_graph_embedding
        self.state_dim = state_dim
        self.state_input_dim = state_input_dim
        # Dimension of state fed into state_ff_init (graph_embedding_dim if using graph embedding,
        # else the provided state_input_dim, else state_dim). Used for the zero-fallback in forward.
        self._state_ff_init_dim = graph_embedding_dim if use_graph_embedding else (
            state_input_dim if state_input_dim > 0 else state_dim
        )
        self.depth = depth
        self.has_ff = has_ff
        self.dropout = dropout

        # Node embedding (Keras: input_node_embedding with separate output_dim)
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_embedding_dim)
            keras_uniform_init_embedding_(self.node_embedding)

        # Graph state embedding (Keras: input_graph_embedding with separate output_dim)
        if use_graph_embedding:
            self.graph_embedding = nn.Embedding(graph_num_embeddings, graph_embedding_dim)
            keras_uniform_init_embedding_(self.graph_embedding)
            self.state_dense_in = None
        else:
            self.graph_embedding = None

        # Input FFN: initial projection from embedding dim to internal dim.
        # Keras FFN output dimension is simply units[-1]; no back-projection.
        node_ff_full = list(node_ff_units)
        edge_ff_full = list(edge_ff_units)
        state_ff_full = list(state_ff_units)
        node_ff_init_input = node_embedding_dim if use_node_embedding else node_dim
        state_ff_init_input = graph_embedding_dim if use_graph_embedding else (
            state_input_dim if state_input_dim > 0 else state_dim
        )
        self.node_ff_init = MLP(units=node_ff_full,
                                input_dim=node_ff_init_input, activation=activation)
        self.edge_ff_init = MLP(units=edge_ff_full,
                                input_dim=edge_input_dim, activation=activation)
        self.state_ff_init = MLP(units=state_ff_full,
                                 input_dim=state_ff_init_input,
                                 activation=activation)

        # Pre-block FFNs for i>0 (separate weights per block)
        # Input dim is the effective internal dim (FF output dim), not the original node_dim.
        if has_ff:
            self.node_ffs = nn.ModuleList()
            self.edge_ffs = nn.ModuleList()
            self.state_ffs = nn.ModuleList()
            for _ in range(depth - 1):  # Only for blocks 1..depth-1
                self.node_ffs.append(MLP(units=node_ff_full,
                                         input_dim=eff_node_dim, activation=activation))
                self.edge_ffs.append(MLP(units=edge_ff_full,
                                         input_dim=eff_edge_dim, activation=activation))
                self.state_ffs.append(MLP(units=state_ff_full,
                                          input_dim=eff_state_dim, activation=activation))

        # Dropout layers for skip connections (separate per stream)
        if dropout is not None and dropout > 0:
            self.dropout_node = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(depth)])
            self.dropout_edge = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(depth)])
            self.dropout_state = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(depth)])
            self._has_dropout = True
        else:
            self._has_dropout = False

        # MEGNet interaction blocks (use effective dims from FF output)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(MEGNetBlock(
                edge_dim=eff_edge_dim,
                node_dim=eff_node_dim,
                state_dim=eff_state_dim,
                units_edge=block_units_edge,
                units_node=block_units_node,
                units_state=block_units_state,
                activation=activation
            ))

        # Final graph-level readout: concat(pooled_nodes, pooled_edges, state)
        self.use_set2set = use_set2set
        if use_set2set:
            # Set2Set pooling (matches Keras default)
            self.node_set2set_proj = nn.Linear(eff_node_dim, set2set_channels)
            self.edge_set2set_proj = nn.Linear(eff_edge_dim, set2set_channels)
            self.pool_nodes_final = PoolingSet2SetEncoder(
                channels=set2set_channels, T=set2set_T,
                pooling_method="sum", init_qstar="0"
            )
            self.pool_edges_final = PoolingSet2SetEncoder(
                channels=set2set_channels, T=set2set_T,
                pooling_method="sum", init_qstar="0"
            )
            final_input_dim = 2 * set2set_channels + 2 * set2set_channels + eff_state_dim
        else:
            self.pool_nodes_final = PoolingNodes(pooling_method=node_pooling)
            self.pool_edges_final = PoolingNodes(pooling_method=node_pooling)
            final_input_dim = eff_node_dim + eff_edge_dim + eff_state_dim

        # Final dropout (matches Keras: Dropout applied after concat, before output MLP)
        if dropout is not None and dropout > 0:
            self.final_dropout = nn.Dropout(p=dropout)
        else:
            self.final_dropout = None

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=final_input_dim,
            activation=out_act
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        batch = data.batch

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Edge batch assignment: each edge belongs to the graph of its source node.
        batch_edge = batch[edge_index[0]]

        # Node embedding
        if self.use_node_embedding:
            n = self.node_embedding(z.long())
        else:
            n = z

        # Graph state initialization (to_pyg_list stores as 'state', also check 'graph_state')
        graph_state = getattr(data, 'graph_state', None)
        if graph_state is None:
            graph_state = getattr(data, 'state', None)
        if graph_state is not None:
            if self.graph_embedding is not None:
                idx = graph_state.long().clamp(0, self.graph_embedding.num_embeddings - 1)
                state = self.graph_embedding(idx)
                if state.dim() > 2:
                    state = state.squeeze(-2)
            else:
                state = graph_state
                if state.dim() == 1:
                    state = state.unsqueeze(-1)
                if self.state_input_dim and state.size(-1) != self.state_input_dim:
                    if state.size(-1) > self.state_input_dim:
                        state = state[..., : self.state_input_dim]
                    else:
                        pad = torch.zeros(
                            state.size(0), self.state_input_dim - state.size(-1),
                            device=state.device, dtype=state.dtype
                        )
                        state = torch.cat([state, pad], dim=-1)
                state = state.to(dtype=n.dtype)
        else:
            state = torch.zeros(batch_size, self._state_ff_init_dim, device=n.device, dtype=n.dtype)

        # Initial FFN projections (matches Keras: GraphMLP for node/edge, MLP for state)
        vp = self.node_ff_init(n)
        ep = self.edge_ff_init(edge_attr)
        up = self.state_ff_init(state)

        # External accumulator skip connection pattern (matches Keras _model.py)
        # vp2=vp, ep2=ep, up2=up initially
        vp2 = vp
        ep2 = ep
        up2 = up

        for i in range(self.depth):
            # Pre-block FFN for blocks after the first
            if self.has_ff and i > 0:
                vp2 = self.node_ffs[i - 1](vp)
                ep2 = self.edge_ffs[i - 1](ep)
                up2 = self.state_ffs[i - 1](up)

            # MEGNetBlock
            vp2, ep2, up2 = self.blocks[i](
                vp2, ep2, up2, edge_index, batch, batch_edge, batch_size
            )

            # Optional dropout on block output (separate per stream)
            if self._has_dropout:
                vp2 = self.dropout_node[i](vp2)
                ep2 = self.dropout_edge[i](ep2)
                up2 = self.dropout_state[i](up2)

            # Add to accumulators (skip connection)
            vp = vp + vp2
            ep = ep + ep2
            up = up + up2

        # Final readout: concatenate pooled nodes, pooled edges, and graph state
        if self.output_embedding == "graph":
            if self.use_set2set:
                vp_proj = self.node_set2set_proj(vp)
                ep_proj = self.edge_set2set_proj(ep)
                pooled_n = self.pool_nodes_final(vp_proj, batch, batch_size).squeeze(1)  # (B, 2*set2set_channels)
                pooled_e = self.pool_edges_final(ep_proj, batch_edge, batch_size).squeeze(1)  # (B, 2*set2set_channels)
            else:
                pooled_n = self.pool_nodes_final(vp, batch, batch_size)       # (B, node_dim)
                pooled_e = self.pool_edges_final(ep, batch_edge, batch_size)  # (B, edge_dim)
            out = torch.cat([pooled_n, pooled_e, up], dim=-1)
        else:
            out = vp

        if self.final_dropout is not None:
            out = self.final_dropout(out)

        out = self.output_mlp(out)
        return out


class MEGNetCrystalModel(nn.Module):
    """MEGNet model for crystalline materials with periodic boundary conditions.

    Unlike the molecular MEGNetModel which expects pre-computed edge features
    (data.edge_attr), this crystal variant computes edge distances from atomic
    positions using periodic lattice shifts, then expands them with a Gaussian
    basis to produce edge features. The rest of the architecture (MEGNet blocks,
    pooling, output MLP) is identical to the molecular version, using the
    external accumulator skip connection pattern.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - data.lattice: Lattice matrix per graph (B, 3, 3).
        - data.edge_image: Periodic image shift vectors per edge (M, 3).
        - data.graph_state: Optional graph-level state features (B, state_input_dim).
    """

    def __init__(self,
                 node_dim: int = 64,
                 edge_dim: int = 64,
                 state_dim: int = 32,
                 state_input_dim: int = 0,
                 depth: int = 3,
                 block_units_edge: list = None,
                 block_units_node: list = None,
                 block_units_state: list = None,
                 node_ff_units: list = None,
                 edge_ff_units: list = None,
                 state_ff_units: list = None,
                 activation: str = "softplus2",
                 has_ff: bool = True,
                 dropout: float = None,
                 node_pooling: str = "sum",
                 use_set2set: bool = True,
                 set2set_channels: int = 16,
                 set2set_T: int = 3,
                 output_units: list = None,
                 output_activation: str = "softplus2",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 use_graph_embedding: bool = False,
                 graph_num_embeddings: int = 100,
                 graph_embedding_dim: int = None,
                 gauss_bins: int = 20,
                 gauss_distance: float = 4.0,
                 gauss_sigma: float = 0.4,
                 gauss_offset: float = 0.0,
                 expand_distance: bool = True):
        """Initialize MEGNetCrystalModel.

        Args:
            node_dim: Internal node feature dimension.
            edge_dim: Internal edge feature dimension.
            state_dim: Internal graph state dimension.
            state_input_dim: Dimension of input graph state. If 0, graph state is
                initialized to zeros of size state_dim.
            depth: Number of MEGNet interaction blocks.
            block_units_edge: Hidden dims for edge MLP in each block (3 layers).
            block_units_node: Hidden dims for node MLP in each block (3 layers).
            block_units_state: Hidden dims for state MLP in each block (3 layers).
            activation: Activation function name.
            has_ff: Whether to apply pre-block feedforward (Keras has_ff pattern).
            dropout: Dropout rate for skip connections. None means no dropout.
            node_pooling: Pooling method for final graph-level readout. Default 'sum'
                matches Keras PoolingNodes().
            output_units: Hidden dims for output MLP. If None, [32, 16].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            output_embedding: Output embedding mode, "graph" for graph-level or "node" for node-level.
            use_node_embedding: Whether to embed integer node features.
            num_embeddings: Vocabulary size for node embedding.
            use_graph_embedding: Whether to embed graph state as integer.
            graph_num_embeddings: Vocabulary size for graph embedding.
            graph_embedding_dim: Output dim for graph embedding. If None, defaults to 64.
            gauss_bins: Number of Gaussian basis functions for distance expansion.
            gauss_distance: Maximum distance for Gaussian expansion.
            gauss_sigma: Width of Gaussian basis functions.
            gauss_offset: Offset for Gaussian basis functions.
            expand_distance: Whether to expand distances with Gaussian basis.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if graph_embedding_dim is None:
            graph_embedding_dim = 64
        if output_units is None:
            output_units = [32, 16]
        if node_ff_units is None:
            node_ff_units = [64, 32]
        if edge_ff_units is None:
            edge_ff_units = [64, 32]
        if state_ff_units is None:
            state_ff_units = [64, 32]

        # Effective internal dimensions are determined by FF output (no back-projection).
        eff_node_dim = node_ff_units[-1] if node_ff_units else node_dim
        eff_edge_dim = edge_ff_units[-1] if edge_ff_units else edge_dim
        eff_state_dim = state_ff_units[-1] if state_ff_units else state_dim

        # Keras default meg_block_args: {node_embed: [64, 32, 32], edge_embed: [64, 32, 32], ...}
        if block_units_edge is None:
            block_units_edge = [64, eff_edge_dim, eff_edge_dim]
        if block_units_node is None:
            block_units_node = [64, eff_node_dim, eff_node_dim]
        if block_units_state is None:
            block_units_state = [64, eff_state_dim, eff_state_dim]

        self.use_node_embedding = use_node_embedding
        self.use_graph_embedding = use_graph_embedding
        self.state_dim = state_dim
        self.state_input_dim = state_input_dim
        self._state_ff_init_dim = graph_embedding_dim if use_graph_embedding else (
            state_input_dim if state_input_dim > 0 else state_dim
        )
        self.depth = depth
        self.expand_distance = expand_distance
        self.has_ff = has_ff
        self.dropout = dropout

        # Node embedding
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)
        if use_graph_embedding:
            self.graph_embedding = nn.Embedding(graph_num_embeddings, graph_embedding_dim)
            keras_uniform_init_embedding_(self.graph_embedding)
            self.state_dense_in = None
        else:
            self.graph_embedding = None

        # Gaussian basis layer for distance expansion
        if expand_distance:
            self.gauss_basis = GaussBasisLayer(
                bins=gauss_bins, distance=gauss_distance,
                sigma=gauss_sigma, offset=gauss_offset
            )
            edge_input_dim = gauss_bins
        else:
            edge_input_dim = 1

        # Input FFN projections (no back-projection, output is units[-1])
        node_ff_full = list(node_ff_units)
        edge_ff_full = list(edge_ff_units)
        state_ff_full = list(state_ff_units)
        state_ff_init_input = graph_embedding_dim if use_graph_embedding else (
            state_input_dim if state_input_dim > 0 else state_dim
        )
        self.node_ff_init = MLP(units=node_ff_full,
                                input_dim=node_dim, activation=activation)
        self.edge_ff_init = MLP(units=edge_ff_full,
                                input_dim=edge_input_dim, activation=activation)
        self.state_ff_init = MLP(units=state_ff_full,
                                 input_dim=state_ff_init_input,
                                 activation=activation)

        # Pre-block FFNs for i>0
        if has_ff:
            self.node_ffs = nn.ModuleList()
            self.edge_ffs = nn.ModuleList()
            self.state_ffs = nn.ModuleList()
            for _ in range(depth - 1):
                self.node_ffs.append(MLP(units=node_ff_full,
                                         input_dim=eff_node_dim, activation=activation))
                self.edge_ffs.append(MLP(units=edge_ff_full,
                                         input_dim=eff_edge_dim, activation=activation))
                self.state_ffs.append(MLP(units=state_ff_full,
                                          input_dim=eff_state_dim, activation=activation))

        # Dropout layers (separate per stream)
        if dropout is not None and dropout > 0:
            self.dropout_node = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(depth)])
            self.dropout_edge = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(depth)])
            self.dropout_state = nn.ModuleList([nn.Dropout(p=dropout) for _ in range(depth)])
            self._has_dropout = True
        else:
            self._has_dropout = False

        # MEGNet interaction blocks (use effective dims from FF output)
        self.blocks = nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(MEGNetBlock(
                edge_dim=eff_edge_dim,
                node_dim=eff_node_dim,
                state_dim=eff_state_dim,
                units_edge=block_units_edge,
                units_node=block_units_node,
                units_state=block_units_state,
                activation=activation
            ))

        # Final graph-level readout
        self.use_set2set = use_set2set
        if use_set2set:
            self.node_set2set_proj = nn.Linear(eff_node_dim, set2set_channels)
            self.edge_set2set_proj = nn.Linear(eff_edge_dim, set2set_channels)
            self.pool_nodes_final = PoolingSet2SetEncoder(
                channels=set2set_channels, T=set2set_T,
                pooling_method="sum", init_qstar="0"
            )
            self.pool_edges_final = PoolingSet2SetEncoder(
                channels=set2set_channels, T=set2set_T,
                pooling_method="sum", init_qstar="0"
            )
            final_input_dim = 2 * set2set_channels + 2 * set2set_channels + eff_state_dim
        else:
            self.pool_nodes_final = PoolingNodes(pooling_method=node_pooling)
            self.pool_edges_final = PoolingNodes(pooling_method=node_pooling)
            final_input_dim = eff_node_dim + eff_edge_dim + eff_state_dim

        # Final dropout (matches Keras: Dropout applied after concat, before output MLP)
        if dropout is not None and dropout > 0:
            self.final_dropout = nn.Dropout(p=dropout)
        else:
            self.final_dropout = None

        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=final_input_dim,
            activation=out_act
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass for periodic crystal systems.

        Args:
            data: PyG Data batch object with lattice and edge_image attributes.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        edge_image = data.edge_image
        lattice = data.lattice

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Edge batch assignment
        batch_edge = batch[edge_index[0]]

        # Node embedding
        if self.use_node_embedding:
            n = self.node_embedding(z.long())
        else:
            n = z

        # Compute edge distances with periodic boundary conditions.
        # Distance is symmetric: |source - target| == |target - source|.
        pos_j = pos[edge_index[0]]
        pos_j = shift_periodic_lattice(pos_j, edge_image, lattice, batch_edge)
        pos_i = pos[edge_index[1]]
        diff = pos_j - pos_i
        ed = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + torch.finfo(diff.dtype).eps)

        # Expand distances with Gaussian basis
        if self.expand_distance:
            ed = self.gauss_basis(ed)

        # Graph state initialization (to_pyg_list stores as 'state', also check 'graph_state')
        graph_state = getattr(data, 'graph_state', None)
        if graph_state is None:
            graph_state = getattr(data, 'state', None)
        if graph_state is not None:
            if self.graph_embedding is not None:
                idx = graph_state.long().clamp(0, self.graph_embedding.num_embeddings - 1)
                state = self.graph_embedding(idx)
                if state.dim() > 2:
                    state = state.squeeze(-2)
            else:
                state = graph_state
                if state.dim() == 1:
                    state = state.unsqueeze(-1)
                if self.state_input_dim and state.size(-1) != self.state_input_dim:
                    if state.size(-1) > self.state_input_dim:
                        state = state[..., : self.state_input_dim]
                    else:
                        pad = torch.zeros(
                            state.size(0), self.state_input_dim - state.size(-1),
                            device=state.device, dtype=state.dtype
                        )
                        state = torch.cat([state, pad], dim=-1)
                state = state.to(dtype=n.dtype)
        else:
            state = torch.zeros(batch_size, self._state_ff_init_dim, device=n.device, dtype=n.dtype)

        # Initial FFN projections
        vp = self.node_ff_init(n)
        ep = self.edge_ff_init(ed)
        up = self.state_ff_init(state)

        # External accumulator skip connection pattern
        vp2 = vp
        ep2 = ep
        up2 = up

        for i in range(self.depth):
            if self.has_ff and i > 0:
                vp2 = self.node_ffs[i - 1](vp)
                ep2 = self.edge_ffs[i - 1](ep)
                up2 = self.state_ffs[i - 1](up)

            vp2, ep2, up2 = self.blocks[i](
                vp2, ep2, up2, edge_index, batch, batch_edge, batch_size
            )

            if self._has_dropout:
                vp2 = self.dropout_node[i](vp2)
                ep2 = self.dropout_edge[i](ep2)
                up2 = self.dropout_state[i](up2)

            vp = vp + vp2
            ep = ep + ep2
            up = up + up2

        # Final readout: concatenate pooled nodes, pooled edges, and graph state
        if self.output_embedding == "graph":
            if self.use_set2set:
                vp_proj = self.node_set2set_proj(vp)
                ep_proj = self.edge_set2set_proj(ep)
                pooled_n = self.pool_nodes_final(vp_proj, batch, batch_size).squeeze(1)
                pooled_e = self.pool_edges_final(ep_proj, batch_edge, batch_size).squeeze(1)
            else:
                pooled_n = self.pool_nodes_final(vp, batch, batch_size)
                pooled_e = self.pool_edges_final(ep, batch_edge, batch_size)
            out = torch.cat([pooled_n, pooled_e, up], dim=-1)
        else:
            out = vp

        if self.final_dropout is not None:
            out = self.final_dropout(out)

        out = self.output_mlp(out)
        return out
