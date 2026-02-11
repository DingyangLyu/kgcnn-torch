"""MXMNet (Molecular Mechanics-Driven Machine Learning) model.

Reference: Zhang et al., Molecular Mechanics-Driven Graph Neural Network with
Multiplex Graph Attention for Molecular Property Prediction (2020).

Faithfully implements the Global and Local message passing blocks as described
in the Keras kgcnn reference implementation including:
- Global MP: two propagation steps with edge-feature multiplication and
  residual blocks.
- Local MP: angular message passing with two sets of spherical basis features,
  two angle indices, and per-layer output accumulation.
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.aggr import AggregateLocalEdges, Aggregate
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.geom import (
    BesselBasisLayer, SphericalBasisLayer, EdgeAngle,
    compute_edge_distances, shift_periodic_lattice
)
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.update import ResidualLayer
from kgcnn_torch.ops.activ import get_activation
from math import sqrt as _sqrt


class MXMNetGlobalMP(nn.Module):
    """Global message passing block for MXMNet.

    Matches the Keras MXMGlobalMP layer:
    1. h_mlp on node features (cross-layer mapping).
    2. propagate() x2: concat(x_i, x_j, edge_attr) -> MLP -> multiply by
       linear(edge_attr) -> aggregate -> add x.
    3. Residual blocks between the two propagation steps.
    """

    def __init__(self, units: int, pooling_method: str = "mean"):
        """Initialize MXMNetGlobalMP.

        Args:
            units: Node feature dimension.
            pooling_method: Aggregation method for messages.
        """
        super().__init__()
        self.dim = units

        # Cross-layer mapping MLP applied to h before message passing.
        self.h_mlp = nn.Sequential(
            nn.Linear(units, units),
            get_activation("swish"),
        )

        # Residual blocks for update function f_u.
        self.res1 = ResidualLayer(units)
        self.res2 = ResidualLayer(units)
        self.res3 = ResidualLayer(units)

        # MLP after first propagation + residual.
        self.mlp = nn.Sequential(
            nn.Linear(units, units),
            get_activation("swish"),
        )

        # Edge message MLP: concat(x_i, x_j, edge_attr) -> units.
        self.x_edge_mlp = nn.Sequential(
            nn.Linear(units * 3, units),
            get_activation("swish"),
        )

        # Linear projection of edge_attr for multiplicative gating.
        self.linear = nn.Linear(units, units, bias=False)

        self.pool = AggregateLocalEdges(pooling_method=pooling_method)

    def propagate(self, x, edge_attr, edge_index):
        """Single propagation step.

        Args:
            x: Node features (N, units).
            edge_attr: Edge features (M, units).
            edge_index: Edge indices (2, M).

        Returns:
            Updated node features (N, units).
        """
        num_nodes = x.size(0)

        # Gather source (x_j) and target (x_i) node features.
        x_i = x[edge_index[1]]  # target / ingoing
        x_j = x[edge_index[0]]  # source / outgoing

        # Prepare message: MLP(concat(x_i, x_j, edge_attr)).
        x_edge = torch.cat([x_i, x_j, edge_attr], dim=-1)
        x_edge = self.x_edge_mlp(x_edge)

        # Multiplicative gating by linear projection of edge_attr.
        edge_attr_lin = self.linear(edge_attr)
        x_edge = edge_attr_lin * x_edge

        # Aggregate messages to target nodes.
        x_p = self.pool(x_edge, edge_index, num_nodes)

        # Add skip connection from node features.
        return x_p + x

    def forward(self, x: torch.Tensor, edge_attr: torch.Tensor,
                edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Node features of shape (N, units).
            edge_attr: Edge features of shape (M, units).
            edge_index: Edge indices of shape (2, M).

        Returns:
            Updated node features of shape (N, units).
        """
        # Keep for residual skip connection.
        res_h = x

        # Cross-layer mapping.
        h = self.h_mlp(x)

        # First message passing.
        h = self.propagate(h, edge_attr, edge_index)

        # Update function f_u.
        h = self.res1(h)
        h = self.mlp(h)
        h = h + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Second message passing.
        h = self.propagate(h, edge_attr, edge_index)

        return h


class MXMNetLocalMP(nn.Module):
    """Local message passing block for MXMNet.

    Matches the Keras MXMLocalMP layer faithfully:
    - Two angular message passing paths using sbf1/sbf2 and angle_idx_1/angle_idx_2.
    - Per-layer output module producing intermediate predictions that are
      accumulated (summed) in the main model loop.
    """

    def __init__(self, units: int, output_units: int = 1,
                 activation: str = "swish",
                 pooling_method: str = "sum"):
        """Initialize MXMNetLocalMP.

        Args:
            units: Node feature dimension.
            output_units: Dimension of per-layer output prediction.
            activation: Activation function name.
            pooling_method: Aggregation method for messages.
        """
        super().__init__()
        self.dim = units
        self.output_dim = output_units

        # Cross-layer mapping MLP.
        self.h_mlp = nn.Sequential(
            nn.Linear(units, units),
            get_activation(activation),
        )

        # MLPs for message passing path 1.
        self.mlp_kj = nn.Sequential(
            nn.Linear(units * 3, units),
            get_activation(activation),
        )
        self.mlp_ji_1 = nn.Sequential(
            nn.Linear(units * 3, units),
            get_activation(activation),
        )

        # MLPs for message passing path 2.
        # After MP1, m is overwritten to (M, units), so input dim is units.
        self.mlp_jj = nn.Sequential(
            nn.Linear(units, units),
            get_activation(activation),
        )
        self.mlp_ji_2 = nn.Sequential(
            nn.Linear(units, units),
            get_activation(activation),
        )

        # SBF projections for the two angular paths.
        self.mlp_sbf1 = nn.Sequential(
            nn.Linear(units, units),
            get_activation(activation),
            nn.Linear(units, units),
            get_activation(activation),
        )
        self.mlp_sbf2 = nn.Sequential(
            nn.Linear(units, units),
            get_activation(activation),
            nn.Linear(units, units),
            get_activation(activation),
        )

        # Linear RBF projections for multiplicative weighting.
        self.lin_rbf1 = nn.Linear(units, units, bias=False)
        self.lin_rbf2 = nn.Linear(units, units, bias=False)

        # Residual layers.
        self.res1 = ResidualLayer(units, activation=activation)
        self.res2 = ResidualLayer(units, activation=activation)
        self.res3 = ResidualLayer(units, activation=activation)

        # Final RBF linear for aggregation weighting.
        self.lin_rbf_out = nn.Linear(units, units, bias=False)

        # Output module.
        self.y_mlp = MLP(
            units=[units, units, units], input_dim=units,
            activation=activation,
        )
        self.y_W = nn.Linear(units, output_units)
        # Initialize y_W with zeros as in Keras version.
        nn.init.zeros_(self.y_W.weight)
        if self.y_W.bias is not None:
            nn.init.zeros_(self.y_W.bias)

        # Aggregation layers.
        self.pool_mkj = Aggregate(pooling_method=pooling_method)
        self.pool_mjj = Aggregate(pooling_method=pooling_method)
        self.pool_h = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, h: torch.Tensor, rbf: torch.Tensor,
                sbf1: torch.Tensor, sbf2: torch.Tensor,
                edge_index: torch.Tensor,
                angle_idx_1: torch.Tensor,
                angle_idx_2: torch.Tensor):
        """Forward pass.

        Args:
            h: Node features (N, units).
            rbf: Radial basis features projected to (M, units).
            sbf1: Spherical basis features for angle set 1 (K1, units).
            sbf2: Spherical basis features for angle set 2 (K2, units).
            edge_index: Edge indices (2, M).
            angle_idx_1: Angle indices set 1 (2, K1), referencing edge pairs.
            angle_idx_2: Angle indices set 2 (2, K2), referencing edge pairs.

        Returns:
            Tuple of (updated_h, y):
                - updated_h: Updated node features (N, units).
                - y: Per-layer output (N, output_units).
        """
        num_nodes = h.size(0)
        num_edges = edge_index.size(1)
        res_h = h

        # Cross-layer mapping.
        h = self.h_mlp(h)

        # Build edge-level message features: concat(hi, hj, rbf).
        # In Keras: hi, hj = GatherNodes(split_indices=[0,1])
        # Keras: index 0 = receive/target, index 1 = send/source.
        # PyG: index 0 = source, index 1 = target.
        hi = h[edge_index[1]]  # target node features
        hj = h[edge_index[0]]  # source node features
        m = torch.cat([hi, hj, rbf], dim=-1)  # (M, 3*units)

        # === Message Passing 1 ===
        # m_kj path: apply MLP, weight by RBF, gather for angles, weight by SBF, aggregate.
        m_kj = self.mlp_kj(m)  # (M, units)
        w_rbf1 = self.lin_rbf1(rbf)  # (M, units)
        m_kj = m_kj * w_rbf1  # (M, units)

        # Gather m_kj for angle pairs.
        # Convention: angle_idx = [ji, kj] (DimeNetPP/Keras convention).
        # Keras GatherNodesOutgoing gathers at index 1 (send/outgoing = kj).
        m_kj_angle = m_kj[angle_idx_1[1]]  # (K1, units)

        # Weight by spherical basis features.
        sw_sbf1 = self.mlp_sbf1(sbf1)  # (K1, units)
        m_kj_angle = m_kj_angle * sw_sbf1  # (K1, units)

        # Aggregate back to edges.
        # Keras PoolingLocalMessages pools to index 0 (receive = ji).
        m_kj_agg = self.pool_mkj(m_kj_angle, angle_idx_1[0], num_edges)  # (M, units)

        # Direct path.
        m_ji_1 = self.mlp_ji_1(m)  # (M, units)

        m = m_ji_1 + m_kj_agg  # (M, units)

        # === Message Passing 2 (j'i path) ===
        m_jj = self.mlp_jj(m)  # (M, units)
        w_rbf2 = self.lin_rbf2(rbf)  # (M, units)
        m_jj = m_jj * w_rbf2  # (M, units)

        # Gather for angle set 2 (at index 1 = kj).
        m_jj_angle = m_jj[angle_idx_2[1]]  # (K2, units)

        # Weight by spherical basis features set 2.
        sw_sbf2 = self.mlp_sbf2(sbf2)  # (K2, units)
        m_jj_angle = m_jj_angle * sw_sbf2  # (K2, units)

        # Aggregate back to edges (pool to index 0 = ji).
        m_jj_agg = self.pool_mjj(m_jj_angle, angle_idx_2[0], num_edges)  # (M, units)

        # Direct path.
        m_ji_2 = self.mlp_ji_2(m)  # (M, units)

        m = m_ji_2 + m_jj_agg  # (M, units)

        # === Aggregation to nodes ===
        w_rbf = self.lin_rbf_out(rbf)  # (M, units)
        m = w_rbf * m  # (M, units)
        h = self.pool_h(m, edge_index, num_nodes)  # (N, units)

        # Update function f_u.
        h = self.res1(h)
        h = self.h_mlp(h)
        h = h + res_h
        h = self.res2(h)
        h = self.res3(h)

        # Output module.
        y = self.y_mlp(h)
        y = self.y_W(y)

        return h, y


class MXMNetModel(nn.Module):
    """MXMNet model for molecular property prediction.

    Combines global message passing (using radial basis features) with
    local message passing (using spherical basis features for angular
    information). Per-layer outputs from LocalMP are summed for the final
    readout, matching the Keras reference implementation.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M) for local edges.
        - data.batch: Batch assignment (N,).
        - data.angle_index_1: Angle indices set 1 (2, K1) for local MP.
        - data.angle_index_2: Angle indices set 2 (2, K2) for local MP.
        - data.range_index: Optional range edge indices (2, R) for global MP.
          If not provided, edge_index is used for global MP as well.
        - Optional (crystal): data.edge_image, data.lattice, data.batch_edge.

    When use_local_mp=False (default), a simplified mode is used with only
    GlobalMP blocks and a single edge index, matching the test expectations.
    """

    def __init__(self,
                 node_dim: int = 32,
                 depth: int = 3,
                 units: int = 32,
                 num_radial: int = 16,
                 num_spherical: int = 7,
                 num_radial_spherical: int = 6,
                 cutoff: float = 5.0,
                 cutoff_global: float = None,
                 envelope_exponent: int = 5,
                 activation: str = "swish",
                 mp_pooling: str = "sum",
                 global_mp_pooling: str = "mean",
                 use_local_mp: bool = True,
                 node_pooling: str = "sum",
                 last_mlp_units: list = None,
                 last_mlp_activation: str = "swish",
                 output_units: list = None,
                 output_activation: str = "swish",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 make_distance: bool = True,
                 use_output_mlp: bool = True):
        """Initialize MXMNet model.

        Args:
            node_dim: Embedding dimension for atomic numbers.
            depth: Number of message passing iterations.
            units: Hidden dimension for all layers.
            num_radial: Number of radial Bessel basis functions.
            num_spherical: Number of spherical harmonics (for local MP).
            num_radial_spherical: Number of radial basis functions inside the
                SphericalBasisLayer. Keras default is 6 (separate from the 16
                used in BesselBasisLayer).
            cutoff: Cutoff distance for local Bessel basis.
            cutoff_global: Cutoff distance for global Bessel basis. Defaults
                to cutoff if not specified.
            envelope_exponent: Exponent for envelope function in Bessel basis.
            activation: Activation function name.
            mp_pooling: Aggregation method for local message passing blocks.
            global_mp_pooling: Aggregation method for global MP blocks.
            use_local_mp: Whether to use local (angular) message passing blocks
                in addition to global MP.
            node_pooling: Pooling method for graph-level readout.
            last_mlp_units: Hidden dims for per-node MLP after interactions.
                If None, defaults to [units, units].
            last_mlp_activation: Activation for last MLP.
            output_units: Hidden dims for the output MLP (after pooling).
                If None, defaults to [units].
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            make_distance: Whether to compute distances from positions.
            use_output_mlp: Whether to apply output MLP after pooling.
        """
        super().__init__()
        if last_mlp_units is None:
            last_mlp_units = [units, units]
        if output_units is None:
            output_units = []
        if cutoff_global is None:
            cutoff_global = cutoff

        self.output_embedding = output_embedding
        self.depth = depth
        self.units = units
        self.use_node_embedding = use_node_embedding
        self.use_local_mp = use_local_mp
        self.make_distance = make_distance
        self.use_output_mlp = use_output_mlp
        self.num_targets = num_targets

        # Node embedding (Uniform[-sqrt(3), sqrt(3)] matching Keras EmbeddingDimeBlock).
        # Keras EmbeddingDimeBlock uses vocab size = input_dim + 1 (line 294 of
        # DimeNetPP/_layers.py), so we add 1 to num_embeddings.
        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings + 1, node_dim)
            nn.init.uniform_(self.node_embedding.weight, -_sqrt(3), _sqrt(3))

        # Keras MXMNet does not apply a separate Dense before message passing.
        # Keep a projection only when dimensions differ.
        if node_dim == units:
            self.dense_in = nn.Identity()
        else:
            self.dense_in = nn.Linear(node_dim, units)

        # Radial basis expansion for local edges.
        self.bessel_basis_local = BesselBasisLayer(
            num_radial=num_radial, cutoff=cutoff,
            envelope_exponent=envelope_exponent
        )

        # Radial basis expansion for global/range edges.
        self.bessel_basis_global = BesselBasisLayer(
            num_radial=num_radial, cutoff=cutoff_global,
            envelope_exponent=envelope_exponent
        )

        if use_local_mp:
            # Spherical basis expansion for local MP.
            # Keras uses a separate num_radial for spherical basis (default 6)
            # distinct from the Bessel basis num_radial (default 16).
            self.spherical_basis = SphericalBasisLayer(
                num_spherical=num_spherical, num_radial=num_radial_spherical,
                cutoff=cutoff, envelope_exponent=envelope_exponent
            )

            # MLPs to project RBF and SBF features to units dimension.
            self.mlp_rbf = nn.Sequential(
                nn.Linear(num_radial, units),
                get_activation(activation),
            )
            # Keras uses two independent GraphMLP instances for sbf_1 and sbf_2.
            self.mlp_sbf_1 = nn.Sequential(
                nn.Linear(num_spherical * num_radial_spherical, units),
                get_activation(activation),
            )
            self.mlp_sbf_2 = nn.Sequential(
                nn.Linear(num_spherical * num_radial_spherical, units),
                get_activation(activation),
            )
        # MLP to project global RBF features to units dimension.
        self.mlp_rbf_global = nn.Sequential(
            nn.Linear(num_radial, units),
            get_activation(activation),
        )

        # Global message passing blocks.
        self.global_mp_blocks = nn.ModuleList()
        for _ in range(depth):
            self.global_mp_blocks.append(MXMNetGlobalMP(
                units=units, pooling_method=global_mp_pooling
            ))

        # Local message passing blocks.
        if use_local_mp:
            self.local_mp_blocks = nn.ModuleList()
            for _ in range(depth):
                self.local_mp_blocks.append(MXMNetLocalMP(
                    units=units, output_units=num_targets,
                    activation=activation, pooling_method=mp_pooling
                ))

        # Per-node MLP and output (used only in non-local-MP mode).
        if not use_local_mp:
            self.last_mlp = MLP(
                units=last_mlp_units, input_dim=units,
                activation=last_mlp_activation
            )

        # Edge angle computation modules (created once, not per forward call).
        self.edge_angle_1 = EdgeAngle()
        self.edge_angle_2 = EdgeAngle(vector_scale=[1.0, -1.0])

        # Graph-level pooling.
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # Output MLP.
        if use_output_mlp:
            if use_local_mp:
                out_units = output_units + [num_targets]
                out_act = [output_activation] * len(output_units) + ["linear"]
                self.output_mlp = MLP(
                    units=out_units,
                    input_dim=num_targets,
                    activation=out_act
                )
            else:
                out_units = output_units + [num_targets]
                out_act = [output_activation] * len(output_units) + ["linear"]
                self.output_mlp = MLP(
                    units=out_units,
                    input_dim=last_mlp_units[-1],
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
        batch = data.batch

        # Node embedding.
        if self.use_node_embedding:
            n = self.node_embedding(z.long())
        else:
            n = z

        # Compute edge distances.
        if self.make_distance:
            pos = data.pos
            # Handle periodic systems.
            if hasattr(data, 'edge_image') and data.edge_image is not None:
                pos_j = pos[edge_index[0]]
                batch_edge = data.batch_edge if hasattr(data, 'batch_edge') else batch[edge_index[0]]
                pos_j = shift_periodic_lattice(pos_j, data.edge_image, data.lattice, batch_edge)
                pos_i = pos[edge_index[1]]
                diff = pos_j - pos_i
                dist_local = torch.sqrt((diff * diff).sum(dim=-1, keepdim=True) + 1e-8)
            else:
                dist_local = compute_edge_distances(pos, edge_index)
        else:
            dist_local = data.edge_attr

        # Local RBF expansion.
        rbf_local = self.bessel_basis_local(dist_local)

        # Determine range/global edge index.
        range_index = getattr(data, 'range_index', None)
        if range_index is not None:
            # Compute global distances.
            if self.make_distance:
                dist_global = compute_edge_distances(pos, range_index)
            else:
                dist_global = dist_local
            rbf_global = self.bessel_basis_global(dist_global)
        else:
            # Use local edges for global MP as well.
            range_index = edge_index
            rbf_global = rbf_local

        # Project RBF features for global MP.
        rbf_g = self.mlp_rbf_global(rbf_global)

        if self.use_local_mp:
            # Project RBF features for local MP.
            rbf_l = self.mlp_rbf(rbf_local)

            # Compute direction vectors for angles.
            if self.make_distance:
                pos_i = pos[edge_index[1]]
                pos_j = pos[edge_index[0]]
                v12 = pos_i - pos_j  # (M, 3)
            else:
                v12 = None

            # Compute spherical basis features for angle set 1.
            angle_idx_1 = getattr(data, 'angle_index_1', None)
            angle_idx_2 = getattr(data, 'angle_index_2', None)

            # Fall back to single angle_index if separate ones not provided.
            if angle_idx_1 is None:
                angle_idx_1 = getattr(data, 'angle_index', None)
            if angle_idx_2 is None:
                angle_idx_2 = getattr(data, 'angle_index', None)

            if angle_idx_1 is not None and v12 is not None:
                a_l_1 = self.edge_angle_1(v12, angle_idx_1)  # (K1, 1)
                sbf_1 = self.spherical_basis(dist_local, a_l_1, angle_idx_1)  # (K1, S*R)
                sbf_1 = self.mlp_sbf_1(sbf_1)  # (K1, units)
            else:
                sbf_1 = None

            if angle_idx_2 is not None and v12 is not None:
                a_l_2 = self.edge_angle_2(v12, angle_idx_2)  # (K2, 1)
                sbf_2 = self.spherical_basis(dist_local, a_l_2, angle_idx_2)  # (K2, S*R)
                sbf_2 = self.mlp_sbf_2(sbf_2)  # (K2, units)
            else:
                sbf_2 = None

        # Initial projection.
        n = self.dense_in(n)

        if self.use_local_mp and sbf_1 is not None and sbf_2 is not None:
            # Full MXMNet with local MP: accumulate per-layer outputs.
            h = n
            nodes_list = []
            for i in range(self.depth):
                h = self.global_mp_blocks[i](h, rbf_g, range_index)
                h, t = self.local_mp_blocks[i](
                    h, rbf_l, sbf_1, sbf_2,
                    edge_index, angle_idx_1, angle_idx_2
                )
                nodes_list.append(t)

            # Sum per-layer outputs.
            out = torch.stack(nodes_list, dim=0).sum(dim=0)  # (N, num_targets)

            # Graph-level pooling.
            if self.output_embedding == "graph":
                batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
                out = self.pooling(out, batch, batch_size)

            if hasattr(self, 'output_mlp') and self.use_output_mlp:
                out = self.output_mlp(out)

        else:
            # Simplified mode without local MP.
            h = n
            for i in range(self.depth):
                h = self.global_mp_blocks[i](h, rbf_g, range_index)

            # Per-node MLP.
            h = self.last_mlp(h)

            # Graph-level pooling.
            if self.output_embedding == "graph":
                batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
                out = self.pooling(h, batch, batch_size)
            else:
                out = h

            if hasattr(self, 'output_mlp') and self.use_output_mlp:
                out = self.output_mlp(out)

        return out
