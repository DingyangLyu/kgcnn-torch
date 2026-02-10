"""PAiNN (Polarizable Atom Interaction Neural Network) model.

Reference: Schutt et al., Equivariant message passing for the prediction of tensorial
properties and molecular spectra (2021).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.geom import (
    BesselBasisLayer, CosCutOffEnvelope,
    compute_edge_distances, compute_edge_direction_normalized,
    shift_periodic_lattice
)
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.norm import GraphLayerNorm, GraphBatchNorm
from kgcnn_torch.layers.modules import keras_uniform_init_embedding_


class EquivariantInitialize(nn.Module):
    """Initialize equivariant vector features (zeros)."""

    def __init__(self, dim: int = 3, units: int = 128, method: str = "zeros"):
        super().__init__()
        self.dim = dim
        self.units = units
        self.method = method

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Initialize vector features.

        Args:
            z: Scalar node features of shape (N, F) or (N,).

        Returns:
            Vector features of shape (N, 3, F).
        """
        if z.dim() < 2:
            n_nodes = z.size(0)
            f = self.units
        else:
            n_nodes = z.size(0)
            f = z.size(1)

        if self.method == "zeros":
            return torch.zeros(n_nodes, self.dim, f, device=z.device, dtype=torch.float)
        elif self.method == "ones":
            return torch.ones(n_nodes, self.dim, f, device=z.device, dtype=torch.float)
        elif self.method == "normal":
            return torch.randn(n_nodes, self.dim, f, device=z.device, dtype=torch.float)
        elif self.method == "eye":
            eye = torch.eye(self.dim, f, device=z.device, dtype=torch.float)
            return eye.unsqueeze(0).expand(n_nodes, -1, -1)
        elif self.method == "eps":
            return torch.full(
                (n_nodes, self.dim, f), fill_value=torch.finfo(torch.float).eps,
                device=z.device, dtype=torch.float
            )
        elif self.method == "node":
            if z.dim() < 2:
                z_expanded = z.unsqueeze(-1).expand(-1, f).float()
            else:
                z_expanded = z.float()
            return z_expanded.unsqueeze(1).expand(-1, self.dim, -1)
        else:
            raise ValueError(f"Unknown initialization method: {self.method}")


class PAiNNConv(nn.Module):
    """PAiNN message passing block.

    Updates both scalar features z (N, F) and vector features v (N, 3, F).
    """

    def __init__(self, units: int, num_radial: int = 20,
                 cutoff: float = None,
                 activation: str = "swish",
                 pooling_method: str = "sum"):
        super().__init__()
        self.units = units
        self.cutoff = cutoff
        from kgcnn_torch.ops.activ import get_activation

        self.dense1 = nn.Linear(units, units)
        self.act1 = get_activation(activation)
        self.phi = nn.Linear(units, units * 3)
        self.w = nn.Linear(num_radial, units * 3)  # RBF -> filter weights

        self.aggr_s = AggregateLocalEdges(pooling_method=pooling_method)
        self.aggr_v = AggregateLocalEdges(pooling_method=pooling_method)

    def forward(self, z: torch.Tensor, v: torch.Tensor,
                rbf: torch.Tensor, envelope: torch.Tensor,
                rij: torch.Tensor, edge_index: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            z: Scalar node features (N, F).
            v: Vector node features (N, 3, F).
            rbf: Radial basis features (M, num_radial).
            envelope: Distance envelope (M, 1).
            rij: Normalized edge direction (M, 3).
            edge_index: Edge indices (2, M), PyG convention.

        Returns:
            (ds, dv): Scalar and vector updates of shape (N, F) and (N, 3, F).
        """
        num_nodes = z.size(0)

        # Process scalar features
        s = self.act1(self.dense1(z))  # (N, F)
        s = self.phi(s)  # (N, 3F)
        s = gather_nodes_outgoing(s, edge_index)  # (M, 3F)

        # RBF filter
        w = self.w(rbf)  # (M, 3F)
        if self.cutoff is not None:
            w = w * envelope  # (M, 3F)
        sw = s * w  # (M, 3F)

        # Split into 3 channels
        sw1, sw2, sw3 = torch.chunk(sw, 3, dim=-1)  # each (M, F)

        # Scalar update: aggregate sw1
        ds = self.aggr_s(sw1, edge_index, num_nodes)  # (N, F)

        # Vector update: vj * sw2 + rij * sw3
        vj = gather_nodes_outgoing(v, edge_index)  # (M, 3, F)
        dv1 = sw2.unsqueeze(1) * vj  # (M, 3, F)
        dv2 = sw3.unsqueeze(1) * rij.unsqueeze(2)  # (M, 3, F)
        dv = dv1 + dv2  # (M, 3, F)

        # Aggregate vector messages - reshape for aggregation
        M = dv.size(0)
        F = dv.size(2)
        dv_flat = dv.reshape(M, 3 * F)  # (M, 3F)
        dv_agg = self.aggr_v(dv_flat, edge_index, num_nodes)  # (N, 3F)
        dv = dv_agg.reshape(num_nodes, 3, F)  # (N, 3, F)

        return ds, dv


class PAiNNUpdate(nn.Module):
    """PAiNN update block.

    Scalar-vector interaction to update both channels.
    """

    def __init__(self, units: int, activation: str = "swish",
                 add_eps: bool = False):
        super().__init__()
        self.units = units
        self.add_eps = add_eps
        from kgcnn_torch.ops.activ import get_activation

        self.lin_u = nn.Linear(units, units, bias=False)
        self.lin_v = nn.Linear(units, units, bias=False)
        self.dense1 = nn.Linear(2 * units, units)
        self.act1 = get_activation(activation)
        self.dense_a = nn.Linear(units, 3 * units)

    def forward(self, z: torch.Tensor, v: torch.Tensor) -> tuple:
        """Forward pass.

        Args:
            z: Scalar node features (N, F).
            v: Vector node features (N, 3, F).

        Returns:
            (ds, dv): Scalar and vector updates of shape (N, F) and (N, 3, F).
        """
        # Linear transformations of vector features
        # v shape: (N, 3, F), lin_v/lin_u applied on last dim
        v_v = self.lin_v(v)  # (N, 3, F)
        v_u = self.lin_u(v)  # (N, 3, F)

        # Scalar product: <v_u, v_v> summed over spatial dim
        v_prod = (v_u * v_v).sum(dim=1)  # (N, F)

        # Euclidean norm of v_v
        v_norm_sq = (v_v * v_v).sum(dim=1)  # (N, F)
        if self.add_eps:
            v_norm = torch.sqrt(v_norm_sq + torch.finfo(v_norm_sq.dtype).eps)
        else:
            v_norm = torch.sqrt(v_norm_sq)

        # Concatenate scalar features and vector norm
        a = torch.cat([z, v_norm], dim=-1)  # (N, 2F)
        a = self.act1(self.dense1(a))  # (N, F)
        a = self.dense_a(a)  # (N, 3F)

        # Split into 3 channels
        a_vv, a_sv, a_ss = torch.chunk(a, 3, dim=-1)  # each (N, F)

        # Vector update
        dv = a_vv.unsqueeze(1) * v_u  # (N, 3, F)

        # Scalar update
        ds = v_prod * a_sv + a_ss  # (N, F)

        return ds, dv


class PAiNNModel(nn.Module):
    """PAiNN model for molecular property prediction.

    Equivariant message passing maintaining scalar z and vector v features.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - Optional (crystal): data.edge_image, data.lattice, data.batch_edge.
    """

    def __init__(self,
                 node_dim: int = 128,
                 depth: int = 3,
                 units: int = 128,
                 num_radial: int = 20,
                 cutoff: float = 5.0,
                 conv_cutoff: float = None,
                 envelope_exponent: int = 5,
                 conv_activation: str = "swish",
                 conv_pooling: str = "sum",
                 update_activation: str = "swish",
                 update_add_eps: bool = False,
                 equiv_normalization: bool = False,
                 node_normalization: bool = False,
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "swish",
                 num_targets: int = 1,
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95):
        """Initialize PAiNN model.

        Args:
            node_dim: Embedding dimension.
            depth: Number of PAiNN message-passing + update blocks.
            units: Hidden dimension for scalar/vector features.
            num_radial: Number of Bessel radial basis functions.
            cutoff: Cutoff distance for Bessel radial basis.
            conv_cutoff: Cutoff for cosine envelope in convolutions. None means
                no envelope damping (matching Keras default).
            envelope_exponent: Exponent for Bessel envelope.
            conv_activation: Activation for convolution.
            conv_pooling: Pooling method in message aggregation.
            update_activation: Activation for update block.
            equiv_normalization: Whether to apply layer norm on vector features.
            node_normalization: Whether to apply batch norm on scalar features.
            node_pooling: Pooling for graph-level readout.
            output_units: Hidden dims for output MLP.
            output_activation: Activation for output MLP.
            num_targets: Number of output targets.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size.
        """
        super().__init__()
        self.output_embedding = output_embedding
        if output_units is None:
            output_units = [128]

        self.use_node_embedding = use_node_embedding
        self.equiv_normalization = equiv_normalization
        self.node_normalization = node_normalization
        self.depth = depth
        self.units = units

        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, node_dim)
            keras_uniform_init_embedding_(self.node_embedding)

        self.equiv_init = EquivariantInitialize(dim=3, units=units, method="zeros")
        self.bessel_basis = BesselBasisLayer(
            num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )
        self.cutoff_env = CosCutOffEnvelope(cutoff=conv_cutoff if conv_cutoff is not None else cutoff)

        self.convs = nn.ModuleList()
        self.updates = nn.ModuleList()
        for _ in range(depth):
            self.convs.append(PAiNNConv(
                units=units, num_radial=num_radial, cutoff=conv_cutoff,
                activation=conv_activation, pooling_method=conv_pooling
            ))
            self.updates.append(PAiNNUpdate(
                units=units, activation=update_activation,
                add_eps=update_add_eps
            ))

        if equiv_normalization:
            # Keras: GraphLayerNormalization(axis=2) on (N, 3, F) normalizes the F dim.
            self.equiv_norms = nn.ModuleList([GraphLayerNorm(units) for _ in range(depth)])
        if node_normalization:
            self.node_norms = nn.ModuleList([GraphBatchNorm(units) for _ in range(depth)])

        self.pooling = PoolingNodes(pooling_method=node_pooling)
        out_units = output_units + [num_targets]
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
        z_in = data.z
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch

        # Initialize equivariant features from raw z (before embedding, matching Keras)
        v = self.equiv_init(z_in)  # (N, 3, units)

        # Node embedding
        if self.use_node_embedding:
            z = self.node_embedding(z_in.long())
        else:
            z = z_in

        # Compute edge features
        if hasattr(data, 'edge_image') and data.edge_image is not None:
            pos_j = pos[edge_index[0]]  # source
            batch_edge = data.batch_edge if hasattr(data, 'batch_edge') else batch[edge_index[0]]
            pos_j = shift_periodic_lattice(pos_j, data.edge_image, data.lattice, batch_edge)
            pos_i = pos[edge_index[1]]  # target
            diff = pos_i - pos_j  # target - source (matches Keras convention)
            dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
            dist = torch.sqrt(dist_sq + torch.finfo(dist_sq.dtype).eps)
            rij = diff / dist.clamp(min=torch.finfo(dist.dtype).eps)
        else:
            # Distance is symmetric: |source - target| == |target - source|.
            # Direction convention: PAiNN uses (target - source) for equivariance.
            dist = compute_edge_distances(pos, edge_index)
            rij = compute_edge_direction_normalized(pos, edge_index)

        env = self.cutoff_env(dist)  # (M, 1)
        rbf = self.bessel_basis(dist)  # (M, num_radial)

        for i in range(self.depth):
            # Message passing
            ds, dv = self.convs[i](z, v, rbf, env, rij, edge_index)
            z = z + ds
            v = v + dv

            # Update
            ds, dv = self.updates[i](z, v)
            z = z + ds
            v = v + dv

            # Optional normalization
            if self.equiv_normalization:
                v = self._apply_equiv_norm(v, self.equiv_norms[i])
            if self.node_normalization:
                z = self.node_norms[i](z, batch)

        # Graph-level pooling on scalar features
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(z, batch, batch_size)
        else:
            out = z
        out = self.output_mlp(out)
        return out

    def _apply_equiv_norm(self, v, norm_layer):
        """Apply equivariant normalization to vector features.

        Reshapes (N, D, F) -> (N*D, F), applies LayerNorm, reshapes back.
        """
        N, D, F = v.shape
        v = v.reshape(N * D, F)
        v = norm_layer(v)
        v = v.reshape(N, D, F)
        return v


class PAiNNCrystalModel(PAiNNModel):
    """PAiNN model for crystalline materials with periodic boundary conditions.

    This is a dedicated crystal variant that always computes edge distances and
    directions using periodic lattice shifts. It inherits all architecture from
    PAiNNModel but overrides the forward pass to require lattice and edge_image.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.batch: Batch assignment (N,).
        - data.lattice: Lattice matrix per graph (B, 3, 3).
        - data.edge_image: Periodic image shift vectors per edge (M, 3).
    """

    def __init__(self, **kwargs):
        """Initialize PAiNNCrystalModel.

        Accepts all the same keyword arguments as PAiNNModel.
        """
        super().__init__(**kwargs)

    def forward(self, data) -> torch.Tensor:
        """Forward pass for periodic crystal systems.

        Args:
            data: PyG Data batch object with lattice and edge_image attributes.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z_in = data.z
        pos = data.pos
        edge_index = data.edge_index
        batch = data.batch
        edge_image = data.edge_image
        lattice = data.lattice

        # Initialize equivariant features from raw z (before embedding, matching Keras)
        v = self.equiv_init(z_in)  # (N, 3, units)

        # Node embedding
        if self.use_node_embedding:
            z = self.node_embedding(z_in.long())
        else:
            z = z_in

        # Compute edge features with periodic boundary conditions
        batch_edge = batch[edge_index[0]]
        pos_j = pos[edge_index[0]]  # source
        pos_j = shift_periodic_lattice(pos_j, edge_image, lattice, batch_edge)
        pos_i = pos[edge_index[1]]  # target
        diff = pos_i - pos_j  # target - source (matches Keras convention)
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + torch.finfo(dist_sq.dtype).eps)
        rij = diff / dist.clamp(min=torch.finfo(dist.dtype).eps)

        env = self.cutoff_env(dist)  # (M, 1)
        rbf = self.bessel_basis(dist)  # (M, num_radial)

        for i in range(self.depth):
            # Message passing
            ds, dv = self.convs[i](z, v, rbf, env, rij, edge_index)
            z = z + ds
            v = v + dv

            # Update
            ds, dv = self.updates[i](z, v)
            z = z + ds
            v = v + dv

            # Optional normalization (use shared helper for consistency)
            if self.equiv_normalization:
                v = self._apply_equiv_norm(v, self.equiv_norms[i])
            if self.node_normalization:
                z = self.node_norms[i](z, batch)

        # Graph-level pooling on scalar features
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(z, batch, batch_size)
        else:
            out = z
        out = self.output_mlp(out)
        return out
