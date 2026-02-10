"""DimeNet++ model.

Reference: Klicpera et al., Fast and Uncertainty-Aware Directional Message Passing for
Non-Equilibrium Molecules (2020).
"""
import torch
import torch.nn as nn
from kgcnn_torch.layers.gather import gather_nodes_outgoing
from kgcnn_torch.layers.aggr import Aggregate, AggregateLocalEdges
from kgcnn_torch.layers.geom import (
    BesselBasisLayer, SphericalBasisLayer,
    compute_edge_distances, shift_periodic_lattice
)
from kgcnn_torch.layers.mlp import MLP
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.update import ResidualLayer
from kgcnn_torch.initializers.initializers import glorot_orthogonal_
from kgcnn_torch.ops.activ import get_activation


def _init_glorot_orthogonal(linear: nn.Linear):
    """Apply glorot-orthogonal initialization to a Linear layer."""
    glorot_orthogonal_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


class EmbeddingDimeBlock(nn.Module):
    """Embedding layer for DimeNet (uniform initialization like original)."""

    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        limit = (3.0 ** 0.5)  # sqrt(3) for uniform ~= glorot uniform
        self.embedding = nn.Embedding(input_dim + 1, output_dim)
        nn.init.uniform_(self.embedding.weight, -limit, limit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x.long())


class DimNetInteractionPPBlock(nn.Module):
    """DimeNet++ interaction block with down/up projections.

    Processes edge embeddings using radial and spherical basis,
    with triplet interactions.
    """

    def __init__(self, emb_size: int, int_emb_size: int,
                 basis_emb_size: int, num_before_skip: int,
                 num_after_skip: int, num_radial: int = 6,
                 num_spherical: int = 7, activation: str = "swish",
                 pooling_method: str = "sum"):
        super().__init__()
        self.emb_size = emb_size

        # Basis transformations
        self.dense_rbf1 = nn.Linear(num_radial, basis_emb_size, bias=False)
        self.dense_rbf2 = nn.Linear(basis_emb_size, emb_size, bias=False)
        self.dense_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.dense_sbf2 = nn.Linear(basis_emb_size, int_emb_size, bias=False)

        # Edge transformations (each nn.Sequential gets its own activation instance)
        self.dense_ji = nn.Sequential(nn.Linear(emb_size, emb_size), get_activation(activation))
        self.dense_kj = nn.Sequential(nn.Linear(emb_size, emb_size), get_activation(activation))

        # Down/up projections (no bias, matching Keras and original paper)
        self.down_projection = nn.Sequential(nn.Linear(emb_size, int_emb_size, bias=False), get_activation(activation))
        self.up_projection = nn.Sequential(nn.Linear(int_emb_size, emb_size, bias=False), get_activation(activation))

        # Residual layers before skip (with glorot_orthogonal init matching Keras)
        self.layers_before_skip = nn.ModuleList()
        for _ in range(num_before_skip):
            self.layers_before_skip.append(ResidualLayer(emb_size, activation=activation,
                                                          kernel_initializer="glorot_orthogonal"))
        self.final_before_skip = nn.Sequential(nn.Linear(emb_size, emb_size), get_activation(activation))

        # Residual layers after skip
        self.layers_after_skip = nn.ModuleList()
        for _ in range(num_after_skip):
            self.layers_after_skip.append(ResidualLayer(emb_size, activation=activation,
                                                         kernel_initializer="glorot_orthogonal"))

        self.gather = gather_nodes_outgoing  # function, not module
        self.aggr = Aggregate(pooling_method=pooling_method)

        # Initialize with glorot-orthogonal
        for module in [self.dense_rbf1, self.dense_rbf2, self.dense_sbf1, self.dense_sbf2]:
            glorot_orthogonal_(module.weight)
        for seq in [self.dense_ji, self.dense_kj, self.down_projection, self.up_projection, self.final_before_skip]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    glorot_orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, rbf: torch.Tensor,
                sbf: torch.Tensor, angle_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Edge embeddings (M, emb_size).
            rbf: Radial basis features (M, num_radial).
            sbf: Spherical basis features (K, num_spherical * num_radial).
            angle_index: Angle indices (2, K), referencing edge pairs.

        Returns:
            Updated edge embeddings (M, emb_size).
        """
        num_edges = x.size(0)

        # Initial transformations
        x_ji = self.dense_ji(x)  # (M, emb_size)
        x_kj = self.dense_kj(x)  # (M, emb_size)

        # Transform via Bessel basis
        rbf_w = self.dense_rbf2(self.dense_rbf1(rbf))  # (M, emb_size)
        x_kj = x_kj * rbf_w

        # Down-project and gather for triplets
        x_kj = self.down_projection(x_kj)  # (M, int_emb_size)
        # angle_index follows KGCNN convention: [0]=target(ji), [1]=source(kj)
        # Gather kj edge features using source edge indices
        x_kj = x_kj[angle_index[1]]  # (K, int_emb_size) - kj edges

        # Transform via spherical basis
        sbf_w = self.dense_sbf2(self.dense_sbf1(sbf))  # (K, int_emb_size)
        x_kj = x_kj * sbf_w  # (K, int_emb_size)

        # Aggregate triplet interactions to target edges (ji)
        target_edge_idx = angle_index[0]  # (K,) - ji edges
        x_kj_agg = self.aggr(x_kj, target_edge_idx, num_edges)  # (M, int_emb_size)

        # Up-project
        x_kj_agg = self.up_projection(x_kj_agg)  # (M, emb_size)

        # Before skip: add ji and kj contributions
        x2 = x_ji + x_kj_agg
        for layer in self.layers_before_skip:
            x2 = layer(x2)
        x2 = self.final_before_skip(x2)

        # Skip connection
        x = x + x2

        # After skip
        for layer in self.layers_after_skip:
            x = layer(x)

        return x


class DimNetOutputBlock(nn.Module):
    """DimeNet++ output block.

    Transforms edge embeddings and pools to node level for output.
    """

    def __init__(self, emb_size: int, out_emb_size: int,
                 num_dense: int, num_targets: int = 12,
                 num_radial: int = 6,
                 activation: str = "swish",
                 output_init: str = "zeros",
                 pooling_method: str = "sum"):
        super().__init__()

        self.dense_rbf = nn.Linear(num_radial, emb_size, bias=False)
        self.up_projection = nn.Linear(emb_size, out_emb_size, bias=False)

        layers = []
        in_dim = out_emb_size
        for _ in range(num_dense):
            layers.append(nn.Linear(in_dim, out_emb_size))
            layers.append(get_activation(activation))
            in_dim = out_emb_size
        self.dense_mlp = nn.Sequential(*layers)

        self.dense_final = nn.Linear(out_emb_size, num_targets, bias=False)

        self.aggr = AggregateLocalEdges(pooling_method=pooling_method)

        # Initialize
        glorot_orthogonal_(self.dense_rbf.weight)
        glorot_orthogonal_(self.up_projection.weight)
        for m in self.dense_mlp:
            if isinstance(m, nn.Linear):
                glorot_orthogonal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        if output_init == "zeros":
            nn.init.zeros_(self.dense_final.weight)
        else:
            glorot_orthogonal_(self.dense_final.weight)

    def forward(self, n: torch.Tensor, x: torch.Tensor,
                rbf: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            n: Node embeddings (N, F), used only for dim_size.
            x: Edge embeddings (M, emb_size).
            rbf: Radial basis features (M, num_radial).
            edge_index: Edge indices (2, M).

        Returns:
            Node-level output (N, num_targets).
        """
        num_nodes = n.size(0)

        # Weight edge embeddings by transformed RBF
        g = self.dense_rbf(rbf)  # (M, emb_size)
        x = g * x  # (M, emb_size)

        # Aggregate to target nodes
        x = self.aggr(x, edge_index, num_nodes)  # (N, emb_size)

        # Project and MLP
        x = self.up_projection(x)  # (N, out_emb_size)
        x = self.dense_mlp(x)  # (N, out_emb_size)
        x = self.dense_final(x)  # (N, num_targets)
        return x


class DimeNetPPModel(nn.Module):
    """DimeNet++ model for molecular property prediction.

    Three-body directional message passing using radial and spherical basis.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.angle_index: Angle/triplet indices (2, K), referencing edge pairs.
        - data.batch: Batch assignment (N,).
        - Optional (crystal): data.edge_image, data.lattice, data.batch_edge.
    """

    def __init__(self,
                 emb_size: int = 128,
                 out_emb_size: int = 256,
                 int_emb_size: int = 64,
                 basis_emb_size: int = 8,
                 num_blocks: int = 4,
                 num_spherical: int = 7,
                 num_radial: int = 6,
                 cutoff: float = 5.0,
                 envelope_exponent: int = 5,
                 num_before_skip: int = 1,
                 num_after_skip: int = 2,
                 num_dense_output: int = 3,
                 num_targets: int = 64,
                 activation: str = "swish",
                 extensive: bool = True,
                 output_init: str = "zeros",
                 output_embedding: str = "graph",
                 use_node_embedding: bool = True,
                 num_embeddings: int = 95,
                 use_output_mlp: bool = True,
                 output_mlp_units: list = None,
                 output_mlp_activation: str = "swish"):
        """Initialize DimeNet++ model.

        Args:
            emb_size: Main embedding dimension.
            out_emb_size: Output block embedding dimension.
            int_emb_size: Interaction triplet embedding dimension.
            basis_emb_size: Basis transformation embedding dimension.
            num_blocks: Number of interaction blocks.
            num_spherical: Number of spherical harmonics.
            num_radial: Number of radial basis functions.
            cutoff: Cutoff distance.
            envelope_exponent: Exponent for envelope function.
            num_before_skip: Residual layers before skip connection.
            num_after_skip: Residual layers after skip connection.
            num_dense_output: Dense layers in output block.
            num_targets: Number of output targets.
            activation: Activation function name.
            extensive: If True, use sum pooling (extensive); else mean (intensive).
            output_init: Initialization for final output layer.
            use_node_embedding: Whether to embed atomic numbers.
            num_embeddings: Vocabulary size for embedding.
            use_output_mlp: Whether to apply additional output MLP.
            output_mlp_units: Units for optional output MLP.
            output_mlp_activation: Activation for optional output MLP.
        """
        super().__init__()
        self.output_embedding = output_embedding
        self.extensive = extensive
        self.use_node_embedding = use_node_embedding
        self.use_output_mlp = use_output_mlp
        self.num_blocks = num_blocks
        self.num_targets = num_targets

        # Match Keras implementation: DimeNetPP currently supports graph-level output only.
        if self.output_embedding != "graph":
            raise ValueError("Unsupported output embedding for mode `DimeNetPP`.")

        if use_node_embedding:
            self.node_embedding = EmbeddingDimeBlock(num_embeddings, emb_size)

        # Basis layers
        self.bessel_basis = BesselBasisLayer(
            num_radial=num_radial, cutoff=cutoff, envelope_exponent=envelope_exponent
        )
        self.spherical_basis = SphericalBasisLayer(
            num_spherical=num_spherical, num_radial=num_radial,
            cutoff=cutoff, envelope_exponent=envelope_exponent
        )

        # Embedding block: RBF -> Dense, [n_i || n_j || rbf_emb] -> Dense
        self.rbf_emb = nn.Sequential(nn.Linear(num_radial, emb_size), get_activation(activation))
        self.edge_emb = nn.Sequential(nn.Linear(emb_size * 2 + emb_size, emb_size), get_activation(activation))

        # Initialize embedding layers
        for seq in [self.rbf_emb, self.edge_emb]:
            for m in seq:
                if isinstance(m, nn.Linear):
                    glorot_orthogonal_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        # Initial output block
        self.output_block_0 = DimNetOutputBlock(
            emb_size, out_emb_size, num_dense_output,
            num_targets=num_targets, num_radial=num_radial,
            activation=activation, output_init=output_init
        )

        # Interaction + output blocks
        self.interaction_blocks = nn.ModuleList()
        self.output_blocks = nn.ModuleList()
        for _ in range(num_blocks):
            self.interaction_blocks.append(DimNetInteractionPPBlock(
                emb_size, int_emb_size, basis_emb_size,
                num_before_skip, num_after_skip,
                num_radial=num_radial, num_spherical=num_spherical,
                activation=activation
            ))
            self.output_blocks.append(DimNetOutputBlock(
                emb_size, out_emb_size, num_dense_output,
                num_targets=num_targets, num_radial=num_radial,
                activation=activation, output_init=output_init
            ))

        # Graph pooling
        pool_method = "sum" if extensive else "mean"
        self.pooling = PoolingNodes(pooling_method=pool_method)

        # Optional output MLP
        # Use num_targets as final default output dimension.
        if use_output_mlp:
            if output_mlp_units is None:
                output_mlp_units = [64, 12]
            out_act = [output_mlp_activation] * (len(output_mlp_units) - 1) + ["linear"]
            self.output_mlp = MLP(
                units=output_mlp_units, input_dim=num_targets,
                activation=out_act,
                use_bias=[True] * (len(output_mlp_units) - 1) + [False]
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
        angle_index = data.angle_index
        batch = data.batch

        # Node embedding
        if self.use_node_embedding:
            n = self.node_embedding(z_in)
        else:
            n = z_in

        # Compute distances
        if hasattr(data, 'edge_image') and data.edge_image is not None:
            pos_j = pos[edge_index[0]]
            batch_edge = data.batch_edge if hasattr(data, 'batch_edge') else batch[edge_index[0]]
            pos_j = shift_periodic_lattice(pos_j, data.edge_image, data.lattice, batch_edge)
            pos_i = pos[edge_index[1]]
            diff = pos_j - pos_i
            dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
            dist = torch.sqrt(dist_sq + torch.finfo(dist_sq.dtype).eps)
        else:
            pos_i = pos[edge_index[1]]
            pos_j = pos[edge_index[0]]
            diff = pos_j - pos_i
            # Distance is symmetric: |source - target| == |target - source|.
            dist = compute_edge_distances(pos, edge_index)

        rbf = self.bessel_basis(dist)  # (M, num_radial)

        # Compute angles between edge pairs
        # v12 = pos_target - pos_source for each edge (incoming direction)
        v12 = pos_i - pos_j  # (M, 3) - reversed for angle computation

        # Angle between vectors using atan2 for numerical stability (matches Keras)
        vec_a = v12[angle_index[0]]  # (K, 3)
        vec_b = v12[angle_index[1]]  # (K, 3)
        dot = (vec_a * vec_b).sum(dim=-1)  # (K,)
        cross = torch.cross(vec_a, vec_b, dim=-1)  # (K, 3)
        cross_norm = torch.norm(cross, dim=-1)  # (K,)
        angles = torch.atan2(cross_norm, dot).unsqueeze(-1)  # (K, 1)

        sbf = self.spherical_basis(dist, angles, angle_index)  # (K, S*R)

        # Embedding block
        rbf_emb = self.rbf_emb(rbf)  # (M, emb_size)
        # Match Keras GatherNodes() order: [target, source]
        n_target = n[edge_index[1]]  # target node features
        n_source = n[edge_index[0]]  # source node features
        n_pairs = torch.cat([n_target, n_source], dim=-1)  # (M, 2*emb_size)
        x = torch.cat([n_pairs, rbf_emb], dim=-1)  # (M, 3*emb_size)
        x = self.edge_emb(x)  # (M, emb_size)

        # Initial output
        ps = self.output_block_0(n, x, rbf, edge_index)

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.interaction_blocks[i](x, rbf, sbf, angle_index)
            p_update = self.output_blocks[i](n, x, rbf, edge_index)
            ps = ps + p_update

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(ps, batch, batch_size)
        else:
            out = ps

        if self.use_output_mlp:
            out = self.output_mlp(out)

        return out


class DimeNetPPCrystalModel(DimeNetPPModel):
    """DimeNet++ model for crystalline materials with periodic boundary conditions.

    This is a dedicated crystal variant that always computes edge distances
    using periodic lattice shifts. It inherits all architecture from DimeNetPPModel
    but overrides the forward pass to require lattice and edge_image inputs.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Atom positions (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.angle_index: Angle/triplet indices (2, K), referencing edge pairs.
        - data.batch: Batch assignment (N,).
        - data.lattice: Lattice matrix per graph (B, 3, 3).
        - data.edge_image: Periodic image shift vectors per edge (M, 3).
    """

    def __init__(self, **kwargs):
        """Initialize DimeNetPPCrystalModel.

        Accepts all the same keyword arguments as DimeNetPPModel.
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
        angle_index = data.angle_index
        batch = data.batch
        edge_image = data.edge_image
        lattice = data.lattice

        # Node embedding
        if self.use_node_embedding:
            n = self.node_embedding(z_in)
        else:
            n = z_in

        # Compute distances with periodic boundary conditions
        batch_edge = batch[edge_index[0]]
        pos_j = pos[edge_index[0]]
        pos_j = shift_periodic_lattice(pos_j, edge_image, lattice, batch_edge)
        pos_i = pos[edge_index[1]]
        diff = pos_j - pos_i
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + torch.finfo(dist_sq.dtype).eps)

        rbf = self.bessel_basis(dist)  # (M, num_radial)

        # Compute angles between edge pairs
        # v12 = pos_target - pos_source for each edge (incoming direction)
        v12 = pos_i - pos_j  # (M, 3) - reversed for angle computation

        # Angle between vectors using atan2 for numerical stability (matches Keras)
        vec_a = v12[angle_index[0]]  # (K, 3)
        vec_b = v12[angle_index[1]]  # (K, 3)
        dot = (vec_a * vec_b).sum(dim=-1)  # (K,)
        cross = torch.cross(vec_a, vec_b, dim=-1)  # (K, 3)
        cross_norm = torch.norm(cross, dim=-1)  # (K,)
        angles = torch.atan2(cross_norm, dot).unsqueeze(-1)  # (K, 1)

        sbf = self.spherical_basis(dist, angles, angle_index)  # (K, S*R)

        # Embedding block
        rbf_emb = self.rbf_emb(rbf)  # (M, emb_size)
        # Match Keras GatherNodes() order: [target, source]
        n_target = n[edge_index[1]]  # target node features
        n_source = n[edge_index[0]]  # source node features
        n_pairs = torch.cat([n_target, n_source], dim=-1)  # (M, 2*emb_size)
        x = torch.cat([n_pairs, rbf_emb], dim=-1)  # (M, 3*emb_size)
        x = self.edge_emb(x)  # (M, emb_size)

        # Initial output
        ps = self.output_block_0(n, x, rbf, edge_index)

        # Interaction blocks
        for i in range(self.num_blocks):
            x = self.interaction_blocks[i](x, rbf, sbf, angle_index)
            p_update = self.output_blocks[i](n, x, rbf, edge_index)
            ps = ps + p_update

        # Graph-level pooling
        if self.output_embedding == "graph":
            batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
            out = self.pooling(ps, batch, batch_size)
        else:
            out = ps

        if self.use_output_mlp:
            out = self.output_mlp(out)

        return out
