"""HDNNP2nd (2nd generation High-Dimensional Neural Network Potential) model.

Reference: Behler, J. Atom-centered symmetry functions for constructing
high-dimensional neural network potentials. J. Chem. Phys. 134, 074106 (2011).

Implements the 2nd generation Behler-Parrinello HDNNP with weighted
Atom-Centered Symmetry Functions (wACSF) as descriptors and element-specific
(relational) MLPs for energy prediction.

Architecture:
    1. Compute radial wACSF descriptors from pairwise distances.
    2. Compute angular wACSF descriptors from triplet angles.
    3. Concatenate radial and angular representations.
    4. Optionally apply batch normalization.
    5. Apply element-specific MLP (RelationalMLP) to produce per-atom energies.
    6. Pool per-atom energies to graph-level total energy.
    7. Optionally apply an output MLP.
"""
import math
import torch
import torch.nn as nn
import numpy as np
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.aggr import AggregateLocalEdges
from kgcnn_torch.layers.pooling import PoolingNodes
from kgcnn_torch.layers.mlp import MLP, RelationalMLP
from kgcnn_torch.ops.scatter import scatter_reduce_sum


def cosine_cutoff(dist: torch.Tensor, cutoff: float) -> torch.Tensor:
    """Cosine cutoff function: fc(r) = 0.5 * (cos(pi * r / Rc) + 1) for r < Rc.

    Smoothly decays to zero at the cutoff radius.

    Args:
        dist: Distances of shape (...).
        cutoff: Cutoff radius Rc.

    Returns:
        Cutoff values of shape (...), zero for r >= Rc.
    """
    return torch.where(
        dist < cutoff,
        0.5 * (torch.cos(math.pi * dist / cutoff) + 1.0),
        torch.zeros_like(dist),
    )


class wACSFRad(nn.Module):
    """Weighted radial Atom-Centered Symmetry Function (wACSF).

    For each atom i of element type t, computes:

        G^rad_{t,f}(i) = sum_{j in neighbors(i)} g(Z_j) *
            exp(-eta_{t,f} * (r_ij - mu_{t,f})^2) * fc(r_ij)

    where:
        - eta, mu are fixed (non-trainable) parameters indexed by (element_type, feature_index),
          matching the Keras implementation where trainable=False.
        - g(Z_j) = Z_j is the atomic number weighting.
        - fc is the cosine cutoff function.

    Args:
        n_types: Number of distinct element types.
        n_features: Number of radial symmetry function features per type.
        cutoff: Cutoff radius for the cosine cutoff function.
        eta_init_range: (low, high) for uniform initialization of eta.
        mu_init_range: (low, high) for uniform initialization of mu.
    """

    def __init__(self, n_types: int, n_features: int, cutoff: float,
                 eta_init_range: tuple = (0.5, 8.0),
                 mu_init_range: tuple = (0.0, None)):
        super().__init__()
        self.n_types = n_types
        self.n_features = n_features
        self.cutoff = cutoff

        # Non-trainable parameters: eta and mu per (type, feature).
        # Matching Keras wACSFRad where trainable=False.
        eta = torch.empty(n_types, n_features)
        mu = torch.empty(n_types, n_features)

        nn.init.uniform_(eta, eta_init_range[0], eta_init_range[1])

        mu_high = mu_init_range[1] if mu_init_range[1] is not None else cutoff
        nn.init.uniform_(mu, mu_init_range[0], mu_high)

        self.register_buffer("eta", eta)
        self.register_buffer("mu", mu)

    def forward(self, z: torch.Tensor, pos: torch.Tensor,
                edge_index: torch.Tensor, type_map: torch.Tensor) -> torch.Tensor:
        """Compute weighted radial ACSF descriptors.

        Args:
            z: Atomic numbers of shape (N,).
            pos: Atom positions of shape (N, 3).
            edge_index: Edge indices of shape (2, M), PyG convention
                (edge_index[0] = source/neighbor j, edge_index[1] = target/center i).
            type_map: Mapping from atomic number to type index, shape (N,).
                Each entry is an integer in [0, n_types).

        Returns:
            Radial descriptors of shape (N, n_features).
        """
        num_nodes = z.size(0)

        # Compute pairwise distances.
        diff = pos[edge_index[0]] - pos[edge_index[1]]  # (M, 3)
        dist = torch.sqrt((diff * diff).sum(dim=-1))  # (M,)

        # Cutoff envelope.
        fc = cosine_cutoff(dist, self.cutoff)  # (M,)

        # Atomic number weighting: g(Z_j) = Z_j.
        z_j = z[edge_index[0]].float()  # (M,)

        # Gather the type index for each center atom i.
        type_i = type_map[edge_index[1]]  # (M,)

        # Look up eta and mu for each edge based on the center atom type.
        eta_edge = self.eta[type_i]  # (M, n_features)
        mu_edge = self.mu[type_i]  # (M, n_features)

        # Compute Gaussian: exp(-eta * (r_ij - mu)^2).
        dist_expanded = dist.unsqueeze(-1)  # (M, 1)
        gauss = torch.exp(-eta_edge * (dist_expanded - mu_edge).pow(2))  # (M, n_features)

        # Weight by Z_j and cutoff.
        edge_features = z_j.unsqueeze(-1) * gauss * fc.unsqueeze(-1)  # (M, n_features)

        # Aggregate (sum) over neighbors to get per-atom descriptors.
        target_idx = edge_index[1]  # (M,)
        out = scatter_reduce_sum(target_idx, edge_features, num_nodes)  # (N, n_features)
        return out


class wACSFAng(nn.Module):
    """Weighted angular Atom-Centered Symmetry Function (wACSF).

    For each atom i of element type t, computes:

        G^ang_{t,f}(i) = 2^(1-zeta_{t,f}) *
            sum_{j,k in neighbors(i), j!=k}
                h(Z_j, Z_k) * (1 + lambda_{t,f} * cos(theta_ijk))^zeta_{t,f} *
                exp(-eta_{t,f} * (r_ij - mu_{t,f})^2) *
                exp(-eta_{t,f} * (r_ik - mu_{t,f})^2) *
                exp(-eta_{t,f} * (r_jk - mu_{t,f})^2) *
                fc(r_ij) * fc(r_ik) * fc(r_jk)

    where:
        - theta_ijk is the angle at atom i between neighbors j and k.
        - h(Z_j, Z_k) = Z_j * Z_k is the element weighting for wACSF.
        - eta, mu, lambda, zeta are fixed (non-trainable) parameters per (type, feature),
          matching the Keras implementation where trainable=False.

    The angle_index tensor has shape (3, K) with rows [center_i, neighbor_j, neighbor_k].

    Args:
        n_types: Number of distinct element types.
        n_features: Number of angular symmetry function features per type.
        cutoff: Cutoff radius for the cosine cutoff function.
        eta_init_range: (low, high) for uniform initialization of eta.
        mu_init_range: (low, high) for uniform initialization of mu.
        zeta_init_range: (low, high) for uniform initialization of zeta.
    """

    def __init__(self, n_types: int, n_features: int, cutoff: float,
                 eta_init_range: tuple = (0.01, 1.0),
                 mu_init_range: tuple = (0.0, None),
                 zeta_init_range: tuple = (1.0, 16.0)):
        super().__init__()
        self.n_types = n_types
        self.n_features = n_features
        self.cutoff = cutoff

        # Non-trainable parameters per (type, feature).
        # Matching Keras wACSFAng where trainable=False.
        eta = torch.empty(n_types, n_features)
        mu = torch.empty(n_types, n_features)
        lam = torch.empty(n_types, n_features)
        zeta = torch.empty(n_types, n_features)

        nn.init.uniform_(eta, eta_init_range[0], eta_init_range[1])
        mu_high = mu_init_range[1] if mu_init_range[1] is not None else cutoff
        nn.init.uniform_(mu, mu_init_range[0], mu_high)
        nn.init.uniform_(lam, -1.0, 1.0)
        nn.init.uniform_(zeta, zeta_init_range[0], zeta_init_range[1])

        self.register_buffer("eta", eta)
        self.register_buffer("mu", mu)
        self.register_buffer("lam", lam)
        self.register_buffer("zeta", zeta)

    def forward(self, z: torch.Tensor, pos: torch.Tensor,
                angle_index: torch.Tensor, type_map: torch.Tensor) -> torch.Tensor:
        """Compute weighted angular ACSF descriptors.

        Args:
            z: Atomic numbers of shape (N,).
            pos: Atom positions of shape (N, 3).
            angle_index: Angle triplet indices of shape (3, K), where
                angle_index[0] = center atom i,
                angle_index[1] = neighbor atom j,
                angle_index[2] = neighbor atom k.
            type_map: Mapping from atomic number to type index, shape (N,).

        Returns:
            Angular descriptors of shape (N, n_features).
        """
        num_nodes = z.size(0)

        idx_i = angle_index[0]  # (K,) center atoms
        idx_j = angle_index[1]  # (K,) neighbor j
        idx_k = angle_index[2]  # (K,) neighbor k

        # Position vectors.
        pos_i = pos[idx_i]  # (K, 3)
        pos_j = pos[idx_j]  # (K, 3)
        pos_k = pos[idx_k]  # (K, 3)

        # Displacement vectors and distances.
        vec_ij = pos_j - pos_i  # (K, 3)
        vec_ik = pos_k - pos_i  # (K, 3)
        r_ij = torch.sqrt((vec_ij * vec_ij).sum(dim=-1))  # (K,)
        r_ik = torch.sqrt((vec_ik * vec_ik).sum(dim=-1))  # (K,)
        vec_jk = pos_k - pos_j  # (K, 3)
        r_jk = torch.sqrt((vec_jk * vec_jk).sum(dim=-1))  # (K,)

        # Cosine of angle at center atom i.
        dot = (vec_ij * vec_ik).sum(dim=-1)  # (K,)
        den = (r_ij * r_ik).clamp(min=1e-12)
        cos_theta = dot / den  # (K,)
        cos_theta = cos_theta.clamp(-1.0, 1.0)

        # Cutoff values for all three edges (ij, ik, jk).
        fc_ij = cosine_cutoff(r_ij, self.cutoff)  # (K,)
        fc_ik = cosine_cutoff(r_ik, self.cutoff)  # (K,)
        fc_jk = cosine_cutoff(r_jk, self.cutoff)  # (K,)

        # Element weighting: h(Z_j, Z_k) = Z_j * Z_k.
        z_j = z[idx_j].float()  # (K,)
        z_k = z[idx_k].float()  # (K,)
        h_weight = z_j * z_k  # (K,)

        # Gather learnable parameters for center atom types.
        type_i = type_map[idx_i]  # (K,)
        eta_t = self.eta[type_i]  # (K, n_features)
        mu_t = self.mu[type_i]  # (K, n_features)
        lam_t = self.lam[type_i]  # (K, n_features)
        zeta_t = self.zeta[type_i]  # (K, n_features)

        # Gaussian terms for r_ij, r_ik, r_jk: exp(-eta * (r - mu)^2) each.
        r_ij_exp = r_ij.unsqueeze(-1)  # (K, 1)
        r_ik_exp = r_ik.unsqueeze(-1)  # (K, 1)
        r_jk_exp = r_jk.unsqueeze(-1)  # (K, 1)
        gij = torch.exp(-eta_t * (r_ij_exp - mu_t).pow(2))  # (K, n_features)
        gik = torch.exp(-eta_t * (r_ik_exp - mu_t).pow(2))  # (K, n_features)
        gjk = torch.exp(-eta_t * (r_jk_exp - mu_t).pow(2))  # (K, n_features)

        # Angular term: (1 + lambda * cos(theta))^zeta.
        # Use clamp to ensure the base is non-negative before raising to power,
        # since zeta can be non-integer.
        cos_expanded = cos_theta.unsqueeze(-1)  # (K, 1)
        angular_base = (1.0 + lam_t * cos_expanded).clamp(min=0.0)  # (K, n_features)
        angular_term = angular_base.pow(zeta_t)  # (K, n_features)

        # Normalization prefactor: 2^(1-zeta).
        prefactor = torch.pow(2.0, 1.0 - zeta_t)  # (K, n_features)

        # Combine all terms: includes r_jk Gaussian and cutoff per Behler formula.
        fc_product = (fc_ij * fc_ik * fc_jk).unsqueeze(-1)  # (K, 1)
        triplet_features = (
            prefactor * h_weight.unsqueeze(-1) * angular_term * gij * gik * gjk * fc_product
        )  # (K, n_features)

        # Aggregate (sum) over triplets to center atoms.
        out = scatter_reduce_sum(idx_i, triplet_features, num_nodes)  # (N, n_features)
        return out


class _IndependentRelationalMLP(nn.Module):
    """Element-specific MLP using fully independent MLPs per element type.

    DEPRECATED: The Keras HDNNP2nd uses ``RelationalDense`` (shared weight matrices
    indexed by relation type with shared bias), which is matched by
    ``kgcnn_torch.layers.mlp.RelationalMLP``. This class is kept only for
    backward compatibility. New code should use the imported ``RelationalMLP``.
    """

    def __init__(self, n_types: int, input_dim: int, hidden_units: list,
                 activation: str = "tanh", use_bias: bool = True):
        super().__init__()
        self.n_types = n_types
        self.input_dim = input_dim

        self.mlps = nn.ModuleList()
        for _ in range(n_types):
            self.mlps.append(
                MLP(units=hidden_units, input_dim=input_dim, activation=activation,
                    use_bias=use_bias)
            )

    def forward(self, x: torch.Tensor, type_map: torch.Tensor) -> torch.Tensor:
        num_nodes = x.size(0)
        out_dim = self.mlps[0].output_dim
        out = torch.zeros(num_nodes, out_dim, device=x.device, dtype=x.dtype)

        for t in range(self.n_types):
            mask = (type_map == t)
            if mask.any():
                out[mask] = self.mlps[t](x[mask])

        return out


class HDNNP2ndModel(nn.Module):
    """2nd generation High-Dimensional Neural Network Potential (Behler-Parrinello).

    Uses weighted Atom-Centered Symmetry Functions (wACSF) for radial and angular
    descriptors, followed by element-specific MLPs (RelationalMLP) to predict
    per-atom energies that are summed to give total energy.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Node coordinates (N, 3).
        - data.edge_index: Edge indices (2, M), PyG convention.
        - data.angle_index: Angle indices (3, K) -- [center, neighbor1, neighbor2].
        - data.batch: Batch assignment (N,).

    Args:
        element_types: List of atomic numbers for the elements in the dataset,
            e.g. [1, 6, 7, 8] for H, C, N, O. Determines n_types and the
            mapping from atomic number to type index.
        n_rad_features: Number of radial symmetry function features per type.
        n_ang_features: Number of angular symmetry function features per type.
        cutoff: Cutoff radius for the cosine cutoff function (Angstroms).
        relational_units: List of hidden layer sizes for the element-specific MLP.
        relational_activation: Activation function for the element-specific MLP.
        use_batch_norm: Whether to apply batch normalization after concatenating
            radial and angular descriptors.
        node_pooling: Pooling method for graph-level readout ('sum' or 'mean').
        output_units: Hidden dimensions for the optional output MLP after pooling.
            If None, no output MLP is applied (direct pooling of per-atom energies).
        output_activation: Activation function for the output MLP.
        num_targets: Number of output targets.
    """

    def __init__(self,
                 element_types: list = None,
                 n_rad_features: int = 22,
                 n_ang_features: int = 10,
                 cutoff: float = 8.0,
                 num_relations: int = 96,
                 relational_units: list = None,
                 relational_activation = None,
                 use_batch_norm: bool = False,
                 node_pooling: str = "sum",
                 use_output_mlp: bool = False,
                 output_units: list = None,
                 output_activation: str = "linear",
                 num_targets: int = 1,
                 output_embedding: str = "graph"):
        super().__init__()

        if element_types is None:
            # Match Keras wACSF defaults: parameter tables are provided for 118 species indices.
            # Keras expects atomic numbers (incl. 0 padding) to be usable as indices.
            element_types = list(range(118))
        if relational_units is None:
            relational_units = [64, 64, 64]
        if relational_activation is None:
            relational_activation = ["swish", "swish", "linear"]

        self.output_embedding = output_embedding
        self.use_output_mlp = use_output_mlp
        self.element_types = sorted(element_types)
        self.n_types = len(self.element_types)
        self.cutoff = cutoff
        self.n_rad_features = n_rad_features
        self.n_ang_features = n_ang_features
        self.use_batch_norm = use_batch_norm

        # Build a lookup table: atomic_number -> type index.
        # We register it as a buffer so it moves to device with the model.
        max_z = max(self.element_types) + 1
        z_to_type = torch.full((max_z,), -1, dtype=torch.long)
        for idx, z_val in enumerate(self.element_types):
            z_to_type[z_val] = idx
        self.register_buffer("z_to_type", z_to_type)

        # Radial wACSF layer.
        self.acsf_rad = wACSFRad(
            n_types=self.n_types,
            n_features=n_rad_features,
            cutoff=cutoff,
        )

        # Angular wACSF layer.
        self.acsf_ang = wACSFAng(
            n_types=self.n_types,
            n_features=n_ang_features,
            cutoff=cutoff,
        )

        # Optional batch normalization on concatenated descriptors.
        descriptor_dim = n_rad_features + n_ang_features
        if use_batch_norm:
            from kgcnn_torch.layers.norm import GraphBatchNorm
            self.batch_norm = GraphBatchNorm(descriptor_dim)

        # Element-specific MLP using RelationalDense (shared bias per layer,
        # weight matrices indexed by relation type), matching Keras RelationalMLP.
        self.relational_mlp = RelationalMLP(
            units=relational_units,
            input_dim=descriptor_dim,
            num_relations=num_relations,
            activation=relational_activation,
        )

        # Graph-level pooling.
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # Optional output MLP (default False, matching Keras which doesn't apply
        # output_mlp by default -- raw pooled features are the output).
        pool_out_dim = relational_units[-1]
        if use_output_mlp:
            if output_units is None:
                output_units = []
            out_units = output_units + [num_targets]
            out_act = [output_activation] * len(output_units) + ["linear"]
            self.output_mlp = MLP(
                units=out_units,
                input_dim=pool_out_dim,
                activation=out_act,
            )
        else:
            # Keras: no output MLP at all when use_output_mlp=False.
            self.output_mlp = None

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object with attributes z, pos, edge_index,
                angle_index, and batch.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        angle_index = data.angle_index
        batch = data.batch

        # Support both node-triplet indices (3, K) and edge-pair indices (2, K).
        # Some converted datasets store angle pairs as indices into edge_index.
        if angle_index.dim() == 2 and angle_index.size(0) == 2:
            e_ij = angle_index[0].long()
            e_ik = angle_index[1].long()
            idx_i = edge_index[1, e_ij]
            idx_j = edge_index[0, e_ij]
            idx_k = edge_index[0, e_ik]
            angle_index = torch.stack([idx_i, idx_j, idx_k], dim=0)

        # Map atomic numbers to type indices.
        type_map = self.z_to_type[z.long()]  # (N,)

        # Compute radial descriptors.
        rep_rad = self.acsf_rad(z, pos, edge_index, type_map)  # (N, n_rad_features)

        # Compute angular descriptors.
        rep_ang = self.acsf_ang(z, pos, angle_index, type_map)  # (N, n_ang_features)

        # Concatenate radial and angular representations.
        rep = torch.cat([rep_rad, rep_ang], dim=-1)  # (N, n_rad + n_ang)

        # Compute batch_size for pooling and normalization.
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Optional batch normalization.
        if self.use_batch_norm:
            rep = self.batch_norm(rep, batch, batch_size)

        # Element-specific MLP for per-atom predictions.
        # Keras RelationalMLP takes [features, relation_index, batch_id, count].
        # The imported RelationalMLP takes (x, relations, batch, batch_size).
        n = self.relational_mlp(rep, type_map, batch, batch_size)  # (N, relational_units[-1])

        # Graph-level pooling (sum of per-atom energies).
        if self.output_embedding == "graph":
            out = self.pooling(n, batch, batch_size)  # (B, relational_units[-1])
        else:
            out = n

        if self.output_mlp is not None:
            out = self.output_mlp(out)

        return out


class HDNNP2ndBehlerModel(nn.Module):
    """2nd generation HDNNP using original Behler symmetry functions (G2/G4).

    Unlike the weighted-ACSF variant (HDNNP2ndModel) which uses learned
    wACSF descriptors, this model uses fixed-parameter Behler G2 (radial)
    and G4 (angular) symmetry functions. The G2 and G4 parameters (eta, rs,
    rc, lambda, zeta, elements) are specified at construction time and
    remain fixed during training.

    The overall architecture is:
        1. Compute G2 radial symmetry functions from pairwise distances.
        2. Compute G4 angular symmetry functions from triplet angles.
        3. Concatenate radial + angular descriptors.
        4. Optionally apply batch normalization.
        5. Apply element-specific MLP (RelationalMLP) for per-atom energies.
        6. Pool per-atom energies to graph-level total energy.
        7. Optionally apply output MLP.

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.pos: Node coordinates (N, 3).
        - data.edge_index: Edge indices (2, M).
        - data.angle_index: Angle indices (3, K) -- [center, neighbor1, neighbor2].
        - data.batch: Batch assignment (N,).

    Args:
        element_types: List of atomic numbers for elements in the dataset.
        g2_eta: List of eta values for G2 radial functions.
        g2_rs: List of Rs (shift) values for G2 radial functions.
        g2_rc: Cutoff radius for G2 functions.
        g4_eta: List of eta values for G4 angular functions.
        g4_zeta: List of zeta values for G4 angular functions.
        g4_lamda: List of lambda (+1/-1) values for G4 angular functions.
        g4_rc: Cutoff radius for G4 functions.
        g4_multiplicity: Optional divisor applied to G4 angular term.
        relational_units: Hidden layer sizes for element-specific MLP.
        relational_activation: Activation function for element-specific MLP.
        use_batch_norm: Whether to apply batch normalization on descriptors.
        node_pooling: Pooling method for graph-level readout.
        output_units: Hidden dimensions for optional output MLP.
        output_activation: Activation function for output MLP.
        num_targets: Number of output targets.
    """

    def __init__(self,
                 element_types: list = None,
                 g2_eta: list = None,
                 g2_rs: list = None,
                 g2_rc: float = 6.0,
                 g4_eta: list = None,
                 g4_zeta: list = None,
                 g4_lamda: list = None,
                 g4_rc: float = 6.0,
                 g4_multiplicity: float = 2.0,
                 relational_units: list = None,
                 relational_activation = None,
                 num_relations: int = 96,
                 use_batch_norm: bool = False,
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "linear",
                 num_targets: int = 1,
                 output_embedding: str = "graph"):
        super().__init__()

        if element_types is None:
            element_types = [1, 6, 7, 8]
        if relational_units is None:
            relational_units = [64, 64, 64]
        if relational_activation is None:
            relational_activation = ["swish", "swish", "linear"]
        if g2_eta is None:
            g2_eta = [0.0, 0.3]
        if g2_rs is None:
            g2_rs = [0.0, 3.0]
        if g4_eta is None:
            g4_eta = [0.0, 0.3]
        if g4_zeta is None:
            g4_zeta = [1.0, 8.0]
        if g4_lamda is None:
            g4_lamda = [-1.0, 1.0]

        self.output_embedding = output_embedding
        self.element_types = sorted(element_types)
        self.n_types = len(self.element_types)
        self.g2_rc = g2_rc
        self.g4_rc = g4_rc
        self.g4_multiplicity = g4_multiplicity

        # Build lookup table: atomic_number -> type index.
        max_z = max(self.element_types) + 1
        z_to_type = torch.full((max_z,), -1, dtype=torch.long)
        for idx, z_val in enumerate(self.element_types):
            z_to_type[z_val] = idx
        self.register_buffer("z_to_type", z_to_type)

        # Register fixed G2 parameters.
        self.register_buffer("g2_eta_params", torch.tensor(g2_eta, dtype=torch.float))
        self.register_buffer("g2_rs_params", torch.tensor(g2_rs, dtype=torch.float))
        self.n_g2 = len(g2_eta) * len(g2_rs) * len(element_types)

        # Register fixed G4 parameters.
        self.register_buffer("g4_eta_params", torch.tensor(g4_eta, dtype=torch.float))
        self.register_buffer("g4_zeta_params", torch.tensor(g4_zeta, dtype=torch.float))
        self.register_buffer("g4_lamda_params", torch.tensor(g4_lamda, dtype=torch.float))
        # Keras ACSFG4 uses relation channels for unordered element pairs (j, k).
        pair_relations = []
        for idx_a, elem_a in enumerate(self.element_types):
            for idx_b in range(idx_a, self.n_types):
                elem_b = self.element_types[idx_b]
                pair_relations.append((elem_a, elem_b))
        self.pair_relations = pair_relations
        self.n_pair_relations = len(pair_relations)
        self.n_g4 = self.n_pair_relations * len(g4_eta) * len(g4_zeta) * len(g4_lamda)

        # Descriptor dimension.
        descriptor_dim = self.n_g2 + self.n_g4

        # Optional batch normalization (graph-aware).
        self.use_batch_norm = use_batch_norm
        if use_batch_norm:
            from kgcnn_torch.layers.norm import GraphBatchNorm
            self.batch_norm = GraphBatchNorm(descriptor_dim)

        # Element-specific MLP using RelationalDense (matching Keras).
        self.relational_mlp = RelationalMLP(
            units=relational_units,
            input_dim=descriptor_dim,
            num_relations=num_relations,
            activation=relational_activation,
        )

        # Graph-level pooling.
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # Always project pooled/atomwise representation to num_targets.
        if output_units is None:
            output_units = []
        pool_out_dim = relational_units[-1]
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=pool_out_dim,
            activation=out_act,
        )

    def _compute_g2(self, z, pos, edge_index):
        """Compute G2 radial symmetry functions.

        Args:
            z: Atomic numbers (N,).
            pos: Positions (N, 3).
            edge_index: Edge indices (2, M).

        Returns:
            Radial descriptors of shape (N, n_g2).
        """
        num_nodes = z.size(0)
        diff = pos[edge_index[0]] - pos[edge_index[1]]
        dist = torch.sqrt((diff * diff).sum(dim=-1) + 1e-8)
        fc = cosine_cutoff(dist, self.g2_rc)

        features = []
        # Match Keras ACSFG2 flatten order:
        # relation(element) major, then parameter table order (rs major, eta minor).
        for elem in self.element_types:
            z_mask = (z[edge_index[0]] == elem).float()
            for rs_val in self.g2_rs_params:
                for eta_val in self.g2_eta_params:
                    gauss = torch.exp(-eta_val * (dist - rs_val) ** 2)
                    edge_feat = gauss * fc * z_mask
                    feat = scatter_reduce_sum(edge_index[1], edge_feat.unsqueeze(-1), num_nodes)
                    features.append(feat)

        return torch.cat(features, dim=-1)  # (N, n_g2)

    def _compute_g4(self, z, pos, angle_index):
        """Compute G4 angular symmetry functions.

        Args:
            z: Atomic numbers (N,).
            pos: Positions (N, 3).
            angle_index: Angle indices (3, K).

        Returns:
            Angular descriptors of shape (N, n_g4).
        """
        num_nodes = z.size(0)
        idx_i = angle_index[0]
        idx_j = angle_index[1]
        idx_k = angle_index[2]

        vec_ij = pos[idx_j] - pos[idx_i]
        vec_ik = pos[idx_k] - pos[idx_i]
        r_ij = torch.sqrt((vec_ij * vec_ij).sum(dim=-1) + 1e-8)
        r_ik = torch.sqrt((vec_ik * vec_ik).sum(dim=-1) + 1e-8)
        vec_jk = pos[idx_k] - pos[idx_j]
        r_jk = torch.sqrt((vec_jk * vec_jk).sum(dim=-1) + 1e-8)

        # Match Keras ACSFG4 exactly (no epsilon or clipping in cosine term).
        cos_theta = (vec_ij * vec_ik).sum(dim=-1) / (r_ij * r_ik)

        fc_ij = cosine_cutoff(r_ij, self.g4_rc)
        fc_ik = cosine_cutoff(r_ik, self.g4_rc)
        fc_jk = cosine_cutoff(r_jk, self.g4_rc)

        # Build unordered pair relation id per (z_j, z_k) on atomic numbers.
        zj = z[idx_j]
        zk = z[idx_k]
        pair_rel = torch.full_like(zj, fill_value=-1, dtype=torch.long)
        for rel_id, (elem_a, elem_b) in enumerate(self.pair_relations):
            pair_mask = ((zj == elem_a) & (zk == elem_b)) | ((zj == elem_b) & (zk == elem_a))
            pair_rel[pair_mask] = rel_id

        features = []
        # Match Keras ACSFG4 flatten order:
        # relation(jk pair) major, parameter table order minor (eta, zeta, lamda).
        for rel_id in range(self.n_pair_relations):
            rel_mask = (pair_rel == rel_id).float()
            for eta_val in self.g4_eta_params:
                for zeta_val in self.g4_zeta_params:
                    for lam_val in self.g4_lamda_params:
                        angular = (1.0 + lam_val * cos_theta) ** zeta_val
                        prefactor = 2.0 ** (1.0 - zeta_val)
                        gauss = torch.exp(-eta_val * (r_ij ** 2 + r_ik ** 2 + r_jk ** 2))
                        triplet_feat = prefactor * angular * gauss * fc_ij * fc_ik * fc_jk
                        if self.g4_multiplicity is not None:
                            triplet_feat = triplet_feat / self.g4_multiplicity
                        triplet_feat = triplet_feat * rel_mask
                        feat = scatter_reduce_sum(idx_i, triplet_feat.unsqueeze(-1), num_nodes)
                        features.append(feat)

        return torch.cat(features, dim=-1)  # (N, n_g4)

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object with z, pos, edge_index, angle_index, batch.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        pos = data.pos
        edge_index = data.edge_index
        angle_index = data.angle_index
        batch = data.batch

        if angle_index.dim() == 2 and angle_index.size(0) == 2:
            e_ij = angle_index[0].long()
            e_ik = angle_index[1].long()
            idx_i = edge_index[1, e_ij]
            idx_j = edge_index[0, e_ij]
            idx_k = edge_index[0, e_ik]
            angle_index = torch.stack([idx_i, idx_j, idx_k], dim=0)

        type_map = self.z_to_type[z.long()]

        # Compute symmetry function descriptors.
        rep_rad = self._compute_g2(z, pos, edge_index)
        rep_ang = self._compute_g4(z, pos, angle_index)
        rep = torch.cat([rep_rad, rep_ang], dim=-1)

        # Compute batch_size for pooling and normalization.
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        if self.use_batch_norm:
            rep = self.batch_norm(rep, batch, batch_size)

        n = self.relational_mlp(rep, type_map, batch, batch_size)

        if self.output_embedding == "graph":
            out = self.pooling(n, batch, batch_size)
        else:
            out = n

        out = self.output_mlp(out)

        return out


class HDNNP2ndAtomWiseModel(nn.Module):
    """2nd generation HDNNP for pre-computed atom-wise representations.

    This simplified variant takes pre-computed per-atom representations
    (e.g., from external symmetry function computation) and applies
    element-specific MLPs (RelationalMLP) to produce per-atom energies,
    which are summed to give the total energy.

    This is useful when the symmetry function descriptors are computed
    externally (e.g., for very large systems or when using pre-computed
    ACSF/SOAP/etc. descriptors).

    Expects PyG Data batch with:
        - data.z: Atomic numbers (N,).
        - data.x: Pre-computed representations (N, F).
        - data.batch: Batch assignment (N,).

    Args:
        element_types: List of atomic numbers for elements in the dataset.
        input_dim: Dimension of pre-computed per-atom representations.
        relational_units: Hidden layer sizes for element-specific MLP.
        relational_activation: Activation function for element-specific MLP.
        node_pooling: Pooling method for graph-level readout.
        output_units: Hidden dimensions for optional output MLP.
        output_activation: Activation function for output MLP.
        num_targets: Number of output targets.
    """

    def __init__(self,
                 element_types: list = None,
                 input_dim: int = 40,
                 relational_units: list = None,
                 relational_activation = None,
                 num_relations: int = 96,
                 node_pooling: str = "sum",
                 output_units: list = None,
                 output_activation: str = "linear",
                 num_targets: int = 1,
                 output_embedding: str = "graph"):
        super().__init__()

        if element_types is None:
            element_types = [1, 6, 7, 8]
        if relational_units is None:
            relational_units = [64, 64, 64]
        if relational_activation is None:
            relational_activation = ["swish", "swish", "linear"]

        self.output_embedding = output_embedding
        self.element_types = sorted(element_types)
        self.n_types = len(self.element_types)

        # Build lookup table: atomic_number -> type index.
        max_z = max(self.element_types) + 1
        z_to_type = torch.full((max_z,), -1, dtype=torch.long)
        for idx, z_val in enumerate(self.element_types):
            z_to_type[z_val] = idx
        self.register_buffer("z_to_type", z_to_type)

        # Element-specific MLP using RelationalDense (matching Keras).
        self.relational_mlp = RelationalMLP(
            units=relational_units,
            input_dim=input_dim,
            num_relations=num_relations,
            activation=relational_activation,
        )

        # Graph-level pooling.
        self.pooling = PoolingNodes(pooling_method=node_pooling)

        # Always project pooled/atomwise representation to num_targets.
        if output_units is None:
            output_units = []
        pool_out_dim = relational_units[-1]
        out_units = output_units + [num_targets]
        out_act = [output_activation] * len(output_units) + ["linear"]
        self.output_mlp = MLP(
            units=out_units,
            input_dim=pool_out_dim,
            activation=out_act,
        )

    def forward(self, data) -> torch.Tensor:
        """Forward pass.

        Args:
            data: PyG Data batch object with z, x (pre-computed representations), batch.

        Returns:
            Graph-level predictions of shape (B, num_targets).
        """
        z = data.z
        x = data.x
        batch = data.batch

        type_map = self.z_to_type[z.long()]
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Apply element-specific MLP.
        n = self.relational_mlp(x, type_map, batch, batch_size)

        # Graph-level pooling.
        if self.output_embedding == "graph":
            out = self.pooling(n, batch, batch_size)
        else:
            out = n

        out = self.output_mlp(out)

        return out
