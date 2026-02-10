"""Geometric layers for graph neural networks.

Includes distance/angle computation, Gaussian/Bessel/Spherical basis expansions,
and cutoff envelope functions.
"""
import math
import torch
import torch.nn as nn

# ---- Reuse implementations from polynom.py to avoid duplication ----
from kgcnn_torch.layers.polynom import (
    spherical_bessel_jn_zeros,
    spherical_bessel_jn_normalization_prefactor,
    torch_spherical_bessel_jn,
    torch_spherical_harmonics_yl,
)


# ---- Distance and direction functions ----

def compute_edge_distances(pos: torch.Tensor, edge_index: torch.Tensor,
                           eps: bool = True) -> torch.Tensor:
    """Compute euclidean distances for edges.

    Args:
        pos: Node positions of shape (N, 3).
        edge_index: Edge indices of shape (2, M), PyG convention.
        eps: Whether to add epsilon for numerical stability.

    Returns:
        Edge distances of shape (M, 1).
    """
    diff = pos[edge_index[1]] - pos[edge_index[0]]  # target - source (consistent)
    dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
    if eps:
        dist_sq = dist_sq + torch.finfo(dist_sq.dtype).eps
    return torch.sqrt(dist_sq)


def compute_edge_direction_normalized(pos: torch.Tensor,
                                      edge_index: torch.Tensor) -> torch.Tensor:
    """Compute normalized direction vectors for edges.

    Direction is target - source (i.e., receiving - sending), matching the
    Keras convention: r_ij = pos_i - pos_j where i=target, j=source.

    Args:
        pos: Node positions of shape (N, 3).
        edge_index: Edge indices of shape (2, M), PyG convention
            (edge_index[0]=source, edge_index[1]=target).

    Returns:
        Normalized direction vectors of shape (M, 3).
    """
    diff = pos[edge_index[1]] - pos[edge_index[0]]  # target - source
    norm = torch.norm(diff, dim=-1, keepdim=True).clamp(min=torch.finfo(diff.dtype).eps)
    return diff / norm


def shift_periodic_lattice(pos: torch.Tensor, edge_image: torch.Tensor,
                           lattice: torch.Tensor, batch_edge: torch.Tensor) -> torch.Tensor:
    """Shift positions by lattice vectors for periodic systems.

    Args:
        pos: Positions of shape (M, 3).
        edge_image: Image indices of shape (M, 3).
        lattice: Lattice matrices of shape (B, 3, 3).
        batch_edge: Batch assignment for edges of shape (M,).

    Returns:
        Shifted positions of shape (M, 3).
    """
    lattice_rep = lattice[batch_edge]  # (M, 3, 3)
    shift = torch.sum(lattice_rep * edge_image.unsqueeze(-1), dim=1)  # (M, 3)
    return pos + shift


# ---- nn.Module layers ----

class GaussBasisLayer(nn.Module):
    """Expand distances into Gaussian basis functions (SchNet).

    e_k(d) = exp(-gamma * (d - mu_k)^2)
    """

    def __init__(self, bins: int = 20, distance: float = 4.0, sigma: float = 0.4,
                 offset: float = 0.0):
        super().__init__()
        self.bins = bins
        self.distance = distance
        self.sigma = sigma
        self.offset = offset
        self.gamma = 1.0 / (sigma * sigma * 2.0)
        # Register centers as buffer (not a parameter)
        # Use arange/bins*distance to match Keras convention (excludes endpoint)
        centers = torch.arange(0, bins, dtype=torch.float) / bins * distance
        self.register_buffer("centers", centers)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances into Gaussian basis.

        Args:
            dist: Distances of shape (M, 1).

        Returns:
            Expanded distances of shape (M, bins).
        """
        return torch.exp(-self.gamma * (dist - self.offset - self.centers).pow(2))


class BesselBasisLayer(nn.Module):
    """Radial Bessel basis expansion (DimeNet).

    RBF_n(d) = sqrt(2/c) * sin(n*pi*d/c) / d * envelope(d/c)
    """

    def __init__(self, num_radial: int, cutoff: float, envelope_exponent: int = 5):
        super().__init__()
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent
        # Trainable frequencies
        freq = torch.arange(1, num_radial + 1, dtype=torch.float) * math.pi
        self.frequencies = nn.Parameter(freq)

    def envelope(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Polynomial envelope for smooth cutoff."""
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / d_scaled + a * d_scaled.pow(p - 1) + b * d_scaled.pow(p) + c * d_scaled.pow(p + 1)
        return torch.where(d_scaled < 1.0, env_val, torch.zeros_like(d_scaled))

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Expand distances into Bessel basis.

        Args:
            dist: Distances of shape (M, 1).

        Returns:
            Expanded distances of shape (M, num_radial).
        """
        d_scaled = dist / self.cutoff  # (M, 1)
        d_cutoff = self.envelope(d_scaled)  # (M, 1)
        return d_cutoff * torch.sin(self.frequencies * d_scaled)  # (M, num_radial)


class CosCutOffEnvelope(nn.Module):
    """Cosine cutoff envelope: f_c(r) = 0.5 * (cos(pi*r/r_c) + 1)."""

    def __init__(self, cutoff: float = None):
        super().__init__()
        if cutoff is None:
            cutoff = 1e8
        self.cutoff = abs(cutoff)

    def forward(self, dist: torch.Tensor) -> torch.Tensor:
        """Apply cutoff envelope.

        Args:
            dist: Distances of shape (M, 1).

        Returns:
            Envelope values of shape (M, 1).
        """
        # Match Keras reference implementation: clip to [-cutoff, cutoff].
        dist_clamped = torch.clamp(dist, min=-self.cutoff, max=self.cutoff)
        return 0.5 * (torch.cos(dist_clamped * math.pi / self.cutoff) + 1.0)


class SphericalBasisLayer(nn.Module):
    """Spherical basis expansion (DimeNet).

    Combines radial Bessel functions with spherical harmonics.
    """

    def __init__(self, num_spherical: int, num_radial: int, cutoff: float,
                 envelope_exponent: int = 5):
        super().__init__()
        assert num_radial <= 64
        self.num_radial = num_radial
        self.num_spherical = num_spherical
        self.cutoff = cutoff
        self.envelope_exponent = envelope_exponent

        # Precompute bessel zeros and normalization
        bessel_zeros = spherical_bessel_jn_zeros(num_spherical, num_radial)
        bessel_norm = spherical_bessel_jn_normalization_prefactor(num_spherical, num_radial)
        self.register_buffer("bessel_zeros", torch.tensor(bessel_zeros, dtype=torch.float))
        self.register_buffer("bessel_norm", torch.tensor(bessel_norm, dtype=torch.float))

    def envelope(self, d_scaled: torch.Tensor) -> torch.Tensor:
        """Polynomial envelope for smooth cutoff."""
        p = self.envelope_exponent + 1
        a = -(p + 1) * (p + 2) / 2
        b = p * (p + 2)
        c = -p * (p + 1) / 2
        env_val = 1.0 / d_scaled + a * d_scaled.pow(p - 1) + b * d_scaled.pow(p) + c * d_scaled.pow(p + 1)
        return torch.where(d_scaled < 1.0, env_val, torch.zeros_like(d_scaled))

    def forward(self, dist: torch.Tensor, angles: torch.Tensor,
                angle_index: torch.Tensor) -> torch.Tensor:
        """Expand distance/angle pairs into spherical basis.

        Args:
            dist: Edge distances of shape (M, 1).
            angles: Angle values of shape (K, 1).
            angle_index: Angle edge indices of shape (2, K). angle_index[1] is the
                source edge index (kj) for the RBF lookup.

        Returns:
            Expanded basis of shape (K, num_radial * num_spherical).
        """
        d_scaled = dist[:, 0] / self.cutoff  # (M,)

        # Compute radial basis functions
        rbf = []
        for n in range(self.num_spherical):
            for k in range(self.num_radial):
                rbf.append(
                    self.bessel_norm[n, k] *
                    torch_spherical_bessel_jn(d_scaled * self.bessel_zeros[n, k], n)
                )
        rbf = torch.stack(rbf, dim=1)  # (M, num_spherical * num_radial)

        # Apply envelope
        d_cutoff = self.envelope(d_scaled)  # (M,)
        rbf_env = d_cutoff.unsqueeze(1) * rbf  # (M, S*R)

        # Gather for angle pairs (source/kj edge of angle)
        rbf_env = rbf_env[angle_index[1]]  # (K, S*R)

        # Compute angular/spherical basis
        cbf = []
        for n in range(self.num_spherical):
            cbf.append(torch_spherical_harmonics_yl(angles[:, 0], n))
        cbf = torch.stack(cbf, dim=1)  # (K, S)
        cbf = cbf.repeat_interleave(self.num_radial, dim=1)  # (K, S*R)

        return rbf_env * cbf


class NodePosition(nn.Module):
    """Get source and target node positions for each edge."""

    def forward(self, pos: torch.Tensor, edge_index: torch.Tensor):
        """Forward pass.

        Args:
            pos: Node positions (N, 3).
            edge_index: Edge indices (2, M).

        Returns:
            Tuple of (pos_target, pos_source), each (M, 3).
        """
        return pos[edge_index[1]], pos[edge_index[0]]


class EuclideanNorm(nn.Module):
    """Compute Euclidean norm along an axis."""

    def __init__(self, axis: int = -1, keepdims: bool = False,
                 add_eps: bool = True, square_norm: bool = False,
                 invert_norm: bool = False, no_nan: bool = True):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims
        self.add_eps = add_eps
        self.square_norm = square_norm
        self.invert_norm = invert_norm
        self.no_nan = no_nan

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sq = torch.relu((x * x).sum(dim=self.axis, keepdim=self.keepdims))
        if self.add_eps:
            sq = sq + torch.finfo(sq.dtype).eps
        if self.square_norm:
            out = sq
        else:
            out = torch.sqrt(sq)
        if self.invert_norm:
            out = 1.0 / out
            if self.no_nan:
                out = torch.where(torch.isnan(out), torch.zeros_like(out), out)
        return out


class ScalarProduct(nn.Module):
    """Compute dot product along an axis."""

    def __init__(self, axis: int = -1, keepdims: bool = False):
        super().__init__()
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        return (vec1 * vec2).sum(dim=self.axis, keepdim=self.keepdims)


class EdgeAngle(nn.Module):
    """Compute angles between edge pairs (for triplet-based models).

    Given edge vectors and an angle index that maps to pairs of edges,
    computes the angle between each pair.
    """

    def __init__(self, vector_scale: list = None):
        super().__init__()
        self.vector_scale = vector_scale

    def forward(self, vectors: torch.Tensor,
                angle_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            vectors: Edge direction vectors (M, 3).
            angle_index: Indices into vectors for angle pairs (2, K).

        Returns:
            Angles in radians (K, 1).
        """
        v1 = vectors[angle_index[0]]
        v2 = vectors[angle_index[1]]
        if self.vector_scale is not None:
            v1 = v1 * self.vector_scale[0]
            v2 = v2 * self.vector_scale[1]
        # Use atan2(||cross||, dot) for numerical stability (matches Keras)
        dot = (v1 * v2).sum(dim=-1)  # (K,)
        cross = torch.cross(v1, v2, dim=-1)  # (K, 3)
        cross_norm = torch.norm(cross, dim=-1)  # (K,)
        angle = torch.atan2(cross_norm, dot)  # (K,)
        return angle.unsqueeze(-1)  # (K, 1)


class NodeDistanceEuclidean(nn.Module):
    """Compute Euclidean distance between node position pairs (Module version)."""

    def __init__(self, add_eps: bool = True):
        super().__init__()
        self.add_eps = add_eps

    def forward(self, pos_start: torch.Tensor, pos_stop: torch.Tensor) -> torch.Tensor:
        """Compute distance ||pos_start - pos_stop||.

        Args:
            pos_start: Positions of shape (M, 3).
            pos_stop: Positions of shape (M, 3).

        Returns:
            Distances of shape (M, 1).
        """
        diff = pos_start - pos_stop
        dist_sq = (diff * diff).sum(dim=-1, keepdim=True)
        if self.add_eps:
            dist_sq = dist_sq + torch.finfo(dist_sq.dtype).eps
        return torch.sqrt(dist_sq)


class EdgeDirectionNormalized(nn.Module):
    """Compute normalized direction vectors between position pairs (Module version)."""

    def __init__(self, add_eps: bool = True):
        super().__init__()
        self.add_eps = add_eps

    def forward(self, pos_start: torch.Tensor, pos_stop: torch.Tensor) -> torch.Tensor:
        """Compute normalized direction (pos_start - pos_stop) / ||...||.

        Args:
            pos_start: Positions of shape (M, 3).
            pos_stop: Positions of shape (M, 3).

        Returns:
            Normalized direction vectors of shape (M, 3).
        """
        diff = pos_start - pos_stop
        norm = torch.norm(diff, dim=-1, keepdim=True)
        if self.add_eps:
            norm = norm.clamp(min=torch.finfo(norm.dtype).eps)
        return diff / norm


class VectorAngle(nn.Module):
    """Compute angle between two vectors using arctan2(cross, dot)."""

    def forward(self, vec1: torch.Tensor, vec2: torch.Tensor) -> torch.Tensor:
        """Compute angle between vec1 and vec2.

        Args:
            vec1: Vectors of shape (M, 3).
            vec2: Vectors of shape (M, 3).

        Returns:
            Angles in radians of shape (M, 1).
        """
        dot = (vec1 * vec2).sum(dim=-1)  # (M,)
        cross = torch.cross(vec1, vec2, dim=-1)  # (M, 3)
        cross_norm = torch.norm(cross, dim=-1)  # (M,)
        angle = torch.atan2(cross_norm, dot)  # (M,)
        return angle.unsqueeze(-1)  # (M, 1)


class CosCutOff(nn.Module):
    """Apply cosine cutoff envelope to input values.

    out = 0.5 * (cos(pi * d / r_c) + 1) * input
    """

    def __init__(self, cutoff: float = None):
        super().__init__()
        if cutoff is None:
            cutoff = 1e8
        self.cutoff = abs(cutoff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply cutoff envelope to values.

        Args:
            x: Input values of shape (M, D).

        Returns:
            Cutoff-modulated values of shape (M, D).
        """
        fc = torch.clamp(x, -self.cutoff, self.cutoff)
        envelope = 0.5 * (torch.cos(fc * math.pi / self.cutoff) + 1.0)
        return envelope * x


class DisplacementVectorsUnitCell(nn.Module):
    """Compute displacement vectors for periodic unit cell systems.

    d_ij = frac_i - (frac_j + cell_translation_ij)
    """

    def forward(self, frac_coords: torch.Tensor, edge_index: torch.Tensor,
                cell_translations: torch.Tensor) -> torch.Tensor:
        """Compute displacement vectors.

        Args:
            frac_coords: Fractional coordinates of shape (N, 3).
            edge_index: Edge indices of shape (2, M).
            cell_translations: Cell translation vectors of shape (M, 3).

        Returns:
            Displacement vectors of shape (M, 3).
        """
        frac_i = frac_coords[edge_index[1]]  # target/receiving
        frac_j = frac_coords[edge_index[0]]  # source/sending
        return frac_i - (frac_j + cell_translations)


class DisplacementVectorsASU(nn.Module):
    """Compute displacement vectors for asymmetric unit (ASU) with symmetry operations.

    Applies symmetry operations to outgoing coordinates before displacement.
    """

    def forward(self, frac_coords: torch.Tensor, edge_index: torch.Tensor,
                symmetry_ops: torch.Tensor, cell_translations: torch.Tensor) -> torch.Tensor:
        """Compute displacement vectors with symmetry operations.

        Args:
            frac_coords: Fractional coordinates of shape (N, 3).
            edge_index: Edge indices of shape (2, M).
            symmetry_ops: Affine symmetry operations of shape (M, 4, 4).
            cell_translations: Cell translation vectors of shape (M, 3).

        Returns:
            Displacement vectors of shape (M, 3).
        """
        frac_i = frac_coords[edge_index[1]]  # target/receiving
        frac_j = frac_coords[edge_index[0]]  # source/sending
        # Apply affine transformation: [x, y, z, 1] @ sym_op.T -> [x', y', z', 1]
        ones = torch.ones(frac_j.size(0), 1, device=frac_j.device, dtype=frac_j.dtype)
        homo = torch.cat([frac_j, ones], dim=-1)  # (M, 4)
        transformed = torch.einsum('mi,mji->mj', homo, symmetry_ops)[:, :3]  # (M, 3)
        # Remove integer parts (wrap into unit cell)
        transformed = transformed - torch.floor(transformed)
        # Add cell translations
        transformed = transformed + cell_translations
        return frac_i - transformed


class FracToRealCoordinates(nn.Module):
    """Convert fractional coordinates to real-space using lattice matrix.

    real = frac @ lattice_matrix
    """

    def forward(self, frac_coords: torch.Tensor, lattice: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """Convert fractional to real coordinates.

        Args:
            frac_coords: Fractional coordinates of shape (N, 3).
            lattice: Lattice matrices of shape (B, 3, 3).
            batch: Batch assignment of shape (N,).

        Returns:
            Real-space coordinates of shape (N, 3).
        """
        lat = lattice[batch]  # (N, 3, 3)
        return torch.einsum('ni,nij->nj', frac_coords, lat)


class RealToFracCoordinates(nn.Module):
    """Convert real-space coordinates to fractional using inverse lattice matrix.

    frac = real @ inv_lattice_matrix
    """

    def __init__(self, is_inverse_lattice_matrix: bool = False):
        super().__init__()
        self.is_inverse_lattice_matrix = is_inverse_lattice_matrix

    def forward(self, real_coords: torch.Tensor, lattice: torch.Tensor,
                batch: torch.Tensor) -> torch.Tensor:
        """Convert real to fractional coordinates.

        Args:
            real_coords: Real-space coordinates of shape (N, 3).
            lattice: Lattice matrices of shape (B, 3, 3). If is_inverse_lattice_matrix
                is True, this is the inverse lattice matrix directly.
            batch: Batch assignment of shape (N,).

        Returns:
            Fractional coordinates of shape (N, 3).
        """
        lat = lattice[batch]  # (N, 3, 3)
        if not self.is_inverse_lattice_matrix:
            lat = torch.linalg.inv(lat)
        return torch.einsum('ni,nij->nj', real_coords, lat)


class PositionEncodingBasisLayer(nn.Module):
    """Fourier / sinusoidal position encoding for distances.

    Maps scalar distances to sinusoidal features.
    """

    def __init__(self, dim_half: int = 10, wave_length_min: float = 1.0,
                 num_mult: int = 100, include_frequencies: bool = False,
                 interleave_sin_cos: bool = False):
        super().__init__()
        self.dim_half = dim_half
        self.wave_length_min = wave_length_min
        self.num_mult = num_mult
        self.include_frequencies = include_frequencies
        self.interleave_sin_cos = interleave_sin_cos
        if num_mult <= 1:
            raise ValueError("`num_mult` must be >1. Reduce `wave_length_min` if necessary.")
        if dim_half <= 1:
            raise ValueError("`dim_half` must be > 1.")
        # Precompute frequencies to match Keras:
        # freq = exp(-log(num_mult) * i/(dim_half-1) - log(wave_length_min))
        # scales = freq * 2 * pi
        steps = torch.arange(dim_half, dtype=torch.float) / (dim_half - 1)
        log_num = -math.log(num_mult)
        log_wave = -math.log(wave_length_min)
        freq = torch.exp(log_num * steps + log_wave)
        scales = freq * math.pi * 2.0
        self.register_buffer("scales", scales)
        self.register_buffer("freq", freq)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Distances of shape (..., 1) or (...,).

        Returns:
            Encoded features of shape (..., 2*dim_half) or (..., 3*dim_half) if
            include_frequencies is True.
        """
        if x.dim() >= 1 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        arg = x.unsqueeze(-1) * self.scales  # (..., dim_half)
        sin_enc = torch.sin(arg)
        cos_enc = torch.cos(arg)
        if self.interleave_sin_cos:
            out = torch.stack([sin_enc, cos_enc], dim=-1).flatten(-2)
        else:
            out = torch.cat([sin_enc, cos_enc], dim=-1)
        if self.include_frequencies:
            out = torch.cat([out, self.freq.expand(arg.shape)], dim=-1)
        return out
