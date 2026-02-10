#!/usr/bin/env python3
"""Step-by-step comparison of SphericalBasisLayer in Torch vs Keras."""
import os, sys
os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn_torch.layers.geom import SphericalBasisLayer as TorchSBL
from kgcnn.layers.geom import SphericalBasisLayer as KerasSBL
from kgcnn_torch.layers.polynom import (
    torch_spherical_bessel_jn, spherical_bessel_jn_zeros,
    spherical_bessel_jn_normalization_prefactor)
from kgcnn.layers.polynom import tf_spherical_bessel_jn
from model_alignment_utils import make_disjoint_graph
from align_dimenetpp_model import Config, generate_angle_index
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, EdgeAngle
from keras.layers import Subtract
from kgcnn_torch.layers.geom import compute_edge_distances


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_index = generate_angle_index(torch_data.edge_index)

    num_spherical = cfg.num_spherical  # 3
    num_radial = cfg.num_radial        # 4
    cutoff = cfg.cutoff                # 5.0
    envelope_exponent = cfg.envelope_exponent  # 5

    # Build both layers
    torch_sbl = TorchSBL(num_spherical, num_radial, cutoff, envelope_exponent)
    keras_sbl = KerasSBL(num_spherical, num_radial, cutoff, envelope_exponent)

    # Compare bessel_zeros
    print("=== Bessel zeros comparison ===")
    print(f"Torch bessel_zeros: shape={torch_sbl.bessel_zeros.shape}")
    print(f"Keras bessel_n_zeros: type={type(keras_sbl.bessel_n_zeros)}, shape={keras_sbl.bessel_n_zeros.shape}")
    diff = np.abs(torch_sbl.bessel_zeros.numpy() - keras_sbl.bessel_n_zeros)
    print(f"Max diff: {diff.max():.2e}")
    print(f"Torch:\n{torch_sbl.bessel_zeros}")
    print(f"Keras:\n{keras_sbl.bessel_n_zeros}")

    # Compare bessel_norm
    print("\n=== Bessel norm comparison ===")
    print(f"Torch bessel_norm: shape={torch_sbl.bessel_norm.shape}")
    print(f"Keras bessel_norm: type={type(keras_sbl.bessel_norm)}, shape={keras_sbl.bessel_norm.shape}")
    diff_norm = np.abs(torch_sbl.bessel_norm.numpy() - keras_sbl.bessel_norm)
    print(f"Max diff: {diff_norm.max():.2e}")

    # Compute distances and angles
    print("\n=== Input data ===")
    # Torch geometry
    pos_t = torch_data.pos
    ei_t = torch_data.edge_index
    dist_t = compute_edge_distances(pos_t, ei_t)  # (M, 1)

    pos_i_t = pos_t[ei_t[1]]
    pos_j_t = pos_t[ei_t[0]]
    v12_t = pos_i_t - pos_j_t
    vec_a_t = v12_t[angle_index[0]]
    vec_b_t = v12_t[angle_index[1]]
    dot_t = (vec_a_t * vec_b_t).sum(dim=-1)
    cross_t = torch.cross(vec_a_t, vec_b_t, dim=-1)
    cross_norm_t = torch.norm(cross_t, dim=-1)
    angles_t = torch.atan2(cross_norm_t, dot_t).unsqueeze(-1)

    # Keras geometry
    pos_k = keras_data["pos"]
    ei_k = keras_data["edge_index"]
    pos1_k, pos2_k = NodePosition()([pos_k, ei_k])
    dist_k = NodeDistanceEuclidean()([pos1_k, pos2_k])
    v12_k = Subtract()([pos1_k, pos2_k])
    angles_k = EdgeAngle()([v12_k, angle_index])

    print(f"dist_t range: [{dist_t.min():.4f}, {dist_t.max():.4f}]")
    print(f"dist_k range: [{dist_k.min():.4f}, {dist_k.max():.4f}]")
    dist_diff = (dist_t - dist_k).abs()
    print(f"dist diff: MAE={dist_diff.mean():.3e}, MAX={dist_diff.max():.3e}")
    print(f"angles_t range: [{angles_t.min():.4f}, {angles_t.max():.4f}]")
    print(f"angles_k range: [{angles_k.min():.4f}, {angles_k.max():.4f}]")
    angle_diff = (angles_t - angles_k).abs()
    print(f"angle diff: MAE={angle_diff.mean():.3e}, MAX={angle_diff.max():.3e}")

    # Step by step SBF computation
    print("\n=== Step-by-step SBF comparison ===")

    d_scaled_t = dist_t[:, 0] / cutoff
    d_scaled_k = dist_k[:, 0] * keras_sbl.inv_cutoff
    print(f"d_scaled diff: MAE={(d_scaled_t - d_scaled_k).abs().mean():.3e}")

    # Compare individual rbf components
    print(f"\n--- rbf components (before stack) ---")
    for n in range(num_spherical):
        for k in range(num_radial):
            x_t = d_scaled_t * torch_sbl.bessel_zeros[n, k]
            x_k = d_scaled_k * keras_sbl.bessel_n_zeros[n][k]
            x_diff = (x_t - x_k).abs()

            jn_t = torch_spherical_bessel_jn(x_t, n)
            jn_k = tf_spherical_bessel_jn(x_k, n)
            jn_diff = (jn_t - jn_k).abs()

            rbf_t = float(torch_sbl.bessel_norm[n, k]) * jn_t
            rbf_k = float(keras_sbl.bessel_norm[n, k]) * jn_k
            rbf_diff = (rbf_t - rbf_k).abs()

            print(f"  n={n},k={k}: x_diff={x_diff.mean():.3e}  "
                  f"jn_diff={jn_diff.mean():.3e}  "
                  f"rbf_diff={rbf_diff.mean():.3e}  "
                  f"rbf_t_norm={rbf_t.norm():.3e}  rbf_k_norm={rbf_k.norm():.3e}")

    # Full rbf stack
    rbf_t_list = []
    rbf_k_list = []
    for n in range(num_spherical):
        for k in range(num_radial):
            rbf_t_list.append(
                torch_sbl.bessel_norm[n, k] *
                torch_spherical_bessel_jn(d_scaled_t * torch_sbl.bessel_zeros[n, k], n))
            rbf_k_list.append(
                keras_sbl.bessel_norm[n, k] *
                tf_spherical_bessel_jn(d_scaled_k * keras_sbl.bessel_n_zeros[n][k], n))
    rbf_t = torch.stack(rbf_t_list, dim=1)
    rbf_k = torch.stack(rbf_k_list, dim=1)
    rbf_diff_all = (rbf_t - rbf_k).abs()
    print(f"\nrbf stack diff: MAE={rbf_diff_all.mean():.3e}, MAX={rbf_diff_all.max():.3e}")

    # Envelope
    d_cutoff_t = torch_sbl.envelope(d_scaled_t)
    d_cutoff_k = keras_sbl.envelope(d_scaled_k)
    env_diff = (d_cutoff_t - d_cutoff_k).abs()
    print(f"envelope diff: MAE={env_diff.mean():.3e}, MAX={env_diff.max():.3e}")

    # rbf_env = d_cutoff * rbf
    rbf_env_t = d_cutoff_t.unsqueeze(1) * rbf_t
    rbf_env_k = d_cutoff_k[:, None] * rbf_k
    env_rbf_diff = (rbf_env_t - rbf_env_k).abs()
    print(f"rbf_env diff: MAE={env_rbf_diff.mean():.3e}, MAX={env_rbf_diff.max():.3e}")

    # Gather for angle pairs
    print(f"\nangle_index[1] (source edge): {angle_index[1][:5]}...")
    rbf_env_t_gathered = rbf_env_t[angle_index[1]]
    # Keras uses GatherNodesOutgoing which gathers at index 1
    from kgcnn.layers.gather import GatherNodesOutgoing
    rbf_env_k_gathered = GatherNodesOutgoing()([rbf_env_k, angle_index])
    gather_diff = (rbf_env_t_gathered - rbf_env_k_gathered).abs()
    print(f"rbf_env gathered diff: MAE={gather_diff.mean():.3e}, MAX={gather_diff.max():.3e}")

    # Spherical harmonics
    print(f"\n--- Spherical harmonics ---")
    from kgcnn_torch.layers.polynom import torch_spherical_harmonics_yl
    from kgcnn.layers.polynom import tf_spherical_harmonics_yl
    for n in range(num_spherical):
        yl_t = torch_spherical_harmonics_yl(angles_t[:, 0], n)
        yl_k = tf_spherical_harmonics_yl(angles_k[:, 0], n)
        yl_diff = (yl_t - yl_k).abs()
        print(f"  Y_{n}: diff MAE={yl_diff.mean():.3e}, MAX={yl_diff.max():.3e}")

    # Full SBF
    sbf_t = torch_sbl(dist_t, angles_t, angle_index)
    # For Keras SBF, need to call it properly
    _ = keras_sbl([dist_k, angles_k, angle_index])  # build
    sbf_k = keras_sbl([dist_k, angles_k, angle_index])
    sbf_diff = (sbf_t - sbf_k).abs()
    print(f"\nFull SBF diff: MAE={sbf_diff.mean():.3e}, MAX={sbf_diff.max():.3e}")
    print(f"SBF norm: torch={sbf_t.norm():.3e}, keras={sbf_k.norm():.3e}")

    # Check for any NaN or Inf
    print(f"\nTorch SBF has NaN: {sbf_t.isnan().any()}, Inf: {sbf_t.isinf().any()}")
    print(f"Keras SBF has NaN: {sbf_k.isnan().any()}, Inf: {sbf_k.isinf().any()}")


if __name__ == "__main__":
    main()
