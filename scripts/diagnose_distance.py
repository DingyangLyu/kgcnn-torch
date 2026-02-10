#!/usr/bin/env python3
"""Diagnose why DimeNetPP distances differ between Torch and Keras."""
import os, sys
os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import keras

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from model_alignment_utils import make_disjoint_graph
from align_dimenetpp_model import Config

def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)

    # Check if positions are identical
    pos_t = torch_data.pos
    pos_k = keras_data["pos"]
    print(f"pos identical: {torch.equal(pos_t, pos_k)}")
    print(f"pos diff: {(pos_t - pos_k).abs().max():.2e}")

    # Check edge indices map to same pairs
    ei_t = torch_data.edge_index  # [src, dst]
    ei_k = keras_data["edge_index"]  # [dst, src]
    print(f"\nTorch edge_index [src, dst]:\n  row0(src): {ei_t[0][:5]}...\n  row1(dst): {ei_t[1][:5]}...")
    print(f"Keras edge_index [dst, src]:\n  row0(dst): {ei_k[0][:5]}...\n  row1(src): {ei_k[1][:5]}...")
    print(f"Torch dst == Keras dst: {torch.equal(ei_t[1], ei_k[0])}")
    print(f"Torch src == Keras src: {torch.equal(ei_t[0], ei_k[1])}")

    # Compute distances both ways
    # Torch: target - source
    diff_t = pos_t[ei_t[1]] - pos_t[ei_t[0]]  # dst - src
    dist_sq_t = (diff_t * diff_t).sum(dim=-1, keepdim=True)
    torch_eps = torch.finfo(dist_sq_t.dtype).eps
    dist_t_with_eps = torch.sqrt(dist_sq_t + torch_eps)
    dist_t_no_eps = torch.sqrt(dist_sq_t)

    # Keras: NodePosition gathers pos at [ei[0], ei[1]] = [dst, src]
    # Then Subtract([pos_dst, pos_src]) = dst - src
    diff_k = pos_k[ei_k[0]] - pos_k[ei_k[1]]  # dst - src in keras convention
    dist_sq_k = (diff_k * diff_k).sum(dim=-1, keepdim=True)
    # Keras EuclideanNorm: relu(dist_sq) + keras.backend.epsilon()
    keras_eps = keras.backend.epsilon()
    dist_k_with_eps = torch.sqrt(torch.relu(dist_sq_k) + keras_eps)
    dist_k_no_eps = torch.sqrt(dist_sq_k)

    print(f"\ntorch.finfo.eps = {torch_eps}")
    print(f"keras.backend.epsilon() = {keras_eps}")
    print(f"epsilon difference = {abs(torch_eps - keras_eps):.2e}")

    # Compare
    print(f"\ndiff vectors identical: {torch.equal(diff_t, diff_k)}")
    print(f"dist_sq identical: {torch.equal(dist_sq_t, dist_sq_k)}")
    print(f"dist (no eps) identical: {torch.equal(dist_t_no_eps, dist_k_no_eps)}")
    print(f"dist (torch_eps) vs (keras_eps): MAE={(dist_t_with_eps - dist_k_with_eps).abs().mean():.3e}")
    print(f"dist (torch_eps) vs (keras_eps): MAX={(dist_t_with_eps - dist_k_with_eps).abs().max():.3e}")

    # Check smallest distance
    min_dist = dist_t_no_eps.min()
    min_idx = dist_t_no_eps.argmin()
    print(f"\nSmallest distance: {min_dist.item():.6f} at edge {min_idx.item()}")
    print(f"  dist_sq = {dist_sq_t[min_idx].item():.12e}")
    print(f"  sqrt(dist_sq + torch_eps) = {dist_t_with_eps[min_idx].item():.12e}")
    print(f"  sqrt(relu(dist_sq) + keras_eps) = {dist_k_with_eps[min_idx].item():.12e}")
    print(f"  sqrt(dist_sq) = {dist_t_no_eps[min_idx].item():.12e}")

    # Show the critical near-zero edge
    edge_src = ei_t[0, min_idx].item()
    edge_dst = ei_t[1, min_idx].item()
    print(f"  nodes: {edge_src} -> {edge_dst}")
    print(f"  pos[{edge_src}] = {pos_t[edge_src].tolist()}")
    print(f"  pos[{edge_dst}] = {pos_t[edge_dst].tolist()}")

    # Now check: does the actual forward pass compute distances differently?
    from kgcnn_torch.layers.geom import compute_edge_distances
    from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean

    dist_torch_fn = compute_edge_distances(pos_t, ei_t)
    pos1_k, pos2_k = NodePosition()([pos_k, ei_k])
    dist_keras_fn = NodeDistanceEuclidean()([pos1_k, pos2_k])

    print(f"\nActual function outputs:")
    print(f"  compute_edge_distances MAE vs manual: {(dist_torch_fn - dist_t_with_eps).abs().mean():.3e}")
    print(f"  NodeDistanceEuclidean MAE vs manual: {(dist_keras_fn - dist_k_with_eps).abs().mean():.3e}")
    print(f"  Torch fn vs Keras fn: MAE={(dist_torch_fn - dist_keras_fn).abs().mean():.3e}, "
          f"MAX={(dist_torch_fn - dist_keras_fn).abs().max():.3e}")


if __name__ == "__main__":
    main()
