#!/usr/bin/env python3
"""Layerwise numeric alignment check: MAT distance + single attention head."""
import os
import sys
from dataclasses import dataclass
from typing import Dict

import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.literature.MAT._layers import MATDistanceMatrix as KerasMATDistanceMatrix
from kgcnn.literature.MAT._layers import MATAttentionHead as KerasMATAttentionHead
from kgcnn_torch.models.mat import MATDistanceMatrix, MATAttentionHead


@dataclass
class Config:
    batch: int = 2
    n_nodes: int = 7
    in_dim: int = 16
    units: int = 8
    seed: int = 42


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def compare_stage_dicts(ref: Dict[str, torch.Tensor], got: Dict[str, torch.Tensor]):
    print("Stage alignment report (Keras vs Torch):")
    worst_mae = 0.0
    worst_abs = 0.0
    for key in ref.keys():
        r, g = ref[key], got[key]
        diff = (r - g).abs()
        mae = float(diff.mean().item())
        rmse = float(torch.sqrt(((r - g) ** 2).mean()).item())
        max_abs = float(diff.max().item())
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")
        worst_mae = max(worst_mae, mae)
        worst_abs = max(worst_abs, max_abs)
    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    h = torch.randn(cfg.batch, cfg.n_nodes, cfg.in_dim)
    xyz = torch.randn(cfg.batch, cfg.n_nodes, 3)
    a_g = torch.randint(0, 2, (cfg.batch, cfg.n_nodes, cfg.n_nodes, 1)).float()

    node_mask = (torch.rand(cfg.batch, cfg.n_nodes, 1) > 0.2).float()
    h = h * node_mask
    xyz_mask = node_mask.repeat(1, 1, 3)

    t_dist = MATDistanceMatrix(trafo="exp")
    k_dist = KerasMATDistanceMatrix(trafo="exp")

    t_a_d, t_a_d_mask = t_dist(xyz, node_mask)
    k_a_d, k_a_d_mask = k_dist(ops.convert_to_tensor(xyz.numpy()), mask=ops.convert_to_tensor(xyz_mask.numpy()))

    t_head = MATAttentionHead(
        units=cfg.units,
        input_dim=cfg.in_dim,
        lambda_attention=0.3,
        lambda_distance=0.3,
        lambda_adjacency=0.4,
        add_identity=False,
        dropout=0.0,
    )
    k_head = KerasMATAttentionHead(
        units=cfg.units,
        lambda_attention=0.3,
        lambda_distance=0.3,
        lambda_adjacency=0.4,
        add_identity=False,
        dropout=None,
    )

    _ = k_head([
        ops.convert_to_tensor(h.numpy()),
        k_a_d,
        ops.convert_to_tensor(a_g.numpy()),
    ], mask=[ops.convert_to_tensor(node_mask.numpy()), k_a_d_mask, ops.convert_to_tensor((a_g > 0).numpy())])

    copy_dense_torch_to_keras(t_head.dense_q, k_head.dense_q)
    copy_dense_torch_to_keras(t_head.dense_k, k_head.dense_k)
    copy_dense_torch_to_keras(t_head.dense_v, k_head.dense_v)

    t_hp = t_head(h, t_a_d, a_g, node_mask, t_a_d_mask, (a_g > 0).float()).detach().cpu()
    k_hp = k_head([
        ops.convert_to_tensor(h.numpy()),
        k_a_d,
        ops.convert_to_tensor(a_g.numpy()),
    ], mask=[ops.convert_to_tensor(node_mask.numpy()), k_a_d_mask, ops.convert_to_tensor((a_g > 0).numpy())])

    ref = {
        "dist": torch.as_tensor(ops.convert_to_numpy(k_a_d)),
        "head": torch.as_tensor(ops.convert_to_numpy(k_hp)),
    }
    got = {
        "dist": t_a_d.detach().cpu(),
        "head": t_hp,
    }
    compare_stage_dicts(ref, got)


if __name__ == "__main__":
    main()
