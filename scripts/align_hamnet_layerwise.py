#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras HamNaiveDynMessage vs kgcnn-torch."""
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

from kgcnn.literature.HamNet._layers import HamNaiveDynMessage as KerasHamNaiveDynMessage
from kgcnn_torch.models.hamnet import HamNaiveDynMessage


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    units: int = 16
    edge_dim: int = 8
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

    h = torch.randn(cfg.n_nodes, cfg.units)
    p = torch.randn(cfg.n_nodes, 3)
    q = torch.randn(cfg.n_nodes, 3)
    e = torch.randn(cfg.n_edges, cfg.edge_dim)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    edge_torch = torch.stack([src, dst], dim=0)
    edge_keras = torch.stack([dst, src], dim=0)

    t_layer = HamNaiveDynMessage(units=cfg.units, edge_dim=cfg.edge_dim, activation="leaky_relu2", activation_last="elu")
    k_layer = KerasHamNaiveDynMessage(units=cfg.units, units_edge=cfg.edge_dim,
                                      activation={"class_name": "function", "config": "kgcnn>leaky_relu2"},
                                      activation_last="elu")

    _ = k_layer([
        ops.convert_to_tensor(h.numpy()),
        ops.convert_to_tensor(e.numpy()),
        ops.convert_to_tensor(p.numpy()),
        ops.convert_to_tensor(q.numpy()),
        ops.convert_to_tensor(edge_keras.numpy()),
    ])

    copy_dense_torch_to_keras(t_layer.align_dense, k_layer.dense_align)
    copy_dense_torch_to_keras(t_layer.attend_dense[0], k_layer.dense_attend)
    copy_dense_torch_to_keras(t_layer.edge_dense, k_layer.dense_e)

    t_node, t_edge = t_layer(h, p, q, e, edge_torch, cfg.n_nodes)
    k_node, k_edge = k_layer([
        ops.convert_to_tensor(h.numpy()),
        ops.convert_to_tensor(e.numpy()),
        ops.convert_to_tensor(p.numpy()),
        ops.convert_to_tensor(q.numpy()),
        ops.convert_to_tensor(edge_keras.numpy()),
    ])

    ref = {
        "node_msg": torch.as_tensor(ops.convert_to_numpy(k_node)),
        "edge_msg": torch.as_tensor(ops.convert_to_numpy(k_edge)),
    }
    got = {
        "node_msg": t_node.detach().cpu(),
        "edge_msg": t_edge.detach().cpu(),
    }
    compare_stage_dicts(ref, got)


if __name__ == "__main__":
    main()
