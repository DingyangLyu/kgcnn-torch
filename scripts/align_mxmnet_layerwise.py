#!/usr/bin/env python3
"""Layerwise numeric alignment check: MXMNet GlobalMP block."""
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

from kgcnn.literature.MXMNet._layers import MXMGlobalMP as KerasMXMGlobalMP
from kgcnn_torch.models.mxmnet import MXMNetGlobalMP


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    units: int = 16
    seed: int = 42


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def copy_residual(t_res, k_res):
    copy_dense_torch_to_keras(t_res.dense_1, k_res.dense_1)
    copy_dense_torch_to_keras(t_res.dense_2, k_res.dense_2)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


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
        worst_mae = max(worst_mae, mae)
        worst_abs = max(worst_abs, max_abs)
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    x = torch.randn(cfg.n_nodes, cfg.units)
    e = torch.randn(cfg.n_edges, cfg.units)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    e_t = torch.stack([src, dst], dim=0)
    e_k = torch.stack([dst, src], dim=0)

    t_layer = MXMNetGlobalMP(units=cfg.units, pooling_method="mean")
    k_layer = KerasMXMGlobalMP(units=cfg.units, pooling_method="mean")

    _ = k_layer([
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(e.numpy()),
        ops.convert_to_tensor(e_k.numpy()),
    ])

    copy_dense_torch_to_keras(t_layer.h_mlp[0], k_layer.h_mlp.mlp_dense_layer_list[0])
    copy_dense_torch_to_keras(t_layer.mlp[0], k_layer.mlp.mlp_dense_layer_list[0])
    copy_dense_torch_to_keras(t_layer.x_edge_mlp[0], k_layer.x_edge_mlp.mlp_dense_layer_list[0])
    copy_dense_torch_to_keras(t_layer.linear, k_layer.linear)
    copy_residual(t_layer.res1, k_layer.res1)
    copy_residual(t_layer.res2, k_layer.res2)
    copy_residual(t_layer.res3, k_layer.res3)

    t_out = t_layer(x, e, e_t).detach().cpu()
    k_out = k_layer([
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(e.numpy()),
        ops.convert_to_tensor(e_k.numpy()),
    ])

    compare_stage_dicts({"global_mp": torch.as_tensor(ops.convert_to_numpy(k_out))}, {"global_mp": t_out})


if __name__ == "__main__":
    main()
