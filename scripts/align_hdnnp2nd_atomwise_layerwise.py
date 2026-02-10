#!/usr/bin/env python3
"""Layerwise numeric alignment check: HDNNP2nd AtomWise relational MLP."""
import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.layers.mlp import RelationalMLP as KerasRelationalMLP
from kgcnn_torch.models.hdnnp2nd import HDNNP2ndAtomWiseModel


@dataclass
class Config:
    n_nodes: int = 14
    input_dim: int = 20
    seed: int = 42


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

    element_types = [1, 6, 7, 8]
    n_types = len(element_types)

    z_vals = torch.tensor(element_types, dtype=torch.long)
    z = z_vals[torch.randint(0, n_types, (cfg.n_nodes,), dtype=torch.long)]
    x = torch.randn(cfg.n_nodes, cfg.input_dim)

    # Build manual type_map matching torch model
    z_to_t = {zv: i for i, zv in enumerate(element_types)}
    type_map = torch.tensor([z_to_t[int(v.item())] for v in z], dtype=torch.long)

    t_model = HDNNP2ndAtomWiseModel(
        element_types=element_types,
        input_dim=cfg.input_dim,
        relational_units=[16, 16, 16],
        relational_activation=["swish", "swish", "linear"],
        num_relations=n_types,
        output_embedding="node",
    )

    k_rel = KerasRelationalMLP(
        units=[16, 16, 16],
        activation=["swish", "swish", "linear"],
        num_relations=n_types,
    )

    _ = k_rel([
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(type_map.numpy()),
    ])

    # Copy RelationalDense weights into Keras RelationalDense kernels.
    # Torch RelationalMLP uses .layers (list of RelationalDense), each with
    # .weight of shape (R, in, out) and .bias of shape (out,) -- already shared.
    for layer_idx, k_dense in enumerate(k_rel.mlp_dense_layer_list):
        t_layer = t_model.relational_mlp.layers[layer_idx]
        kernel = t_layer.weight.detach().cpu().numpy()  # (R, in, out)
        bias = t_layer.bias.detach().cpu().numpy() if t_layer.bias is not None else None
        if bias is not None:
            k_dense.set_weights([kernel, bias])
        else:
            k_dense.set_weights([kernel])

    t_node = t_model.relational_mlp(x, type_map).detach().cpu()
    k_node = k_rel([
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(type_map.numpy()),
    ])

    ref = {"node_out": torch.as_tensor(ops.convert_to_numpy(k_node))}
    got = {"node_out": t_node}
    compare_stage_dicts(ref, got)


if __name__ == "__main__":
    main()
