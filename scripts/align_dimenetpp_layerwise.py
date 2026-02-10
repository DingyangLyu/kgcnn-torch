#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras DimeNetPP interaction block vs torch."""
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

from kgcnn.literature.DimeNetPP._layers import DimNetInteractionPPBlock as KerasDimNetInteractionPPBlock
from kgcnn_torch.models.dimenetpp import DimNetInteractionPPBlock


@dataclass
class Config:
    n_edges: int = 36
    n_triplets: int = 72
    emb_size: int = 16
    int_emb_size: int = 8
    basis_emb_size: int = 6
    num_radial: int = 6
    num_spherical: int = 3
    seed: int = 42


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_edges, cfg.emb_size, dtype=torch.float32)
    rbf = torch.randn(cfg.n_edges, cfg.num_radial, dtype=torch.float32)
    sbf = torch.randn(cfg.n_triplets, cfg.num_spherical * cfg.num_radial, dtype=torch.float32)
    target_edges = torch.randint(0, cfg.n_edges, (cfg.n_triplets,), dtype=torch.long)
    source_edges = torch.randint(0, cfg.n_edges, (cfg.n_triplets,), dtype=torch.long)
    angle_index = torch.stack([target_edges, source_edges], dim=0)  # [ji, kj]
    return x, rbf, sbf, angle_index


def torch_forward(block: DimNetInteractionPPBlock, x, rbf, sbf, angle_index) -> Dict[str, torch.Tensor]:
    return {"interaction": block(x, rbf, sbf, angle_index).detach().cpu()}


def keras_forward(block: KerasDimNetInteractionPPBlock, x, rbf, sbf, angle_index) -> Dict[str, torch.Tensor]:
    out = block([
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(rbf.numpy()),
        ops.convert_to_tensor(sbf.numpy()),
        ops.convert_to_tensor(angle_index.numpy()),
    ])
    return {"interaction": torch.as_tensor(ops.convert_to_numpy(out))}


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
        print(f"- {key:12s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    x, rbf, sbf, angle_index = build_inputs(cfg)

    torch_block = DimNetInteractionPPBlock(
        emb_size=cfg.emb_size,
        int_emb_size=cfg.int_emb_size,
        basis_emb_size=cfg.basis_emb_size,
        num_before_skip=1,
        num_after_skip=2,
        num_radial=cfg.num_radial,
        num_spherical=cfg.num_spherical,
        activation="swish",
        pooling_method="sum",
    )
    keras_block = KerasDimNetInteractionPPBlock(
        emb_size=cfg.emb_size,
        int_emb_size=cfg.int_emb_size,
        basis_emb_size=cfg.basis_emb_size,
        num_before_skip=1,
        num_after_skip=2,
        activation="swish",
        pooling_method="sum",
        kernel_initializer="kgcnn>glorot_orthogonal",
    )

    _ = keras_forward(keras_block, x, rbf, sbf, angle_index)

    # Basis transforms
    copy_dense_torch_to_keras(torch_block.dense_rbf1, keras_block.dense_rbf1)
    copy_dense_torch_to_keras(torch_block.dense_rbf2, keras_block.dense_rbf2)
    copy_dense_torch_to_keras(torch_block.dense_sbf1, keras_block.dense_sbf1)
    copy_dense_torch_to_keras(torch_block.dense_sbf2, keras_block.dense_sbf2)
    # Main transforms
    copy_dense_torch_to_keras(torch_block.dense_ji[0], keras_block.dense_ji)
    copy_dense_torch_to_keras(torch_block.dense_kj[0], keras_block.dense_kj)
    copy_dense_torch_to_keras(torch_block.down_projection[0], keras_block.down_projection)
    copy_dense_torch_to_keras(torch_block.up_projection[0], keras_block.up_projection)
    copy_dense_torch_to_keras(torch_block.final_before_skip[0], keras_block.final_before_skip)
    # Residual stacks
    for i in range(1):
        copy_dense_torch_to_keras(torch_block.layers_before_skip[i].dense_1, keras_block.layers_before_skip[i].dense_1)
        copy_dense_torch_to_keras(torch_block.layers_before_skip[i].dense_2, keras_block.layers_before_skip[i].dense_2)
    for i in range(2):
        copy_dense_torch_to_keras(torch_block.layers_after_skip[i].dense_1, keras_block.layers_after_skip[i].dense_1)
        copy_dense_torch_to_keras(torch_block.layers_after_skip[i].dense_2, keras_block.layers_after_skip[i].dense_2)

    torch_out = torch_forward(torch_block, x, rbf, sbf, angle_index)
    keras_out = keras_forward(keras_block, x, rbf, sbf, angle_index)
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
