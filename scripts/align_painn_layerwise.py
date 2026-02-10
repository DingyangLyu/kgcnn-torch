#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras PAiNN blocks vs kgcnn-torch PAiNN."""
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

from kgcnn.literature.PAiNN._layers import PAiNNconv as KerasPAiNNconv
from kgcnn.literature.PAiNN._layers import PAiNNUpdate as KerasPAiNNUpdate
from kgcnn_torch.models.painn import PAiNNModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    units: int = 16
    num_radial: int = 8
    depth: int = 2
    seed: int = 42


class KerasPAiNNStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.convs = [
            KerasPAiNNconv(
                units=cfg.units,
                conv_pool="scatter_sum",
                activation="swish",
                cutoff=5.0,
                use_bias=True,
            )
            for _ in range(cfg.depth)
        ]
        self.updates = [
            KerasPAiNNUpdate(units=cfg.units, activation="swish", add_eps=False, use_bias=True)
            for _ in range(cfg.depth)
        ]

    def forward(self, z, v, rbf, env, rij, edge_index) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        zk = z
        vk = v
        out["z_0"] = torch.as_tensor(ops.convert_to_numpy(zk))
        out["v_0"] = torch.as_tensor(ops.convert_to_numpy(vk))
        for i in range(self.cfg.depth):
            ds, dv = self.convs[i]([zk, vk, rbf, env, rij, edge_index])
            zk = zk + ds
            vk = vk + dv
            ds2, dv2 = self.updates[i]([zk, vk])
            zk = zk + ds2
            vk = vk + dv2
            out[f"z_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(zk))
            out[f"v_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(vk))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_random_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    z = torch.randn(cfg.n_nodes, cfg.units, dtype=torch.float32)
    v = torch.randn(cfg.n_nodes, 3, cfg.units, dtype=torch.float32)
    rbf = torch.randn(cfg.n_edges, cfg.num_radial, dtype=torch.float32)
    env = torch.rand(cfg.n_edges, 1, dtype=torch.float32)
    rij = torch.randn(cfg.n_edges, 3, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    return z, v, rbf, env, rij, edge_index_torch, edge_index_keras


def torch_forward_stages(model: PAiNNModel, z, v, rbf, env, rij, edge_index):
    out: Dict[str, torch.Tensor] = {}
    zk = z
    vk = v
    out["z_0"] = zk.detach().cpu()
    out["v_0"] = vk.detach().cpu()
    for i in range(model.depth):
        ds, dv = model.convs[i](zk, vk, rbf, env, rij, edge_index)
        zk = zk + ds
        vk = vk + dv
        ds2, dv2 = model.updates[i](zk, vk)
        zk = zk + ds2
        vk = vk + dv2
        out[f"z_{i+1}"] = zk.detach().cpu()
        out[f"v_{i+1}"] = vk.detach().cpu()
    return out


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
        print(f"- {key:6s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    z, v, rbf, env, rij, edge_index_torch, edge_index_keras = build_random_inputs(cfg)

    torch_model = PAiNNModel(
        node_dim=cfg.units,
        depth=cfg.depth,
        units=cfg.units,
        num_radial=cfg.num_radial,
        cutoff=5.0,
        conv_cutoff=5.0,
        conv_activation="swish",
        update_activation="swish",
        update_add_eps=False,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_stack = KerasPAiNNStack(cfg)

    _ = keras_stack.forward(
        ops.convert_to_tensor(z.numpy()),
        ops.convert_to_tensor(v.numpy()),
        ops.convert_to_tensor(rbf.numpy()),
        ops.convert_to_tensor(env.numpy()),
        ops.convert_to_tensor(rij.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    for i in range(cfg.depth):
        tc = torch_model.convs[i]
        kc = keras_stack.convs[i]
        copy_dense_torch_to_keras(tc.dense1, kc.lay_dense1)
        copy_dense_torch_to_keras(tc.phi, kc.lay_phi)
        copy_dense_torch_to_keras(tc.w, kc.lay_w)

        tu = torch_model.updates[i]
        ku = keras_stack.updates[i]
        copy_dense_torch_to_keras(tu.dense1, ku.lay_dense1)
        copy_dense_torch_to_keras(tu.lin_u, ku.lay_lin_u)
        copy_dense_torch_to_keras(tu.lin_v, ku.lay_lin_v)
        copy_dense_torch_to_keras(tu.dense_a, ku.lay_a)

    torch_stages = torch_forward_stages(torch_model, z, v, rbf, env, rij, edge_index_torch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(z.numpy()),
        ops.convert_to_tensor(v.numpy()),
        ops.convert_to_tensor(rbf.numpy()),
        ops.convert_to_tensor(env.numpy()),
        ops.convert_to_tensor(rij.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
