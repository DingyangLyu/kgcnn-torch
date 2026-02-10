#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras GIN stack vs kgcnn-torch GIN stack."""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.layers.conv import GIN as KerasGIN
from kgcnn.layers.mlp import GraphMLP as KerasGraphMLP
from kgcnn_torch.models.gin import GINModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    depth: int = 3
    units: int = 16
    seed: int = 42


class KerasGINStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.convs: List[KerasGIN] = [
            KerasGIN(pooling_method="scatter_sum", epsilon_learnable=False) for _ in range(cfg.depth)
        ]
        self.mlps: List[KerasGraphMLP] = [
            KerasGraphMLP(
                units=[cfg.units, cfg.units],
                activation=["relu", "linear"],
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            )
            for _ in range(cfg.depth)
        ]

    def forward(self, x, edge_index, batch, count_nodes) -> Dict[str, torch.Tensor]:
        out = {}
        h = self.dense_in(x)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(h))
        for i in range(self.cfg.depth):
            h = self.convs[i]([h, edge_index])
            h = self.mlps[i]([h, batch, count_nodes])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    bias = torch_linear.bias.detach().cpu().numpy() if torch_linear.bias is not None else None
    keras_dense.set_weights([kernel, bias] if bias is not None else [kernel])


def build_random_graph(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    batch = torch.cat([torch.zeros(cfg.n_nodes // 2, dtype=torch.long),
                       torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long)])
    count_nodes = torch.tensor(cfg.n_nodes, dtype=torch.long)
    return x, edge_index_torch, edge_index_keras, batch, count_nodes


def torch_forward_stages(model: GINModel, x, edge_index, batch) -> Dict[str, torch.Tensor]:
    out = {}
    h = model.dense_in(x)
    out["dense_in"] = h.detach().cpu()
    for i in range(model.depth):
        h = model.convs[i](h, edge_index)
        h = model.gin_mlps[i](h)
        out[f"layer_{i+1}"] = h.detach().cpu()
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
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    x, edge_index_torch, edge_index_keras, batch, count_nodes = build_random_graph(cfg)
    torch_model = GINModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        gin_mlp_units=[cfg.units, cfg.units],
        gin_mlp_activation="relu",
        gin_mlp_use_normalization=False,
        output_units=[],
        output_activation="linear",
        output_final_activation="linear",
        output_embedding="node",
        num_targets=1,
        use_node_embedding=False,
    )
    keras_stack = KerasGINStack(cfg)
    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
    )

    copy_dense_torch_to_keras(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth):
        # epsilon scalar
        keras_stack.convs[i].set_weights([torch_model.convs[i].eps.detach().cpu().numpy().reshape(())])
        for j in range(len(torch_model.gin_mlps[i].linears)):
            copy_dense_torch_to_keras(
                torch_model.gin_mlps[i].linears[j],
                keras_stack.mlps[i].mlp_dense_layer_list[j],
            )

    torch_stages = torch_forward_stages(torch_model, x, edge_index_torch, batch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
