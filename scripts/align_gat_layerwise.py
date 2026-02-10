#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras GAT vs kgcnn-torch GAT.

This script compares intermediate activations (dense_in and each GAT layer)
between:
  - gcnn_keras-master/kgcnn/layers AttentionHeadGAT stack
  - kgcnn-torch/kgcnn_torch/models/GATModel

Weights are copied from torch -> keras for matched sublayers to isolate
implementation differences.
"""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

from alignment_thresholds import get_thresholds

# Must be set before importing keras.
os.environ.setdefault("KERAS_BACKEND", "torch")
import keras
from keras import ops


ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.layers.attention import AttentionHeadGAT as KerasAttentionHeadGAT
from kgcnn_torch.models.gat import GATModel
from kgcnn_torch.ops.activ import get_activation


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    edge_dim: int = 8
    depth: int = 3
    attention_units: int = 8
    heads: int = 3
    concat: bool = False
    attention_activation: str = "leaky_relu2"
    pooling: str = "mean"
    seed: int = 42


class KerasGATStack:
    """Minimal eager Keras GAT stack exposing layerwise outputs."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_in = keras.layers.Dense(cfg.attention_units, activation="linear")
        self.heads: List[List[KerasAttentionHeadGAT]] = []
        for _ in range(cfg.depth):
            layer_heads = []
            for _ in range(cfg.heads):
                layer_heads.append(
                    KerasAttentionHeadGAT(
                        units=cfg.attention_units,
                        use_edge_features=True,
                        use_final_activation=False,
                        activation={"class_name": "function", "config": "kgcnn>leaky_relu2"},
                        use_bias=True,
                    )
                )
            self.heads.append(layer_heads)
        self.avg = keras.layers.Average()
        self.concat = keras.layers.Concatenate(axis=-1)
        self.after_avg_act = keras.layers.Activation(
            activation={"class_name": "function", "config": "kgcnn>leaky_relu2"}
        )

    def forward(self, x, edge_attr, edge_index) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        nk = self.dense_in(x)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(nk))
        for layer_id, heads in enumerate(self.heads):
            h = [head([nk, edge_attr, edge_index]) for head in heads]
            if self.cfg.concat:
                nk = self.concat(h)
            else:
                nk = self.avg(h)
                nk = self.after_avg_act(nk)
            out[f"layer_{layer_id+1}"] = torch.as_tensor(ops.convert_to_numpy(nk))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_random_graph(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    edge_attr = torch.randn(cfg.n_edges, cfg.edge_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    # Keras convention in kgcnn: edge_index[0]=target(receive), edge_index[1]=source(send).
    edge_index_keras = torch.stack([dst, src], dim=0)
    return x, edge_attr, edge_index_torch, edge_index_keras


def torch_forward_stages(model: GATModel, x, edge_attr, edge_index) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    h = model.dense_in(x)
    out["dense_in"] = h.detach().cpu()
    for layer_id, heads in enumerate(model.attention_layers):
        head_outs = [head(h, edge_index, edge_attr) for head in heads]
        if model.attention_heads_concat:
            h = torch.cat(head_outs, dim=-1)
        else:
            h = torch.stack(head_outs, dim=0).mean(dim=0)
            if model.activation_after_average is not None:
                h = model.activation_after_average(h)
        out[f"layer_{layer_id+1}"] = h.detach().cpu()
    return out


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def compare_stage_dicts(ref: Dict[str, torch.Tensor], got: Dict[str, torch.Tensor]):
    keys = list(ref.keys())
    print("Stage alignment report (Keras vs Torch):")
    worst_mae = 0.0
    worst_abs = 0.0
    for key in keys:
        r = ref[key]
        g = got[key]
        diff = (r - g).abs()
        mae = float(diff.mean().item())
        max_abs = float(diff.max().item())
        worst_mae = max(worst_mae, mae)
        worst_abs = max(worst_abs, max_abs)
        rmse = float(torch.sqrt(((r - g) ** 2).mean()).item())
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    x, edge_attr, edge_index_torch, edge_index_keras = build_random_graph(cfg)

    torch_model = GATModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        attention_units=cfg.attention_units,
        attention_heads_num=cfg.heads,
        attention_heads_concat=cfg.concat,
        attention_activation=cfg.attention_activation,
        use_edge_features=True,
        edge_dim=cfg.edge_dim,
        output_units=[],
        output_activation="linear",
        output_final_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_stack = KerasGATStack(cfg)

    # Build keras weights by one dry run.
    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    # Copy matched weights torch -> keras.
    copy_dense_torch_to_keras(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth):
        for j in range(cfg.heads):
            t_head = torch_model.attention_layers[i][j]
            k_head = keras_stack.heads[i][j]
            copy_dense_torch_to_keras(t_head.linear_trafo, k_head.lay_linear_trafo)
            copy_dense_torch_to_keras(t_head.linear_alpha, k_head.lay_alpha)

    torch_stages = torch_forward_stages(torch_model, x, edge_attr, edge_index_torch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
