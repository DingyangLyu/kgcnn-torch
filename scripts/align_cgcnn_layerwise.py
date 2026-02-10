#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras CGCNN layer vs kgcnn-torch CGCNN."""
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

from kgcnn.literature.CGCNN._layers import CGCNNLayer as KerasCGCNNLayer
from kgcnn_torch.models.cgcnn import CGCNNModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 40
    node_dim: int = 16
    edge_dim: int = 20
    depth: int = 3
    seed: int = 42


class KerasCGCNNStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_in = keras.layers.Dense(cfg.node_dim, activation="linear", use_bias=True)
        self.layers: List[KerasCGCNNLayer] = [
            KerasCGCNNLayer(
                units=cfg.node_dim,
                activation_s="softplus",
                activation_out="softplus",
                batch_normalization=False,
                pooling_method="scatter_mean",
                use_bias=True,
            )
            for _ in range(cfg.depth)
        ]

    def forward(self, x, edge_attr, edge_index, batch, batch_edge, count_nodes, count_edges) -> Dict[str, torch.Tensor]:
        out = {}
        h = self.dense_in(x)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(h))
        for i, layer in enumerate(self.layers):
            h = layer([h, edge_attr, edge_index, batch, batch_edge, count_nodes, count_edges])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))
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
    edge_index_keras = torch.stack([dst, src], dim=0)
    batch = torch.cat([
        torch.zeros(cfg.n_nodes // 2, dtype=torch.long),
        torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long),
    ], dim=0)
    batch_edge = batch[src]
    count_nodes = torch.bincount(batch, minlength=2)
    count_edges = torch.bincount(batch_edge, minlength=2)
    return x, edge_attr, edge_index_torch, edge_index_keras, batch, batch_edge, count_nodes, count_edges


def torch_forward_stages(model: CGCNNModel, x, edge_attr, edge_index) -> Dict[str, torch.Tensor]:
    out = {}
    h = model.dense_in(x)
    out["dense_in"] = h.detach().cpu()
    for i, layer in enumerate(model.convs):
        h = layer(h, edge_attr, edge_index)
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
    x, edge_attr, edge_index_torch, edge_index_keras, batch, batch_edge, count_nodes, count_edges = build_random_graph(cfg)

    torch_model = CGCNNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        gauss_bins=cfg.edge_dim,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
        expand_distance=False,
        make_distance=False,
        edge_input_dim=cfg.edge_dim,
    )
    # Disable batchnorm in torch to match Keras stack here.
    for layer in torch_model.convs:
        layer.batch_normalization = False

    keras_stack = KerasCGCNNStack(cfg)

    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(batch_edge.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(count_edges.numpy()),
    )

    copy_dense_torch_to_keras(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth):
        t = torch_model.convs[i]
        k = keras_stack.layers[i]
        # Keras naming uses f->sigmoid gate and s->softplus filter.
        copy_dense_torch_to_keras(t.linear_gate, k.f)
        copy_dense_torch_to_keras(t.linear_filter, k.s)

    torch_stages = torch_forward_stages(torch_model, x, edge_attr, edge_index_torch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(batch_edge.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(count_edges.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
