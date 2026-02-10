#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras RGCN block vs kgcnn-torch RGCN."""
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

from kgcnn.layers.aggr import AggregateLocalEdges as KerasAggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing as KerasGatherNodesOutgoing
from kgcnn.layers.relational import RelationalDense as KerasRelationalDense
from kgcnn_torch.models.rgcn import RGCNModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 40
    node_dim: int = 16
    units: int = 16
    depth: int = 3
    num_relations: int = 5
    seed: int = 42


class KerasRGCNStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_self: List[keras.layers.Dense] = [
            keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
            for _ in range(cfg.depth)
        ]
        self.rel_dense: List[KerasRelationalDense] = [
            KerasRelationalDense(
                units=cfg.units,
                num_relations=cfg.num_relations,
                activation="linear",
                use_bias=False,
            )
            for _ in range(cfg.depth)
        ]
        self.gather = KerasGatherNodesOutgoing()
        self.mult = keras.layers.Multiply()
        self.aggr = KerasAggregateLocalEdges(pooling_method="scatter_sum")
        self.act = keras.layers.Activation("swish")
        self.add = keras.layers.Add()

    def forward(self, x, edge_w, edge_type, edge_index) -> Dict[str, torch.Tensor]:
        out = {}
        h = x
        for i in range(self.cfg.depth):
            n_j = self.gather([h, edge_index])
            h0 = self.dense_self[i](h)
            h_j = self.rel_dense[i]([n_j, edge_type])
            msg = self.mult([h_j, edge_w])
            agg = self.aggr([h, msg, edge_index])
            h = self.add([agg, h0])
            h = self.act(h)
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
    edge_w = torch.randn(cfg.n_edges, 1, dtype=torch.float32)
    edge_type = torch.randint(0, cfg.num_relations, (cfg.n_edges,), dtype=torch.long)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    return x, edge_w, edge_type, edge_index_torch, edge_index_keras


def torch_forward_stages(model: RGCNModel, x, edge_w, edge_type, edge_index) -> Dict[str, torch.Tensor]:
    out = {}
    h = x
    for i, conv in enumerate(model.convs):
        h = conv(h, edge_index, edge_type, edge_w)
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
    x, edge_w, edge_type, edge_index_torch, edge_index_keras = build_random_graph(cfg)

    torch_model = RGCNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        num_relations=cfg.num_relations,
        rgcn_activation="swish",
        rgcn_pooling="sum",
        use_residual=False,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_stack = KerasRGCNStack(cfg)

    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_w.numpy()),
        ops.convert_to_tensor(edge_type.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    for i in range(cfg.depth):
        t = torch_model.convs[i]
        copy_dense_torch_to_keras(t.self_loop, keras_stack.dense_self[i])
        rel_weights = t.weight.detach().cpu().numpy()
        keras_stack.rel_dense[i].set_weights([rel_weights])

    torch_stages = torch_forward_stages(torch_model, x, edge_w, edge_type, edge_index_torch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_w.numpy()),
        ops.convert_to_tensor(edge_type.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
