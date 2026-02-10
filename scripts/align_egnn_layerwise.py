#!/usr/bin/env python3
"""Layerwise numeric alignment check: EGNN single layer Keras-emulated vs torch."""
import os
import sys
from dataclasses import dataclass
from typing import Dict

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
from kgcnn.layers.gather import GatherNodesIngoing as KerasGatherNodesIngoing
from kgcnn_torch.models.egnn import EGNNLayer


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    units: int = 16
    seed: int = 42


class KerasEGNNLayerEmu:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gather_in = KerasGatherNodesIngoing()
        self.gather_out = KerasGatherNodesOutgoing()
        self.aggr_msg = KerasAggregateLocalEdges(pooling_method="scatter_sum")
        self.aggr_coord = KerasAggregateLocalEdges(pooling_method="scatter_mean")

        self.edge_dense_1 = keras.layers.Dense(cfg.units, activation="swish", use_bias=True)
        self.edge_dense_2 = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)

        self.coord_dense_1 = keras.layers.Dense(cfg.units, activation="swish", use_bias=True)
        self.coord_dense_2 = keras.layers.Dense(1, activation="linear", use_bias=True)

        self.node_dense_1 = keras.layers.Dense(cfg.units, activation="swish", use_bias=True)
        self.node_dense_2 = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)

    def forward(self, h, pos, edge_index) -> Dict[str, torch.Tensor]:
        h_i = self.gather_in([h, edge_index])
        h_j = self.gather_out([h, edge_index])

        pos_i = ops.take(pos, ops.take(edge_index, 0, axis=0), axis=0)
        pos_j = ops.take(pos, ops.take(edge_index, 1, axis=0), axis=0)
        diff_x = pos_i - pos_j
        norm_x = ops.sqrt(ops.sum(diff_x * diff_x, axis=-1, keepdims=True) + 1e-8)

        edge_input = ops.concatenate([h_i, h_j, norm_x], axis=-1)
        m_ij = self.edge_dense_2(self.edge_dense_1(edge_input))

        coord_weight = self.coord_dense_2(self.coord_dense_1(m_ij))
        coord_msg = diff_x * coord_weight
        coord_agg = self.aggr_coord([h, coord_msg, edge_index])
        pos_updated = pos + coord_agg

        m_agg = self.aggr_msg([h, m_ij, edge_index])
        node_input = ops.concatenate([h, m_agg], axis=-1)
        h_updated = h + self.node_dense_2(self.node_dense_1(node_input))
        return {
            "h": torch.as_tensor(ops.convert_to_numpy(h_updated)),
            "pos": torch.as_tensor(ops.convert_to_numpy(pos_updated)),
        }


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    h = torch.randn(cfg.n_nodes, cfg.units, dtype=torch.float32)
    pos = torch.randn(cfg.n_nodes, 3, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    return h, pos, edge_index_torch, edge_index_keras


def torch_forward(layer: EGNNLayer, h, pos, edge_index):
    h_new, pos_new = layer(h, pos, edge_index, edge_attr=None)
    return {"h": h_new.detach().cpu(), "pos": pos_new.detach().cpu()}


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
        print(f"- {key:8s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    h, pos, edge_index_torch, edge_index_keras = build_inputs(cfg)

    torch_layer = EGNNLayer(
        units=cfg.units,
        edge_mlp_units=[cfg.units, cfg.units],
        edge_mlp_activation="swish",
        coord_mlp_units=[cfg.units, 1],
        coord_mlp_activation="swish",
        node_mlp_units=[cfg.units, cfg.units],
        node_mlp_activation="swish",
        use_edge_attr=False,
        use_attention=False,
        use_normalize=False,
        pooling_method="sum",
        coord_pooling_method="mean",
    )
    keras_layer = KerasEGNNLayerEmu(cfg)

    _ = keras_layer.forward(
        ops.convert_to_tensor(h.numpy()),
        ops.convert_to_tensor(pos.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    copy_dense_torch_to_keras(torch_layer.edge_mlp.linears[0], keras_layer.edge_dense_1)
    copy_dense_torch_to_keras(torch_layer.edge_mlp.linears[1], keras_layer.edge_dense_2)
    copy_dense_torch_to_keras(torch_layer.coord_mlp.linears[0], keras_layer.coord_dense_1)
    copy_dense_torch_to_keras(torch_layer.coord_mlp.linears[1], keras_layer.coord_dense_2)
    copy_dense_torch_to_keras(torch_layer.node_mlp.linears[0], keras_layer.node_dense_1)
    copy_dense_torch_to_keras(torch_layer.node_mlp.linears[1], keras_layer.node_dense_2)

    torch_out = torch_forward(torch_layer, h, pos, edge_index_torch)
    keras_out = keras_layer.forward(
        ops.convert_to_tensor(h.numpy()),
        ops.convert_to_tensor(pos.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
