#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras MoGAT atom stack vs kgcnn-torch."""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
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

from kgcnn.layers.update import GRUUpdate as KerasGRUUpdate
from kgcnn.literature.MoGAT._layers import AttentiveHeadFP_ as KerasAttentiveHeadFP_
from kgcnn_torch.models.mogat import MoGATModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    edge_dim: int = 8
    units: int = 16
    depth_ato: int = 3
    seed: int = 42


class KerasMoGATAtomStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.heads: List[KerasAttentiveHeadFP_] = []
        self.grus: List[KerasGRUUpdate] = []
        for i in range(cfg.depth_ato):
            self.heads.append(
                KerasAttentiveHeadFP_(
                    units=cfg.units,
                    use_edge_features=(i == 0),
                    activation={"class_name": "function", "config": "kgcnn>leaky_relu2"},
                    activation_context="elu",
                    use_bias=True,
                )
            )
            self.grus.append(KerasGRUUpdate(units=cfg.units))

    def forward(self, x, edge_attr, edge_index) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        h = self.dense_in(x)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(h))
        for i in range(self.cfg.depth_ato):
            c = self.heads[i]([h, edge_attr, edge_index])
            h = self.grus[i]([h, c])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def copy_gru_torch_to_keras(torch_gru_cell: torch.nn.GRUCell, keras_gru_update: KerasGRUUpdate):
    """Copy GRUCell weights PyTorch(r,z,n) -> Keras(z,r,h)."""
    w_ih = torch_gru_cell.weight_ih.detach().cpu().numpy()
    w_hh = torch_gru_cell.weight_hh.detach().cpu().numpy()
    b_ih = torch_gru_cell.bias_ih.detach().cpu().numpy()
    b_hh = torch_gru_cell.bias_hh.detach().cpu().numpy()

    def _reorder_chunks(arr, axis=0):
        r, z, n = np.split(arr, 3, axis=axis)
        return np.concatenate([z, r, n], axis=axis)

    kernel = _reorder_chunks(w_ih, axis=0).T
    recurrent_kernel = _reorder_chunks(w_hh, axis=0).T
    bias_input = _reorder_chunks(b_ih, axis=0)
    bias_recurrent = _reorder_chunks(b_hh, axis=0)
    bias = np.stack([bias_input, bias_recurrent], axis=0)

    keras_gru_update.gru_cell.set_weights([kernel, recurrent_kernel, bias])


def build_random_graph(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    edge_attr = torch.randn(cfg.n_edges, cfg.edge_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    return x, edge_attr, edge_index_torch, edge_index_keras


def torch_forward_stages(model: MoGATModel, x, edge_attr, edge_index) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    h = model.dense_in(x)
    out["dense_in"] = h.detach().cpu()
    for i in range(model.depthato):
        c = model.attention_layers[i](h, edge_index, edge_attr)
        h = model.gru_layers[i](c, h)
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
    x, edge_attr, edge_index_torch, edge_index_keras = build_random_graph(cfg)

    torch_model = MoGATModel(
        node_dim=cfg.node_dim,
        depthato=cfg.depth_ato,
        depthmol=2,
        units=cfg.units,
        edge_dim=cfg.edge_dim,
        use_edge_features=True,
        activation="leaky_relu2",
        dropout=0.0,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_stack = KerasMoGATAtomStack(cfg)

    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    copy_dense_torch_to_keras(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth_ato):
        t_head = torch_model.attention_layers[i]
        k_head = keras_stack.heads[i]
        copy_dense_torch_to_keras(t_head.linear_trafo, k_head.lay_linear_trafo)
        copy_dense_torch_to_keras(t_head.alpha_activation[0], k_head.lay_alpha_activation)
        copy_dense_torch_to_keras(t_head.linear_alpha, k_head.lay_alpha)
        if i == 0:
            copy_dense_torch_to_keras(t_head.fc1[0], k_head.lay_fc1)
            copy_dense_torch_to_keras(t_head.fc2[0], k_head.lay_fc2)
        copy_gru_torch_to_keras(torch_model.gru_layers[i].gru_cell, keras_stack.grus[i])

    torch_stages = torch_forward_stages(torch_model, x, edge_attr, edge_index_torch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
