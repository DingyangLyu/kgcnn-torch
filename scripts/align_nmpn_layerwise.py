#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras NMPN core vs kgcnn-torch NMPN."""
import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops
import keras

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.layers.aggr import AggregateLocalEdges as KerasAggregateLocalEdges
from kgcnn.layers.gather import GatherNodesOutgoing as KerasGatherNodesOutgoing
from kgcnn.layers.gather import GatherNodesIngoing as KerasGatherNodesIngoing
from kgcnn.layers.message import MatMulMessages as KerasMatMulMessages
from kgcnn.layers.update import GRUUpdate as KerasGRUUpdate
from kgcnn.literature.NMPN._layers import TrafoEdgeNetMessages as KerasTrafoEdgeNetMessages
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
from kgcnn_torch.layers.message import MatMulMessages
from kgcnn_torch.models.nmpn import NMPNModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    units: int = 16
    edge_dim: int = 10
    edge_hidden: int = 20
    depth: int = 3
    seed: int = 42


class KerasNMPNCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.edge_dense_in = keras.layers.Dense(cfg.edge_hidden, activation="swish", use_bias=True)
        self.edge_dense_out = keras.layers.Dense(cfg.edge_hidden, activation="swish", use_bias=True)
        self.trafo_in = KerasTrafoEdgeNetMessages(target_shape=(cfg.units, cfg.units))
        self.trafo_out = KerasTrafoEdgeNetMessages(target_shape=(cfg.units, cfg.units))
        self.gather_out = KerasGatherNodesOutgoing()
        self.gather_in = KerasGatherNodesIngoing()
        self.matmul = KerasMatMulMessages()
        self.aggr = KerasAggregateLocalEdges(pooling_method="scatter_sum")
        self.gru = KerasGRUUpdate(cfg.units)
        self.concat = keras.layers.Concatenate(axis=-1)

    def forward(self, n0, ed, edge_index) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        n = self.dense_in(n0)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(n))

        edge_net_in = self.trafo_in(self.edge_dense_in(ed))
        edge_net_out = self.trafo_out(self.edge_dense_out(ed))
        out["edge_net_in"] = torch.as_tensor(ops.convert_to_numpy(edge_net_in))
        out["edge_net_out"] = torch.as_tensor(ops.convert_to_numpy(edge_net_out))

        for i in range(self.cfg.depth):
            n_in = self.gather_out([n, edge_index])
            n_out = self.gather_in([n, edge_index])
            m_in = self.matmul([edge_net_in, n_in])
            m_out = self.matmul([edge_net_out, n_out])
            eu = self.concat([m_in, m_out])
            eu = self.aggr([n, eu, edge_index])
            n = self.gru([n, eu])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(n))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def copy_gru_torch_to_keras(torch_gru_cell: torch.nn.GRUCell, keras_gru_update: KerasGRUUpdate):
    units = torch_gru_cell.hidden_size
    w_ih = torch_gru_cell.weight_ih.detach().cpu().numpy()
    w_hh = torch_gru_cell.weight_hh.detach().cpu().numpy()
    b_ih = torch_gru_cell.bias_ih.detach().cpu().numpy()
    b_hh = torch_gru_cell.bias_hh.detach().cpu().numpy()

    def _reorder(arr, axis=0):
        r, z, n = np.split(arr, 3, axis=axis)
        return np.concatenate([z, r, n], axis=axis)

    kernel = _reorder(w_ih, axis=0).T
    recurrent_kernel = _reorder(w_hh, axis=0).T
    bias = np.stack([_reorder(b_ih, axis=0), _reorder(b_hh, axis=0)], axis=0)
    keras_gru_update.gru_cell.set_weights([kernel, recurrent_kernel, bias])


def build_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    n0 = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    ed = torch.randn(cfg.n_edges, cfg.edge_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    return n0, ed, edge_index_torch, edge_index_keras


def torch_forward(model: NMPNModel, n0, ed, edge_index) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    n = model.dense_in(n0)
    out["dense_in"] = n.detach().cpu()

    edge_net_in = model.edge_trafo_in(model.edge_mlp_in(ed))
    edge_net_out = model.edge_trafo_out(model.edge_mlp_out(ed))
    out["edge_net_in"] = edge_net_in.detach().cpu()
    out["edge_net_out"] = edge_net_out.detach().cpu()

    matmul = MatMulMessages()
    num_nodes = n.size(0)
    for i in range(model.depth):
        n_in = gather_nodes_outgoing(n, edge_index)
        n_out = gather_nodes_ingoing(n, edge_index)
        m_in = matmul(edge_net_in, n_in)
        m_out = matmul(edge_net_out, n_out)
        eu = torch.cat([m_in, m_out], dim=-1)
        eu = model.aggr(eu, edge_index, num_nodes)
        n = model.gru_update(eu, n)
        out[f"layer_{i+1}"] = n.detach().cpu()
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
    n0, ed, edge_index_torch, edge_index_keras = build_inputs(cfg)

    torch_model = NMPNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        edge_dim=cfg.edge_dim,
        edge_mlp_units=[cfg.edge_hidden],
        edge_mlp_activation="swish",
        message_pooling="sum",
        use_set2set=False,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_core = KerasNMPNCore(cfg)

    _ = keras_core.forward(
        ops.convert_to_tensor(n0.numpy()),
        ops.convert_to_tensor(ed.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    copy_dense_torch_to_keras(torch_model.dense_in, keras_core.dense_in)
    copy_dense_torch_to_keras(torch_model.edge_mlp_in.linears[0], keras_core.edge_dense_in)
    copy_dense_torch_to_keras(torch_model.edge_mlp_out.linears[0], keras_core.edge_dense_out)
    copy_dense_torch_to_keras(torch_model.edge_trafo_in.dense, keras_core.trafo_in.lay_dense)
    copy_dense_torch_to_keras(torch_model.edge_trafo_out.dense, keras_core.trafo_out.lay_dense)
    copy_gru_torch_to_keras(torch_model.gru_update.gru_cell, keras_core.gru)

    torch_out = torch_forward(torch_model, n0, ed, edge_index_torch)
    keras_out = keras_core.forward(
        ops.convert_to_tensor(n0.numpy()),
        ops.convert_to_tensor(ed.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
