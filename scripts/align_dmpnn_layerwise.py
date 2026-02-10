#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras DMPNN core vs kgcnn-torch DMPNN."""
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
from kgcnn.literature.DMPNN._layers import DMPNNPPoolingEdgesDirected as KerasDirectedPool
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_edges_pairs
from kgcnn_torch.models.dmpnn import DMPNNModel


@dataclass
class Config:
    n_nodes: int = 12
    n_pairs: int = 20
    node_dim: int = 16
    edge_dim: int = 14
    units: int = 16
    depth: int = 3
    seed: int = 42


class KerasDMPNNCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.gather = KerasGatherNodesOutgoing()
        self.concat = keras.layers.Concatenate(axis=-1)
        self.add = keras.layers.Add()
        self.activation = keras.layers.Activation("relu")
        self.dense_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.dense_step = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.directed_pool = KerasDirectedPool()
        self.aggr = KerasAggregateLocalEdges(pooling_method="scatter_sum")
        self.dense_node = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)

    def forward(self, n, ed, edi, ed_pairs) -> Dict[str, torch.Tensor]:
        out = {}
        n_j = self.gather([n, edi])
        h0 = self.concat([n_j, ed])
        h0 = self.dense_init(h0)
        h0 = self.activation(h0)
        out["h0"] = torch.as_tensor(ops.convert_to_numpy(h0))

        h = h0
        for i in range(self.cfg.depth):
            mvw = self.directed_pool([n, h, edi, ed_pairs])
            h = self.dense_step(mvw)
            h = self.add([h, h0])
            h = self.activation(h)
            out[f"edge_step_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))

        mv = self.aggr([n, h, edi])
        mv = self.concat([mv, n])  # Keras order: [mv, n]
        hv = self.dense_node(mv)
        hv = self.activation(hv)
        out["node_readout"] = torch.as_tensor(ops.convert_to_numpy(hv))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_directed_graph(cfg: Config):
    torch.manual_seed(cfg.seed)
    n = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_pairs,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_pairs,), dtype=torch.long)
    src_all = torch.cat([src, dst], dim=0)
    dst_all = torch.cat([dst, src], dim=0)
    edge_index_torch = torch.stack([src_all, dst_all], dim=0)
    edge_index_keras = torch.stack([dst_all, src_all], dim=0)
    m = edge_index_torch.size(1)
    edge_pairs = torch.cat([torch.arange(cfg.n_pairs, m), torch.arange(0, cfg.n_pairs)]).long()
    edge_pairs_keras = edge_pairs.unsqueeze(0)
    ed = torch.randn(m, cfg.edge_dim, dtype=torch.float32)
    return n, ed, edge_index_torch, edge_index_keras, edge_pairs, edge_pairs_keras


def torch_forward_stages(model: DMPNNModel, n, ed, edge_index, edge_pairs) -> Dict[str, torch.Tensor]:
    out = {}
    num_nodes = n.size(0)
    node = model.node_embedding(n.long()) if model.use_node_embedding else n
    edge_aggr = model.aggr

    h0 = model.activation(model.message_init(torch.cat([gather_nodes_outgoing(node, edge_index), ed], dim=-1)))
    out["h0"] = h0.detach().cpu()

    h = h0
    for i in range(model.depth):
        a = edge_aggr(h, edge_index, num_nodes)
        a_source = gather_nodes_outgoing(a, edge_index)
        m_rev = gather_edges_pairs(h, edge_pairs)
        a_corr = a_source - m_rev
        h = model.W_h(a_corr)
        h = model.activation(h + h0)
        out[f"edge_step_{i+1}"] = h.detach().cpu()

    m_agg = edge_aggr(h, edge_index, num_nodes)
    h_v = model.activation(model.node_readout(torch.cat([m_agg, node], dim=-1)))
    out["node_readout"] = h_v.detach().cpu()
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
    n, ed, edge_index_torch, edge_index_keras, edge_pairs, edge_pairs_keras = build_directed_graph(cfg)

    torch_model = DMPNNModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        depth=cfg.depth,
        units=cfg.units,
        message_activation="relu",
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
        dropout_rate=0.0,
    )
    keras_core = KerasDMPNNCore(cfg)

    _ = keras_core.forward(
        ops.convert_to_tensor(n.numpy()),
        ops.convert_to_tensor(ed.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(edge_pairs_keras.numpy()),
    )

    copy_dense_torch_to_keras(torch_model.message_init, keras_core.dense_init)
    copy_dense_torch_to_keras(torch_model.W_h, keras_core.dense_step)
    copy_dense_torch_to_keras(torch_model.node_readout, keras_core.dense_node)

    torch_stages = torch_forward_stages(torch_model, n, ed, edge_index_torch, edge_pairs)
    keras_stages = keras_core.forward(
        ops.convert_to_tensor(n.numpy()),
        ops.convert_to_tensor(ed.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(edge_pairs_keras.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
