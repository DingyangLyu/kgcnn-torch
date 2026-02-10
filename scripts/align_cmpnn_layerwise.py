#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras CMPNN core vs kgcnn-torch CMPNN."""
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
from kgcnn.layers.gather import GatherEdgesPairs as KerasGatherEdgesPairs
from kgcnn_torch.layers.aggr import AggregateLocalEdges as TorchAggregateLocalEdges
from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_edges_pairs
from kgcnn_torch.models.cmpnn import CMPNNModel


@dataclass
class Config:
    n_nodes: int = 12
    n_pairs: int = 20
    node_dim: int = 16
    edge_dim: int = 14
    units: int = 16
    depth: int = 4
    seed: int = 42


class KerasCMPNNCore:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.act = keras.layers.Activation("relu")
        self.add = keras.layers.Add()
        self.sub = keras.layers.Subtract()
        self.mul = keras.layers.Multiply()
        self.concat = keras.layers.Concatenate(axis=-1)
        self.dense_node_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.dense_edge_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.dense_edge_steps = [
            keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
            for _ in range(max(cfg.depth - 1, 1))
        ]
        self.dense_node_final = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.aggr_sum = KerasAggregateLocalEdges(pooling_method="scatter_sum")
        self.aggr_max = KerasAggregateLocalEdges(pooling_method="scatter_max")
        self.gather_out = KerasGatherNodesOutgoing()
        self.gather_pairs = KerasGatherEdgesPairs()

    def forward(self, n, ed, edi, e_pairs) -> Dict[str, torch.Tensor]:
        out = {}
        h0 = self.act(self.dense_node_init(n))
        he0 = self.act(self.dense_edge_init(ed))
        out["h0"] = torch.as_tensor(ops.convert_to_numpy(h0))
        out["he0"] = torch.as_tensor(ops.convert_to_numpy(he0))

        h = h0
        he = he0
        for i in range(self.cfg.depth - 1):
            m_pool = self.aggr_sum([h, he, edi])
            m_max = self.aggr_max([h, he, edi])
            m = self.mul([m_pool, m_max])
            h = self.add([h, m])

            h_out = self.gather_out([h, edi])
            e_rev = self.gather_pairs([he, e_pairs])
            he = self.sub([h_out, e_rev])
            he = self.dense_edge_steps[i](he)
            he = self.add([he, he0])
            he = self.act(he)
            out[f"edge_step_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(he))
            out[f"node_step_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))

        m_pool = self.aggr_sum([h, he, edi])
        m_max = self.aggr_max([h, he, edi])
        m = self.mul([m_pool, m_max])
        h_final = self.concat([m, h, h0])
        h_final = self.dense_node_final(h_final)
        out["node_final"] = torch.as_tensor(ops.convert_to_numpy(h_final))
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


def torch_forward_stages(model: CMPNNModel, n, ed, edge_index, edge_pairs) -> Dict[str, torch.Tensor]:
    out = {}
    num_nodes = n.size(0)
    h0 = model.node_init_act(model.node_init(n))
    he0 = model.edge_init_act(model.edge_init(ed))
    out["h0"] = h0.detach().cpu()
    out["he0"] = he0.detach().cpu()

    h = h0
    he = he0
    for i in range(model.depth - 1):
        m_pool = model.aggr_sum(he, edge_index, num_nodes)
        m_max = model.aggr_max(he, edge_index, num_nodes)
        m = m_pool * m_max
        h = h + m

        h_out = gather_nodes_outgoing(h, edge_index)
        e_rev = gather_edges_pairs(he, edge_pairs)
        he = h_out - e_rev
        he = model.edge_denses[i](he)
        he = he + he0
        he = model.activation(he)
        out[f"edge_step_{i+1}"] = he.detach().cpu()
        out[f"node_step_{i+1}"] = h.detach().cpu()

    m_pool = model.aggr_sum(he, edge_index, num_nodes)
    m_max = model.aggr_max(he, edge_index, num_nodes)
    m = m_pool * m_max
    h_final = model.node_dense(torch.cat([m, h, h0], dim=-1))
    out["node_final"] = h_final.detach().cpu()
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

    torch_model = CMPNNModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        depth=cfg.depth,
        units=cfg.units,
        activation="relu",
        dropout=0.0,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_core = KerasCMPNNCore(cfg)

    _ = keras_core.forward(
        ops.convert_to_tensor(n.numpy()),
        ops.convert_to_tensor(ed.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(edge_pairs_keras.numpy()),
    )

    copy_dense_torch_to_keras(torch_model.node_init, keras_core.dense_node_init)
    copy_dense_torch_to_keras(torch_model.edge_init, keras_core.dense_edge_init)
    for i in range(max(cfg.depth - 1, 1)):
        copy_dense_torch_to_keras(torch_model.edge_denses[i], keras_core.dense_edge_steps[i])
    copy_dense_torch_to_keras(torch_model.node_dense, keras_core.dense_node_final)

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
