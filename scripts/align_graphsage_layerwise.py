#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras GraphSAGE vs kgcnn-torch GraphSAGE."""
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

from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.norm import GraphLayerNormalization
from kgcnn.layers.mlp import GraphMLP as KerasGraphMLP
from kgcnn_torch.models.graphsage import GraphSAGEModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    edge_dim: int = 8
    depth: int = 2
    units: int = 16
    seed: int = 42


class KerasGraphSAGEStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        if cfg.node_dim != cfg.units:
            self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        else:
            self.dense_in = None
        self.gather = GatherNodesOutgoing()
        self.concat = keras.layers.Concatenate(axis=-1)
        self.edge_mlps: List[KerasGraphMLP] = []
        self.node_mlps: List[KerasGraphMLP] = []
        self.aggrs: List[AggregateLocalEdges] = []
        self.norms: List[GraphLayerNormalization] = []
        for _ in range(cfg.depth):
            self.edge_mlps.append(KerasGraphMLP(
                units=[cfg.units, cfg.units],
                activation=["relu", "linear"],
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))
            self.node_mlps.append(KerasGraphMLP(
                units=[cfg.units, cfg.units],
                activation=["relu", "linear"],
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))
            self.aggrs.append(AggregateLocalEdges(pooling_method="scatter_mean"))
            self.norms.append(GraphLayerNormalization())

    def forward(self, x, edge_attr, edge_index, batch_node, batch_edge, count_nodes, count_edges):
        out = {}
        if self.dense_in is not None:
            h = self.dense_in(x)
            out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(h))
        else:
            h = x
        for i in range(self.cfg.depth):
            eu = self.gather([h, edge_index])
            eu = self.concat([eu, edge_attr])
            eu = self.edge_mlps[i]([eu, batch_edge, count_edges])
            nu = self.aggrs[i]([h, eu, edge_index])
            hu = self.concat([h, nu])
            h = self.node_mlps[i]([hu, batch_node, count_nodes])
            h = self.norms[i]([h, batch_node, count_nodes])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    bias = torch_linear.bias.detach().cpu().numpy() if torch_linear.bias is not None else None
    keras_dense.set_weights([kernel, bias] if bias is not None else [kernel])


def build_random_graph(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    edge_attr = torch.randn(cfg.n_edges, cfg.edge_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    batch = torch.cat([torch.zeros(cfg.n_nodes // 2, dtype=torch.long),
                       torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long)])
    batch_edge = batch[edge_index_torch[0]]
    count_nodes = torch.tensor(cfg.n_nodes, dtype=torch.long)
    count_edges = torch.tensor(cfg.n_edges, dtype=torch.long)
    return x, edge_attr, edge_index_torch, edge_index_keras, batch, batch_edge, count_nodes, count_edges


def torch_forward_stages(model: GraphSAGEModel, x, edge_attr, edge_index, batch):
    out = {}
    if model.dense_in is not None:
        h = model.dense_in(x)
        out["dense_in"] = h.detach().cpu()
    else:
        h = x

    for i in range(model.depth):
        x_j = h[edge_index[0]]
        msg_input = torch.cat([x_j, edge_attr], dim=-1)
        msg = model.edge_mlps[i](msg_input)
        agg = model.aggregators[i](msg, edge_index, h.size(0))
        h = model.node_mlps[i](torch.cat([h, agg], dim=-1))
        h = model.layer_norms[i](h)
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
    x, edge_attr, edge_index_t, edge_index_k, batch, batch_edge, count_nodes, count_edges = build_random_graph(cfg)
    torch_model = GraphSAGEModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        node_mlp_units=[cfg.units, cfg.units],
        edge_mlp_units=[cfg.units, cfg.units],
        edge_dim=cfg.edge_dim,
        use_edge_features=True,
        pooling_method="mean",
        output_units=[],
        output_embedding="node",
        num_targets=1,
        use_node_embedding=False,
    )
    keras_stack = KerasGraphSAGEStack(cfg)
    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_k.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(batch_edge.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(count_edges.numpy()),
    )

    if torch_model.dense_in is not None:
        copy_dense_torch_to_keras(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth):
        for j in range(len(torch_model.edge_mlps[i].linears)):
            copy_dense_torch_to_keras(torch_model.edge_mlps[i].linears[j], keras_stack.edge_mlps[i].mlp_dense_layer_list[j])
            copy_dense_torch_to_keras(torch_model.node_mlps[i].linears[j], keras_stack.node_mlps[i].mlp_dense_layer_list[j])
        ln = torch_model.layer_norms[i].ln
        gamma = ln.weight.detach().cpu().numpy()
        beta = ln.bias.detach().cpu().numpy()
        keras_stack.norms[i].set_weights([gamma, beta])

    torch_stages = torch_forward_stages(torch_model, x, edge_attr, edge_index_t, batch)
    keras_stages = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_k.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(batch_edge.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(count_edges.numpy()),
    )
    compare_stage_dicts(keras_stages, torch_stages)


if __name__ == "__main__":
    main()
