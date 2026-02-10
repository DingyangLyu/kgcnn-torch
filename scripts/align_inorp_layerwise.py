#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras INorp interaction stack vs kgcnn-torch."""
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

from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import GraphMLP
from kgcnn_torch.models.inorp import INorpModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    edge_dim: int = 8
    edge_mlp_units: tuple = (16, 16)
    node_mlp_units: tuple = (16, 16)
    depth: int = 3
    seed: int = 42


MAX_MAE, MAX_ABS = get_thresholds(__file__)


class KerasINorpStack:
    """Manual Keras recreation matching INorpModel architecture (no edge projection)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        # Matches model.node_projection: maps node_dim -> node_dim
        self.node_projection = keras.layers.Dense(cfg.node_dim, activation="linear")
        self.gi = GatherNodesIngoing()
        self.go = GatherNodesOutgoing()
        self.cat = keras.layers.Concatenate(axis=-1)
        self.pool = AggregateLocalEdges(pooling_method="sum")
        self.edge_mlps: List[GraphMLP] = []
        self.node_mlps: List[GraphMLP] = []
        edge_out_dim = cfg.edge_mlp_units[-1]
        for i in range(cfg.depth):
            # Match model's _norm_act: ["relu", "linear"] for 2-layer MLP
            e_act = ["relu"] * (len(cfg.edge_mlp_units) - 1) + ["linear"]
            n_act = ["relu"] * (len(cfg.node_mlp_units) - 1) + ["linear"]
            self.edge_mlps.append(GraphMLP(
                units=list(cfg.edge_mlp_units), activation=e_act))
            self.node_mlps.append(GraphMLP(
                units=list(cfg.node_mlp_units), activation=n_act))

    def forward(self, x, edge_attr, edge_index, batch, count_nodes, count_edges) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        n = self.node_projection(x)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(n))
        for i in range(self.cfg.depth):
            eu1 = self.gi([n, edge_index])
            eu2 = self.go([n, edge_index])
            # Edge input: [n_j, n_i, raw_edge_features] (no edge projection)
            eu = self.cat([eu2, eu1, edge_attr])
            eu = self.edge_mlps[i]([eu, batch[edge_index[0]], count_edges])
            nu = self.pool([n, eu, edge_index])
            n = self.node_mlps[i]([self.cat([n, nu]), batch, count_nodes])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(n))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.node_dim)
    edge_attr = torch.randn(cfg.n_edges, cfg.edge_dim)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    batch = torch.cat([torch.zeros(cfg.n_nodes // 2, dtype=torch.long), torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long)])
    count_nodes = torch.bincount(batch, minlength=2)
    edge_batch = batch[dst]
    count_edges = torch.bincount(edge_batch, minlength=2)
    return x, edge_attr, edge_index_torch, edge_index_keras, batch, count_nodes, count_edges


def torch_forward(model: INorpModel, x, edge_attr, edge_index, batch) -> Dict[str, torch.Tensor]:
    from kgcnn_torch.layers.gather import gather_nodes_outgoing, gather_nodes_ingoing
    out = {}
    n = model.node_projection(x)
    out["dense_in"] = n.detach().cpu()
    num_nodes = n.size(0)
    for i in range(model.depth):
        n_j = gather_nodes_outgoing(n, edge_index)
        n_i = gather_nodes_ingoing(n, edge_index)
        eu = model.blocks[i]["edge_mlp"](torch.cat([n_j, n_i, edge_attr], dim=-1))
        agg = model.aggr(eu, edge_index, num_nodes)
        n = model.blocks[i]["node_mlp"](torch.cat([n, agg], dim=-1))
        out[f"layer_{i+1}"] = n.detach().cpu()
    return out


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
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")
        worst_mae = max(worst_mae, mae)
        worst_abs = max(worst_abs, max_abs)
    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    x, edge_attr, e_t, e_k, batch, count_nodes, count_edges = build_inputs(cfg)

    t_model = INorpModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        edge_dim=cfg.edge_dim,
        edge_mlp_units=list(cfg.edge_mlp_units),
        node_mlp_units=list(cfg.node_mlp_units),
        message_pooling="sum",
        use_set2set=False,
        use_graph_state=False,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
        node_input_dim=cfg.node_dim,
        use_edge_embedding=False,
    )
    k_stack = KerasINorpStack(cfg)

    _ = k_stack.forward(ops.convert_to_tensor(x.numpy()), ops.convert_to_tensor(edge_attr.numpy()),
                        ops.convert_to_tensor(e_k.numpy()), ops.convert_to_tensor(batch.numpy()),
                        ops.convert_to_tensor(count_nodes.numpy()), ops.convert_to_tensor(count_edges.numpy()))

    copy_dense_torch_to_keras(t_model.node_projection, k_stack.node_projection)
    for i in range(cfg.depth):
        for j, kd in enumerate(k_stack.edge_mlps[i].mlp_dense_layer_list):
            copy_dense_torch_to_keras(t_model.blocks[i]["edge_mlp"].linears[j], kd)
        for j, kd in enumerate(k_stack.node_mlps[i].mlp_dense_layer_list):
            copy_dense_torch_to_keras(t_model.blocks[i]["node_mlp"].linears[j], kd)

    t_out = torch_forward(t_model, x, edge_attr, e_t, batch)
    k_out = k_stack.forward(ops.convert_to_tensor(x.numpy()), ops.convert_to_tensor(edge_attr.numpy()),
                            ops.convert_to_tensor(e_k.numpy()), ops.convert_to_tensor(batch.numpy()),
                            ops.convert_to_tensor(count_nodes.numpy()), ops.convert_to_tensor(count_edges.numpy()))
    compare_stage_dicts(k_out, t_out)


if __name__ == "__main__":
    main()
