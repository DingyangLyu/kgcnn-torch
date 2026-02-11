#!/usr/bin/env python3
"""Model-level alignment: RGCN (full model, Torch -> Keras weight transfer).

Builds both Torch RGCNModel and a KerasRGCNFullStack that mirrors the full
architecture (Embedding + per-layer (GatherOutgoing + RelationalDense +
Multiply + AggregateLocal + Dense(self_loop) + Add + Activation) +
PoolingNodes + output MLP), transfers ALL weights, and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, copy_relational_dense,
    compare_outputs, keras_to_torch, make_disjoint_graph_relational,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.relational import RelationalDense as KerasRelationalDense
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.rgcn import RGCNModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    units: int = 32
    num_relations: int = 5
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    output_activation: str = "relu"
    output_final_activation: str = "linear"  # alignment uses linear; Keras literature default is "softmax"
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [32, 16]


class KerasRGCNFullStack:
    """Full Keras RGCN model stack mirroring RGCNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        self.gather = GatherNodesOutgoing()

        # Per-layer: Dense(self_loop) + RelationalDense + Activation
        self.dense_self = []
        self.rel_dense = []
        self.activations = []
        self.aggrs = []
        for i in range(cfg.depth):
            in_dim = cfg.node_dim if i == 0 else cfg.units
            self.dense_self.append(
                keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
            )
            self.rel_dense.append(KerasRelationalDense(
                units=cfg.units,
                num_relations=cfg.num_relations,
                activation=None,
                use_bias=True,
            ))
            self.activations.append(keras.layers.Activation("swish"))
            self.aggrs.append(AggregateLocalEdges(pooling_method="scatter_sum"))

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + [cfg.output_final_activation]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
        )

    def forward(self, z, edge_index, edge_type, edge_attr, batch_id_node, count_nodes):
        n = self.node_embedding(z)

        for i in range(self.cfg.depth):
            n_j = self.gather([n, edge_index])
            h_j = self.rel_dense[i]([n_j, edge_type])
            if edge_attr is not None:
                h_j = keras.layers.Multiply()([h_j, edge_attr])
            h = self.aggrs[i]([n, h_j, edge_index])
            h0 = self.dense_self[i](n)
            n = keras.layers.Add()([h, h0])
            n = self.activations[i](n)

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: RGCNModel,
                         keras_stack: KerasRGCNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    for i in range(cfg.depth):
        t = torch_model.convs[i]
        # Self-loop: nn.Linear -> Dense
        copy_dense(t.self_loop, keras_stack.dense_self[i])
        # Relational weights: (R, in, out) — direct copy (no transpose!)
        rel_w = t.weight.detach().cpu().numpy()
        weights = [rel_w]
        if t.rel_bias is not None:
            weights.append(t.rel_bias.detach().cpu().numpy())
        keras_stack.rel_dense[i].set_weights(weights)

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph_relational(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, num_relations=cfg.num_relations,
        seed=cfg.seed, include_edge_weight=True,
    )

    torch_model = RGCNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        num_relations=cfg.num_relations,
        rgcn_activation="swish",
        rgcn_pooling="sum",
        use_residual=False,
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        output_final_activation=cfg.output_final_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasRGCNFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    et_k = keras_data["edge_type"]
    ea_k = keras_data["edge_attr"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, et_k, ea_k, bid_k, cn_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ei_k, et_k, ea_k, bid_k, cn_k))

    print(f"RGCN model-level alignment (Torch -> Keras):")
    compare_outputs("RGCN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("RGCN model alignment PASSED.")


if __name__ == "__main__":
    main()
