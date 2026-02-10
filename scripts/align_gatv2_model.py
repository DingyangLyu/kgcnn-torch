#!/usr/bin/env python3
"""Model-level alignment: GATv2 (full model, Torch -> Keras weight transfer).

Builds both Torch GATv2Model and a KerasGATv2FullStack that mirrors the full
architecture (Embedding + Dense + multi-head AttentionHeadGATV2 + Average +
Activation + PoolingNodes + output MLP), transfers ALL weights, and verifies
final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass
from typing import List

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, compare_outputs,
    keras_to_torch, make_disjoint_graph,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.attention import AttentionHeadGATV2 as KerasAttentionHeadGATV2
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.gatv2 import GATv2Model


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    attention_units: int = 16
    heads: int = 3
    concat: bool = False
    edge_dim: int = 8
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    output_activation: str = "relu"
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [32, 16]


class KerasGATv2FullStack:
    """Full Keras GATv2 model stack mirroring GATv2Model architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = keras.layers.Dense(cfg.attention_units, activation="linear", use_bias=True)
        self.heads: List[List[KerasAttentionHeadGATV2]] = []
        for _ in range(cfg.depth):
            layer_heads = []
            for _ in range(cfg.heads):
                layer_heads.append(KerasAttentionHeadGATV2(
                    units=cfg.attention_units,
                    use_edge_features=True,
                    use_final_activation=False,
                    activation={"class_name": "function", "config": "kgcnn>leaky_relu2"},
                    use_bias=True,
                ))
            self.heads.append(layer_heads)

        self.avg = keras.layers.Average()
        self.after_avg_act = keras.layers.Activation(
            activation={"class_name": "function", "config": "kgcnn>leaky_relu2"}
        )
        self.pooling = KerasPoolingNodes(pooling_method="scatter_mean")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["sigmoid"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
            use_bias=out_bias,
        )

    def forward(self, z, edge_attr, edge_index, batch_id_node, count_nodes):
        n = self.node_embedding(z)
        nk = self.dense_in(n)

        for layer_heads in self.heads:
            h = [head([nk, edge_attr, edge_index]) for head in layer_heads]
            if self.cfg.concat:
                nk = keras.layers.Concatenate(axis=-1)(h)
            else:
                nk = self.avg(h)
                nk = self.after_avg_act(nk)

        out = self.pooling([count_nodes, nk, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: GATv2Model, keras_stack: KerasGATv2FullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    for i in range(cfg.depth):
        for j in range(cfg.heads):
            t_head = torch_model.attention_layers[i][j]
            k_head = keras_stack.heads[i][j]
            copy_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
            copy_dense(t_head.alpha_activation[0], k_head.lay_alpha_activation)
            copy_dense(t_head.linear_alpha, k_head.lay_alpha)

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_edge_attr=True,
    )

    torch_model = GATv2Model(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        attention_units=cfg.attention_units,
        attention_heads_num=cfg.heads,
        attention_heads_concat=cfg.concat,
        attention_activation="leaky_relu2",
        use_edge_features=True,
        edge_dim=cfg.edge_dim,
        node_pooling="mean",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasGATv2FullStack(cfg)
    z_k = keras_data["z"]
    ea_k = keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k))

    print(f"GATv2 model-level alignment (Torch -> Keras):")
    compare_outputs("GATv2_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("GATv2 model alignment PASSED.")


if __name__ == "__main__":
    main()
