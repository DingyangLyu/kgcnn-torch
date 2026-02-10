#!/usr/bin/env python3
"""Model-level alignment: CMPNN (full model, Torch -> Keras weight transfer).

Builds both Torch CMPNNModel and a KerasCMPNNFullStack that mirrors the full
architecture (Embedding + node_init + edge_init + (communicative aggregation
sum*max + edge update with reverse subtraction + Dense + residual)×(depth-1)
+ final communicative agg + concat + node_dense + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Simplified version without GRU pooling (use_final_gru=False).
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp,
    compare_outputs, keras_to_torch, make_disjoint_graph_directed,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.gather import GatherNodesOutgoing, GatherEdgesPairs
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.cmpnn import CMPNNModel


@dataclass
class Config:
    node_dim: int = 32
    edge_dim: int = 8
    depth: int = 4
    units: int = 32
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    n_nodes: int = 20
    n_edges_per_dir: int = 30
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [32, 16]


class KerasCMPNNFullStack:
    """Full Keras CMPNN model stack mirroring CMPNNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        self.node_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.node_init_act = keras.layers.Activation("relu")
        self.edge_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.edge_init_act = keras.layers.Activation("relu")

        # Per-step edge Dense layers (depth-1)
        self.edge_denses = [
            keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
            for _ in range(max(cfg.depth - 1, 1))
        ]

        self.aggr_sum = AggregateLocalEdges(pooling_method="scatter_sum")
        self.aggr_max = AggregateLocalEdges(pooling_method="scatter_max")
        self.gather_out = GatherNodesOutgoing()
        self.gather_pairs = GatherEdgesPairs()
        self.activation = keras.layers.Activation("relu")

        # Final node Dense: [m, h, h0] -> (3*units) -> units
        self.node_dense = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["relu"] * len(cfg.output_units) + ["linear"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act, use_bias=out_bias)

    def forward(self, z, edge_attr, edge_index, edge_pair_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        n = self.node_embedding(z)

        h0 = self.node_init_act(self.node_init(n))
        he0 = self.edge_init_act(self.edge_init(edge_attr))
        h = h0
        he = he0

        for i in range(self.cfg.depth - 1):
            # Communicative node update: sum * max
            m_pool = self.aggr_sum([n, he, edge_index])
            m_max = self.aggr_max([n, he, edge_index])
            m = keras.layers.Multiply()([m_pool, m_max])
            h = keras.layers.Add()([h, m])

            # Edge update: source_node - reverse_edge
            h_out = self.gather_out([h, edge_index])
            e_rev = self.gather_pairs([he, edge_pair_index])
            he = keras.layers.Subtract()([h_out, e_rev])
            he = self.edge_denses[i](he)
            he = keras.layers.Add()([he, he0])
            he = self.activation(he)

        # Final communicative aggregation
        m_pool = self.aggr_sum([n, he, edge_index])
        m_max = self.aggr_max([n, he, edge_index])
        m = keras.layers.Multiply()([m_pool, m_max])

        # Final node features
        h_final = keras.layers.Concatenate(axis=-1)([m, h, h0])
        h_final = self.node_dense(h_final)

        out = self.pooling([count_nodes, h_final, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: CMPNNModel,
                         keras_stack: KerasCMPNNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.node_init, keras_stack.node_init)
    copy_dense(torch_model.edge_init, keras_stack.edge_init)

    for i in range(max(cfg.depth - 1, 1)):
        copy_dense(torch_model.edge_denses[i], keras_stack.edge_denses[i])

    copy_dense(torch_model.node_dense, keras_stack.node_dense)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph_directed(
        n_nodes=cfg.n_nodes, n_edges_per_dir=cfg.n_edges_per_dir,
        batch_size=cfg.batch_size, node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_edge_attr=True,
    )

    torch_model = CMPNNModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        depth=cfg.depth,
        units=cfg.units,
        dropout=0.0,
        activation="relu",
        node_dense_activation="linear",
        use_final_gru=False,
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation="relu",
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasCMPNNFullStack(cfg)
    z_k = keras_data["z"]
    ea_k = keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    ep_k = keras_data["edge_pair_index"]
    bid_k = keras_data["batch_id_node"]
    bie_k = keras_data["batch_id_edge"]
    cn_k = keras_data["count_nodes"]
    ce_k = keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k))

    print(f"CMPNN model-level alignment (Torch -> Keras):")
    compare_outputs("CMPNN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("CMPNN model alignment PASSED.")


if __name__ == "__main__":
    main()
