#!/usr/bin/env python3
"""Model-level alignment: DMPNN (full model, Torch -> Keras weight transfer).

Builds both Torch DMPNNModel and a KerasDMPNNFullStack that mirrors the full
architecture (Embedding + message_init + (DMPNNPPoolingEdgesDirected + shared Dense
+ residual + activation)×depth + AggregateLocalEdges + node_readout + PoolingNodes
+ output MLP), transfers ALL weights, and verifies final output.

NOTE: Simplified version without graph state or edge embedding.
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

from kgcnn.layers.gather import GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.literature.DMPNN._layers import DMPNNPPoolingEdgesDirected
from kgcnn_torch.models.dmpnn import DMPNNModel


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


class KerasDMPNNFullStack:
    """Full Keras DMPNN model stack mirroring DMPNNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        self.gather_out = GatherNodesOutgoing()
        self.concat_init = keras.layers.Concatenate(axis=-1)

        # message_init: Dense(node_dim + edge_dim, units)
        self.message_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.init_act = keras.layers.Activation("relu")

        # Shared Dense for all message passing steps
        self.W_h = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.activation = keras.layers.Activation("relu")

        self.dmpnn_pool = []
        for _ in range(cfg.depth):
            self.dmpnn_pool.append(DMPNNPPoolingEdgesDirected())

        # Final aggregation + node readout
        self.aggr = AggregateLocalEdges(pooling_method="scatter_sum")
        self.concat_readout = keras.layers.Concatenate(axis=-1)
        self.node_readout = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.node_act = keras.layers.Activation("relu")

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["relu"] * len(cfg.output_units) + ["linear"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act, use_bias=out_bias)

    def forward(self, z, edge_attr, edge_index, edge_pair_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        n = self.node_embedding(z)

        # Initial edge messages: concat(source_node, edge_attr) -> Dense
        n_j = self.gather_out([n, edge_index])
        h0 = self.concat_init([n_j, edge_attr])
        h0 = self.init_act(self.message_init(h0))
        h = h0

        # Directed message passing loop
        for i in range(self.cfg.depth):
            # DMPNNPPoolingEdgesDirected: aggregate, gather, subtract reverse
            m_vw = self.dmpnn_pool[i]([n, h, edge_index, edge_pair_index])
            h = self.W_h(m_vw)
            h = keras.layers.Add()([h, h0])
            h = self.activation(h)

        # Node readout
        m_agg = self.aggr([n, h, edge_index])
        h_v = self.concat_readout([m_agg, n])
        h_v = self.node_act(self.node_readout(h_v))

        out = self.pooling([count_nodes, h_v, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: DMPNNModel,
                         keras_stack: KerasDMPNNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.message_init, keras_stack.message_init)
    copy_dense(torch_model.W_h, keras_stack.W_h)
    copy_dense(torch_model.node_readout, keras_stack.node_readout)
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

    torch_model = DMPNNModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        depth=cfg.depth,
        units=cfg.units,
        message_activation="relu",
        init_activation="relu",
        node_activation="relu",
        message_pooling="sum",
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation="relu",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        use_edge_embedding=False,
        dropout_rate=0.0,
        use_graph_state=False,
    )
    torch_model.eval()

    keras_stack = KerasDMPNNFullStack(cfg)
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

    print(f"DMPNN model-level alignment (Torch -> Keras):")
    compare_outputs("DMPNN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("DMPNN model alignment PASSED.")


if __name__ == "__main__":
    main()
