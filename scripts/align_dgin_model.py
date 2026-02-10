#!/usr/bin/env python3
"""Model-level alignment: DGIN (full model, Torch -> Keras weight transfer).

Builds both Torch DGINModel and a KerasDGINFullStack that mirrors the full
architecture:
  DMPNN stage: Embedding + edge_init + (DMPNNPPoolingEdgesDirected + shared Dense
    + residual + activation)×depth_dmpnn + AggregateLocalEdges + node_dense
  GIN_D stage: (GIN_D + GraphMLP)×depth_gin + per-layer (PoolingNodes + last_mlp)
    + sum + output MLP
Transfers ALL weights and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass
from typing import List

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
from kgcnn.layers.mlp import MLP as KerasMLP, GraphMLP as KerasGraphMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.literature.DMPNN._layers import DMPNNPPoolingEdgesDirected
from kgcnn.literature.DGIN._layers import GIN_D as KerasGIN_D
from kgcnn_torch.models.dgin import DGINModel


@dataclass
class Config:
    node_dim: int = 32
    edge_dim: int = 8
    depth_dmpnn: int = 3
    depth_gin: int = 3
    units: int = 32
    gin_mlp_units: list = None
    last_mlp_units: list = None
    output_units: list = None
    num_targets: int = 2
    num_embeddings: int = 95
    n_nodes: int = 20
    n_edges_per_dir: int = 30
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.gin_mlp_units is None:
            self.gin_mlp_units = [32, 32]
        if self.last_mlp_units is None:
            self.last_mlp_units = [32, 32]
        if self.output_units is None:
            self.output_units = []


class KerasDGINFullStack:
    """Full Keras DGIN model stack mirroring DGINModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        # DMPNN stage
        self.gather_out = GatherNodesOutgoing()
        self.concat_init = keras.layers.Concatenate(axis=-1)
        self.edge_init = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.edge_init_act = keras.layers.Activation("relu")
        self.dmpnn_dense = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.activation = keras.layers.Activation("relu")
        self.dmpnn_pools = [DMPNNPPoolingEdgesDirected() for _ in range(cfg.depth_dmpnn)]
        self.aggr = AggregateLocalEdges(pooling_method="scatter_sum")
        self.concat_node = keras.layers.Concatenate(axis=-1)
        # node_dense: (node_dim + units) -> gin_mlp_units[-1]
        gin_out_dim = cfg.gin_mlp_units[-1] if cfg.gin_mlp_units else cfg.units
        self.node_dense = keras.layers.Dense(gin_out_dim, activation="linear", use_bias=True)

        # GIN_D stage
        self.gin_convs: List[KerasGIN_D] = []
        self.gin_mlps: List[KerasGraphMLP] = []
        gin_act = ["relu"] * max(len(cfg.gin_mlp_units) - 1, 0) + ["linear"]
        for _ in range(cfg.depth_gin):
            self.gin_convs.append(KerasGIN_D(
                pooling_method="scatter_sum",
                epsilon_learnable=False,
            ))
            self.gin_mlps.append(KerasGraphMLP(
                units=cfg.gin_mlp_units,
                activation=gin_act,
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))

        # Per-layer readout: (depth_gin + 1) pools + MLPs
        self.poolings: List[KerasPoolingNodes] = [
            KerasPoolingNodes(pooling_method="scatter_mean")
            for _ in range(cfg.depth_gin + 1)
        ]
        # Torch uses activation=output_activation (scalar "relu") -> broadcast to all layers
        self.last_mlps: List[KerasMLP] = [
            KerasMLP(units=cfg.last_mlp_units, activation="relu")
            for _ in range(cfg.depth_gin + 1)
        ]

        # Output MLP
        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["relu"] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, edge_attr, edge_index, edge_pair_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        n = self.node_embedding(z)

        # DMPNN stage: initialize edge messages
        n_j = self.gather_out([n, edge_index])
        h0 = self.concat_init([n_j, edge_attr])
        h0 = self.edge_init_act(self.edge_init(h0))
        h = h0

        for i in range(self.cfg.depth_dmpnn):
            m_vw = self.dmpnn_pools[i]([n, h, edge_index, edge_pair_index])
            h = self.dmpnn_dense(m_vw)
            h = keras.layers.Add()([h, h0])
            h = self.activation(h)

        # Transition to GIN: aggregate final edge messages
        a_i = self.aggr([n, h, edge_index])
        m_v = self.concat_node([n, a_i])
        h_v = self.node_dense(m_v)  # linear, no activation
        h_v_0 = h_v

        # GIN_D stage
        list_embeddings = [h_v_0]
        for i in range(self.cfg.depth_gin):
            h_v = self.gin_convs[i]([h_v, edge_index, h_v_0])
            h_v = self.gin_mlps[i]([h_v, batch_id_node, count_nodes])
            list_embeddings.append(h_v)

        # Per-layer readout
        out = None
        for i, emb in enumerate(list_embeddings):
            p = self.poolings[i]([count_nodes, emb, batch_id_node])
            p = self.last_mlps[i](p)
            if out is None:
                out = p
            else:
                out = out + p

        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: DGINModel,
                         keras_stack: KerasDGINFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    # DMPNN stage
    copy_dense(torch_model.edge_init, keras_stack.edge_init)
    copy_dense(torch_model.dmpnn_dense, keras_stack.dmpnn_dense)
    copy_dense(torch_model.node_dense, keras_stack.node_dense)

    # GIN_D stage
    for i in range(cfg.depth_gin):
        # GIN_D epsilon
        keras_stack.gin_convs[i].eps_k.assign(
            float(torch_model.gin_convs[i].eps.detach().cpu().item())
        )
        copy_mlp(torch_model.gin_mlps[i], keras_stack.gin_mlps[i])

    # Per-layer readout MLPs
    for i in range(cfg.depth_gin + 1):
        copy_mlp(torch_model.last_mlps[i], keras_stack.last_mlps[i])

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

    torch_model = DGINModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        depth_dmpnn=cfg.depth_dmpnn,
        depth_gin=cfg.depth_gin,
        units=cfg.units,
        dropout_dmpnn=0.0,
        dropout_gin=0.0,
        activation="relu",
        gin_mlp_units=cfg.gin_mlp_units,
        gin_mlp_activation=["relu", "linear"],
        gin_mlp_use_normalization=False,
        last_mlp_units=cfg.last_mlp_units,
        node_pooling="mean",
        output_units=cfg.output_units,
        output_activation="relu",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasDGINFullStack(cfg)
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

    print(f"DGIN model-level alignment (Torch -> Keras):")
    compare_outputs("DGIN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("DGIN model alignment PASSED.")


if __name__ == "__main__":
    main()
