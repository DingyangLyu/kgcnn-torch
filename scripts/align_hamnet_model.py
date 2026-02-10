#!/usr/bin/env python3
"""Model-level alignment: HamNet (full model, Torch -> Keras weight transfer).

Builds both Torch HamNetModel and a KerasHamNetFullStack that mirrors the full
architecture (Embedding + node_init + edge_init +
(HamNaiveDynMessage + GRU/NaiveUnion update)×depth +
HamNetFingerprintGenerator + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Uses use_gru_update=True, use_gru_update_edge=False (default config).
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, copy_gru_cell,
    compare_outputs, keras_to_torch, make_disjoint_graph,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.layers.update import GRUUpdate as KerasGRUUpdate
from kgcnn.literature.HamNet._layers import (
    HamNaiveDynMessage as KerasHamNaiveDynMessage,
    HamNetNaiveUnion as KerasHamNetNaiveUnion,
    HamNetFingerprintGenerator as KerasHamNetFingerprintGenerator,
)
from kgcnn_torch.models.hamnet import HamNetModel


@dataclass
class Config:
    node_dim: int = 32
    edge_dim: int = 16
    depth: int = 2
    units: int = 32
    fingerprint_dim: int = 32
    fingerprint_depth: int = 2
    use_gru_update: bool = True
    activation: str = "leaky_relu2"
    output_units: list = None
    output_activation: str = "relu"
    num_targets: int = 2
    num_embeddings: int = 95
    n_nodes: int = 15
    n_edges: int = 40
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [25, 10]


class KerasHamNetFullStack:
    """Full Keras HamNet model stack mirroring HamNetModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.node_init = keras.layers.Dense(cfg.units, activation="tanh")
        self.edge_init = keras.layers.Dense(cfg.units, activation="tanh")

        # Internal edge dim after edge_init becomes units
        internal_edge_dim = cfg.units

        self.message_layers = []
        self.node_updates = []
        for _ in range(cfg.depth):
            self.message_layers.append(KerasHamNaiveDynMessage(
                units=cfg.units, units_edge=internal_edge_dim,
                activation="kgcnn>leaky_relu2", activation_last="elu",
            ))
            if cfg.use_gru_update:
                self.node_updates.append(KerasGRUUpdate(units=cfg.units))
            else:
                self.node_updates.append(KerasHamNetNaiveUnion(
                    units=cfg.units, activation="kgcnn>leaky_relu2",
                ))

        self.fingerprint = KerasHamNetFingerprintGenerator(
            units=cfg.fingerprint_dim,
            units_attend=cfg.fingerprint_dim,
            depth=cfg.fingerprint_depth,
            activation="kgcnn>leaky_relu2",
            pooling_method="mean",
        )

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["linear"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act, use_bias=out_bias)

    def forward(self, z, pos, edge_attr, edge_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        h = self.node_embedding(z)
        h = self.node_init(h)
        e = self.edge_init(edge_attr)

        q = pos
        p = torch.zeros_like(q)

        for i in range(self.cfg.depth):
            mv, me = self.message_layers[i]([h, e, p, q, edge_index])
            if self.cfg.use_gru_update:
                h = self.node_updates[i]([h, mv])
            else:
                h = self.node_updates[i]([h, mv])
            e = me

        out = self.fingerprint([count_nodes, h, batch_id_node])
        out = self.output_mlp(out)
        return out


def copy_hamnet_message(torch_msg, keras_msg):
    """Copy HamNaiveDynMessage weights."""
    copy_dense(torch_msg.align_dense, keras_msg.dense_align)
    # attend_dense is nn.Sequential([Linear, activation]) -> extract [0]
    copy_dense(torch_msg.attend_dense[0], keras_msg.dense_attend)
    copy_dense(torch_msg.edge_dense, keras_msg.dense_e)


def copy_hamnet_fingerprint(torch_fp, keras_fp, cfg):
    """Copy HamNetFingerprintGenerator weights."""
    # init_dense is nn.Sequential([Linear, activation]) -> extract [0]
    copy_dense(torch_fp.init_dense[0], keras_fp.vertex2mol)
    for i in range(cfg.fingerprint_depth):
        # attend_denses[i] is nn.Sequential([Linear, activation])
        copy_dense(torch_fp.attend_denses[i][0], keras_fp.readouts[i].dense_attend)
        copy_dense(torch_fp.align_denses[i], keras_fp.readouts[i].dense_align)
        # GRU cells: torch nn.GRUCell -> keras GRUCell (raw, not wrapped)
        copy_gru_cell(torch_fp.grus[i], keras_fp.unions[i])


def transfer_all_weights(torch_model: HamNetModel,
                         keras_stack: KerasHamNetFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    # node_init/edge_init: extract Linear from Sequential
    copy_dense(torch_model.node_init[0], keras_stack.node_init)
    copy_dense(torch_model.edge_init[0], keras_stack.edge_init)

    for i in range(cfg.depth):
        copy_hamnet_message(torch_model.message_layers[i],
                            keras_stack.message_layers[i])
        if cfg.use_gru_update:
            copy_gru_cell(torch_model.node_update_layers[i].gru_cell,
                          keras_stack.node_updates[i])
        else:
            copy_dense(torch_model.node_update_layers[i].dense,
                       keras_stack.node_updates[i].lay_dense)

    copy_hamnet_fingerprint(torch_model.fingerprint, keras_stack.fingerprint, cfg)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_pos=True, include_edge_attr=True,
    )

    torch_model = HamNetModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        depth=cfg.depth,
        units=cfg.units,
        fingerprint_dim=cfg.fingerprint_dim,
        fingerprint_depth=cfg.fingerprint_depth,
        activation=cfg.activation,
        activation_last="elu",
        fingerprint_activation=cfg.activation,
        fingerprint_activation_context=cfg.activation,
        use_gru_update=cfg.use_gru_update,
        use_gru_update_edge=False,
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasHamNetFullStack(cfg)
    z_k = keras_data["z"]
    pos_k = keras_data["pos"]
    ea_k = keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    bie_k = keras_data["batch_id_edge"]
    cn_k = keras_data["count_nodes"]
    ce_k = keras_data["count_edges"]
    _ = keras_stack.forward(z_k, pos_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, pos_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k))

    print("HamNet model-level alignment (Torch -> Keras):")
    compare_outputs("HamNet_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("HamNet model alignment PASSED.")


if __name__ == "__main__":
    main()
