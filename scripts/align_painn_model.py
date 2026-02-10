#!/usr/bin/env python3
"""Model-level alignment: PAiNN (full model, Torch -> Keras weight transfer).

Builds both Torch PAiNNModel and a KerasPAiNNFullStack that mirrors the full
architecture (Embedding + BesselBasis + CosCutOff + EquivariantInit +
(PAiNNConv + residual + PAiNNUpdate + residual)×depth + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Simplified version without equiv_normalization or node_normalization.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp,
    compare_outputs, keras_to_torch, make_disjoint_graph,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.geom import (
    NodePosition, NodeDistanceEuclidean, BesselBasisLayer as KerasBesselBasis,
    CosCutOffEnvelope as KerasCosCutOff, EdgeDirectionNormalized,
)
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.literature.PAiNN._layers import (
    PAiNNconv as KerasPAiNNConv,
    PAiNNUpdate as KerasPAiNNUpdate,
    EquivariantInitialize as KerasEquivInit,
)
from kgcnn_torch.models.painn import PAiNNModel


@dataclass
class Config:
    node_dim: int = 16
    depth: int = 2
    units: int = 16
    num_radial: int = 8
    cutoff: float = 5.0
    conv_cutoff: float = 5.0
    envelope_exponent: int = 5
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    n_nodes: int = 10
    n_edges: int = 30
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [8]


class KerasPAiNNFullStack:
    """Full Keras PAiNN model stack mirroring PAiNNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.equiv_init = KerasEquivInit(dim=3, units=cfg.units, method="zeros")
        self.bessel_basis = KerasBesselBasis(
            num_radial=cfg.num_radial, cutoff=cfg.cutoff,
            envelope_exponent=cfg.envelope_exponent,
        )
        self.cos_cutoff = KerasCosCutOff(cutoff=cfg.conv_cutoff)
        self.node_pos = NodePosition()
        self.dist = NodeDistanceEuclidean()
        self.dir_norm = EdgeDirectionNormalized()

        self.convs = []
        self.updates = []
        for _ in range(cfg.depth):
            self.convs.append(KerasPAiNNConv(
                units=cfg.units, conv_pool="scatter_sum",
                activation="swish", cutoff=cfg.conv_cutoff,
            ))
            self.updates.append(KerasPAiNNUpdate(
                units=cfg.units, activation="swish",
            ))

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["swish"] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, pos, edge_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        # Initialize equivariant features from raw z (before embedding)
        v = self.equiv_init(z)  # (N, 3, units)

        # Node embedding
        n = self.node_embedding(z)

        # Compute edge features
        pos1, pos2 = self.node_pos([pos, edge_index])
        d = self.dist([pos1, pos2])   # (M, 1)
        rij = self.dir_norm([pos1, pos2])  # (M, 3)
        rbf = self.bessel_basis(d)    # (M, num_radial)
        env = self.cos_cutoff(d)      # (M, 1)

        for i in range(self.cfg.depth):
            # Message passing
            ds, dv = self.convs[i]([n, v, rbf, env, rij, edge_index])
            n = keras.layers.Add()([n, ds])
            v = keras.layers.Add()([v, dv])

            # Update
            ds, dv = self.updates[i]([n, v])
            n = keras.layers.Add()([n, ds])
            v = keras.layers.Add()([v, dv])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def copy_bessel_basis(torch_bessel, keras_bessel):
    """Copy BesselBasisLayer frequencies."""
    freq = torch_bessel.frequencies.detach().cpu().numpy()
    keras_bessel.set_weights([freq])


def copy_painn_conv(torch_conv, keras_conv):
    """Copy PAiNNConv weights."""
    copy_dense(torch_conv.dense1, keras_conv.lay_dense1)
    copy_dense(torch_conv.phi, keras_conv.lay_phi)
    copy_dense(torch_conv.w, keras_conv.lay_w)


def copy_painn_update(torch_update, keras_update):
    """Copy PAiNNUpdate weights."""
    copy_dense(torch_update.lin_u, keras_update.lay_lin_u)
    copy_dense(torch_update.lin_v, keras_update.lay_lin_v)
    copy_dense(torch_update.dense1, keras_update.lay_dense1)
    copy_dense(torch_update.dense_a, keras_update.lay_a)


def transfer_all_weights(torch_model: PAiNNModel,
                         keras_stack: KerasPAiNNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_bessel_basis(torch_model.bessel_basis, keras_stack.bessel_basis)

    for i in range(cfg.depth):
        copy_painn_conv(torch_model.convs[i], keras_stack.convs[i])
        copy_painn_update(torch_model.updates[i], keras_stack.updates[i])

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False,
    )

    torch_model = PAiNNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        num_radial=cfg.num_radial,
        cutoff=cfg.cutoff,
        conv_cutoff=cfg.conv_cutoff,
        envelope_exponent=cfg.envelope_exponent,
        conv_activation="swish",
        conv_pooling="sum",
        update_activation="swish",
        update_add_eps=False,
        equiv_normalization=False,
        node_normalization=False,
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation="swish",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasPAiNNFullStack(cfg)
    z_k = keras_data["z"]
    pos_k = keras_data["pos"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    bie_k = keras_data["batch_id_edge"]
    cn_k = keras_data["count_nodes"]
    ce_k = keras_data["count_edges"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, bid_k, bie_k, cn_k, ce_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, pos_k, ei_k, bid_k, bie_k, cn_k, ce_k))

    print(f"PAiNN model-level alignment (Torch -> Keras):")
    compare_outputs("PAiNN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("PAiNN model alignment PASSED.")


if __name__ == "__main__":
    main()
