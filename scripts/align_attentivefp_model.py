#!/usr/bin/env python3
"""Model-level alignment: AttentiveFP (full model, Torch -> Keras weight transfer).

Builds both Torch AttentiveFPModel and a KerasAttentiveFPFullStack that mirrors
the full architecture (Embedding + Dense + AttentiveHeadFP + GRU + Dropout +
PoolingNodesAttentive + output MLP), transfers ALL weights including GRU gate
reorder, and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, copy_gru_cell,
    compare_outputs, keras_to_torch, make_disjoint_graph,
)

import keras as ks
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.attention import AttentiveHeadFP as KerasAttentiveHeadFP
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodesAttentive as KerasPoolingNodesAttentive
from kgcnn.layers.update import GRUUpdate as KerasGRUUpdate
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.attentivefp import AttentiveFPModel


@dataclass
class Config:
    node_dim: int = 32
    depth_ato: int = 2
    depth_mol: int = 2
    units: int = 32
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


def _copy_gru_cell_torch_to_keras(torch_gru_cell, keras_gru_cell):
    """Copy torch.nn.GRUCell weights to keras.layers.GRUCell.

    Gate reorder: PyTorch (r,z,n) -> Keras (z,r,h).
    """
    w_ih = torch_gru_cell.weight_ih.detach().cpu().numpy()
    w_hh = torch_gru_cell.weight_hh.detach().cpu().numpy()
    b_ih = torch_gru_cell.bias_ih.detach().cpu().numpy()
    b_hh = torch_gru_cell.bias_hh.detach().cpu().numpy()

    def _reorder(arr, axis=0):
        r, z, n = np.split(arr, 3, axis=axis)
        return np.concatenate([z, r, n], axis=axis)

    kernel = _reorder(w_ih, axis=0).T
    recurrent_kernel = _reorder(w_hh, axis=0).T
    bias = np.stack([_reorder(b_ih), _reorder(b_hh)], axis=0)
    keras_gru_cell.set_weights([kernel, recurrent_kernel, bias])


class KerasAttentiveFPFullStack:
    """Full Keras AttentiveFP model stack mirroring AttentiveFPModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = ks.layers.Dense(cfg.units, activation="linear", use_bias=True)

        # Atom-level: attention heads + GRU updates
        self.heads = []
        self.grus = []
        for i in range(cfg.depth_ato):
            self.heads.append(KerasAttentiveHeadFP(
                units=cfg.units,
                use_edge_features=(i == 0),
                activation={"class_name": "function", "config": "kgcnn>leaky_relu2"},
                activation_context="elu",
                use_bias=True,
            ))
            self.grus.append(KerasGRUUpdate(units=cfg.units))

        # Molecule-level attentive pooling
        self.attentive_pooling = KerasPoolingNodesAttentive(
            units=cfg.units, depth=cfg.depth_mol,
        )

        # Output MLP
        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["sigmoid"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
            use_bias=out_bias,
        )

    def forward(self, z, edge_attr, edge_index, batch_id_node, count_nodes):
        """Full forward pass returning graph-level output."""
        n = self.node_embedding(z)
        nk = self.dense_in(n)

        # First atom-level iteration (with edge features, no dropout)
        ck = self.heads[0]([nk, edge_attr, edge_index])
        nk = self.grus[0]([nk, ck])

        # Remaining atom-level iterations (no edge features, with dropout skipped in eval)
        for i in range(1, self.cfg.depth_ato):
            ck = self.heads[i]([nk, edge_attr, edge_index])
            nk = self.grus[i]([nk, ck])
            # Dropout skipped during eval (dropout=0 or not applied)

        # Molecule-level attentive pooling
        out = self.attentive_pooling([count_nodes, nk, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: AttentiveFPModel,
                         keras_stack: KerasAttentiveFPFullStack, cfg: Config):
    """Transfer ALL weights from Torch model to Keras stack."""
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    # Atom-level: attention heads + GRU
    for i in range(cfg.depth_ato):
        t_head = torch_model.attention_layers[i]
        k_head = keras_stack.heads[i]

        copy_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
        # alpha_activation is Sequential(Linear, Activation) in torch
        copy_dense(t_head.alpha_activation[0], k_head.lay_alpha_activation)
        copy_dense(t_head.linear_alpha, k_head.lay_alpha)

        if i == 0 and t_head.use_edge_features:
            copy_dense(t_head.fc1[0], k_head.lay_fc1)
            copy_dense(t_head.fc2[0], k_head.lay_fc2)

        # GRU: torch GRUUpdate.gru_cell -> keras GRUUpdate.gru_cell
        copy_gru_cell(torch_model.gru_updates[i].gru_cell, keras_stack.grus[i])

    # Attentive pooling weights
    t_pool = torch_model.attentive_pooling
    k_pool = keras_stack.attentive_pooling
    copy_dense(t_pool.linear_trafo, k_pool.lay_linear_trafo)
    # lay_alpha in Torch is Sequential(Linear, Activation)
    copy_dense(t_pool.lay_alpha[0], k_pool.lay_alpha)
    # GRU in attentive pooling
    _copy_gru_cell_torch_to_keras(t_pool.gru, k_pool.lay_gru)

    # Output MLP
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_edge_attr=True,
    )

    # Build Torch model (dropout=0 for deterministic comparison)
    torch_model = AttentiveFPModel(
        node_dim=cfg.node_dim,
        depth_ato=cfg.depth_ato,
        depth_mol=cfg.depth_mol,
        units=cfg.units,
        use_edge_features=True,
        edge_dim=cfg.edge_dim,
        attention_activation="leaky_relu2",
        attention_activation_context="elu",
        pooling_activation="leaky_relu2",
        pooling_activation_context="elu",
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        dropout=0.0,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    # Build Keras stack (dry run)
    keras_stack = KerasAttentiveFPFullStack(cfg)
    z_k = keras_data["z"]
    ea_k = keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)

    # Transfer weights
    transfer_all_weights(torch_model, keras_stack, cfg)

    # Forward pass: Torch
    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    # Forward pass: Keras
    keras_out = keras_to_torch(keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k))

    # Compare
    print(f"AttentiveFP model-level alignment (Torch -> Keras):")
    compare_outputs("AttentiveFP_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("AttentiveFP model alignment PASSED.")


if __name__ == "__main__":
    main()
