#!/usr/bin/env python3
"""Model-level alignment: MoGAT (full model, Torch -> Keras weight transfer).

Builds both Torch MoGATModel and a KerasMoGATFullStack that mirrors the full
architecture (Embedding + Dense + (AttentiveHeadFP_ + GRU + Dropout) x depthato +
per-layer PoolingNodesAttentive + self-attention + Multiply + Flatten + output MLP),
transfers ALL weights, and verifies final output.
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

from kgcnn.literature.MoGAT._layers import AttentiveHeadFP_ as KerasAttentiveHeadFP_
from kgcnn.layers.update import GRUUpdate as KerasGRUUpdate
from kgcnn.layers.pooling import PoolingNodesAttentive as KerasPoolingNodesAttentive
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.modules import Embedding as KerasEmbedding, ExpandDims
from kgcnn_torch.models.mogat import MoGATModel


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


@dataclass
class Config:
    node_dim: int = 32
    depthato: int = 2
    depthmol: int = 2
    units: int = 32
    edge_dim: int = 8
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [32, 16]


class KerasMoGATFullStack:
    """Full Keras MoGAT model stack mirroring MoGATModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = ks.layers.Dense(cfg.units, activation="linear", use_bias=True)

        # Attention layers + GRU updates (no dropout during eval)
        self.heads = []
        self.grus = []
        for i in range(cfg.depthato):
            self.heads.append(KerasAttentiveHeadFP_(
                units=cfg.units,
                use_edge_features=(i == 0),
                activation="kgcnn>leaky_relu2",
                activation_context="elu",
                use_bias=True,
            ))
            self.grus.append(KerasGRUUpdate(units=cfg.units))

        # Per-layer attentive pooling
        self.layer_pools = []
        for _ in range(cfg.depthato):
            self.layer_pools.append(KerasPoolingNodesAttentive(
                units=cfg.units, depth=cfg.depthmol,
            ))

        self.expand = ExpandDims(axis=1)
        self.concat_ax1 = ks.layers.Concatenate(axis=1)

        # Self-attention: keras.layers.Attention(use_scale=True)
        self.self_attention = ks.layers.Attention(
            use_scale=True, score_mode="dot",
        )

        self.flatten = ks.layers.Flatten()

        # Output MLP
        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["relu"] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
        )

    def forward(self, z, edge_attr, edge_index, batch_id_node, count_nodes):
        n = self.node_embedding(z)
        nk = self.dense_in(n)

        list_emb = []
        # First layer: with edge features
        ck = self.heads[0]([nk, edge_attr, edge_index])
        nk = self.grus[0]([nk, ck])
        # No dropout during eval
        list_emb.append(nk)

        for i in range(1, self.cfg.depthato):
            ck = self.heads[i]([nk, edge_attr, edge_index])
            nk = self.grus[i]([nk, ck])
            list_emb.append(nk)

        # Per-layer attentive pooling
        out = [
            self.layer_pools[i]([count_nodes, ni, batch_id_node])
            for i, ni in enumerate(list_emb)
        ]
        out = [self.expand(x) for x in out]
        out = self.concat_ax1(out)  # (B, depthato, units)

        # Self-attention
        at = self.self_attention([out, out])

        # Element-wise multiply
        out = at * out

        # Flatten
        out = self.flatten(out)

        # Output MLP
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: MoGATModel,
                         keras_stack: KerasMoGATFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    for i in range(cfg.depthato):
        t_head = torch_model.attention_layers[i]
        k_head = keras_stack.heads[i]

        copy_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
        copy_dense(t_head.alpha_activation[0], k_head.lay_alpha_activation)
        copy_dense(t_head.linear_alpha, k_head.lay_alpha)

        if i == 0 and t_head.use_edge_features:
            copy_dense(t_head.fc1[0], k_head.lay_fc1)
            copy_dense(t_head.fc2[0], k_head.lay_fc2)

        # GRU
        copy_gru_cell(torch_model.gru_layers[i].gru_cell, keras_stack.grus[i])

    # Attentive pooling weights
    for i in range(cfg.depthato):
        t_pool = torch_model.layer_pools[i]
        k_pool = keras_stack.layer_pools[i]
        copy_dense(t_pool.linear_trafo, k_pool.lay_linear_trafo)
        copy_dense(t_pool.lay_alpha[0], k_pool.lay_alpha)
        _copy_gru_cell_torch_to_keras(t_pool.gru, k_pool.lay_gru)

    # Self-attention scale: Torch attn_scale (scalar) -> Keras Attention.scale
    scale_val = torch_model.attn_scale.detach().cpu().numpy()
    keras_stack.self_attention.set_weights([scale_val.reshape(())])

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

    torch_model = MoGATModel(
        node_dim=cfg.node_dim,
        depthato=cfg.depthato,
        depthmol=cfg.depthmol,
        units=cfg.units,
        edge_dim=cfg.edge_dim,
        use_edge_features=True,
        activation="leaky_relu2",
        dropout=0.0,  # no dropout for deterministic comparison
        output_units=cfg.output_units,
        output_activation="relu",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasMoGATFullStack(cfg)
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

    print(f"MoGAT model-level alignment (Torch -> Keras):")
    compare_outputs("MoGAT_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("MoGAT model alignment PASSED.")


if __name__ == "__main__":
    main()
