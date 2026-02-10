#!/usr/bin/env python3
"""Model-level alignment: GIN (full model, Torch -> Keras weight transfer).

Builds both Torch GINModel and a KerasGINFullStack that mirrors the full
architecture (Embedding + Dense + GINConv + GraphMLP per layer +
per-layer PoolingNodes + readout MLP + Dropout + Add + output MLP),
transfers ALL weights, and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import keras as ks
import torch
from dataclasses import dataclass
from typing import List

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, compare_outputs,
    keras_to_torch, make_disjoint_graph,
)

from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.conv import GIN as KerasGIN
from kgcnn.layers.mlp import MLP as KerasMLP, GraphMLP as KerasGraphMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.gin import GINModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    units: int = 32
    gin_mlp_units: list = None
    last_mlp_units: list = None
    output_units: list = None
    num_targets: int = 2
    num_embeddings: int = 95
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.gin_mlp_units is None:
            self.gin_mlp_units = [32, 32]
        if self.last_mlp_units is None:
            self.last_mlp_units = [32, 32, 32]
        if self.output_units is None:
            self.output_units = []


class KerasGINFullStack:
    """Full Keras GIN model stack mirroring GINModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = ks.layers.Dense(cfg.units, activation="linear", use_bias=True)

        # GIN convolutions + per-layer MLPs
        self.convs: List[KerasGIN] = [
            KerasGIN(pooling_method="scatter_sum", epsilon_learnable=False)
            for _ in range(cfg.depth)
        ]
        # gin_mlp: activation = ["relu", "linear"] (last layer linear)
        gin_act = ["relu"] * max(len(cfg.gin_mlp_units) - 1, 0) + ["linear"]
        self.gin_mlps: List[KerasGraphMLP] = [
            KerasGraphMLP(
                units=cfg.gin_mlp_units,
                activation=gin_act,
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            )
            for _ in range(cfg.depth)
        ]

        # Per-layer readout: (depth + 1) MLPs
        self.poolings: List[KerasPoolingNodes] = [
            KerasPoolingNodes(pooling_method="scatter_sum")
            for _ in range(cfg.depth + 1)
        ]
        # last_mlp: activation = ["relu", ..., "linear"]
        last_act = ["relu"] * max(len(cfg.last_mlp_units) - 1, 0) + ["linear"]
        self.readout_mlps: List[KerasMLP] = [
            KerasMLP(units=cfg.last_mlp_units, activation=last_act)
            for _ in range(cfg.depth + 1)
        ]

        # Output MLP
        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["relu"] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, edge_index, batch_id_node, count_nodes):
        """Full forward pass returning graph-level output."""
        n = self.node_embedding(z)
        n = self.dense_in(n)

        list_embeddings = [n]
        for i in range(self.cfg.depth):
            n = self.convs[i]([n, edge_index])
            n = self.gin_mlps[i]([n, batch_id_node, count_nodes])
            list_embeddings.append(n)

        # Per-layer readout: pool -> MLP, then sum
        # No dropout during eval (dropout_rate=0 in config)
        out = None
        for i, emb in enumerate(list_embeddings):
            h = self.poolings[i]([count_nodes, emb, batch_id_node])
            h = self.readout_mlps[i](h)
            if out is None:
                out = h
            else:
                out = out + h

        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: GINModel, keras_stack: KerasGINFullStack, cfg: Config):
    """Transfer ALL weights from Torch model to Keras stack."""
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    for i in range(cfg.depth):
        # GIN epsilon (scalar)
        keras_stack.convs[i].set_weights(
            [torch_model.convs[i].eps.detach().cpu().numpy().reshape(())]
        )
        # GIN per-layer MLP
        copy_mlp(torch_model.gin_mlps[i], keras_stack.gin_mlps[i])

    # Per-layer readout MLPs (depth + 1)
    for i in range(cfg.depth + 1):
        copy_mlp(torch_model.readout_mlps[i], keras_stack.readout_mlps[i])

    # Output MLP
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_edge_attr=False,
    )

    # Build Torch model
    torch_model = GINModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        gin_mlp_units=cfg.gin_mlp_units,
        gin_mlp_activation="relu",
        gin_mlp_use_normalization=False,
        gin_pooling="sum",
        epsilon_learnable=False,
        use_edge_features=False,
        node_pooling="sum",
        last_mlp_units=cfg.last_mlp_units,
        last_mlp_activation="relu",
        dropout_rate=0.0,
        output_units=cfg.output_units,
        output_activation="relu",
        output_final_activation="linear",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    # Build Keras stack (dry run)
    keras_stack = KerasGINFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)

    # Transfer weights
    transfer_all_weights(torch_model, keras_stack, cfg)

    # Forward pass: Torch
    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    # Forward pass: Keras
    keras_out = keras_to_torch(keras_stack.forward(z_k, ei_k, bid_k, cn_k))

    # Compare
    print(f"GIN model-level alignment (Torch -> Keras):")
    compare_outputs("GIN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("GIN model alignment PASSED.")


if __name__ == "__main__":
    main()
