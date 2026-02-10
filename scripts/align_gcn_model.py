#!/usr/bin/env python3
"""Model-level alignment: GCN (full model, Torch -> Keras weight transfer).

Builds both Torch GCNModel and a KerasGCNFullStack that mirrors the full
architecture (Embedding + Dense + GCNConv + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output matches.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass

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

from kgcnn.layers.conv import GCN as KerasGCN
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.gcn import GCNModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    gcn_units: int = 32
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    output_activation: str = "relu"
    output_final_activation: str = "linear"
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = [32, 16]


class KerasGCNFullStack:
    """Full Keras GCN model stack mirroring GCNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim
        )
        self.dense_in = keras.layers.Dense(cfg.gcn_units, activation="linear", use_bias=True)
        self.convs = [
            KerasGCN(
                units=cfg.gcn_units,
                pooling_method="scatter_sum",
                activation="kgcnn>leaky_relu2",
                use_bias=True,
                normalize_by_weights=False,
            )
            for _ in range(cfg.depth)
        ]
        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + [cfg.output_final_activation]
        # Match Torch: use_bias=[True,...,True,False]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
            use_bias=out_bias,
        )

    def forward(self, z, edge_weight, edge_index, batch_id_node, count_nodes):
        """Full forward pass returning graph-level output."""
        n = self.node_embedding(z)
        n = self.dense_in(n)
        for conv in self.convs:
            n = conv([n, edge_weight, edge_index])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: GCNModel, keras_stack: KerasGCNFullStack, cfg: Config):
    """Transfer ALL weights from Torch model to Keras stack."""
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth):
        copy_dense(torch_model.convs[i].linear, keras_stack.convs[i].layer_dense)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    # Build test data
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_edge_attr=False,
    )

    # Build Torch model
    torch_model = GCNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        gcn_units=cfg.gcn_units,
        gcn_activation="leaky_relu2",
        gcn_pooling="sum",
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        output_final_activation=cfg.output_final_activation,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    # Build Keras stack (dry run to initialize weights)
    keras_stack = KerasGCNFullStack(cfg)
    z_k = keras_data["z"]
    ew_k = keras_data["edge_weight"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ew_k, ei_k, bid_k, cn_k)

    # Transfer weights
    transfer_all_weights(torch_model, keras_stack, cfg)

    # Forward pass: Torch
    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    # Forward pass: Keras
    keras_out = keras_to_torch(keras_stack.forward(z_k, ew_k, ei_k, bid_k, cn_k))

    # Compare
    print(f"GCN model-level alignment (Torch -> Keras):")
    compare_outputs("GCN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("GCN model alignment PASSED.")


if __name__ == "__main__":
    main()
