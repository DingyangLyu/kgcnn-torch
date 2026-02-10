#!/usr/bin/env python3
"""Model-level alignment: SchNet (full model, Torch -> Keras weight transfer).

Builds both Torch SchNetModel and a KerasSchNetFullStack that mirrors the full
architecture (Embedding + GaussBasis + Dense + SchNetInteraction + GraphMLP +
PoolingNodes + output MLP), transfers ALL weights, and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, copy_graph_mlp,
    compare_outputs, keras_to_torch, make_disjoint_graph,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.conv import SchNetInteraction as KerasSchNetInteraction
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, GaussBasisLayer as KerasGaussBasis
from kgcnn.layers.mlp import MLP as KerasMLP, GraphMLP as KerasGraphMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.schnet import SchNetModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    units: int = 32
    gauss_bins: int = 20
    gauss_distance: float = 4.0
    gauss_sigma: float = 0.4
    gauss_offset: float = 0.0
    num_targets: int = 2
    num_embeddings: int = 95
    last_mlp_units: list = None
    output_units: list = None
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.last_mlp_units is None:
            self.last_mlp_units = [32, 32]
        if self.output_units is None:
            self.output_units = [32]


class KerasSchNetFullStack:
    """Full Keras SchNet model stack mirroring SchNetModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim
        )
        self.node_position = NodePosition()
        self.node_distance = NodeDistanceEuclidean()
        self.gauss_basis = KerasGaussBasis(
            bins=cfg.gauss_bins, distance=cfg.gauss_distance,
            sigma=cfg.gauss_sigma, offset=cfg.gauss_offset,
        )
        self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.interactions = [
            KerasSchNetInteraction(
                units=cfg.units,
                cfconv_pool="scatter_sum",
                activation={"class_name": "function", "config": "kgcnn>shifted_softplus"},
                use_bias=True,
            )
            for _ in range(cfg.depth)
        ]
        # last_mlp (applied per-node before pooling) - GraphMLP in Keras
        last_act = ["kgcnn>shifted_softplus"] * len(cfg.last_mlp_units)
        self.last_mlp = KerasGraphMLP(
            units=cfg.last_mlp_units,
            activation=last_act,
        )
        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")
        # output_mlp (applied after pooling) - MLP in Keras
        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["kgcnn>shifted_softplus"] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
        )

    def forward(self, z, pos, edge_index, batch_id_node, count_nodes):
        """Full forward pass returning graph-level output."""
        n = self.node_embedding(z)

        # Compute distances
        pos1, pos2 = self.node_position([pos, edge_index])
        ed = self.node_distance([pos1, pos2])
        ed = self.gauss_basis(ed)

        # Core model
        n = self.dense_in(n)
        for interaction in self.interactions:
            n = interaction([n, ed, edge_index])

        n = self.last_mlp([n, batch_id_node, count_nodes])

        # Graph-level output
        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: SchNetModel, keras_stack: KerasSchNetFullStack, cfg: Config):
    """Transfer ALL weights from Torch model to Keras stack."""
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    for i in range(cfg.depth):
        t = torch_model.interactions[i]
        k = keras_stack.interactions[i]
        copy_dense(t.dense_in, k.lay_dense1)
        copy_dense(t.cfconv.dense1, k.lay_cfconv.lay_dense1)
        copy_dense(t.cfconv.dense2, k.lay_cfconv.lay_dense2)
        copy_dense(t.dense1, k.lay_dense2)
        copy_dense(t.dense2, k.lay_dense3)

    copy_mlp(torch_model.last_mlp, keras_stack.last_mlp)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    # Build test data (with positions for distance computation)
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False,
    )

    # Build Torch model
    torch_model = SchNetModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        gauss_bins=cfg.gauss_bins,
        gauss_distance=cfg.gauss_distance,
        gauss_sigma=cfg.gauss_sigma,
        gauss_offset=cfg.gauss_offset,
        interaction_activation="shifted_softplus",
        interaction_pooling="sum",
        node_pooling="sum",
        last_mlp_units=cfg.last_mlp_units,
        last_mlp_activation="shifted_softplus",
        output_units=cfg.output_units,
        output_activation="shifted_softplus",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        make_distance=True,
        expand_distance=True,
        use_output_mlp=True,
    )
    torch_model.eval()

    # Build Keras stack (dry run)
    keras_stack = KerasSchNetFullStack(cfg)
    z_k = keras_data["z"]
    pos_k = keras_data["pos"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, bid_k, cn_k)

    # Transfer weights
    transfer_all_weights(torch_model, keras_stack, cfg)

    # Forward pass: Torch
    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    # Forward pass: Keras
    keras_out = keras_to_torch(keras_stack.forward(z_k, pos_k, ei_k, bid_k, cn_k))

    # Compare
    print(f"SchNet model-level alignment (Torch -> Keras):")
    compare_outputs("SchNet_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("SchNet model alignment PASSED.")


if __name__ == "__main__":
    main()
