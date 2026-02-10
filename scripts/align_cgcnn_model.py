#!/usr/bin/env python3
"""Model-level alignment: CGCNN (full model, Torch -> Keras weight transfer).

Builds both Torch CGCNNModel and a KerasCGCNNFullStack that mirrors the full
architecture (Embedding + Dense + GaussBasis + CGCNNLayer x depth +
PoolingNodes + output MLP), transfers ALL weights, and verifies final output.
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

from kgcnn.layers.geom import GaussBasisLayer as KerasGaussBasisLayer
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.literature.CGCNN._layers import CGCNNLayer as KerasCGCNNLayer
from kgcnn_torch.models.cgcnn import CGCNNModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    conv_units: int = 32
    gauss_bins: int = 20
    gauss_distance: float = 5.0
    gauss_offset: float = 0.0
    gauss_sigma: float = 0.4
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


class KerasCGCNNFullStack:
    """Full Keras CGCNN model stack mirroring CGCNNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = keras.layers.Dense(cfg.conv_units, activation="linear", use_bias=True)

        self.gauss_basis = KerasGaussBasisLayer(
            bins=cfg.gauss_bins,
            distance=cfg.gauss_distance,
            offset=cfg.gauss_offset,
            sigma=cfg.gauss_sigma,
        )

        self.convs = []
        for _ in range(cfg.depth):
            self.convs.append(KerasCGCNNLayer(
                units=cfg.conv_units,
                activation_s="softplus",
                activation_out="softplus",
                batch_normalization=False,
                use_bias=True,
                pooling_method="scatter_mean",
            ))

        self.pooling = KerasPoolingNodes(pooling_method="scatter_mean")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["linear"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
            use_bias=out_bias,
        )

    def forward(self, z, edge_distances, edge_index, batch_id_node, batch_id_edge,
                count_nodes, count_edges):
        n = self.node_embedding(z)
        n = self.dense_in(n)

        edge_features = self.gauss_basis(edge_distances)

        for i in range(self.cfg.depth):
            n = self.convs[i]([
                n, edge_features, edge_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges,
            ])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: CGCNNModel,
                         keras_stack: KerasCGCNNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    for i in range(cfg.depth):
        t = torch_model.convs[i]
        k = keras_stack.convs[i]
        # CGCNN: linear_gate -> f (sigmoid gate), linear_filter -> s (softplus)
        copy_dense(t.linear_gate, k.f)
        copy_dense(t.linear_filter, k.s)

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_edge_attr=True, include_pos=False,
    )

    # Use edge_attr as distances (scalar per edge)
    # Generate positive distances for Gaussian basis
    torch.manual_seed(cfg.seed + 1)
    distances = torch.rand(torch_data.edge_index.shape[1], 1) * cfg.gauss_distance
    torch_data.edge_attr = distances
    keras_data["edge_attr"] = distances

    torch_model = CGCNNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        conv_units=cfg.conv_units,
        gauss_bins=cfg.gauss_bins,
        gauss_distance=cfg.gauss_distance,
        gauss_offset=cfg.gauss_offset,
        gauss_sigma=cfg.gauss_sigma,
        node_pooling="mean",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasCGCNNFullStack(cfg)
    z_k = keras_data["z"]
    dist_k = keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    bie_k = keras_data["batch_id_edge"]
    cn_k = keras_data["count_nodes"]
    ce_k = keras_data["count_edges"]
    _ = keras_stack.forward(z_k, dist_k, ei_k, bid_k, bie_k, cn_k, ce_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, dist_k, ei_k, bid_k, bie_k, cn_k, ce_k))

    print(f"CGCNN model-level alignment (Torch -> Keras):")
    compare_outputs("CGCNN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("CGCNN model alignment PASSED.")


if __name__ == "__main__":
    main()
