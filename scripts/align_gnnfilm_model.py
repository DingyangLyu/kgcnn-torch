#!/usr/bin/env python3
"""Model-level alignment: GNNFilm (full model, Torch -> Keras weight transfer).

Builds both Torch GNNFilmModel and a KerasGNNFilmFullStack that mirrors the full
architecture (Embedding + per-layer (GatherNodes + 3x RelationalDense + Multiply +
Add + AggregateLocal + Activation) + PoolingNodes + output MLP), transfers ALL
weights, and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_embedding, copy_mlp, compare_outputs,
    keras_to_torch, make_disjoint_graph_relational,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.relational import RelationalDense as KerasRelationalDense
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.gnnfilm import GNNFilmModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    units: int = 32  # must equal node_dim to avoid dense_in
    num_relations: int = 5
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


class KerasGNNFilmFullStack:
    """Full Keras GNNFilm model stack mirroring GNNFilmModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        self.gather = GatherNodes(split_indices=[0, 1], concat_axis=None)

        # Per-layer: 3 RelationalDense (gamma, beta, hj) + Activation
        self.gammas = []
        self.betas = []
        self.hjs = []
        self.activations = []
        self.aggrs = []
        for _ in range(cfg.depth):
            self.gammas.append(KerasRelationalDense(
                units=cfg.units,
                num_relations=cfg.num_relations,
                activation="sigmoid",
                use_bias=True,
            ))
            self.betas.append(KerasRelationalDense(
                units=cfg.units,
                num_relations=cfg.num_relations,
                activation="sigmoid",
                use_bias=True,
            ))
            self.hjs.append(KerasRelationalDense(
                units=cfg.units,
                num_relations=cfg.num_relations,
                activation=None,
                use_bias=True,
            ))
            self.activations.append(keras.layers.Activation("swish"))
            self.aggrs.append(AggregateLocalEdges(pooling_method="scatter_sum"))

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
        )

    def forward(self, z, edge_index, edge_type, batch_id_node, count_nodes):
        n = self.node_embedding(z)

        for i in range(self.cfg.depth):
            n_i, n_j = self.gather([n, edge_index])
            gamma = self.gammas[i]([n_i, edge_type])
            beta = self.betas[i]([n_i, edge_type])
            h_j = self.hjs[i]([n_j, edge_type])
            m = keras.layers.Multiply()([h_j, gamma])
            m = keras.layers.Add()([m, beta])
            h = self.aggrs[i]([n, m, edge_index])
            n = self.activations[i](h)

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def _copy_relational_dense_torch_to_keras(torch_rd, keras_rd):
    """Copy Torch RelationalDense weights to Keras RelationalDense.

    Torch: .weight (R, in, out) or .bases/.comps, .bias
    Keras: .kernel (R, in, out), .bias
    No transpose needed — shapes match.
    """
    w = torch_rd.weight.detach().cpu().numpy()
    weights = [w]
    if torch_rd.bias is not None:
        weights.append(torch_rd.bias.detach().cpu().numpy())
    keras_rd.set_weights(weights)


def transfer_all_weights(torch_model: GNNFilmModel,
                         keras_stack: KerasGNNFilmFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    for i in range(cfg.depth):
        t_layer = torch_model.film_layers[i]
        _copy_relational_dense_torch_to_keras(t_layer.rel_dense_gamma, keras_stack.gammas[i])
        _copy_relational_dense_torch_to_keras(t_layer.rel_dense_beta, keras_stack.betas[i])
        _copy_relational_dense_torch_to_keras(t_layer.rel_dense_hj, keras_stack.hjs[i])

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph_relational(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, num_relations=cfg.num_relations,
        seed=cfg.seed, include_edge_weight=False,
    )

    torch_model = GNNFilmModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        num_relations=cfg.num_relations,
        activation="swish",
        modulation_activation="sigmoid",
        film_pooling="sum",
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasGNNFilmFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    et_k = keras_data["edge_type"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, et_k, bid_k, cn_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ei_k, et_k, bid_k, cn_k))

    print(f"GNNFilm model-level alignment (Torch -> Keras):")
    compare_outputs("GNNFilm_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("GNNFilm model alignment PASSED.")


if __name__ == "__main__":
    main()
