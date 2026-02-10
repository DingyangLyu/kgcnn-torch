#!/usr/bin/env python3
"""Model-level alignment: MEGAN (full model, Torch -> Keras weight transfer).

Builds both Torch MEGANModel and a KerasMEGANFullStack that mirrors the full
architecture (Embedding + MultiHeadGATV2Layer×depth + edge importance (sigmoid) +
AggregateLocalEdges + node importance MLP + weighted PoolingNodes×K + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Simplified version without edge features, dropout, or explanation loss.
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

from kgcnn.layers.attention import MultiHeadGATV2Layer as KerasMultiHeadGATV2Layer
from kgcnn.layers.aggr import AggregateLocalEdges as KerasAggregateLocalEdges
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.megan import MEGANModel


@dataclass
class Config:
    node_dim: int = 32
    units: list = None
    num_heads: int = 2
    depth: int = 3
    concat_heads: bool = True
    importance_channels: int = 2
    importance_units: list = None
    final_units: list = None
    final_activation: str = "linear"
    final_pooling: str = "sum"
    num_targets: int = 2
    num_embeddings: int = 95
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.units is None:
            self.units = [self.node_dim] * self.depth
        if self.importance_units is None:
            self.importance_units = []
        if self.final_units is None:
            self.final_units = [self.num_targets]


class KerasMEGANFullStack:
    """Full Keras MEGAN model stack mirroring MEGANModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        self.attention_layers = []
        for u in cfg.units:
            self.attention_layers.append(KerasMultiHeadGATV2Layer(
                units=u,
                num_heads=cfg.importance_channels,
                use_edge_features=False,
                activation="kgcnn>leaky_relu2",
                use_bias=True,
                has_self_loops=True,
                concat_heads=cfg.concat_heads,
                normalize_softmax=False,
            ))

        # Edge importance pooling
        self.pool_edges_in = KerasAggregateLocalEdges(
            pooling_method='mean', pooling_index=0)
        self.pool_edges_out = KerasAggregateLocalEdges(
            pooling_method='mean', pooling_index=1)
        self.lay_average = keras.layers.Average()

        # Node importance sub-network
        imp_units = cfg.importance_units + [cfg.importance_channels]
        imp_acts = ['relu'] * len(cfg.importance_units) + ['linear']
        self.node_importance_layers = []
        for u, act in zip(imp_units, imp_acts):
            self.node_importance_layers.append(
                keras.layers.Dense(u, activation=act, use_bias=True))

        self.pooling = KerasPoolingNodes(pooling_method=cfg.final_pooling)

        # Output MLP: Keras MEGAN uses use_bias (constructor param, True) for ALL
        # final Dense layers (the per-layer bias list is prepared but not used).
        final_acts = ['relu'] * len(cfg.final_units)
        final_acts[-1] = cfg.final_activation
        final_biases = [True] * len(cfg.final_units)
        self.final_layers = []
        for u, act, bias in zip(cfg.final_units, final_acts, final_biases):
            self.final_layers.append(
                keras.layers.Dense(u, activation=act, use_bias=bias))

    def forward(self, z, edge_index, batch_id_node, count_nodes):
        node_input = self.node_embedding(z)

        # Dummy edge input (not used since use_edge_features=False)
        n_edges = ops.shape(edge_index)[1]
        edge_input = ops.zeros((n_edges, 1))

        # Attention layers - collect alpha tensors
        x = node_input
        alphas = []
        for lay in self.attention_layers:
            x, alpha = lay([x, edge_input, edge_index])
            alphas.append(alpha)  # alpha: (M, K, 1)

        # Edge importance: concat alphas on axis=-1, sum, sigmoid
        alphas = keras.layers.Concatenate(axis=-1)(alphas)  # (M, K, depth)
        edge_importances = ops.sum(alphas, axis=-1, keepdims=False)  # (M, K)
        edge_importances = keras.layers.Activation("sigmoid")(edge_importances)

        # Pool edge importances to nodes (both directions), average
        pooled_edges_in = self.pool_edges_in(
            [node_input, edge_importances, edge_index])
        pooled_edges_out = self.pool_edges_out(
            [node_input, edge_importances, edge_index])
        pooled_edges = self.lay_average([pooled_edges_out, pooled_edges_in])

        # Node importance: sigmoid(MLP(x)) * pooled_edges
        ni = x
        for lay in self.node_importance_layers:
            ni = lay(ni)
        ni = keras.layers.Activation("sigmoid")(ni)  # (N, K)
        node_importances = ni * pooled_edges  # (N, K)

        # Weighted graph pooling per importance channel
        outs = []
        for k in range(self.cfg.importance_channels):
            w_k = ops.expand_dims(node_importances[:, k], axis=-1)  # (N, 1)
            out = self.pooling([count_nodes, x * w_k, batch_id_node])
            outs.append(out)
        out = keras.layers.Concatenate(axis=-1)(outs)  # (B, F*K)

        # Output MLP
        for lay in self.final_layers:
            out = lay(out)

        return out


def copy_multihead_gatv2(torch_layer, keras_layer, num_heads):
    """Copy MultiHeadGATV2Layer weights."""
    for k in range(num_heads):
        # Torch: head_linears[k] = Sequential(Linear, activation) -> extract [0]
        # Keras: head_layers[k] = (lay_linear, lay_alpha_activation, lay_alpha)
        copy_dense(torch_layer.head_linears[k][0],
                   keras_layer.head_layers[k][0])
        copy_dense(torch_layer.head_alpha_acts[k][0],
                   keras_layer.head_layers[k][1])
        copy_dense(torch_layer.head_alphas[k],
                   keras_layer.head_layers[k][2])


def transfer_all_weights(torch_model: MEGANModel,
                         keras_stack: KerasMEGANFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    for i in range(cfg.depth):
        copy_multihead_gatv2(torch_model.attention_layers[i],
                             keras_stack.attention_layers[i],
                             cfg.num_heads)

    # Node importance MLP: Torch uses MLP, Keras uses list of Dense
    for i in range(len(cfg.importance_units) + 1):
        copy_dense(torch_model.importance_mlp.linears[i],
                   keras_stack.node_importance_layers[i])

    # Output MLP: Torch uses MLP, Keras uses list of Dense
    for i in range(len(cfg.final_units)):
        copy_dense(torch_model.output_mlp.linears[i],
                   keras_stack.final_layers[i])


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_edge_attr=False,
    )

    torch_model = MEGANModel(
        node_dim=cfg.node_dim,
        units=cfg.units,
        num_heads=cfg.num_heads,
        depth=cfg.depth,
        attention_activation="leaky_relu2",
        use_edge_features=False,
        concat_heads=cfg.concat_heads,
        importance_channels=cfg.importance_channels,
        importance_units=cfg.importance_units,
        importance_activation="relu",
        final_units=cfg.final_units,
        final_activation=cfg.final_activation,
        use_bias=True,
        final_pooling=cfg.final_pooling,
        dropout_rate=0.0,
        final_dropout_rate=0.0,
        regression_reference=None,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasMEGANFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ei_k, bid_k, cn_k))

    print("MEGAN model-level alignment (Torch -> Keras):")
    compare_outputs("MEGAN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("MEGAN model alignment PASSED.")


if __name__ == "__main__":
    main()
