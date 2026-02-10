#!/usr/bin/env python3
"""Model-level alignment: EGNN (full model, Torch -> Keras weight transfer).

Builds both Torch EGNNModel and a KerasEGNNFullStack that mirrors the full
architecture (Embedding + optional dense_in + EGNNLayer×depth (edge_mlp +
coord_mlp + node_mlp + aggr) + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Simplified version without attention, normalization, or node_attributes.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
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

from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.geom import NodePosition, EuclideanNorm
from kgcnn.layers.mlp import GraphMLP as KerasGraphMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.egnn import EGNNModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    units: int = 32
    edge_mlp_units: list = None
    coord_mlp_units: list = None
    node_mlp_units: list = None
    edge_mlp_activation: str = "swish"
    coord_mlp_activation: str = "swish"
    node_mlp_activation: str = "swish"
    edge_dim: int = 8
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.edge_mlp_units is None:
            self.edge_mlp_units = [32, 32]
        if self.coord_mlp_units is None:
            self.coord_mlp_units = [32, 1]
        if self.node_mlp_units is None:
            self.node_mlp_units = [32, 32]
        if self.output_units is None:
            self.output_units = [32]


class KerasEGNNFullStack:
    """Full Keras EGNN model stack mirroring EGNNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        # Per-layer edge, coord, node MLPs
        edge_act = [cfg.edge_mlp_activation] * max(len(cfg.edge_mlp_units) - 1, 0) + ["linear"]
        coord_act = [cfg.coord_mlp_activation] * max(len(cfg.coord_mlp_units) - 1, 0) + ["linear"]
        node_act = [cfg.node_mlp_activation] * max(len(cfg.node_mlp_units) - 1, 0) + ["linear"]

        self.edge_mlps = []
        self.coord_mlps = []
        self.node_mlps = []
        self.gather_nodes = []
        self.node_positions = []
        self.euclidean_norms = []
        self.aggr_msgs = []
        self.aggr_coords = []
        self.concats_edge = []
        self.concats_node = []
        self.adds = []
        self.multiplys = []

        for _ in range(cfg.depth):
            self.edge_mlps.append(KerasGraphMLP(
                units=cfg.edge_mlp_units,
                activation=edge_act,
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))
            self.coord_mlps.append(KerasGraphMLP(
                units=cfg.coord_mlp_units,
                activation=coord_act,
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))
            self.node_mlps.append(KerasGraphMLP(
                units=cfg.node_mlp_units,
                activation=node_act,
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))
            self.gather_nodes.append(GatherNodes([0, 1], concat_axis=None))
            self.node_positions.append(NodePosition())
            self.euclidean_norms.append(EuclideanNorm(keepdims=True, axis=-1))
            self.aggr_msgs.append(AggregateLocalEdges(pooling_method="scatter_sum"))
            self.aggr_coords.append(AggregateLocalEdges(pooling_method="scatter_mean"))
            self.concats_edge.append(keras.layers.Concatenate(axis=-1))
            self.concats_node.append(keras.layers.Concatenate(axis=-1))
            self.adds.append(keras.layers.Add())
            self.multiplys.append(keras.layers.Multiply())

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.edge_mlp_activation] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
        )

    def forward(self, z, pos, edge_attr, edge_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        h = self.node_embedding(z)

        for i in range(self.cfg.depth):
            # Gather node pairs
            h_i, h_j = self.gather_nodes[i]([h, edge_index])

            # Compute coordinate differences and distances
            pos1, pos2 = self.node_positions[i]([pos, edge_index])
            diff_x = keras.layers.Subtract()([pos1, pos2])
            norm_x = self.euclidean_norms[i](diff_x)

            # Edge model: [h_i, h_j, norm_x, edge_attr]
            edge_input = self.concats_edge[i]([h_i, h_j, norm_x, edge_attr])
            m_ij = self.edge_mlps[i]([edge_input, batch_id_edge, count_edges])

            # Coordinate model
            coord_weight = self.coord_mlps[i]([m_ij, batch_id_edge, count_edges])
            coord_msg = self.multiplys[i]([coord_weight, diff_x])
            coord_agg = self.aggr_coords[i]([h, coord_msg, edge_index])
            pos = keras.layers.Add()([pos, coord_agg])

            # Node model
            m_agg = self.aggr_msgs[i]([h, m_ij, edge_index])
            node_input = self.concats_node[i]([h, m_agg])
            node_update = self.node_mlps[i]([node_input, batch_id_node, count_nodes])

            # Residual
            h = self.adds[i]([h, node_update])

        out = self.pooling([count_nodes, h, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: EGNNModel,
                         keras_stack: KerasEGNNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    for i in range(cfg.depth):
        t_layer = torch_model.layers[i]
        copy_mlp(t_layer.edge_mlp, keras_stack.edge_mlps[i])
        copy_mlp(t_layer.coord_mlp, keras_stack.coord_mlps[i])
        copy_mlp(t_layer.node_mlp, keras_stack.node_mlps[i])

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_edge_attr=True, include_pos=True,
    )

    torch_model = EGNNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        edge_mlp_units=cfg.edge_mlp_units,
        edge_mlp_activation=cfg.edge_mlp_activation,
        coord_mlp_units=cfg.coord_mlp_units,
        coord_mlp_activation=cfg.coord_mlp_activation,
        node_mlp_units=cfg.node_mlp_units,
        node_mlp_activation=cfg.node_mlp_activation,
        use_edge_attr=True,
        edge_attr_dim=cfg.edge_dim,
        use_attention=False,
        use_normalize=False,
        use_skip=True,
        use_node_attributes=False,
        use_node_normalization=False,
        layer_pooling="sum",
        coord_pooling="mean",
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation=cfg.edge_mlp_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasEGNNFullStack(cfg)
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

    print(f"EGNN model-level alignment (Torch -> Keras):")
    compare_outputs("EGNN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("EGNN model alignment PASSED.")


if __name__ == "__main__":
    main()
