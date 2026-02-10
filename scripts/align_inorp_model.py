#!/usr/bin/env python3
"""Model-level alignment: INorp (full model, Torch -> Keras weight transfer).

Builds both Torch INorpModel and a KerasINorpFullStack that mirrors the full
architecture (Embedding + per-layer
(GatherOut + GatherIn + Concat + edge_mlp + Aggr + Concat + node_mlp) +
PoolingNodes + output MLP), transfers ALL weights, and verifies final output.

NOTE: Simplified version without graph state or Set2Set pooling.
Uses use_edge_embedding=False so edge features are float tensors.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass
from typing import List

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

from kgcnn.layers.gather import GatherNodesIngoing, GatherNodesOutgoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.mlp import GraphMLP as KerasGraphMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.inorp import INorpModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    edge_dim: int = 8
    edge_mlp_units: list = None
    node_mlp_units: list = None
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    output_activation: str = "relu"
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.edge_mlp_units is None:
            self.edge_mlp_units = [32, 32]
        if self.node_mlp_units is None:
            self.node_mlp_units = [32, 32]
        if self.output_units is None:
            self.output_units = [32, 16]


class KerasINorpFullStack:
    """Full Keras INorp model stack mirroring INorpModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        self.gather_in = GatherNodesIngoing()
        self.gather_out = GatherNodesOutgoing()
        self.concat = keras.layers.Concatenate(axis=-1)

        # Match Torch's activation normalization: ["relu", ..., "linear"]
        edge_act = ["relu"] * (len(cfg.edge_mlp_units) - 1) + ["linear"]
        node_act = ["relu"] * (len(cfg.node_mlp_units) - 1) + ["linear"]

        # Compute node dim per layer (matches Torch's variable node_dim logic)
        node_dim0 = cfg.node_dim
        effective_node_dim = cfg.node_mlp_units[-1] if cfg.node_mlp_units else cfg.node_dim
        edge_out_dim = cfg.edge_mlp_units[-1] if cfg.edge_mlp_units else effective_node_dim

        self.edge_mlps: List[KerasGraphMLP] = []
        self.node_mlps: List[KerasGraphMLP] = []
        self.aggrs: List[AggregateLocalEdges] = []
        for i in range(cfg.depth):
            self.edge_mlps.append(KerasGraphMLP(
                units=cfg.edge_mlp_units,
                activation=edge_act,
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
            self.aggrs.append(AggregateLocalEdges(pooling_method="scatter_mean"))

        self.pooling = KerasPoolingNodes(pooling_method="scatter_mean")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["sigmoid"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(
            units=out_units,
            activation=out_act,
            use_bias=out_bias,
        )

    def forward(self, z, edge_attr, edge_index, batch_id_node, batch_id_edge,
                count_nodes, count_edges):
        n = self.node_embedding(z)
        ed = edge_attr

        for i in range(self.cfg.depth):
            # Gather node pairs: source (outgoing) and target (ingoing)
            eu1 = self.gather_in([n, edge_index])   # target
            eu2 = self.gather_out([n, edge_index])   # source
            # Concat order matches Torch: [source, target, edge]
            eu = self.concat([self.concat([eu2, eu1]), ed])
            eu = self.edge_mlps[i]([eu, batch_id_edge, count_edges])
            # Aggregate to target nodes
            nu = self.aggrs[i]([n, eu, edge_index])
            # Node update: [node, aggregated]
            nu = self.concat([n, nu])
            n = self.node_mlps[i]([nu, batch_id_node, count_nodes])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: INorpModel,
                         keras_stack: KerasINorpFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    for i in range(cfg.depth):
        copy_mlp(torch_model.blocks[i]["edge_mlp"], keras_stack.edge_mlps[i])
        copy_mlp(torch_model.blocks[i]["node_mlp"], keras_stack.node_mlps[i])

    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_edge_attr=True,
    )

    torch_model = INorpModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        edge_dim=cfg.edge_dim,
        edge_mlp_units=cfg.edge_mlp_units,
        edge_mlp_activation="relu",
        node_mlp_units=cfg.node_mlp_units,
        node_mlp_activation="relu",
        message_pooling="mean",
        use_set2set=False,
        use_graph_state=False,
        node_pooling="mean",
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        output_final_activation="sigmoid",
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        use_edge_embedding=False,
    )
    torch_model.eval()

    keras_stack = KerasINorpFullStack(cfg)
    z_k = keras_data["z"]
    ea_k = keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    bie_k = keras_data["batch_id_edge"]
    cn_k = keras_data["count_nodes"]
    ce_k = keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k))

    print(f"INorp model-level alignment (Torch -> Keras):")
    compare_outputs("INorp_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("INorp model alignment PASSED.")


if __name__ == "__main__":
    main()
