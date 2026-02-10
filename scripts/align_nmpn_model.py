#!/usr/bin/env python3
"""Model-level alignment: NMPN (full model, Torch -> Keras weight transfer).

Builds both Torch NMPNModel and a KerasNMPNFullStack that mirrors the full
architecture (Embedding + dense_in + 2×(edge MLP + TrafoEdgeNetMessages) +
(GatherNodes + MatMulMessages + Concatenate + Aggregate + GRUUpdate)×depth
+ Concatenate(n0, n) + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Simplified version without Set2Set pooling.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, copy_gru_cell,
    compare_outputs, keras_to_torch, make_disjoint_graph,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.gather import GatherNodesOutgoing, GatherNodesIngoing
from kgcnn.layers.aggr import AggregateLocalEdges
from kgcnn.layers.message import MatMulMessages as KerasMatMulMessages
from kgcnn.layers.mlp import GraphMLP as KerasGraphMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.update import GRUUpdate as KerasGRUUpdate
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.literature.NMPN._layers import TrafoEdgeNetMessages as KerasTrafoEdgeNet
from kgcnn_torch.models.nmpn import NMPNModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 3
    units: int = 32
    edge_dim: int = 8
    edge_mlp_units: list = None
    edge_mlp_activation: str = "swish"
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.edge_mlp_units is None:
            self.edge_mlp_units = [32, 32, 32]
        if self.output_units is None:
            self.output_units = [16, 8]


class KerasNMPNFullStack:
    """Full Keras NMPN model stack mirroring NMPNModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)

        # Two edge network paths (in and out)
        # Torch MLP broadcasts scalar activation to ALL layers
        edge_act = cfg.edge_mlp_activation
        self.edge_mlp_in = KerasGraphMLP(
            units=cfg.edge_mlp_units, activation=edge_act,
            use_bias=True, use_normalization=False, use_dropout=False,
        )
        self.edge_trafo_in = KerasTrafoEdgeNet(
            target_shape=(cfg.units, cfg.units),
        )
        self.edge_mlp_out = KerasGraphMLP(
            units=cfg.edge_mlp_units, activation=cfg.edge_mlp_activation,
            use_bias=True, use_normalization=False, use_dropout=False,
        )
        self.edge_trafo_out = KerasTrafoEdgeNet(
            target_shape=(cfg.units, cfg.units),
        )

        self.gather_out = GatherNodesOutgoing()
        self.gather_in = GatherNodesIngoing()
        self.matmul_msg = KerasMatMulMessages()
        self.aggr = AggregateLocalEdges(pooling_method="scatter_sum")

        self.gru = KerasGRUUpdate(units=cfg.units)

        self.concat_final = keras.layers.Concatenate(axis=-1)
        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = ["selu"] * len(cfg.output_units) + ["sigmoid"]
        out_bias = [True] * len(cfg.output_units) + [False]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act, use_bias=out_bias)

    def forward(self, z, edge_attr, edge_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        n0 = self.node_embedding(z)
        n = self.dense_in(n0)

        # Compute edge networks once (reused each step)
        edge_net_in = self.edge_trafo_in(
            self.edge_mlp_in([edge_attr, batch_id_edge, count_edges])
        )
        edge_net_out = self.edge_trafo_out(
            self.edge_mlp_out([edge_attr, batch_id_edge, count_edges])
        )

        for _ in range(self.cfg.depth):
            # Gather node features for each edge
            n_in = self.gather_out([n, edge_index])   # source (outgoing) nodes
            n_out = self.gather_in([n, edge_index])   # target (ingoing) nodes

            # Matrix multiply messages
            m_in = self.matmul_msg([edge_net_in, n_in])
            m_out = self.matmul_msg([edge_net_out, n_out])

            # Concatenate both directions
            eu = keras.layers.Concatenate(axis=-1)([m_in, m_out])

            # Aggregate to target nodes
            agg = self.aggr([n, eu, edge_index])

            # GRU update
            n = self.gru([n, agg])

        # Concatenate initial embeddings
        n = self.concat_final([n0, n])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: NMPNModel,
                         keras_stack: KerasNMPNFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    # Edge MLPs
    copy_mlp(torch_model.edge_mlp_in, keras_stack.edge_mlp_in)
    copy_mlp(torch_model.edge_mlp_out, keras_stack.edge_mlp_out)

    # TrafoEdgeNetMessages Dense layers
    copy_dense(torch_model.edge_trafo_in.dense, keras_stack.edge_trafo_in.lay_dense)
    copy_dense(torch_model.edge_trafo_out.dense, keras_stack.edge_trafo_out.lay_dense)

    # GRU update
    copy_gru_cell(torch_model.gru_update.gru_cell, keras_stack.gru)

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

    torch_model = NMPNModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        edge_dim=cfg.edge_dim,
        edge_mlp_units=cfg.edge_mlp_units,
        edge_mlp_activation=cfg.edge_mlp_activation,
        message_pooling="sum",
        use_set2set=False,
        node_pooling="sum",
        output_units=cfg.output_units,
        output_activation="selu",
        output_final_activation="sigmoid",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
    )
    torch_model.eval()

    keras_stack = KerasNMPNFullStack(cfg)
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

    print(f"NMPN model-level alignment (Torch -> Keras):")
    compare_outputs("NMPN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("NMPN model alignment PASSED.")


if __name__ == "__main__":
    main()
