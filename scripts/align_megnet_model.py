#!/usr/bin/env python3
"""Model-level alignment: MEGNet (full model, Torch -> Keras weight transfer).

Builds both Torch MEGNetModel and a KerasMEGNetFullStack that mirrors the full
architecture (Embedding + FF_init + (optional FF + MEGNetBlock + skip)×depth +
PoolingNodes + output MLP), transfers ALL weights, and verifies final output.

NOTE: Simplified version without Set2Set pooling, dropout, or graph embedding.
Uses use_set2set=False and mean pooling for the final readout.
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

from kgcnn.literature.Megnet._layers import MEGnetBlock as KerasMEGnetBlock
from kgcnn.layers.mlp import GraphMLP as KerasGraphMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn_torch.models.megnet import MEGNetModel


@dataclass
class Config:
    node_dim: int = 32
    edge_dim: int = 32
    state_dim: int = 16
    edge_input_dim: int = 8
    depth: int = 3
    block_units_edge: list = None
    block_units_node: list = None
    block_units_state: list = None
    node_ff_units: list = None
    edge_ff_units: list = None
    state_ff_units: list = None
    activation: str = "kgcnn>softplus2"
    num_targets: int = 2
    num_embeddings: int = 95
    output_units: list = None
    n_nodes: int = 20
    n_edges: int = 60
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.node_ff_units is None:
            self.node_ff_units = [32, 16]
        if self.edge_ff_units is None:
            self.edge_ff_units = [32, 16]
        if self.state_ff_units is None:
            self.state_ff_units = [32, 16]
        eff_n = self.node_ff_units[-1]
        eff_e = self.edge_ff_units[-1]
        eff_s = self.state_ff_units[-1]
        if self.block_units_edge is None:
            self.block_units_edge = [32, eff_e, eff_e]
        if self.block_units_node is None:
            self.block_units_node = [32, eff_n, eff_n]
        if self.block_units_state is None:
            self.block_units_state = [32, eff_s, eff_s]
        if self.output_units is None:
            self.output_units = [16, 8]


class KerasMEGNetFullStack:
    """Full Keras MEGNet model stack mirroring MEGNetModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )

        act = cfg.activation

        # Initial FFN projections
        self.node_ff_init = KerasGraphMLP(
            units=cfg.node_ff_units, activation=act,
            use_bias=True, use_normalization=False, use_dropout=False,
        )
        self.edge_ff_init = KerasGraphMLP(
            units=cfg.edge_ff_units, activation=act,
            use_bias=True, use_normalization=False, use_dropout=False,
        )
        self.state_ff_init = KerasMLP(
            units=cfg.state_ff_units, activation=act,
        )

        # Pre-block FFNs for blocks 1..depth-1
        self.node_ffs = []
        self.edge_ffs = []
        self.state_ffs = []
        for _ in range(cfg.depth - 1):
            self.node_ffs.append(KerasGraphMLP(
                units=cfg.node_ff_units, activation=act,
                use_bias=True, use_normalization=False, use_dropout=False,
            ))
            self.edge_ffs.append(KerasGraphMLP(
                units=cfg.edge_ff_units, activation=act,
                use_bias=True, use_normalization=False, use_dropout=False,
            ))
            self.state_ffs.append(KerasMLP(
                units=cfg.state_ff_units, activation=act,
            ))

        # MEGNet blocks
        self.blocks = []
        for _ in range(cfg.depth):
            self.blocks.append(KerasMEGnetBlock(
                node_embed=cfg.block_units_node,
                edge_embed=cfg.block_units_edge,
                env_embed=cfg.block_units_state,
                pooling_method="scatter_mean",
                activation=act,
            ))

        # Final pooling (no Set2Set)
        self.pool_nodes = KerasPoolingNodes(pooling_method="scatter_mean")
        self.pool_edges = KerasPoolingNodes(pooling_method="scatter_mean")

        eff_n = cfg.node_ff_units[-1]
        eff_e = cfg.edge_ff_units[-1]
        eff_s = cfg.state_ff_units[-1]
        final_dim = eff_n + eff_e + eff_s

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [act] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, edge_attr, edge_index,
                batch_id_node, batch_id_edge, count_nodes, count_edges):
        n = self.node_embedding(z)

        # Graph state: zeros (no graph_state input)
        state = torch.zeros(self.cfg.batch_size, self.cfg.state_dim,
                            dtype=torch.float32)

        # Initial FFN
        vp = self.node_ff_init([n, batch_id_node, count_nodes])
        ep = self.edge_ff_init([edge_attr, batch_id_edge, count_edges])
        up = self.state_ff_init(state)

        # External accumulator skip pattern
        vp2 = vp
        ep2 = ep
        up2 = up

        for i in range(self.cfg.depth):
            if i > 0:
                vp2 = self.node_ffs[i - 1]([vp, batch_id_node, count_nodes])
                ep2 = self.edge_ffs[i - 1]([ep, batch_id_edge, count_edges])
                up2 = self.state_ffs[i - 1](up)

            vp2, ep2, up2 = self.blocks[i]([
                vp2, ep2, edge_index, up2,
                batch_id_node, batch_id_edge, count_nodes, count_edges
            ])

            vp = keras.layers.Add()([vp, vp2])
            ep = keras.layers.Add()([ep, ep2])
            up = keras.layers.Add()([up, up2])

        # Final pooling
        pooled_n = self.pool_nodes([count_nodes, vp, batch_id_node])
        pooled_e = self.pool_edges([count_edges, ep, batch_id_edge])
        out = keras.layers.Concatenate(axis=-1)([pooled_n, pooled_e, up])
        out = self.output_mlp(out)
        return out


def _copy_block_weights(torch_block, keras_block):
    """Copy MEGNetBlock weights: 3 Dense layers each for edge, node, state."""
    # Edge update: 3 Dense layers
    copy_dense(torch_block.edge_dense_layers[0], keras_block.lay_phi_e)
    copy_dense(torch_block.edge_dense_layers[1], keras_block.lay_phi_e_1)
    copy_dense(torch_block.edge_dense_layers[2], keras_block.lay_phi_e_2)

    # Node update: 3 Dense layers
    copy_dense(torch_block.node_dense_layers[0], keras_block.lay_phi_n)
    copy_dense(torch_block.node_dense_layers[1], keras_block.lay_phi_n_1)
    copy_dense(torch_block.node_dense_layers[2], keras_block.lay_phi_n_2)

    # State update: 3 Dense layers
    copy_dense(torch_block.state_dense_layers[0], keras_block.lay_phi_u)
    copy_dense(torch_block.state_dense_layers[1], keras_block.lay_phi_u_1)
    copy_dense(torch_block.state_dense_layers[2], keras_block.lay_phi_u_2)


def transfer_all_weights(torch_model: MEGNetModel,
                         keras_stack: KerasMEGNetFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    # Initial FFN
    copy_mlp(torch_model.node_ff_init, keras_stack.node_ff_init)
    copy_mlp(torch_model.edge_ff_init, keras_stack.edge_ff_init)
    copy_mlp(torch_model.state_ff_init, keras_stack.state_ff_init)

    # Pre-block FFNs
    for i in range(cfg.depth - 1):
        copy_mlp(torch_model.node_ffs[i], keras_stack.node_ffs[i])
        copy_mlp(torch_model.edge_ffs[i], keras_stack.edge_ffs[i])
        copy_mlp(torch_model.state_ffs[i], keras_stack.state_ffs[i])

    # MEGNet blocks
    for i in range(cfg.depth):
        _copy_block_weights(torch_model.blocks[i], keras_stack.blocks[i])

    # Output MLP
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_input_dim, seed=cfg.seed,
        include_edge_attr=True,
    )

    torch_model = MEGNetModel(
        node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim,
        state_dim=cfg.state_dim,
        edge_input_dim=cfg.edge_input_dim,
        state_input_dim=0,
        depth=cfg.depth,
        block_units_edge=cfg.block_units_edge,
        block_units_node=cfg.block_units_node,
        block_units_state=cfg.block_units_state,
        node_ff_units=cfg.node_ff_units,
        edge_ff_units=cfg.edge_ff_units,
        state_ff_units=cfg.state_ff_units,
        activation="softplus2",
        has_ff=True,
        dropout=None,
        use_set2set=False,
        node_pooling="mean",
        output_units=cfg.output_units,
        output_activation="softplus2",
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        use_graph_embedding=False,
    )
    torch_model.eval()

    keras_stack = KerasMEGNetFullStack(cfg)
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

    print(f"MEGNet model-level alignment (Torch -> Keras):")
    compare_outputs("MEGNet_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("MEGNet model alignment PASSED.")


if __name__ == "__main__":
    main()
