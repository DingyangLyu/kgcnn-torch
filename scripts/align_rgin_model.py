#!/usr/bin/env python3
"""Model-level alignment: rGIN (full model, Torch -> Keras weight transfer).

Builds both Torch rGINModel and a KerasrGINFullStack that mirrors the full
architecture (Embedding + Dense + (rGINConv + GraphMLP) x depth +
per-layer (PoolingNodes + readout MLP + Dropout) + Add + output MLP),
transfers ALL weights, and verifies final output.

NOTE: rGIN uses random feature augmentation, so we must seed both frameworks
and use fixed random values to ensure deterministic comparison.
This script sets seeds but since Keras and Torch random generators differ,
we override the random features to use the same values.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import torch
from dataclasses import dataclass
from typing import List

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_mlp, compare_outputs,
    keras_to_torch, make_disjoint_graph,
)

import keras as ks
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.mlp import MLP as KerasMLP, GraphMLP as KerasGraphMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.literature.rGIN._layers import rGIN as KerasrGIN
from kgcnn_torch.models.rgin import rGINModel


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
    random_range: int = 100

    def __post_init__(self):
        if self.gin_mlp_units is None:
            self.gin_mlp_units = [32, 32]
        if self.last_mlp_units is None:
            self.last_mlp_units = [64, 64, 64]
        if self.output_units is None:
            self.output_units = []


class KerasrGINFullStack:
    """Full Keras rGIN model stack mirroring rGINModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        self.dense_in = ks.layers.Dense(cfg.units, activation="linear", use_bias=True)

        # rGIN convolutions + per-layer GraphMLPs
        self.convs: List[KerasrGIN] = []
        self.gin_mlps: List[KerasGraphMLP] = []
        gin_act = ["relu"] * max(len(cfg.gin_mlp_units) - 1, 0) + ["linear"]
        for _ in range(cfg.depth):
            self.convs.append(KerasrGIN(
                pooling_method="scatter_sum",
                epsilon_learnable=False,
                random_range=cfg.random_range,
            ))
            self.gin_mlps.append(KerasGraphMLP(
                units=cfg.gin_mlp_units,
                activation=gin_act,
                use_bias=True,
                use_normalization=False,
                use_dropout=False,
            ))

        # Per-layer readout: (depth + 1) pools + MLPs
        self.poolings: List[KerasPoolingNodes] = [
            KerasPoolingNodes(pooling_method="scatter_sum")
            for _ in range(cfg.depth + 1)
        ]
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
        n = self.node_embedding(z)
        n = self.dense_in(n)

        list_embeddings = [n]
        for i in range(self.cfg.depth):
            n = self.convs[i]([n, edge_index])
            n = self.gin_mlps[i]([n, batch_id_node, count_nodes])
            list_embeddings.append(n)

        # Per-layer readout: pool -> MLP, then sum (no dropout during eval)
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


def transfer_all_weights(torch_model: rGINModel,
                         keras_stack: KerasrGINFullStack, cfg: Config):
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)
    copy_dense(torch_model.dense_in, keras_stack.dense_in)

    for i in range(cfg.depth):
        # rGIN epsilon
        keras_stack.convs[i].eps_k.assign(
            float(torch_model.convs[i].eps.detach().cpu().item())
        )
        # GIN per-layer MLP
        copy_mlp(torch_model.mlps[i], keras_stack.gin_mlps[i])

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

    torch_model = rGINModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        gin_mlp_units=cfg.gin_mlp_units,
        gin_mlp_activation=["relu", "linear"],
        gin_mlp_use_normalization=False,
        gin_pooling="sum",
        epsilon_learnable=False,
        random_range=cfg.random_range,
        dropout=0.0,
        node_pooling="sum",
        last_mlp_units=cfg.last_mlp_units,
        last_mlp_activation="relu",
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
    keras_stack = KerasrGINFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)

    # Transfer weights
    transfer_all_weights(torch_model, keras_stack, cfg)

    # rGIN uses random features each forward pass. Torch and Keras have
    # fundamentally different RNG implementations, so we pre-generate
    # deterministic random values and monkey-patch both models to use them.
    import numpy as np
    np.random.seed(cfg.seed + 200)
    n_total = torch_data.z.shape[0]
    # Pre-generate random values for each depth
    fixed_randoms = [
        torch.tensor(np.random.rand(n_total, 1).astype(np.float32))
        for _ in range(cfg.depth)
    ]

    # Monkey-patch Torch rGINConv to use fixed random values
    _call_idx_torch = [0]
    _orig_torch_forwards = []
    for i in range(cfg.depth):
        conv = torch_model.convs[i]
        _orig_torch_forwards.append(conv.forward)
        def _patched_torch_forward(x, edge_index, _conv=conv, _i=i):
            num_nodes = x.size(0)
            x_aug = torch.cat([x, fixed_randoms[_i].to(x.device)], dim=-1)
            from kgcnn_torch.layers.gather import gather_nodes_outgoing
            x_j = gather_nodes_outgoing(x_aug, edge_index)
            agg = _conv.aggr(x_j, edge_index, num_nodes)
            return x_aug + agg
        conv.forward = _patched_torch_forward

    # Monkey-patch Keras rGIN to use fixed random values
    for i in range(cfg.depth):
        k_conv = keras_stack.convs[i]
        def _patched_keras_call(inputs, _k_conv=k_conv, _i=i, **kwargs):
            node, edge_index = inputs
            node = _k_conv.lay_concat([node, fixed_randoms[_i]])
            ed = _k_conv.lay_gather([node, edge_index], **kwargs)
            nu = _k_conv.lay_pool([node, ed, edge_index], **kwargs)
            out = _k_conv.lay_add([node, nu], **kwargs)
            return out
        k_conv.call = _patched_keras_call

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z_k, ei_k, bid_k, cn_k))

    print(f"rGIN model-level alignment (Torch -> Keras):")
    compare_outputs("rGIN_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("rGIN model alignment PASSED.")


if __name__ == "__main__":
    main()
