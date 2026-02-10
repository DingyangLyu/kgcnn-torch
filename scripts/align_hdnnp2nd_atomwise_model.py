#!/usr/bin/env python3
"""Model-level alignment: HDNNP2ndAtomWise (full model, Torch -> Keras weight transfer).

Builds both Torch HDNNP2ndAtomWiseModel and a KerasHDNNP2ndAtomWiseFullStack that mirrors
the full architecture (z_to_type mapping + RelationalMLP + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output.

NOTE: This is the simplified "atom-wise" variant that takes pre-computed per-atom
representations (no ACSF/symmetry function computation).
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_relational_mlp, copy_mlp,
    compare_outputs, keras_to_torch,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.mlp import RelationalMLP as KerasRelationalMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn_torch.models.hdnnp2nd import HDNNP2ndAtomWiseModel


@dataclass
class Config:
    element_types: list = None
    input_dim: int = 40
    relational_units: list = None
    relational_activation: list = None
    num_relations: int = 96
    node_pooling: str = "sum"
    output_units: list = None
    output_activation: str = "swish"
    num_targets: int = 2
    n_nodes: int = 20
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.element_types is None:
            self.element_types = [1, 6, 7, 8]
        if self.relational_units is None:
            self.relational_units = [64, 64, 64]
        if self.relational_activation is None:
            self.relational_activation = ["swish", "swish", "linear"]
        if self.output_units is None:
            self.output_units = [32]


class KerasHDNNP2ndAtomWiseFullStack:
    """Full Keras HDNNP2ndAtomWise model stack mirroring HDNNP2ndAtomWiseModel."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Build same z_to_type mapping as Torch
        element_types = sorted(cfg.element_types)
        max_z = max(element_types) + 1
        self.z_to_type = np.full(max_z, -1, dtype=np.int64)
        for idx, z_val in enumerate(element_types):
            self.z_to_type[z_val] = idx

        self.relational_mlp = KerasRelationalMLP(
            units=cfg.relational_units,
            num_relations=cfg.num_relations,
            activation=cfg.relational_activation,
        )

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, x, batch_id_node, count_nodes):
        # Map z to type indices (same as Torch's z_to_type buffer)
        z_np = ops.convert_to_numpy(z)
        type_map = torch.tensor(self.z_to_type[z_np], dtype=torch.long)

        n = self.relational_mlp([x, type_map, batch_id_node, count_nodes])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: HDNNP2ndAtomWiseModel,
                         keras_stack: KerasHDNNP2ndAtomWiseFullStack, cfg: Config):
    copy_relational_mlp(torch_model.relational_mlp, keras_stack.relational_mlp)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # Generate test data (no graph structure needed, just node-level)
    nodes_per_graph = cfg.n_nodes // cfg.batch_size
    batch_node = torch.cat([
        torch.full((nodes_per_graph,), i, dtype=torch.long)
        for i in range(cfg.batch_size)
    ])
    remainder = cfg.n_nodes - nodes_per_graph * cfg.batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node,
                                torch.full((remainder,), cfg.batch_size - 1, dtype=torch.long)])
    total_nodes = batch_node.shape[0]

    count_nodes = torch.zeros(cfg.batch_size, dtype=torch.long)
    for i in range(cfg.batch_size):
        count_nodes[i] = (batch_node == i).sum()

    # Random atomic numbers from element_types
    z_choices = torch.tensor(cfg.element_types, dtype=torch.long)
    z_indices = torch.randint(0, len(cfg.element_types), (total_nodes,))
    z = z_choices[z_indices]

    # Random pre-computed features
    x = torch.randn(total_nodes, cfg.input_dim)

    # Build Torch model
    from types import SimpleNamespace
    torch_data = SimpleNamespace(z=z, x=x, batch=batch_node)

    torch_model = HDNNP2ndAtomWiseModel(
        element_types=cfg.element_types,
        input_dim=cfg.input_dim,
        relational_units=cfg.relational_units,
        relational_activation=cfg.relational_activation,
        node_pooling=cfg.node_pooling,
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
    )
    torch_model.eval()

    # Build and initialize Keras stack
    keras_stack = KerasHDNNP2ndAtomWiseFullStack(cfg)
    _ = keras_stack.forward(z, x, batch_node, count_nodes)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(keras_stack.forward(z, x, batch_node, count_nodes))

    print("HDNNP2ndAtomWise model-level alignment (Torch -> Keras):")
    compare_outputs("HDNNP2ndAtomWise_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("HDNNP2ndAtomWise model alignment PASSED.")


if __name__ == "__main__":
    main()
