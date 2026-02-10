#!/usr/bin/env python3
"""Model-level alignment: HDNNP2ndBehler (full model, Torch -> Keras weight transfer).

Builds both Torch HDNNP2ndBehlerModel and a KerasHDNNP2ndBehlerFullStack that mirrors
the full architecture (ACSFG2 + ACSFG4 + Concatenate + RelationalMLP + PoolingNodes + output MLP),
transfers ALL trainable weights, and verifies final output.

NOTE: Descriptor layers (ACSFG2/ACSFG4) are non-trainable with parameters
generated from the same G2/G4 hyperparameters.
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

from kgcnn.literature.HDNNP2nd._acsf import ACSFG2, ACSFG4
from kgcnn.layers.mlp import RelationalMLP as KerasRelationalMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn_torch.models.hdnnp2nd import HDNNP2ndBehlerModel


@dataclass
class Config:
    element_types: list = None
    g2_eta: list = None
    g2_rs: list = None
    g2_rc: float = 6.0
    g4_eta: list = None
    g4_zeta: list = None
    g4_lamda: list = None
    g4_rc: float = 6.0
    g4_multiplicity: float = 2.0
    relational_units: list = None
    relational_activation: list = None
    node_pooling: str = "sum"
    output_units: list = None
    output_activation: str = "linear"
    num_targets: int = 2
    n_nodes: int = 12
    n_edges: int = 36
    n_angles: int = 30
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.element_types is None:
            self.element_types = [1, 6]
        if self.g2_eta is None:
            self.g2_eta = [0.0, 0.3]
        if self.g2_rs is None:
            self.g2_rs = [0.0, 3.0]
        if self.g4_eta is None:
            self.g4_eta = [0.0, 0.3]
        if self.g4_zeta is None:
            self.g4_zeta = [1.0, 8.0]
        if self.g4_lamda is None:
            self.g4_lamda = [-1.0, 1.0]
        if self.relational_units is None:
            self.relational_units = [64, 64, 64]
        if self.relational_activation is None:
            self.relational_activation = ["swish", "swish", "linear"]
        if self.output_units is None:
            self.output_units = []


class KerasHDNNP2ndBehlerFullStack:
    """Full Keras HDNNP2ndBehler model stack mirroring HDNNP2ndBehlerModel."""

    def __init__(self, cfg: Config):
        self.cfg = cfg

        # Build Keras ACSFG2/G4 from the same parameter tables
        self.g2 = ACSFG2(**ACSFG2.make_param_table(
            eta=cfg.g2_eta, rs=cfg.g2_rs, rc=cfg.g2_rc,
            elements=cfg.element_types))
        self.g4 = ACSFG4(**ACSFG4.make_param_table(
            eta=cfg.g4_eta, zeta=cfg.g4_zeta, lamda=cfg.g4_lamda,
            rc=cfg.g4_rc, elements=cfg.element_types,
            multiplicity=cfg.g4_multiplicity))

        # Build z_to_type mapping (same as Torch model)
        element_types = sorted(cfg.element_types)
        max_z = max(element_types) + 1
        self.z_to_type = np.full(max_z, -1, dtype=np.int64)
        for idx, z_val in enumerate(element_types):
            self.z_to_type[z_val] = idx

        self.relational_mlp = KerasRelationalMLP(
            units=cfg.relational_units,
            num_relations=96,
            activation=cfg.relational_activation,
        )

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, pos, edge_index_keras, angle_index,
                batch_id_node, count_nodes):
        # Descriptor computation (Keras ACSFG2/G4 handle z as atomic numbers)
        rep_g2 = self.g2([z, pos, edge_index_keras])
        rep_g4 = self.g4([z, pos, angle_index])
        rep = keras.layers.Concatenate()([rep_g2, rep_g4])

        # Map z to type indices for RelationalMLP (same as Torch model)
        z_np = ops.convert_to_numpy(z)
        type_map = torch.tensor(self.z_to_type[z_np], dtype=torch.long)

        n = self.relational_mlp([rep, type_map, batch_id_node, count_nodes])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: HDNNP2ndBehlerModel,
                         keras_stack: KerasHDNNP2ndBehlerFullStack, cfg: Config):
    copy_relational_mlp(torch_model.relational_mlp, keras_stack.relational_mlp)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # Generate z as actual atomic numbers from element_types
    z_choices = torch.tensor(sorted(cfg.element_types), dtype=torch.long)
    z = z_choices[torch.randint(0, len(cfg.element_types), (cfg.n_nodes,))]
    pos = torch.randn(cfg.n_nodes, 3)

    # Edge index
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    # Angle index (3, K): [center, neighbor1, neighbor2] - same for both
    center = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr2 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    # Avoid degenerate triplets
    nbr1 = torch.where(nbr1 == center, (nbr1 + 1) % cfg.n_nodes, nbr1)
    nbr2 = torch.where(nbr2 == center, (nbr2 + 2) % cfg.n_nodes, nbr2)
    nbr2 = torch.where(nbr2 == nbr1, (nbr2 + 1) % cfg.n_nodes, nbr2)
    angle_index = torch.stack([center, nbr1, nbr2], dim=0)

    # Batch assignment
    nodes_per_graph = cfg.n_nodes // cfg.batch_size
    batch_node = torch.cat([
        torch.full((nodes_per_graph,), i, dtype=torch.long)
        for i in range(cfg.batch_size)
    ])
    remainder = cfg.n_nodes - nodes_per_graph * cfg.batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node,
                                torch.full((remainder,), cfg.batch_size - 1, dtype=torch.long)])

    count_nodes = torch.zeros(cfg.batch_size, dtype=torch.long)
    for i in range(cfg.batch_size):
        count_nodes[i] = (batch_node == i).sum()

    # Build Torch model
    torch_model = HDNNP2ndBehlerModel(
        element_types=cfg.element_types,
        g2_eta=cfg.g2_eta, g2_rs=cfg.g2_rs, g2_rc=cfg.g2_rc,
        g4_eta=cfg.g4_eta, g4_zeta=cfg.g4_zeta, g4_lamda=cfg.g4_lamda,
        g4_rc=cfg.g4_rc, g4_multiplicity=cfg.g4_multiplicity,
        relational_units=cfg.relational_units,
        relational_activation=cfg.relational_activation,
        use_batch_norm=False,
        node_pooling=cfg.node_pooling,
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
    )
    torch_model.eval()

    # Build Keras stack
    keras_stack = KerasHDNNP2ndBehlerFullStack(cfg)
    _ = keras_stack.forward(z, pos, edge_index_keras, angle_index,
                            batch_node, count_nodes)

    transfer_all_weights(torch_model, keras_stack, cfg)

    # Torch forward
    from types import SimpleNamespace
    torch_data = SimpleNamespace(
        z=z, pos=pos, edge_index=edge_index_torch,
        angle_index=angle_index, batch=batch_node,
    )
    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(
        keras_stack.forward(z, pos, edge_index_keras, angle_index,
                            batch_node, count_nodes))

    print("HDNNP2ndBehler model-level alignment (Torch -> Keras):")
    compare_outputs("HDNNP2ndBehler_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("HDNNP2ndBehler model alignment PASSED.")


if __name__ == "__main__":
    main()
