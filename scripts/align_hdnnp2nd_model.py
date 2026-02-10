#!/usr/bin/env python3
"""Model-level alignment: HDNNP2nd weighted ACSF (full model, Torch -> Keras weight transfer).

Builds both Torch HDNNP2ndModel and a KerasHDNNP2ndFullStack that mirrors the full
architecture (wACSFRad + wACSFAng + Concatenate + RelationalMLP + PoolingNodes + output MLP),
transfers ALL trainable weights, and verifies final output.

NOTE: Uses type indices (0-indexed) as z input, matching the layerwise alignment test.
Descriptor layers (wACSFRad/wACSFAng) are non-trainable with parameters extracted
from Torch model buffers.
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

from kgcnn.literature.HDNNP2nd._wacsf import (
    wACSFRad as KeraswACSFRad, wACSFAng as KeraswACSFAng,
)
from kgcnn.layers.mlp import RelationalMLP as KerasRelationalMLP, MLP as KerasMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn_torch.models.hdnnp2nd import HDNNP2ndModel


@dataclass
class Config:
    n_types: int = 4
    n_rad_features: int = 8
    n_ang_features: int = 6
    cutoff: float = 5.0
    relational_units: list = None
    relational_activation: list = None
    num_relations: int = 96
    node_pooling: str = "sum"
    output_units: list = None
    output_activation: str = "swish"
    num_targets: int = 2
    n_nodes: int = 14
    n_edges: int = 40
    n_angles: int = 36
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.relational_units is None:
            self.relational_units = [64, 64, 64]
        if self.relational_activation is None:
            self.relational_activation = ["swish", "swish", "linear"]
        if self.output_units is None:
            self.output_units = [32]


class KerasHDNNP2ndFullStack:
    """Full Keras HDNNP2nd (wACSF) model stack mirroring HDNNP2ndModel."""

    def __init__(self, cfg: Config, eta_mu, eta_mu_lz):
        self.cfg = cfg

        # Descriptor layers (non-trainable, initialized from Torch buffers)
        self.acsf_rad = KeraswACSFRad(
            eta_mu=eta_mu, cutoff=cfg.cutoff, add_eps=True)
        self.acsf_ang = KeraswACSFAng(
            eta_mu_lambda_zeta=eta_mu_lz, cutoff=cfg.cutoff, add_eps=True)

        # Relational MLP (trainable)
        self.relational_mlp = KerasRelationalMLP(
            units=cfg.relational_units,
            num_relations=cfg.num_relations,
            activation=cfg.relational_activation,
        )

        self.pooling = KerasPoolingNodes(pooling_method="scatter_sum")

        out_units = cfg.output_units + [cfg.num_targets]
        out_act = [cfg.output_activation] * len(cfg.output_units) + ["linear"]
        self.output_mlp = KerasMLP(units=out_units, activation=out_act)

    def forward(self, z, pos, edge_index_keras, angle_index,
                batch_id_node, count_nodes):
        # Descriptor computation (same z for weighting + parameter lookup)
        rep_rad = self.acsf_rad([z, pos, edge_index_keras])
        rep_ang = self.acsf_ang([z, pos, angle_index])
        rep = keras.layers.Concatenate()([rep_rad, rep_ang])

        # RelationalMLP (z serves as type/relation index)
        n = self.relational_mlp([rep, z, batch_id_node, count_nodes])

        out = self.pooling([count_nodes, n, batch_id_node])
        out = self.output_mlp(out)
        return out


def transfer_all_weights(torch_model: HDNNP2ndModel,
                         keras_stack: KerasHDNNP2ndFullStack, cfg: Config):
    copy_relational_mlp(torch_model.relational_mlp, keras_stack.relational_mlp)
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    # Use 0-indexed type indices as z (matching layerwise test convention)
    element_types = list(range(cfg.n_types))

    z = torch.randint(0, cfg.n_types, (cfg.n_nodes,), dtype=torch.long)
    pos = torch.randn(cfg.n_nodes, 3)

    # Edge index (Torch convention)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    # Angle index (3, K): [center, neighbor1, neighbor2] - same for both
    center = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr2 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
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
    torch_model = HDNNP2ndModel(
        element_types=element_types,
        n_rad_features=cfg.n_rad_features,
        n_ang_features=cfg.n_ang_features,
        cutoff=cfg.cutoff,
        num_relations=cfg.num_relations,
        relational_units=cfg.relational_units,
        relational_activation=cfg.relational_activation,
        use_batch_norm=False,
        node_pooling=cfg.node_pooling,
        use_output_mlp=True,
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
    )
    torch_model.eval()

    # Extract descriptor parameters from Torch model buffers
    eta_mu = np.stack([
        torch_model.acsf_rad.eta.detach().cpu().numpy(),
        torch_model.acsf_rad.mu.detach().cpu().numpy(),
    ], axis=-1)
    eta_mu_lz = np.stack([
        torch_model.acsf_ang.eta.detach().cpu().numpy(),
        torch_model.acsf_ang.mu.detach().cpu().numpy(),
        torch_model.acsf_ang.lam.detach().cpu().numpy(),
        torch_model.acsf_ang.zeta.detach().cpu().numpy(),
    ], axis=-1)

    # Build Keras stack (with extracted descriptor params)
    keras_stack = KerasHDNNP2ndFullStack(cfg, eta_mu, eta_mu_lz)
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

    print("HDNNP2nd (wACSF) model-level alignment (Torch -> Keras):")
    compare_outputs("HDNNP2nd_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("HDNNP2nd model alignment PASSED.")


if __name__ == "__main__":
    main()
