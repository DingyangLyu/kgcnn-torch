#!/usr/bin/env python3
"""Model-level alignment: DimeNetPP (full model, Torch -> Keras weight transfer).

Builds both Torch DimeNetPPModel and a KerasDimeNetPPFullStack that mirrors the full
architecture (EmbeddingDimeBlock + BesselBasisLayer + SphericalBasisLayer +
rbf_emb Dense + edge_emb Dense + DimNetOutputBlock×(num_blocks+1) +
DimNetInteractionPPBlock×num_blocks + PoolingNodes + optional output MLP),
transfers ALL weights, and verifies final output.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_mlp,
    compare_outputs, keras_to_torch,
)

import keras
from keras import ops
from keras.layers import Dense, Subtract, Concatenate

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.geom import (
    NodePosition, NodeDistanceEuclidean, EdgeAngle,
    BesselBasisLayer as KerasBesselBasisLayer,
    SphericalBasisLayer as KerasSphericalBasisLayer,
)
from kgcnn.layers.gather import GatherNodes
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.literature.DimeNetPP._layers import (
    EmbeddingDimeBlock as KerasEmbeddingDimeBlock,
    DimNetInteractionPPBlock as KerasDimNetInteractionPPBlock,
    DimNetOutputBlock as KerasDimNetOutputBlock,
)
from kgcnn_torch.models.dimenetpp import DimeNetPPModel


@dataclass
class Config:
    emb_size: int = 32
    out_emb_size: int = 16
    int_emb_size: int = 16
    basis_emb_size: int = 8
    num_blocks: int = 2
    num_spherical: int = 3
    num_radial: int = 4
    cutoff: float = 5.0
    envelope_exponent: int = 5
    num_before_skip: int = 1
    num_after_skip: int = 1
    num_dense_output: int = 2
    num_targets: int = 2
    activation: str = "swish"
    extensive: bool = True
    output_init: str = "zeros"
    use_output_mlp: bool = True
    output_mlp_units: list = None
    output_mlp_activation: str = "swish"
    num_embeddings: int = 95
    n_nodes: int = 16
    n_edges: int = 40
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_mlp_units is None:
            self.output_mlp_units = [16, self.num_targets]


def generate_angle_index(edge_index_torch):
    """Generate triplet angle indices from Torch-convention edge_index [src, dst].

    For each edge ji (src[p]=j, dst[p]=i), find all edges kj (dst[q]=j)
    that share node j. Returns (2, K): [ji_edge_idx, kj_edge_idx].
    """
    src = edge_index_torch[0]
    dst = edge_index_torch[1]
    M = src.shape[0]

    # Build reverse index: node -> list of edge indices with dst==node
    node_to_incoming = {}
    for q in range(M):
        d = dst[q].item()
        if d not in node_to_incoming:
            node_to_incoming[d] = []
        node_to_incoming[d].append(q)

    ji_list, kj_list = [], []
    for p in range(M):
        j = src[p].item()
        if j in node_to_incoming:
            for q in node_to_incoming[j]:
                if q != p:
                    ji_list.append(p)
                    kj_list.append(q)

    return torch.tensor([ji_list, kj_list], dtype=torch.long)


class KerasDimeNetPPFullStack:
    """Full Keras DimeNetPP model stack mirroring DimeNetPPModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbeddingDimeBlock(
            input_dim=cfg.num_embeddings, output_dim=cfg.emb_size,
        )
        self.bessel_basis = KerasBesselBasisLayer(
            num_radial=cfg.num_radial, cutoff=cfg.cutoff,
            envelope_exponent=cfg.envelope_exponent,
        )
        self.spherical_basis = KerasSphericalBasisLayer(
            num_spherical=cfg.num_spherical, num_radial=cfg.num_radial,
            cutoff=cfg.cutoff, envelope_exponent=cfg.envelope_exponent,
        )
        self.rbf_emb_dense = Dense(
            cfg.emb_size, use_bias=True, activation=cfg.activation,
            kernel_initializer="kgcnn>glorot_orthogonal",
        )
        self.edge_emb_dense = Dense(
            cfg.emb_size, use_bias=True, activation=cfg.activation,
            kernel_initializer="kgcnn>glorot_orthogonal",
        )

        self.output_block_0 = KerasDimNetOutputBlock(
            cfg.emb_size, cfg.out_emb_size, cfg.num_dense_output,
            num_targets=cfg.num_targets,
            output_kernel_initializer=cfg.output_init,
            activation=cfg.activation,
        )

        self.interaction_blocks = []
        self.output_blocks = []
        for _ in range(cfg.num_blocks):
            self.interaction_blocks.append(KerasDimNetInteractionPPBlock(
                cfg.emb_size, cfg.int_emb_size, cfg.basis_emb_size,
                cfg.num_before_skip, cfg.num_after_skip,
                activation=cfg.activation,
            ))
            self.output_blocks.append(KerasDimNetOutputBlock(
                cfg.emb_size, cfg.out_emb_size, cfg.num_dense_output,
                num_targets=cfg.num_targets,
                output_kernel_initializer=cfg.output_init,
                activation=cfg.activation,
            ))

        pool_method = "sum" if cfg.extensive else "mean"
        self.pooling = KerasPoolingNodes(pooling_method=pool_method)

        if cfg.use_output_mlp:
            out_act = ([cfg.output_mlp_activation] *
                       (len(cfg.output_mlp_units) - 1) + ["linear"])
            out_bias = [True] * (len(cfg.output_mlp_units) - 1) + [False]
            self.output_mlp = KerasMLP(
                units=cfg.output_mlp_units, activation=out_act,
                use_bias=out_bias,
            )

    def forward(self, z, pos, edge_index, angle_index,
                batch_id_node, count_nodes):
        n = self.node_embedding(z)

        # Geometry: distances and angles
        pos1, pos2 = NodePosition()([pos, edge_index])
        dist = NodeDistanceEuclidean()([pos1, pos2])
        rbf = self.bessel_basis(dist)

        v12 = Subtract()([pos1, pos2])
        angles = EdgeAngle()([v12, angle_index])
        sbf = self.spherical_basis([dist, angles, angle_index])

        # Embedding block
        rbf_emb = self.rbf_emb_dense(rbf)
        n_pairs = GatherNodes()([n, edge_index])
        x = Concatenate(axis=-1)([n_pairs, rbf_emb])
        x = self.edge_emb_dense(x)

        # Initial output
        ps = self.output_block_0([n, x, rbf, edge_index])

        # Interaction blocks
        for i in range(self.cfg.num_blocks):
            x = self.interaction_blocks[i]([x, rbf, sbf, angle_index])
            p_update = self.output_blocks[i]([n, x, rbf, edge_index])
            ps = ps + p_update

        out = self.pooling([count_nodes, ps, batch_id_node])

        if self.cfg.use_output_mlp:
            out = self.output_mlp(out)

        return out


def copy_residual_layer(torch_res, keras_res):
    """Copy ResidualLayer weights (dense_1, dense_2)."""
    copy_dense(torch_res.dense_1, keras_res.dense_1)
    copy_dense(torch_res.dense_2, keras_res.dense_2)


def copy_output_block(torch_block, keras_block, num_dense):
    """Copy DimNetOutputBlock weights."""
    copy_dense(torch_block.dense_rbf, keras_block.dense_rbf)
    copy_dense(torch_block.up_projection, keras_block.up_projection)
    # dense_mlp: Torch nn.Sequential with (Linear, activation) pairs
    # Keras GraphMLP (= MLP) with mlp_dense_layer_list
    for i in range(num_dense):
        copy_dense(torch_block.dense_mlp[2 * i],
                   keras_block.dense_mlp.mlp_dense_layer_list[i])
    copy_dense(torch_block.dense_final, keras_block.dense_final)


def copy_interaction_block(torch_block, keras_block, cfg: Config):
    """Copy DimNetInteractionPPBlock weights."""
    # Basis transformations (no bias)
    copy_dense(torch_block.dense_rbf1, keras_block.dense_rbf1)
    copy_dense(torch_block.dense_rbf2, keras_block.dense_rbf2)
    copy_dense(torch_block.dense_sbf1, keras_block.dense_sbf1)
    copy_dense(torch_block.dense_sbf2, keras_block.dense_sbf2)

    # Edge transformations: Torch Sequential([Linear, activation]) -> Keras Dense
    copy_dense(torch_block.dense_ji[0], keras_block.dense_ji)
    copy_dense(torch_block.dense_kj[0], keras_block.dense_kj)

    # Projections: Torch Sequential([Linear(no bias), activation]) -> Keras Dense
    copy_dense(torch_block.down_projection[0], keras_block.down_projection)
    copy_dense(torch_block.up_projection[0], keras_block.up_projection)

    # Residual layers before skip
    for j in range(cfg.num_before_skip):
        copy_residual_layer(torch_block.layers_before_skip[j],
                            keras_block.layers_before_skip[j])

    # Final before skip: Sequential([Linear, activation]) -> Dense
    copy_dense(torch_block.final_before_skip[0], keras_block.final_before_skip)

    # Residual layers after skip
    for j in range(cfg.num_after_skip):
        copy_residual_layer(torch_block.layers_after_skip[j],
                            keras_block.layers_after_skip[j])


def transfer_all_weights(torch_model: DimeNetPPModel,
                         keras_stack: KerasDimeNetPPFullStack, cfg: Config):
    # EmbeddingDimeBlock: Torch nn.Embedding -> Keras add_weight
    w = torch_model.node_embedding.embedding.weight.detach().cpu().numpy()
    keras_stack.node_embedding.set_weights([w])

    # BesselBasisLayer trainable frequencies
    freq = torch_model.bessel_basis.frequencies.detach().cpu().numpy()
    keras_stack.bessel_basis.set_weights([freq])

    # rbf_emb, edge_emb: extract Linear from Sequential
    copy_dense(torch_model.rbf_emb[0], keras_stack.rbf_emb_dense)
    copy_dense(torch_model.edge_emb[0], keras_stack.edge_emb_dense)

    # Output block 0
    copy_output_block(torch_model.output_block_0,
                      keras_stack.output_block_0, cfg.num_dense_output)

    # Interaction + output blocks
    for i in range(cfg.num_blocks):
        copy_interaction_block(torch_model.interaction_blocks[i],
                               keras_stack.interaction_blocks[i], cfg)
        copy_output_block(torch_model.output_blocks[i],
                          keras_stack.output_blocks[i], cfg.num_dense_output)

    # Optional output MLP
    if cfg.use_output_mlp:
        copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    from model_alignment_utils import make_disjoint_graph
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False,
    )

    # Generate angle indices from Torch edge_index
    angle_index = generate_angle_index(torch_data.edge_index)
    torch_data.angle_index = angle_index
    # angle_index indexes edges (not nodes), same for both conventions

    # Build Torch model
    torch_model = DimeNetPPModel(
        emb_size=cfg.emb_size,
        out_emb_size=cfg.out_emb_size,
        int_emb_size=cfg.int_emb_size,
        basis_emb_size=cfg.basis_emb_size,
        num_blocks=cfg.num_blocks,
        num_spherical=cfg.num_spherical,
        num_radial=cfg.num_radial,
        cutoff=cfg.cutoff,
        envelope_exponent=cfg.envelope_exponent,
        num_before_skip=cfg.num_before_skip,
        num_after_skip=cfg.num_after_skip,
        num_dense_output=cfg.num_dense_output,
        num_targets=cfg.num_targets,
        activation=cfg.activation,
        extensive=cfg.extensive,
        output_init=cfg.output_init,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        use_output_mlp=cfg.use_output_mlp,
        output_mlp_units=cfg.output_mlp_units,
        output_mlp_activation=cfg.output_mlp_activation,
    )
    torch_model.eval()

    # Build and initialize Keras stack
    keras_stack = KerasDimeNetPPFullStack(cfg)
    z_k = keras_data["z"]
    pos_k = keras_data["pos"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(
        keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k))

    print("DimeNetPP model-level alignment (Torch -> Keras):")
    compare_outputs("DimeNetPP_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("DimeNetPP model alignment PASSED.")


if __name__ == "__main__":
    main()
