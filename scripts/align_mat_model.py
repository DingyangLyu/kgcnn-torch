#!/usr/bin/env python3
"""Model-level alignment: MAT (full model, Torch -> Keras weight transfer).

Builds both Torch MATModel and a KerasMATFullStack that mirrors the full
architecture (Embedding + input Dense + MATDistanceMatrix + adj Dense +
(LayerNorm + MATAttentionHead×heads + Dense proj + residual +
 LayerNorm + MLP + Dense proj + mask + residual)×depth +
LayerNorm + MATGlobalPool + output MLP),
transfers ALL weights, and verifies final output.

NOTE: MAT uses padded (dense) representation, NOT disjoint.
Dropout is disabled (set to None) for deterministic alignment.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_embedding, copy_layernorm,
    compare_outputs, keras_to_torch,
)

import keras
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.modules import Embedding as KerasEmbedding
from kgcnn.layers.mlp import MLP as KerasMLP
from kgcnn.literature.MAT._layers import (
    MATAttentionHead as KerasMATAttentionHead,
    MATDistanceMatrix as KerasMATDistanceMatrix,
    MATGlobalPool as KerasMATGlobalPool,
)
from kgcnn_torch.models.mat import MATModel


@dataclass
class Config:
    embedding_units: int = 32
    depth: int = 2
    num_heads: int = 4
    attention_units: int = 8
    merge_heads: str = "concat"
    lambda_attention: float = 0.3
    lambda_distance: float = 0.3
    add_identity: bool = False
    distance_trafo: str = "exp"
    units_ff: list = None
    ff_activations: list = None
    output_units: list = None
    output_activations: list = None
    num_targets: int = 2
    num_embeddings: int = 95
    input_node_dim: int = 64
    n_max: int = 8  # max nodes per graph (padded)
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.units_ff is None:
            self.units_ff = [32, 32, 32]
        if self.ff_activations is None:
            self.ff_activations = ["relu", "relu", "linear"]
        if self.output_units is None:
            self.output_units = [32, 16]
        if self.output_activations is None:
            self.output_activations = (["relu"] * len(self.output_units)
                                       + ["linear"])


def make_padded_data(cfg: Config):
    """Generate random padded graph data for MAT alignment testing.

    Returns:
        node_input: (B, N) int64 atom types
        xyz_input: (B, N, 3) float coordinates
        adjacency: (B, N, N, 1) float adjacency matrix
        node_mask: (B, N) bool mask
        adj_mask: (B, N, N) bool mask
    """
    torch.manual_seed(cfg.seed)
    B, N = cfg.batch_size, cfg.n_max

    # Atom types: valid ones in [1, 29], padded = 0
    node_input = torch.randint(1, 30, (B, N), dtype=torch.long)

    # Mark last 2 nodes in each graph as padded
    n_valid = N - 2
    node_mask = torch.zeros(B, N, dtype=torch.bool)
    node_mask[:, :n_valid] = True
    node_input[~node_mask] = 0

    # Coordinates
    xyz_input = torch.randn(B, N, 3)
    xyz_input[~node_mask] = 0.0

    # Adjacency: random {0, 1} for valid pairs
    adjacency = torch.zeros(B, N, N, 1)
    for b in range(B):
        adj_b = torch.randint(0, 2, (n_valid, n_valid), dtype=torch.float32)
        adjacency[b, :n_valid, :n_valid, 0] = adj_b

    # Adj mask
    adj_mask = torch.zeros(B, N, N, dtype=torch.bool)
    adj_mask[:, :n_valid, :n_valid] = True

    return node_input, xyz_input, adjacency, node_mask, adj_mask


class KerasMATFullStack:
    """Full Keras MAT model stack mirroring MATModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbedding(
            input_dim=cfg.num_embeddings, output_dim=cfg.input_node_dim,
        )
        self.input_projection = keras.layers.Dense(
            cfg.embedding_units, use_bias=False)
        self.adj_projection = keras.layers.Dense(1, use_bias=False)
        self.distance_layer = KerasMATDistanceMatrix(trafo=cfg.distance_trafo)

        self.attention_norms = []
        self.attention_heads = []  # list of lists
        self.attention_projections = []
        self.ff_norms = []
        self.ff_mlps = []
        self.ff_projections = []

        for _ in range(cfg.depth):
            self.attention_norms.append(keras.layers.LayerNormalization())

            heads = []
            for _ in range(cfg.num_heads):
                heads.append(KerasMATAttentionHead(
                    units=cfg.attention_units,
                    lambda_attention=cfg.lambda_attention,
                    lambda_distance=cfg.lambda_distance,
                    add_identity=cfg.add_identity,
                    dropout=None,  # Disable for alignment
                ))
            self.attention_heads.append(heads)

            if cfg.merge_heads in ("add", "sum", "reduce_sum"):
                proj_in = cfg.attention_units
            else:
                proj_in = cfg.attention_units * cfg.num_heads
            self.attention_projections.append(
                keras.layers.Dense(cfg.embedding_units, use_bias=False))

            self.ff_norms.append(keras.layers.LayerNormalization())
            self.ff_mlps.append(KerasMLP(
                units=cfg.units_ff, activation=cfg.ff_activations))
            self.ff_projections.append(
                keras.layers.Dense(cfg.embedding_units, use_bias=False))

        self.final_norm = keras.layers.LayerNormalization()
        self.pool = KerasMATGlobalPool()

        out_units = cfg.output_units + [cfg.num_targets]
        out_acts = cfg.output_activations
        self.output_mlp = KerasMLP(units=out_units, activation=out_acts)

    def forward(self, node_input, xyz_input, adjacency, node_mask, adj_mask):
        # Embedding
        n = self.node_embedding(node_input)

        # Masks: (B, N) -> (B, N, 1) and (B, N, N) -> (B, N, N, 1)
        n_mask = ops.expand_dims(ops.cast(node_mask, dtype="float32"), axis=-1)
        a_mask = ops.expand_dims(ops.cast(adj_mask, dtype="float32"), axis=-1)

        # Distance matrix
        dist, dist_mask = self.distance_layer(xyz_input, mask=n_mask)

        # Adjacency projection (has_edge_dim=True: adj has shape B,N,N,1)
        adj = self.adj_projection(adjacency)

        # Input projection
        h = self.input_projection(n)
        h_mask = n_mask

        for i in range(self.cfg.depth):
            # 1. Norm + Attention + Residual
            hn = self.attention_norms[i](h)
            head_outputs = []
            for head in self.attention_heads[i]:
                ho = head(
                    [hn, dist, adj],
                    mask=[n_mask, dist_mask, a_mask],
                )
                head_outputs.append(ho)

            if self.cfg.merge_heads in ("add", "sum", "reduce_sum"):
                hu = keras.layers.Add()(head_outputs)
            else:
                hu = keras.layers.Concatenate(axis=-1)(head_outputs)
            hu = self.attention_projections[i](hu)
            h = h + hu

            # 2. Norm + MLP + Residual
            hn = self.ff_norms[i](h)
            hu = self.ff_mlps[i](hn)
            hu = self.ff_projections[i](hu)
            hu = hu * h_mask
            h = h + hu

        # Final norm
        out = self.final_norm(h)
        out = out * h_mask
        out = self.pool(out, mask=h_mask)
        out = self.output_mlp(out)

        return out


def copy_seq_to_keras_mlp(torch_seq, keras_mlp):
    """Copy nn.Sequential([Linear, act, ...]) -> Keras MLP weights.

    Finds all nn.Linear layers in the Sequential and copies to
    keras_mlp.mlp_dense_layer_list[i].
    """
    keras_idx = 0
    for module in torch_seq:
        if isinstance(module, torch.nn.Linear):
            copy_dense(module, keras_mlp.mlp_dense_layer_list[keras_idx])
            keras_idx += 1


def copy_attention_head(torch_head, keras_head):
    """Copy MATAttentionHead weights (dense_q, dense_k, dense_v)."""
    copy_dense(torch_head.dense_q, keras_head.dense_q)
    copy_dense(torch_head.dense_k, keras_head.dense_k)
    copy_dense(torch_head.dense_v, keras_head.dense_v)


def transfer_all_weights(torch_model: MATModel,
                         keras_stack: KerasMATFullStack, cfg: Config):
    # Embedding
    copy_embedding(torch_model.node_embedding, keras_stack.node_embedding)

    # Input projection (no bias)
    copy_dense(torch_model.input_projection, keras_stack.input_projection)

    # Adjacency projection (no bias)
    copy_dense(torch_model.adj_projection, keras_stack.adj_projection)

    # Transformer blocks
    for i in range(cfg.depth):
        # Attention norm
        copy_layernorm(torch_model.attention_norms[i],
                       keras_stack.attention_norms[i])

        # Attention heads
        for j in range(cfg.num_heads):
            copy_attention_head(torch_model.attention_heads[i][j],
                                keras_stack.attention_heads[i][j])

        # Attention projection
        copy_dense(torch_model.attention_projections[i],
                   keras_stack.attention_projections[i])

        # FF norm
        copy_layernorm(torch_model.ff_norms[i], keras_stack.ff_norms[i])

        # FF MLP (Sequential -> Keras MLP)
        copy_seq_to_keras_mlp(torch_model.ff_mlps[i],
                              keras_stack.ff_mlps[i])

        # FF projection
        copy_dense(torch_model.ff_projections[i],
                   keras_stack.ff_projections[i])

    # Final norm
    copy_layernorm(torch_model.final_norm, keras_stack.final_norm)

    # Output MLP (Sequential -> Keras MLP)
    copy_seq_to_keras_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()

    node_input, xyz_input, adjacency, node_mask, adj_mask = make_padded_data(cfg)

    # Build Torch model (dropout=None for deterministic alignment)
    torch_model = MATModel(
        embedding_units=cfg.embedding_units,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        attention_units=cfg.attention_units,
        merge_heads=cfg.merge_heads,
        lambda_attention=cfg.lambda_attention,
        lambda_distance=cfg.lambda_distance,
        add_identity=cfg.add_identity,
        attention_dropout=None,
        distance_trafo=cfg.distance_trafo,
        units_ff=cfg.units_ff,
        ff_activations=cfg.ff_activations,
        output_units=cfg.output_units,
        output_activations=cfg.output_activations,
        num_targets=cfg.num_targets,
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        input_node_dim=cfg.input_node_dim,
        output_embedding="graph",
    )
    torch_model.eval()

    # Build and initialize Keras stack
    keras_stack = KerasMATFullStack(cfg)
    _ = keras_stack.forward(node_input, xyz_input, adjacency,
                            node_mask, adj_mask)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(
            node_input, xyz_input, adjacency,
            node_mask.float(), adj_mask.float(),
        ).detach().cpu()

    keras_out = keras_to_torch(
        keras_stack.forward(node_input, xyz_input, adjacency,
                            node_mask, adj_mask))

    print("MAT model-level alignment (Torch -> Keras):")
    compare_outputs("MAT_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("MAT model alignment PASSED.")


if __name__ == "__main__":
    main()
