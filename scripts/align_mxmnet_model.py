#!/usr/bin/env python3
"""Model-level alignment: MXMNet (full model, Torch -> Keras weight transfer).

Builds both Torch MXMNetModel and a KerasMXMNetFullStack that mirrors the full
architecture (EmbeddingDimeBlock + BesselBasisLayer + SphericalBasisLayer +
MLP projections + (GlobalMP + LocalMP)×depth + PoolingNodes + output MLP),
transfers ALL weights, and verifies final output.

NOTE: Both Torch MXMNetLocalMP and Keras MXMLocalMP now use the same
DimeNetPP angle convention: angle_index = [ji, kj]. Both gather at index [1]
(kj / send) and pool at index [0] (ji / receive). The LocalMP is implemented
manually in the Keras stack to keep weight transfer explicit.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np
from dataclasses import dataclass

from alignment_thresholds import get_thresholds
from model_alignment_utils import (
    copy_dense, copy_mlp, copy_graph_mlp,
    compare_outputs, keras_to_torch,
)

import keras
from keras import ops
from keras.layers import Dense, Subtract

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from kgcnn.layers.geom import (
    NodePosition, NodeDistanceEuclidean, EdgeAngle,
    BesselBasisLayer as KerasBesselBasisLayer,
    SphericalBasisLayer as KerasSphericalBasisLayer,
)
from kgcnn.layers.mlp import MLP as KerasMLP, GraphMLP as KerasGraphMLP
from kgcnn.layers.pooling import PoolingNodes as KerasPoolingNodes
from kgcnn.layers.update import ResidualLayer as KerasResidualLayer
from kgcnn.ops.scatter import scatter_reduce_sum
from kgcnn.literature.DimeNetPP._layers import EmbeddingDimeBlock as KerasEmbeddingDimeBlock
from kgcnn.literature.MXMNet._layers import MXMGlobalMP as KerasMXMGlobalMP
from kgcnn_torch.models.mxmnet import MXMNetModel


@dataclass
class Config:
    node_dim: int = 32
    depth: int = 2
    units: int = 32
    num_radial: int = 6
    num_spherical: int = 3
    num_radial_spherical: int = 4
    cutoff: float = 5.0
    envelope_exponent: int = 5
    activation: str = "swish"
    mp_pooling: str = "sum"
    global_mp_pooling: str = "mean"
    node_pooling: str = "sum"
    output_units: list = None
    output_activation: str = "swish"
    num_targets: int = 2
    num_embeddings: int = 95
    n_nodes: int = 16
    n_edges: int = 40
    batch_size: int = 4
    seed: int = 42

    def __post_init__(self):
        if self.output_units is None:
            self.output_units = []


def generate_angle_index(edge_index_torch):
    """Generate DimeNetPP-style angle indices from edge_index [src, dst].

    For each edge ji (src[p]=j, dst[p]=i), find all edges kj (dst[q]=j)
    that share node j. Returns (2, K): [ji_edge_idx, kj_edge_idx].
    """
    src = edge_index_torch[0]
    dst = edge_index_torch[1]
    M = src.shape[0]

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
                    ji_list.append(p)    # ji edge (receive)
                    kj_list.append(q)    # kj edge (send)

    return torch.tensor([ji_list, kj_list], dtype=torch.long)


class KerasLocalMPBlock:
    """Manual Keras LocalMP block matching Torch MXMNetLocalMP convention.

    Uses DimeNetPP angle convention: angle_index = [ji, kj].
    Gathers at angle_index[1] (kj / send) and pools to angle_index[0]
    (ji / receive), matching both Torch and Keras MXMLocalMP.
    """

    def __init__(self, units, output_units, activation, pooling_method="sum"):
        self.dim = units
        self.h_mlp = KerasGraphMLP(units, activation=activation)
        self.mlp_kj = KerasGraphMLP([units], activation=activation)
        self.mlp_ji_1 = KerasGraphMLP([units], activation=activation)
        self.mlp_ji_2 = KerasGraphMLP([units], activation=activation)
        self.mlp_jj = KerasGraphMLP([units], activation=activation)
        self.mlp_sbf1 = KerasGraphMLP([units, units], activation=activation)
        self.mlp_sbf2 = KerasGraphMLP([units, units], activation=activation)
        self.lin_rbf1 = Dense(units, use_bias=False, activation="linear")
        self.lin_rbf2 = Dense(units, use_bias=False, activation="linear")
        self.res1 = KerasResidualLayer(units)
        self.res2 = KerasResidualLayer(units)
        self.res3 = KerasResidualLayer(units)
        self.lin_rbf_out = Dense(units, use_bias=False, activation="linear")
        self.y_mlp = KerasGraphMLP([units, units, units], activation=activation)
        self.y_W = Dense(output_units, activation="linear",
                         kernel_initializer="zeros")
        self.pooling_method = pooling_method

    def forward(self, h, rbf, sbf1, sbf2, edge_index_keras,
                angle_idx_1, angle_idx_2):
        """Forward pass.

        Args:
            h: Node features (N, units).
            rbf: Projected RBF features (M, units).
            sbf1, sbf2: Projected SBF features (K, units).
            edge_index_keras: (2, M) Keras convention [target, source].
            angle_idx_1, angle_idx_2: (2, K) DimeNetPP convention [ji, kj].
        """
        num_nodes = h.shape[0]
        num_edges = edge_index_keras.shape[1]
        res_h = h

        h = self.h_mlp(h)

        # Gather node features for edges (Keras edge convention)
        hi = ops.take(h, edge_index_keras[0], axis=0)  # target
        hj = ops.take(h, edge_index_keras[1], axis=0)  # source
        m = ops.concatenate([hi, hj, rbf], axis=-1)

        # === MP1: m_kj path ===
        m_kj = self.mlp_kj(m)
        w_rbf1 = self.lin_rbf1(rbf)
        m_kj = m_kj * w_rbf1

        # Gather at angle_idx_1[1] (kj / send edge)
        m_kj_angle = ops.take(m_kj, angle_idx_1[1], axis=0)
        sw_sbf1 = self.mlp_sbf1(sbf1)
        m_kj_angle = m_kj_angle * sw_sbf1

        # Aggregate to angle_idx_1[0] (ji / receive edge)
        m_kj_agg = scatter_reduce_sum(
            angle_idx_1[0], m_kj_angle, (num_edges, self.dim))

        m_ji_1 = self.mlp_ji_1(m)
        m = m_ji_1 + m_kj_agg

        # === MP2: m_jj path ===
        m_jj = self.mlp_jj(m)
        w_rbf2 = self.lin_rbf2(rbf)
        m_jj = m_jj * w_rbf2

        m_jj_angle = ops.take(m_jj, angle_idx_2[1], axis=0)
        sw_sbf2 = self.mlp_sbf2(sbf2)
        m_jj_angle = m_jj_angle * sw_sbf2

        m_jj_agg = scatter_reduce_sum(
            angle_idx_2[0], m_jj_angle, (num_edges, self.dim))

        m_ji_2 = self.mlp_ji_2(m)
        m = m_ji_2 + m_jj_agg

        # === Aggregate to nodes ===
        w_rbf = self.lin_rbf_out(rbf)
        m = w_rbf * m

        # Pool to target nodes (Keras edge_index[0] = target)
        h_new = scatter_reduce_sum(
            edge_index_keras[0], m, (num_nodes, self.dim))

        # Update
        h_new = self.res1(h_new)
        h_new = self.h_mlp(h_new)  # Shared h_mlp
        h_new = h_new + res_h
        h_new = self.res2(h_new)
        h_new = self.res3(h_new)

        # Output module
        y = self.y_mlp(h_new)
        y = self.y_W(y)

        return h_new, y


class KerasMXMNetFullStack:
    """Full Keras MXMNet model stack mirroring MXMNetModel architecture."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.node_embedding = KerasEmbeddingDimeBlock(
            input_dim=cfg.num_embeddings, output_dim=cfg.node_dim,
        )
        # Torch uses Identity when node_dim == units
        if cfg.node_dim != cfg.units:
            self.dense_in = Dense(cfg.units)
        else:
            self.dense_in = None

        self.bessel_basis_local = KerasBesselBasisLayer(
            num_radial=cfg.num_radial, cutoff=cfg.cutoff,
            envelope_exponent=cfg.envelope_exponent,
        )
        self.bessel_basis_global = KerasBesselBasisLayer(
            num_radial=cfg.num_radial, cutoff=cfg.cutoff,
            envelope_exponent=cfg.envelope_exponent,
        )
        self.spherical_basis = KerasSphericalBasisLayer(
            num_spherical=cfg.num_spherical,
            num_radial=cfg.num_radial_spherical,
            cutoff=cfg.cutoff, envelope_exponent=cfg.envelope_exponent,
        )

        # MLP projections (separate sbf_1 and sbf_2, matching Torch)
        self.mlp_rbf = KerasGraphMLP([cfg.units], activation=cfg.activation)
        self.mlp_sbf_1 = KerasGraphMLP([cfg.units], activation=cfg.activation)
        self.mlp_sbf_2 = KerasGraphMLP([cfg.units], activation=cfg.activation)
        self.mlp_rbf_global = KerasGraphMLP(
            [cfg.units], activation=cfg.activation)

        # GlobalMP blocks (use Keras MXMGlobalMP with Keras edge convention)
        self.global_mp_blocks = []
        for _ in range(cfg.depth):
            self.global_mp_blocks.append(KerasMXMGlobalMP(
                units=cfg.units, pooling_method=cfg.global_mp_pooling,
            ))

        # LocalMP blocks (manual implementation for angle_index convention)
        self.local_mp_blocks = []
        for _ in range(cfg.depth):
            self.local_mp_blocks.append(KerasLocalMPBlock(
                units=cfg.units, output_units=cfg.num_targets,
                activation=cfg.activation, pooling_method=cfg.mp_pooling,
            ))

        self.pooling = KerasPoolingNodes(pooling_method=cfg.node_pooling)

        if cfg.output_units:
            out_units = cfg.output_units + [cfg.num_targets]
            out_act = ([cfg.output_activation] * len(cfg.output_units)
                       + ["linear"])
            self.output_mlp = KerasMLP(
                units=out_units, activation=out_act)
        else:
            out_units = [cfg.num_targets]
            out_act = ["linear"]
            self.output_mlp = KerasMLP(
                units=out_units, activation=out_act)

    def forward(self, z, pos, edge_index_keras, angle_idx_1, angle_idx_2,
                batch_id_node, count_nodes):
        """Forward pass.

        Args:
            edge_index_keras: (2, M) Keras convention [target, source].
            angle_idx_1, angle_idx_2: (2, K) DimeNetPP convention [ji, kj].
        """
        n = self.node_embedding(z)

        # Geometry: local edges
        pos1_l, pos2_l = NodePosition()([pos, edge_index_keras])
        d_l = NodeDistanceEuclidean()([pos1_l, pos2_l])
        rbf_l_raw = self.bessel_basis_local(d_l)

        v12_l = Subtract()([pos1_l, pos2_l])
        # EdgeAngle uses angle_idx[0] for v1 and [1] for v2
        # Same angle_index as Torch → angles are symmetric → same values
        a_l_1 = EdgeAngle()([v12_l, angle_idx_1])
        a_l_2 = EdgeAngle(vector_scale=[1.0, -1.0])([v12_l, angle_idx_2])
        sbf_l_1_raw = self.spherical_basis([d_l, a_l_1, angle_idx_1])
        sbf_l_2_raw = self.spherical_basis([d_l, a_l_2, angle_idx_2])

        # Global edges (same as local for simplicity)
        rbf_g_raw = rbf_l_raw

        # Project basis features
        rbf_l = self.mlp_rbf(rbf_l_raw)
        sbf_l_1 = self.mlp_sbf_1(sbf_l_1_raw)
        sbf_l_2 = self.mlp_sbf_2(sbf_l_2_raw)
        rbf_g = self.mlp_rbf_global(rbf_g_raw)

        # Initial projection (Identity when node_dim == units)
        if self.dense_in is not None:
            n = self.dense_in(n)

        # Message passing loop
        h = n
        nodes_list = []
        for i in range(self.cfg.depth):
            h = self.global_mp_blocks[i]([h, rbf_g, edge_index_keras])
            h, t = self.local_mp_blocks[i].forward(
                h, rbf_l, sbf_l_1, sbf_l_2,
                edge_index_keras, angle_idx_1, angle_idx_2)
            nodes_list.append(t)

        # Sum per-layer outputs
        out = nodes_list[0]
        for t in nodes_list[1:]:
            out = out + t

        out = self.pooling([count_nodes, out, batch_id_node])
        out = self.output_mlp(out)

        return out


# ---- Weight transfer helpers ----

def copy_residual_layer(torch_res, keras_res):
    """Copy ResidualLayer weights."""
    copy_dense(torch_res.dense_1, keras_res.dense_1)
    copy_dense(torch_res.dense_2, keras_res.dense_2)


def copy_seq_to_graphmlp(torch_seq, keras_gmlp):
    """Copy nn.Sequential([Linear, act, ...]) -> GraphMLP weights.

    For a single-layer Sequential: copy [0] -> mlp_dense_layer_list[0].
    For a two-layer Sequential: copy [0],[2] -> mlp_dense_layer_list[0],[1].
    """
    idx = 0
    keras_idx = 0
    while idx < len(torch_seq):
        module = torch_seq[idx]
        if isinstance(module, torch.nn.Linear):
            copy_dense(module, keras_gmlp.mlp_dense_layer_list[keras_idx])
            keras_idx += 1
        idx += 1


def copy_global_mp(torch_gmp, keras_gmp):
    """Copy MXMNetGlobalMP weights."""
    copy_seq_to_graphmlp(torch_gmp.h_mlp, keras_gmp.h_mlp)
    copy_residual_layer(torch_gmp.res1, keras_gmp.res1)
    copy_residual_layer(torch_gmp.res2, keras_gmp.res2)
    copy_residual_layer(torch_gmp.res3, keras_gmp.res3)
    copy_seq_to_graphmlp(torch_gmp.mlp, keras_gmp.mlp)
    copy_seq_to_graphmlp(torch_gmp.x_edge_mlp, keras_gmp.x_edge_mlp)
    copy_dense(torch_gmp.linear, keras_gmp.linear)


def copy_local_mp(torch_lmp, keras_lmp):
    """Copy MXMNetLocalMP weights to KerasLocalMPBlock."""
    copy_seq_to_graphmlp(torch_lmp.h_mlp, keras_lmp.h_mlp)
    copy_seq_to_graphmlp(torch_lmp.mlp_kj, keras_lmp.mlp_kj)
    copy_seq_to_graphmlp(torch_lmp.mlp_ji_1, keras_lmp.mlp_ji_1)
    copy_seq_to_graphmlp(torch_lmp.mlp_ji_2, keras_lmp.mlp_ji_2)
    copy_seq_to_graphmlp(torch_lmp.mlp_jj, keras_lmp.mlp_jj)
    copy_seq_to_graphmlp(torch_lmp.mlp_sbf1, keras_lmp.mlp_sbf1)
    copy_seq_to_graphmlp(torch_lmp.mlp_sbf2, keras_lmp.mlp_sbf2)
    copy_dense(torch_lmp.lin_rbf1, keras_lmp.lin_rbf1)
    copy_dense(torch_lmp.lin_rbf2, keras_lmp.lin_rbf2)
    copy_dense(torch_lmp.lin_rbf_out, keras_lmp.lin_rbf_out)
    copy_residual_layer(torch_lmp.res1, keras_lmp.res1)
    copy_residual_layer(torch_lmp.res2, keras_lmp.res2)
    copy_residual_layer(torch_lmp.res3, keras_lmp.res3)
    copy_mlp(torch_lmp.y_mlp, keras_lmp.y_mlp)
    copy_dense(torch_lmp.y_W, keras_lmp.y_W)


def transfer_all_weights(torch_model: MXMNetModel,
                         keras_stack: KerasMXMNetFullStack, cfg: Config):
    # EmbeddingDimeBlock
    w = torch_model.node_embedding.weight.detach().cpu().numpy()
    keras_stack.node_embedding.set_weights([w])

    # Initial projection (skip if Identity)
    if keras_stack.dense_in is not None:
        copy_dense(torch_model.dense_in, keras_stack.dense_in)

    # BesselBasisLayer frequencies
    freq_l = torch_model.bessel_basis_local.frequencies.detach().cpu().numpy()
    keras_stack.bessel_basis_local.set_weights([freq_l])
    freq_g = torch_model.bessel_basis_global.frequencies.detach().cpu().numpy()
    keras_stack.bessel_basis_global.set_weights([freq_g])

    # MLP projections (Sequential -> GraphMLP)
    copy_seq_to_graphmlp(torch_model.mlp_rbf, keras_stack.mlp_rbf)
    copy_seq_to_graphmlp(torch_model.mlp_sbf_1, keras_stack.mlp_sbf_1)
    copy_seq_to_graphmlp(torch_model.mlp_sbf_2, keras_stack.mlp_sbf_2)
    copy_seq_to_graphmlp(torch_model.mlp_rbf_global, keras_stack.mlp_rbf_global)

    # GlobalMP blocks
    for i in range(cfg.depth):
        copy_global_mp(torch_model.global_mp_blocks[i],
                       keras_stack.global_mp_blocks[i])

    # LocalMP blocks
    for i in range(cfg.depth):
        copy_local_mp(torch_model.local_mp_blocks[i],
                      keras_stack.local_mp_blocks[i])

    # Output MLP
    copy_mlp(torch_model.output_mlp, keras_stack.output_mlp)


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    from model_alignment_utils import make_disjoint_graph
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False,
    )

    # Generate angle indices in DimeNetPP convention [ji, kj]
    angle_idx = generate_angle_index(torch_data.edge_index)
    # Use same angle_index for both sets
    torch_data.angle_index_1 = angle_idx
    torch_data.angle_index_2 = angle_idx

    # Build Torch model (use_local_mp=True)
    torch_model = MXMNetModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        num_radial=cfg.num_radial,
        num_spherical=cfg.num_spherical,
        num_radial_spherical=cfg.num_radial_spherical,
        cutoff=cfg.cutoff,
        envelope_exponent=cfg.envelope_exponent,
        activation=cfg.activation,
        mp_pooling=cfg.mp_pooling,
        global_mp_pooling=cfg.global_mp_pooling,
        use_local_mp=True,
        node_pooling=cfg.node_pooling,
        output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets,
        output_embedding="graph",
        use_node_embedding=True,
        num_embeddings=cfg.num_embeddings,
        use_output_mlp=True,
    )
    torch_model.eval()

    # Build and initialize Keras stack
    keras_stack = KerasMXMNetFullStack(cfg)
    z_k = keras_data["z"]
    pos_k = keras_data["pos"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    # Initialize all layers with a forward pass
    _ = keras_stack.forward(z_k, pos_k, ei_k, angle_idx, angle_idx, bid_k, cn_k)

    transfer_all_weights(torch_model, keras_stack, cfg)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()

    keras_out = keras_to_torch(
        keras_stack.forward(z_k, pos_k, ei_k, angle_idx, angle_idx, bid_k, cn_k))

    print("MXMNet model-level alignment (Torch -> Keras):")
    compare_outputs("MXMNet_output", torch_out, keras_out, MAX_MAE, MAX_ABS)
    print("MXMNet model alignment PASSED.")


if __name__ == "__main__":
    main()
