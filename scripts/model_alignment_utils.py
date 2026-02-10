#!/usr/bin/env python3
"""Shared utilities for model-level alignment tests (Torch -> Keras weight transfer).

Provides reusable weight-copy helpers, comparison functions, and test-data
generators that are used by every ``align_<model>_model.py`` script.
"""
import os
import sys
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch

os.environ.setdefault("KERAS_BACKEND", "torch")
import keras  # noqa: E402
from keras import ops  # noqa: E402

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
for _p in (TORCH_REPO, KERAS_REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# Weight transfer: Torch -> Keras
# ---------------------------------------------------------------------------

def copy_dense(torch_linear: torch.nn.Linear, keras_dense):
    """Copy nn.Linear weights to Keras Dense (transpose kernel)."""
    kernel = torch_linear.weight.detach().cpu().numpy().T  # (out, in) -> (in, out)
    if torch_linear.bias is not None:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])
    else:
        keras_dense.set_weights([kernel])


def copy_embedding(torch_emb: torch.nn.Embedding, keras_emb):
    """Copy nn.Embedding weights to Keras Embedding (same shape)."""
    w = torch_emb.weight.detach().cpu().numpy()  # (V, D)
    keras_emb.set_weights([w])


def copy_batchnorm(torch_bn: torch.nn.BatchNorm1d, keras_bn):
    """Copy BatchNorm1d -> Keras BatchNormalization.

    Torch order: weight(gamma), bias(beta), running_mean, running_var.
    Keras order: gamma, beta, moving_mean, moving_variance.
    """
    gamma = torch_bn.weight.detach().cpu().numpy()
    beta = torch_bn.bias.detach().cpu().numpy()
    moving_mean = torch_bn.running_mean.detach().cpu().numpy()
    moving_var = torch_bn.running_var.detach().cpu().numpy()
    keras_bn.set_weights([gamma, beta, moving_mean, moving_var])


def copy_mlp(torch_mlp, keras_mlp):
    """Copy Torch MLP -> Keras MLP/GraphMLP weights.

    Torch MLP:  .linears[i] (nn.Linear), .norms[i] (may be nn.Identity or BN)
    Keras MLP:  .mlp_dense_layer_list[i], .mlp_norm_layer_list[i] (may be None)
    """
    depth = len(torch_mlp.linears)
    for i in range(depth):
        copy_dense(torch_mlp.linears[i], keras_mlp.mlp_dense_layer_list[i])
        # Handle normalization layers if present on both sides
        t_norm = torch_mlp.norms[i]
        k_norm = keras_mlp.mlp_norm_layer_list[i] if i < len(keras_mlp.mlp_norm_layer_list) else None
        if k_norm is not None and not isinstance(t_norm, torch.nn.Identity):
            if isinstance(t_norm, torch.nn.BatchNorm1d):
                copy_batchnorm(t_norm, k_norm)
            elif hasattr(t_norm, 'bn') and isinstance(t_norm.bn, torch.nn.BatchNorm1d):
                # GraphBatchNorm wraps .bn (nn.BatchNorm1d)
                copy_batchnorm(t_norm.bn, k_norm)


def copy_graph_mlp(torch_mlp, keras_graph_mlp):
    """Alias: same internal structure as copy_mlp."""
    copy_mlp(torch_mlp, keras_graph_mlp)


def copy_relational_dense(torch_rd, keras_rd):
    """Copy Torch RelationalDense -> Keras RelationalDense weights.

    Torch RelationalDense.weight: (R, in, out) — same shape as Keras kernel.
    """
    w = torch_rd.weight.detach().cpu().numpy()  # (R, in, out)
    weights = [w]
    if torch_rd.bias is not None:
        weights.append(torch_rd.bias.detach().cpu().numpy())
    keras_rd.set_weights(weights)


def copy_relational_mlp(torch_rmlp, keras_rmlp):
    """Copy Torch RelationalMLP -> Keras RelationalMLP weights.

    Torch RelationalMLP:  .layers[i] (RelationalDense)
    Keras RelationalMLP:  .mlp_dense_layer_list[i] (RelationalDense)
    """
    depth = torch_rmlp._depth
    for i in range(depth):
        copy_relational_dense(torch_rmlp.layers[i], keras_rmlp.mlp_dense_layer_list[i])


def copy_layernorm(torch_ln, keras_ln):
    """Copy nn.LayerNorm weights -> Keras GraphLayerNormalization.

    Torch LayerNorm: .weight (gamma), .bias (beta).
    """
    gamma = torch_ln.weight.detach().cpu().numpy()
    beta = torch_ln.bias.detach().cpu().numpy()
    keras_ln.set_weights([gamma, beta])


def copy_gru_cell(torch_gru_cell: torch.nn.GRUCell, keras_target):
    """Copy GRUCell weights: PyTorch (r,z,n) -> Keras (z,r,h) gate reorder.

    Args:
        torch_gru_cell: torch.nn.GRUCell instance.
        keras_target: Keras GRUUpdate layer (has .gru_cell) or raw GRUCell.
    """
    w_ih = torch_gru_cell.weight_ih.detach().cpu().numpy()  # (3u, in)
    w_hh = torch_gru_cell.weight_hh.detach().cpu().numpy()  # (3u, u)
    b_ih = torch_gru_cell.bias_ih.detach().cpu().numpy()    # (3u,)
    b_hh = torch_gru_cell.bias_hh.detach().cpu().numpy()    # (3u,)

    def _reorder(arr, axis=0):
        """Reorder gate chunks: PyTorch (r,z,n) -> Keras (z,r,h)."""
        r, z, n = np.split(arr, 3, axis=axis)
        return np.concatenate([z, r, n], axis=axis)

    kernel = _reorder(w_ih, axis=0).T             # (in, 3u)
    recurrent_kernel = _reorder(w_hh, axis=0).T   # (u, 3u)
    bias_input = _reorder(b_ih, axis=0)            # (3u,)
    bias_recurrent = _reorder(b_hh, axis=0)        # (3u,)
    bias = np.stack([bias_input, bias_recurrent], axis=0)  # (2, 3u)

    # Support both GRUUpdate (has .gru_cell) and raw GRUCell
    cell = getattr(keras_target, 'gru_cell', keras_target)
    cell.set_weights([kernel, recurrent_kernel, bias])


# ---------------------------------------------------------------------------
# Output comparison
# ---------------------------------------------------------------------------

def compare_outputs(name: str, ref: torch.Tensor, got: torch.Tensor,
                    max_mae: float, max_abs: float):
    """Compare two tensors and assert alignment thresholds.

    Args:
        name: Descriptive label for the comparison.
        ref: Reference tensor (Torch model output).
        got: Test tensor (Keras model output, converted to torch).
        max_mae: Maximum allowed mean absolute error.
        max_abs: Maximum allowed absolute difference.

    Raises:
        SystemExit: If thresholds are exceeded.
    """
    diff = (ref - got).abs()
    mae = float(diff.mean().item())
    rmse = float(torch.sqrt(((ref - got) ** 2).mean()).item())
    abs_max = float(diff.max().item())

    print(f"  {name:30s} | shape={tuple(ref.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={abs_max:.6e}")

    if mae > max_mae or abs_max > max_abs:
        raise SystemExit(
            f"Alignment FAILED for '{name}': MAE={mae:.3e} (limit {max_mae:.1e}), "
            f"MAX={abs_max:.3e} (limit {max_abs:.1e})"
        )
    print(f"  => PASS")


def keras_to_torch(keras_tensor) -> torch.Tensor:
    """Convert a Keras output tensor to a plain torch.Tensor on CPU."""
    return torch.as_tensor(ops.convert_to_numpy(keras_tensor))


# ---------------------------------------------------------------------------
# Test data generators
# ---------------------------------------------------------------------------

def make_disjoint_graph(n_nodes: int = 20, n_edges: int = 60,
                        batch_size: int = 4, node_dim: int = 16,
                        edge_dim: int = 8, node_vocab: int = 95,
                        include_pos: bool = False,
                        include_edge_attr: bool = True,
                        seed: int = 42):
    """Generate random disjoint graph data for both Torch and Keras.

    Returns:
        torch_data: SimpleNamespace with .z, .edge_index (src,dst), .edge_attr,
                     .batch, .pos, .edge_weight
        keras_inputs: dict with keys matching model_disjoint() inputs
    """
    from types import SimpleNamespace
    torch.manual_seed(seed)

    # Node assignments: split nodes roughly evenly across batch
    nodes_per_graph = n_nodes // batch_size
    batch_node = torch.cat([
        torch.full((nodes_per_graph,), i, dtype=torch.long)
        for i in range(batch_size)
    ])
    # Handle remainder
    remainder = n_nodes - nodes_per_graph * batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node,
                                torch.full((remainder,), batch_size - 1, dtype=torch.long)])
    total_nodes = batch_node.shape[0]

    count_nodes = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_nodes[i] = (batch_node == i).sum()

    # Atomic numbers
    z = torch.randint(1, min(node_vocab, 30), (total_nodes,), dtype=torch.long)

    # Edge indices (random within graph boundaries, no self-loops)
    src_list, dst_list, batch_edge_list = [], [], []
    edges_per_graph = n_edges // batch_size
    node_offset = 0
    for g in range(batch_size):
        g_nodes = int(count_nodes[g].item())
        n_e = edges_per_graph if g < batch_size - 1 else n_edges - edges_per_graph * (batch_size - 1)
        # Generate edges without self-loops
        s_all, d_all = [], []
        while len(s_all) < n_e:
            n_try = (n_e - len(s_all)) * 2  # oversample
            s_cand = torch.randint(0, g_nodes, (n_try,))
            d_cand = torch.randint(0, g_nodes, (n_try,))
            mask = s_cand != d_cand  # reject self-loops
            s_all.extend(s_cand[mask].tolist())
            d_all.extend(d_cand[mask].tolist())
        s = torch.tensor(s_all[:n_e], dtype=torch.long) + node_offset
        d = torch.tensor(d_all[:n_e], dtype=torch.long) + node_offset
        src_list.append(s)
        dst_list.append(d)
        batch_edge_list.append(torch.full((n_e,), g, dtype=torch.long))
        node_offset += g_nodes

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    batch_edge = torch.cat(batch_edge_list)
    total_edges = src.shape[0]

    count_edges = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_edges[i] = (batch_edge == i).sum()

    # Edge index: Torch = [src, dst], Keras = [dst, src]
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    # Edge attributes
    edge_attr = torch.randn(total_edges, edge_dim) if include_edge_attr else None
    edge_weight = torch.ones(total_edges, 1)

    # Positions
    pos = torch.randn(total_nodes, 3) if include_pos else None

    # Torch SimpleNamespace (PyG-like)
    torch_data = SimpleNamespace(
        z=z,
        edge_index=edge_index_torch,
        edge_attr=edge_attr,
        edge_weight=edge_weight,
        batch=batch_node,
        pos=pos,
    )

    # Keras inputs dict
    keras_data = {
        "z": z,
        "edge_index": edge_index_keras,
        "edge_attr": edge_attr,
        "edge_weight": edge_weight,
        "batch_id_node": batch_node,
        "batch_id_edge": batch_edge,
        "count_nodes": count_nodes,
        "count_edges": count_edges,
        "pos": pos,
    }

    return torch_data, keras_data


def make_disjoint_graph_relational(n_nodes: int = 20, n_edges: int = 60,
                                    batch_size: int = 4, node_dim: int = 16,
                                    num_relations: int = 5, node_vocab: int = 95,
                                    include_edge_weight: bool = True,
                                    seed: int = 42):
    """Generate random disjoint relational graph data for RGCN/GNNFilm models.

    Returns:
        torch_data: SimpleNamespace with .z, .edge_index, .edge_type, .edge_attr, .batch
        keras_data: dict with z, edge_index, edge_type, edge_attr, batch_id_node, count_nodes
    """
    from types import SimpleNamespace
    torch.manual_seed(seed)

    # Node assignments
    nodes_per_graph = n_nodes // batch_size
    batch_node = torch.cat([
        torch.full((nodes_per_graph,), i, dtype=torch.long)
        for i in range(batch_size)
    ])
    remainder = n_nodes - nodes_per_graph * batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node,
                                torch.full((remainder,), batch_size - 1, dtype=torch.long)])
    total_nodes = batch_node.shape[0]

    count_nodes = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_nodes[i] = (batch_node == i).sum()

    z = torch.randint(1, min(node_vocab, 30), (total_nodes,), dtype=torch.long)

    # Edge indices (no self-loops)
    src_list, dst_list, batch_edge_list = [], [], []
    edges_per_graph = n_edges // batch_size
    node_offset = 0
    for g in range(batch_size):
        g_nodes = int(count_nodes[g].item())
        n_e = edges_per_graph if g < batch_size - 1 else n_edges - edges_per_graph * (batch_size - 1)
        s_all, d_all = [], []
        while len(s_all) < n_e:
            n_try = (n_e - len(s_all)) * 2
            s_cand = torch.randint(0, g_nodes, (n_try,))
            d_cand = torch.randint(0, g_nodes, (n_try,))
            mask = s_cand != d_cand
            s_all.extend(s_cand[mask].tolist())
            d_all.extend(d_cand[mask].tolist())
        s = torch.tensor(s_all[:n_e], dtype=torch.long) + node_offset
        d = torch.tensor(d_all[:n_e], dtype=torch.long) + node_offset
        src_list.append(s)
        dst_list.append(d)
        batch_edge_list.append(torch.full((n_e,), g, dtype=torch.long))
        node_offset += g_nodes

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    batch_edge = torch.cat(batch_edge_list)
    total_edges = src.shape[0]

    count_edges = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_edges[i] = (batch_edge == i).sum()

    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    # Edge types (relation indices)
    edge_type = torch.randint(0, num_relations, (total_edges,), dtype=torch.long)

    # Edge weights
    edge_attr = torch.ones(total_edges, 1) if include_edge_weight else None

    torch_data = SimpleNamespace(
        z=z,
        edge_index=edge_index_torch,
        edge_type=edge_type,
        edge_attr=edge_attr,
        batch=batch_node,
    )

    keras_data = {
        "z": z,
        "edge_index": edge_index_keras,
        "edge_type": edge_type,
        "edge_attr": edge_attr,
        "batch_id_node": batch_node,
        "batch_id_edge": batch_edge,
        "count_nodes": count_nodes,
        "count_edges": count_edges,
    }

    return torch_data, keras_data


def make_disjoint_graph_directed(n_nodes: int = 20, n_edges_per_dir: int = 30,
                                  batch_size: int = 4, node_dim: int = 16,
                                  edge_dim: int = 8, node_vocab: int = 95,
                                  include_edge_attr: bool = True,
                                  seed: int = 42):
    """Generate random directed disjoint graph with reverse edge pairs.

    Creates edges (u->v) and their reverse (v->u) side by side so that
    edge_pair_index[i] = i+1 if i is even, i-1 if i is odd (alternating pairs).
    Total edges = 2 * n_edges_per_dir.

    Returns:
        torch_data: SimpleNamespace with .z, .edge_index (src,dst), .edge_attr,
                     .edge_pair_index, .batch
        keras_data: dict with z, edge_index (dst,src), edge_attr, edge_pair_index,
                     batch_id_node, batch_id_edge, count_nodes, count_edges
    """
    from types import SimpleNamespace
    torch.manual_seed(seed)

    # Node assignments
    nodes_per_graph = n_nodes // batch_size
    batch_node = torch.cat([
        torch.full((nodes_per_graph,), i, dtype=torch.long)
        for i in range(batch_size)
    ])
    remainder = n_nodes - nodes_per_graph * batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node,
                                torch.full((remainder,), batch_size - 1, dtype=torch.long)])
    total_nodes = batch_node.shape[0]

    count_nodes = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_nodes[i] = (batch_node == i).sum()

    z = torch.randint(1, min(node_vocab, 30), (total_nodes,), dtype=torch.long)

    # Build directed edges with reverse pairs
    src_list, dst_list, pair_list, batch_edge_list = [], [], [], []
    edges_per_graph = n_edges_per_dir // batch_size
    node_offset = 0
    edge_offset = 0
    for g in range(batch_size):
        g_nodes = int(count_nodes[g].item())
        n_e = edges_per_graph if g < batch_size - 1 else n_edges_per_dir - edges_per_graph * (batch_size - 1)
        # Generate edges without self-loops
        s_all, d_all = [], []
        while len(s_all) < n_e:
            n_try = (n_e - len(s_all)) * 2
            s_cand = torch.randint(0, g_nodes, (n_try,))
            d_cand = torch.randint(0, g_nodes, (n_try,))
            mask = s_cand != d_cand
            s_all.extend(s_cand[mask].tolist())
            d_all.extend(d_cand[mask].tolist())
        s = torch.tensor(s_all[:n_e], dtype=torch.long) + node_offset
        d = torch.tensor(d_all[:n_e], dtype=torch.long) + node_offset
        # Interleave forward and reverse edges
        for j in range(n_e):
            idx_fwd = edge_offset + 2 * j
            idx_rev = edge_offset + 2 * j + 1
            src_list.append(s[j].unsqueeze(0))
            dst_list.append(d[j].unsqueeze(0))
            src_list.append(d[j].unsqueeze(0))  # reverse
            dst_list.append(s[j].unsqueeze(0))  # reverse
            pair_list.append(torch.tensor([idx_rev]))  # fwd -> rev
            pair_list.append(torch.tensor([idx_fwd]))  # rev -> fwd
            batch_edge_list.append(torch.tensor([g]))
            batch_edge_list.append(torch.tensor([g]))
        node_offset += g_nodes
        edge_offset += 2 * n_e

    src = torch.cat(src_list)
    dst = torch.cat(dst_list)
    edge_pair_index = torch.cat(pair_list)
    batch_edge = torch.cat(batch_edge_list)
    total_edges = src.shape[0]

    count_edges = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_edges[i] = (batch_edge == i).sum()

    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    edge_attr = torch.randn(total_edges, edge_dim) if include_edge_attr else None

    # Keras edge_pair_index: shape (1, M) with extra axis
    edge_pair_index_keras = edge_pair_index.unsqueeze(0)  # (1, M)

    torch_data = SimpleNamespace(
        z=z,
        edge_index=edge_index_torch,
        edge_attr=edge_attr,
        edge_pair_index=edge_pair_index,
        batch=batch_node,
    )

    keras_data = {
        "z": z,
        "edge_index": edge_index_keras,
        "edge_attr": edge_attr,
        "edge_pair_index": edge_pair_index_keras,
        "batch_id_node": batch_node,
        "batch_id_edge": batch_edge,
        "count_nodes": count_nodes,
        "count_edges": count_edges,
    }

    return torch_data, keras_data
