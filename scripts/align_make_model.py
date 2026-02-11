#!/usr/bin/env python3
"""Alignment test: kgcnn_torch models vs kgcnn official make_model().

Builds both Torch models and Keras make_model() models with identical
architecture, transfers weights Torch→Keras, feeds the same real data
(MUTAG / ClinTox), and verifies outputs match. Then runs 50-step
training divergence.

Usage:
    KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES="" python scripts/align_make_model.py
"""
import os
import sys
import math
from types import SimpleNamespace

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.functional as F
import numpy as np

ROOT = "/home/yuanbai/Downloads/MLIPs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from train_alignment_utils import collect_keras_params as _collect_keras_params_orig
from model_alignment_utils import keras_to_torch

import keras as _keras
_BACKEND = _keras.backend.backend()


def collect_keras_params(model):
    """Backend-safe wrapper: returns torch Parameters for torch backend, [] otherwise."""
    if _BACKEND == "torch":
        return _collect_keras_params_orig(model)
    return []


PICKLE_DIR = os.path.join(ROOT, "kgcnn-torch", "datasets", "raw")
N_STEPS = 50
LR = 0.01
BATCH_SIZE = 8


# ---- Data loading ----

def load_pickle_as_pyg(pickle_name, edge_attr_keys=None):
    from kgcnn_torch.data.base import MemoryGraphList
    gl = MemoryGraphList()
    gl.load(os.path.join(PICKLE_DIR, pickle_name))
    return gl.to_pyg_list(edge_attr_keys=edge_attr_keys)


def make_batch(pyg_list, batch_size, seed=42):
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)
    indices = torch.randperm(len(pyg_list))[:batch_size]
    subset = [pyg_list[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


# ---- Format converters ----

def pyg_batch_to_torch_data(batch):
    """PyG batch → SimpleNamespace for Torch model."""
    num_edges = batch.edge_index.size(1)
    return SimpleNamespace(
        z=batch.z,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, 'edge_attr', None),
        edge_weight=torch.ones(num_edges, 1),
        batch=batch.batch,
        pos=getattr(batch, 'pos', None),
    )


def compute_batch_edge_pair_index(batch, batch_size):
    """Compute reverse edge mapping for a PyG batched graph.

    For each directed edge (src, dst), finds the edge (dst, src) and returns
    its global index. Returns a flat tensor of shape (total_M,).
    """
    ei = batch.edge_index
    src_np, dst_np = ei[0].numpy(), ei[1].numpy()
    node_batch = batch.batch.numpy()
    M = len(src_np)
    edge_dicts = [{} for _ in range(batch_size)]
    edge_graph = np.zeros(M, dtype=np.int64)
    for i in range(M):
        g = int(node_batch[src_np[i]])
        edge_graph[i] = g
        edge_dicts[g][(src_np[i], dst_np[i])] = i
    pair_idx = np.arange(M, dtype=np.int64)
    for i in range(M):
        g = int(edge_graph[i])
        rev = (dst_np[i], src_np[i])
        if rev in edge_dicts[g]:
            pair_idx[i] = edge_dicts[g][rev]
    return torch.from_numpy(pair_idx)


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
                    ji_list.append(p)
                    kj_list.append(q)
    return torch.tensor([ji_list, kj_list], dtype=torch.long)


def generate_angle_index_nodes(edge_index_torch, num_nodes):
    """Generate HDNNP2nd-style angle indices (atom triplets) from edge_index.

    For each node i, enumerate all (j, k) neighbor pairs where j and k
    are both connected to i. Returns (3, K): [center, nbr1, nbr2].
    PyG convention: edge_index[0]=src, edge_index[1]=dst. A src->dst edge
    means src is a neighbor of dst.
    """
    src = edge_index_torch[0].numpy()
    dst = edge_index_torch[1].numpy()
    neighbors = {}
    for idx in range(len(src)):
        s, d = int(src[idx]), int(dst[idx])
        neighbors.setdefault(d, []).append(s)
    center_list, nbr1_list, nbr2_list = [], [], []
    for i in range(num_nodes):
        nbrs = neighbors.get(i, [])
        for a in range(len(nbrs)):
            for b in range(a + 1, len(nbrs)):
                center_list.append(i)
                nbr1_list.append(nbrs[a])
                nbr2_list.append(nbrs[b])
    return torch.tensor([center_list, nbr1_list, nbr2_list], dtype=torch.long)


def pyg_batch_to_padded_kgcnn(batch, batch_size, input_names):
    """PyG batch → list of padded numpy arrays for Keras make_model().

    The Keras model expects padded inputs like:
      [node_number (B, max_N), edge_weights (B, max_M, 1),
       edge_indices (B, max_M, 2), total_nodes (B,), total_edges (B,)]

    Args:
        batch: PyG Batch object
        batch_size: Number of graphs in batch
        input_names: List of input name strings matching make_model's inputs config

    Returns:
        List of numpy arrays in the order matching input_names
    """
    # Compute per-graph node/edge counts
    nodes_per_graph = []
    edges_per_graph = []
    for i in range(batch_size):
        node_mask = batch.batch == i
        nodes_per_graph.append(int(node_mask.sum()))
        src_mask = batch.batch[batch.edge_index[0]] == i
        edges_per_graph.append(int(src_mask.sum()))

    max_nodes = max(nodes_per_graph)
    max_edges = max(edges_per_graph)

    # Split node/edge data per graph
    node_offsets = [0] + list(np.cumsum(nodes_per_graph[:-1]))
    edge_offsets = [0] + list(np.cumsum(edges_per_graph[:-1]))

    result = {}

    # node_number: (B, max_N) int64
    node_number_padded = np.zeros((batch_size, max_nodes), dtype='int64')
    for i in range(batch_size):
        start = node_offsets[i]
        n = nodes_per_graph[i]
        node_number_padded[i, :n] = batch.z[start:start + n].numpy()
    result["node_number"] = node_number_padded

    # node_attributes: (B, max_N, F) float32 — if x is available
    if hasattr(batch, 'x') and batch.x is not None:
        feat_dim = batch.x.size(-1)
        node_attr_padded = np.zeros((batch_size, max_nodes, feat_dim), dtype='float32')
        for i in range(batch_size):
            start = node_offsets[i]
            n = nodes_per_graph[i]
            node_attr_padded[i, :n] = batch.x[start:start + n].numpy()
        result["node_attributes"] = node_attr_padded

    # edge_weights: (B, max_M, 1) float32
    edge_weights_padded = np.zeros((batch_size, max_edges, 1), dtype='float32')
    for i in range(batch_size):
        e = edges_per_graph[i]
        edge_weights_padded[i, :e, 0] = 1.0  # unit weights
    result["edge_weights"] = edge_weights_padded

    # edge_number: (B, max_M) int64 — for models using edge embedding
    if hasattr(batch, 'edge_type') and batch.edge_type is not None:
        edge_number_padded = np.zeros((batch_size, max_edges), dtype='int64')
        edge_type_flat = batch.edge_type.numpy()
        for i in range(batch_size):
            start = edge_offsets[i]
            e = edges_per_graph[i]
            edge_number_padded[i, :e] = edge_type_flat[start:start + e]
        result["edge_number"] = edge_number_padded

    # edge_attributes: (B, max_M, D) float32
    if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
        edge_dim = batch.edge_attr.size(-1)
        edge_attr_padded = np.zeros((batch_size, max_edges, edge_dim), dtype='float32')
        for i in range(batch_size):
            start = edge_offsets[i]
            e = edges_per_graph[i]
            edge_attr_padded[i, :e] = batch.edge_attr[start:start + e].numpy()
        result["edge_attributes"] = edge_attr_padded

    # edge_indices: (B, max_M, 2) int64
    # KGCNN convention: [target, source] — edge_indices stored as (N_edges, 2) with [dst, src]
    # PyG convention: edge_index[0]=src, edge_index[1]=dst
    edge_indices_padded = np.zeros((batch_size, max_edges, 2), dtype='int64')
    ei = batch.edge_index.numpy()  # (2, total_edges) in [src, dst] order
    for i in range(batch_size):
        start = edge_offsets[i]
        e = edges_per_graph[i]
        node_offset = node_offsets[i]
        local_src = ei[0, start:start + e] - node_offset
        local_dst = ei[1, start:start + e] - node_offset
        # KGCNN expects [target, source] in the 2D array
        edge_indices_padded[i, :e, 0] = local_dst
        edge_indices_padded[i, :e, 1] = local_src
    result["edge_indices"] = edge_indices_padded

    # node_coordinates: (B, max_N, 3) float32
    if hasattr(batch, 'pos') and batch.pos is not None:
        pos_padded = np.zeros((batch_size, max_nodes, 3), dtype='float32')
        for i in range(batch_size):
            start = node_offsets[i]
            n = nodes_per_graph[i]
            pos_padded[i, :n] = batch.pos[start:start + n].numpy()
        result["node_coordinates"] = pos_padded

    # total_nodes, total_edges: (B,) int64
    result["total_nodes"] = np.array(nodes_per_graph, dtype='int64')
    result["total_edges"] = np.array(edges_per_graph, dtype='int64')

    # node_attributes: alias for node_number (used by GNNFilm)
    result["node_attributes"] = node_number_padded

    # edge_indices_reverse: (B, max_M, 1) int64 — local reverse edge index
    if "edge_indices_reverse" in input_names or "total_reverse" in input_names:
        edge_reverse_padded = np.zeros((batch_size, max_edges, 1), dtype='int64')
        for g in range(batch_size):
            start = edge_offsets[g]
            M_g = edges_per_graph[g]
            n_off = node_offsets[g]
            local_dict = {}
            for j in range(M_g):
                gj = start + j
                ls = int(ei[0, gj]) - n_off
                ld = int(ei[1, gj]) - n_off
                local_dict[(ls, ld)] = j
            for j in range(M_g):
                gj = start + j
                ls = int(ei[0, gj]) - n_off
                ld = int(ei[1, gj]) - n_off
                rev = (ld, ls)
                edge_reverse_padded[g, j, 0] = local_dict.get(rev, j)
        result["edge_indices_reverse"] = edge_reverse_padded
        result["total_reverse"] = np.array(edges_per_graph, dtype='int64')

    # edge_relations: (B, max_M) int64 — edge relation types
    if hasattr(batch, 'edge_type') and batch.edge_type is not None:
        edge_rel_padded = np.zeros((batch_size, max_edges), dtype='int64')
        et_flat = batch.edge_type.numpy()
        for g in range(batch_size):
            start = edge_offsets[g]
            e = edges_per_graph[g]
            edge_rel_padded[g, :e] = et_flat[start:start + e]
        result["edge_relations"] = edge_rel_padded

    # graph_attributes: (B, D) float32 — graph-level state
    if hasattr(batch, 'graph_state') and batch.graph_state is not None:
        result["graph_attributes"] = batch.graph_state.numpy().astype('float32')

    # graph_labels: (B, 1) float32 — dummy labels (e.g., for MEGAN explanation loss)
    if "graph_labels" in input_names:
        result["graph_labels"] = np.zeros((batch_size, 1), dtype='float32')

    # charge: (B, 1) float32 — graph-level charge (e.g., for MEGNet state)
    if "charge" in input_names:
        result["charge"] = np.zeros((batch_size, 1), dtype='float32')

    # Build output list in order of input_names
    return [result[name] for name in input_names]


# ---- Weight transfer: Torch → Keras make_model ----

def transfer_weights_to_make_model(torch_model, keras_model, model_type, depth):
    """Transfer weights from a kgcnn_torch model to a Keras make_model().

    Works by matching layer types and positions in the Keras model's layer list.
    """
    layers_by_type = {}
    for layer in keras_model.layers:
        cls = layer.__class__.__name__
        layers_by_type.setdefault(cls, []).append(layer)

    dispatch = {
        "GCN": _transfer_gcn, "GIN": _transfer_gin, "GIN_edge": _transfer_gin_edge,
        "GAT": _transfer_gat,
        "GATv2": _transfer_gatv2, "SchNet": _transfer_schnet,
        "AttentiveFP": _transfer_attentivefp, "GraphSAGE": _transfer_graphsage,
        "MEGAN": _transfer_megan, "rGIN": _transfer_rgin,
        "EGNN": _transfer_egnn, "PAiNN": _transfer_painn, "NMPN": _transfer_nmpn,
        "DMPNN": _transfer_dmpnn, "CMPNN": _transfer_cmpnn, "DGIN": _transfer_dgin,
        "RGCN": _transfer_rgcn, "GNNFilm": _transfer_gnnfilm, "INorp": _transfer_inorp,
        "MEGNet": _transfer_megnet, "CGCNN": _transfer_cgcnn,
        "HamNet": _transfer_hamnet, "DimeNetPP": _transfer_dimenetpp,
        "HDNNP2nd": _transfer_hdnnp2nd, "MAT": _transfer_mat,
        "MoGAT": _transfer_mogat, "MXMNet": _transfer_mxmnet,
    }
    fn = dispatch.get(model_type)
    if fn is None:
        raise ValueError(f"Unsupported model_type: {model_type}")
    fn(torch_model, layers_by_type, depth)


def _w(t):
    """Torch param → numpy, transposed for Dense (out,in) → Keras (in,out)."""
    return t.detach().cpu().numpy().T


def _b(t):
    """Torch bias → numpy."""
    return t.detach().cpu().numpy()


def _set_dense(torch_linear, keras_dense):
    """Copy nn.Linear → keras.layers.Dense."""
    ws = [_w(torch_linear.weight)]
    if torch_linear.bias is not None:
        ws.append(_b(torch_linear.bias))
    keras_dense.set_weights(ws)


def _set_embedding(torch_emb, keras_emb):
    """Copy nn.Embedding → kgcnn Embedding."""
    keras_emb.set_weights([torch_emb.weight.detach().cpu().numpy()])


def _set_mlp(torch_mlp, keras_mlp):
    """Copy Torch MLP → Keras MLP (all Dense layers)."""
    ws = []
    for lin in torch_mlp.linears:
        ws.append(_w(lin.weight))
        if lin.bias is not None:
            ws.append(_b(lin.bias))
    keras_mlp.set_weights(ws)


def _set_relational_dense(torch_rd, keras_rd):
    """Copy Torch RelationalDense → Keras RelationalDense.

    Weight is (num_relations, in_features, out_features) — no transpose needed.
    """
    ws = [torch_rd.weight.detach().cpu().numpy()]
    if torch_rd.bias is not None:
        ws.append(torch_rd.bias.detach().cpu().numpy())
    keras_rd.set_weights(ws)


def _set_layernorm(torch_ln, keras_ln):
    """Copy torch LayerNorm → Keras LayerNormalization."""
    ws = [torch_ln.weight.detach().cpu().numpy(),
          torch_ln.bias.detach().cpu().numpy()]
    keras_ln.set_weights(ws)


def _set_relational_mlp(torch_rmlp, keras_rmlp):
    """Copy Torch RelationalMLP → Keras RelationalMLP (all RelationalDense layers)."""
    for t_rd, k_rd in zip(torch_rmlp.layers, keras_rmlp.mlp_dense_layer_list):
        _set_relational_dense(t_rd, k_rd)


def _transfer_gcn(torch_model, layers_by_type, depth):
    # Embedding
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Dense (input projection)
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # GCN convolutions
    for i in range(depth):
        _set_dense(torch_model.convs[i].linear, layers_by_type["GCN"][i])
    # Output MLP
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_gin(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # GIN conv layers have epsilon parameter
    for i in range(depth):
        layers_by_type["GIN"][i].set_weights(
            [torch_model.convs[i].eps.detach().cpu().numpy().reshape(())])
    # GraphMLP shows up as "MLP" in layers_by_type.
    # MLP order: [gin_mlp_0..gin_mlp_{d-1}, readout_0..readout_d, output]
    all_mlps = layers_by_type["MLP"]
    # gin_mlps: MLP[0..depth-1]
    for i in range(depth):
        _set_mlp(torch_model.gin_mlps[i], all_mlps[i])
    # readout_mlps: MLP[depth..2*depth]
    for i in range(depth + 1):
        _set_mlp(torch_model.readout_mlps[i], all_mlps[depth + i])
    # output_mlp: MLP[2*depth+1]
    _set_mlp(torch_model.output_mlp, all_mlps[2 * depth + 1])


def _transfer_gin_edge(torch_model, layers_by_type, depth):
    embeddings = layers_by_type["Embedding"]
    _set_embedding(torch_model.node_embedding, embeddings[0])
    _set_embedding(torch_model.edge_embedding, embeddings[1])
    denses = layers_by_type["Dense"]
    _set_dense(torch_model.dense_in, denses[0])
    _set_dense(torch_model.edge_proj, denses[1])
    # GINE conv layers have epsilon parameter
    for i in range(depth):
        layers_by_type["GINE"][i].set_weights(
            [torch_model.convs[i].eps.detach().cpu().numpy().reshape(())])
    # MLP order: [gin_mlp_0..gin_mlp_{d-1}, readout_0..readout_d, output]
    all_mlps = layers_by_type["MLP"]
    for i in range(depth):
        _set_mlp(torch_model.gin_mlps[i], all_mlps[i])
    for i in range(depth + 1):
        _set_mlp(torch_model.readout_mlps[i], all_mlps[depth + i])
    _set_mlp(torch_model.output_mlp, all_mlps[2 * depth + 1])


def _transfer_gat(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # AttentionHeadGAT layers
    heads = layers_by_type.get("AttentionHeadGAT", [])
    idx = 0
    for i in range(depth):
        n_heads = len(torch_model.attention_layers[i])
        for j in range(n_heads):
            t_head = torch_model.attention_layers[i][j]
            k_head = heads[idx]
            _set_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
            _set_dense(t_head.linear_alpha, k_head.lay_alpha)
            idx += 1
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_gatv2(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    heads = layers_by_type.get("AttentionHeadGATV2", [])
    idx = 0
    for i in range(depth):
        n_heads = len(torch_model.attention_layers[i])
        for j in range(n_heads):
            t_head = torch_model.attention_layers[i][j]
            k_head = heads[idx]
            _set_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
            _set_dense(t_head.alpha_activation[0], k_head.lay_alpha_activation)
            _set_dense(t_head.linear_alpha, k_head.lay_alpha)
            idx += 1
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_schnet(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Initial Dense projection (model level: embedding_dim → units)
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # Interactions
    interactions = layers_by_type.get("SchNetInteraction", [])
    for i in range(depth):
        t_int = torch_model.interactions[i]
        k_int = interactions[i]
        # Keras SchNetInteraction sublayers:
        #   lay_cfconv (SchNetCFconv) → lay_dense1, lay_dense2 (filter network)
        #   lay_dense1 (atom-wise, no bias)
        #   lay_dense2 (activation, with bias)
        #   lay_dense3 (linear, with bias)
        # Torch SchNetInteraction attrs:
        #   dense_in (no bias) → keras lay_dense1
        #   cfconv.dense1, cfconv.dense2 → keras lay_cfconv.lay_dense1/2
        #   dense1 (with bias) → keras lay_dense2
        #   dense2 (with bias) → keras lay_dense3
        _set_dense(t_int.cfconv.dense1, k_int.lay_cfconv.lay_dense1)
        _set_dense(t_int.cfconv.dense2, k_int.lay_cfconv.lay_dense2)
        _set_dense(t_int.dense_in, k_int.lay_dense1)
        _set_dense(t_int.dense1, k_int.lay_dense2)
        _set_dense(t_int.dense2, k_int.lay_dense3)
    # GraphMLP shows up as "MLP" in layers_by_type
    # MLP[0] = last_mlp, MLP[1] = output_mlp
    _set_mlp(torch_model.last_mlp, layers_by_type["MLP"][0])
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][1])


def _transfer_attentivefp(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # AttentiveHeadFP layers
    heads = layers_by_type.get("AttentiveHeadFP", [])
    for i, k_head in enumerate(heads):
        t_head = torch_model.attention_layers[i]
        _set_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
        _set_dense(t_head.alpha_activation[0], k_head.lay_alpha_activation)
        _set_dense(t_head.linear_alpha, k_head.lay_alpha)
        if i == 0 and t_head.use_edge_features:
            _set_dense(t_head.fc1[0], k_head.lay_fc1)
            _set_dense(t_head.fc2[0], k_head.lay_fc2)
    # GRU updates — pass GRUUpdate wrapper (has .gru_cell), not the cell itself
    gru_updates = layers_by_type.get("GRUUpdate", [])
    for i, k_gru in enumerate(gru_updates):
        _copy_gru_cell(torch_model.gru_updates[i], k_gru)
    # PoolingNodesAttentive — gru is a raw GRUCell, not wrapped
    pools = layers_by_type.get("PoolingNodesAttentive", [])
    if pools:
        t_pool = torch_model.attentive_pooling
        k_pool = pools[0]
        _set_dense(t_pool.linear_trafo, k_pool.lay_linear_trafo)
        _set_dense(t_pool.lay_alpha[0], k_pool.lay_alpha)
        _copy_gru_cell_raw(t_pool.gru, k_pool.lay_gru)
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _copy_gru_cell(torch_gru_update, keras_gru_update):
    """Copy torch GRUUpdate → keras GRUUpdate (wraps GRUCell)."""
    _copy_gru_cell_raw(torch_gru_update.gru_cell, keras_gru_update)


def _copy_gru_cell_raw(torch_gru_cell, keras_gru_layer):
    """Copy torch.nn.GRUCell weights to keras GRUUpdate/GRUCell.

    Gate reorder: PyTorch (r,z,n) -> Keras (z,r,h).
    """
    w_ih = torch_gru_cell.weight_ih.detach().cpu().numpy()
    w_hh = torch_gru_cell.weight_hh.detach().cpu().numpy()
    b_ih = torch_gru_cell.bias_ih.detach().cpu().numpy()
    b_hh = torch_gru_cell.bias_hh.detach().cpu().numpy()

    def _reorder(arr, axis=0):
        r, z, n = np.split(arr, 3, axis=axis)
        return np.concatenate([z, r, n], axis=axis)

    kernel = _reorder(w_ih, axis=0).T
    recurrent_kernel = _reorder(w_hh, axis=0).T
    bias = np.stack([_reorder(b_ih), _reorder(b_hh)], axis=0)
    keras_gru_layer.set_weights([kernel, recurrent_kernel, bias])


def _transfer_graphsage(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # GraphMLP shows up as "MLP" in layers_by_type.
    # MLP order: [edge_mlp_0, node_mlp_0, edge_mlp_1, node_mlp_1, ..., output_mlp]
    all_mlps = layers_by_type["MLP"]
    for i in range(depth):
        _set_mlp(torch_model.edge_mlps[i], all_mlps[i * 2])
        _set_mlp(torch_model.node_mlps[i], all_mlps[i * 2 + 1])
    # LayerNorm
    norms = layers_by_type.get("GraphLayerNormalization", [])
    for i in range(min(depth, len(norms))):
        t_norm = torch_model.layer_norms[i]
        k_norm = norms[i]
        # GraphLayerNorm wraps nn.LayerNorm as self.ln
        ws = [t_norm.ln.weight.detach().cpu().numpy(),
              t_norm.ln.bias.detach().cpu().numpy()]
        k_norm.set_weights(ws)
    _set_mlp(torch_model.output_mlp, all_mlps[2 * depth])


def _transfer_megan(torch_model, layers_by_type, depth):
    # MEGAN make_model wraps the MEGAN class (a Model subclass) as a single layer.
    # Access its sub-layers through the MEGAN layer.
    megan_layers = layers_by_type.get("MEGAN", [])
    if megan_layers:
        lbt = {}
        for layer in megan_layers[0].layers:
            cls = layer.__class__.__name__
            lbt.setdefault(cls, []).append(layer)
    else:
        lbt = layers_by_type
    if "Embedding" in lbt:
        _set_embedding(torch_model.node_embedding, lbt["Embedding"][0])
    # MultiHeadGATV2Layer
    mh_layers = lbt.get("MultiHeadGATV2Layer", [])
    for i, k_mh in enumerate(mh_layers):
        t_mh = torch_model.attention_layers[i]
        n_heads = len(t_mh.head_linears)
        for k in range(n_heads):
            _set_dense(t_mh.head_linears[k][0], k_mh.head_layers[k][0])
            _set_dense(t_mh.head_alpha_acts[k][0], k_mh.head_layers[k][1])
            _set_dense(t_mh.head_alphas[k], k_mh.head_layers[k][2])
    # Node importance Dense layers
    dense_layers = lbt.get("Dense", [])
    imp_len = len(torch_model.importance_mlp.linears)
    final_len = len(torch_model.output_mlp.linears)
    for i in range(imp_len):
        _set_dense(torch_model.importance_mlp.linears[i], dense_layers[i])
    for i in range(final_len):
        _set_dense(torch_model.output_mlp.linears[i], dense_layers[imp_len + i])


def _transfer_rgin(torch_model, layers_by_type, depth):
    # rGIN: same as GIN but uses rGIN layer type; embedding is optional
    if "Embedding" in layers_by_type and hasattr(torch_model, 'node_embedding'):
        _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    for i in range(depth):
        layers_by_type["rGIN"][i].set_weights(
            [torch_model.convs[i].eps.detach().cpu().numpy().reshape(())])
    # MLP order: [gin_mlp_0..d-1, readout_0..d, output]
    all_mlps = layers_by_type["MLP"]
    for i in range(depth):
        _set_mlp(torch_model.mlps[i], all_mlps[i])
    for i in range(depth + 1):
        _set_mlp(torch_model.readout_mlps[i], all_mlps[depth + i])
    _set_mlp(torch_model.output_mlp, all_mlps[2 * depth + 1])


def _transfer_egnn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    all_mlps = layers_by_type.get("MLP", []) + layers_by_type.get("GraphMLP", [])
    idx = 0
    if torch_model.dense_in is not None:
        _set_mlp(torch_model.dense_in, all_mlps[idx]); idx += 1
    # Per layer: edge_mlp, coord_mlp (except last layer), node_mlp
    # Keras model doesn't expose coord_mlp for the last layer in its layers list
    for i in range(depth):
        _set_mlp(torch_model.layers[i].edge_mlp, all_mlps[idx]); idx += 1
        if i < depth - 1:
            _set_mlp(torch_model.layers[i].coord_mlp, all_mlps[idx]); idx += 1
        _set_mlp(torch_model.layers[i].node_mlp, all_mlps[idx]); idx += 1
    _set_mlp(torch_model.output_mlp, all_mlps[idx])


def _transfer_painn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # BesselBasisLayer has freq parameter
    bessel_layers = layers_by_type.get("BesselBasisLayer", [])
    if bessel_layers:
        bessel_layers[0].set_weights(
            [torch_model.bessel_basis.frequencies.detach().cpu().numpy()])
    # PAiNNconv layers
    convs = layers_by_type.get("PAiNNconv", [])
    for i in range(depth):
        t_conv = torch_model.convs[i]
        k_conv = convs[i]
        ws = []
        ws.append(_w(t_conv.dense1.weight)); ws.append(_b(t_conv.dense1.bias))
        ws.append(_w(t_conv.phi.weight)); ws.append(_b(t_conv.phi.bias))
        ws.append(_w(t_conv.w.weight)); ws.append(_b(t_conv.w.bias))
        k_conv.set_weights(ws)
    # PAiNNUpdate layers — lin_u and lin_v have no bias
    updates = layers_by_type.get("PAiNNUpdate", [])
    for i in range(depth):
        t_upd = torch_model.updates[i]
        k_upd = updates[i]
        # Keras order: lay_dense1, lay_lin_u, lay_lin_v, lay_a
        ws = [
            _w(t_upd.dense1.weight), _b(t_upd.dense1.bias),           # lay_dense1 (2*units, units)
            _w(t_upd.lin_u.weight),                                    # lay_lin_u, no bias
            _w(t_upd.lin_v.weight),                                    # lay_lin_v, no bias
            _w(t_upd.dense_a.weight), _b(t_upd.dense_a.bias),         # lay_a
        ]
        k_upd.set_weights(ws)
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_nmpn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # Edge MLPs (in and out)
    all_mlps = layers_by_type["MLP"]
    _set_mlp(torch_model.edge_mlp_in, all_mlps[0])
    _set_mlp(torch_model.edge_mlp_out, all_mlps[1])
    # TrafoEdgeNetMessages
    trafos = layers_by_type.get("TrafoEdgeNetMessages", [])
    _set_dense(torch_model.edge_trafo_in.dense, trafos[0])
    _set_dense(torch_model.edge_trafo_out.dense, trafos[1])
    # GRUUpdate
    gru_layers = layers_by_type.get("GRUUpdate", [])
    _copy_gru_cell(torch_model.gru_update, gru_layers[0])
    # Output MLP
    _set_mlp(torch_model.output_mlp, all_mlps[2])


def _transfer_dmpnn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Dense[0]: edge_initialize → message_init
    # Dense[1]: edge_dense_all → W_h (shared)
    # Dense[2]: node_dense → node_readout
    denses = layers_by_type["Dense"]
    _set_dense(torch_model.message_init, denses[0])
    _set_dense(torch_model.W_h, denses[1])
    _set_dense(torch_model.node_readout, denses[2])
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_cmpnn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Dense[0]: node_initialize, Dense[1]: edge_initialize
    # Dense[2]: edge_dense (shared), Dense[3]: node_dense
    denses = layers_by_type["Dense"]
    _set_dense(torch_model.node_init, denses[0])
    _set_dense(torch_model.edge_init, denses[1])
    _set_dense(torch_model.edge_denses[0], denses[2])
    _set_dense(torch_model.node_dense, denses[3])
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_dgin(torch_model, layers_by_type, depth):
    depth_gin = torch_model.depth_gin
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Dense[0]: edge_init, Dense[1]: dmpnn_dense (shared), Dense[2]: node_dense
    denses = layers_by_type["Dense"]
    _set_dense(torch_model.edge_init, denses[0])
    _set_dense(torch_model.dmpnn_dense, denses[1])
    _set_dense(torch_model.node_dense, denses[2])
    # GIN_D epsilon
    gin_layers = layers_by_type.get("GIN_D", [])
    for i in range(depth_gin):
        gin_layers[i].set_weights(
            [torch_model.gin_convs[i].eps.detach().cpu().numpy().reshape(())])
    # All gin_mlps, last_mlps, output_mlp are class "MLP" in Keras.
    # Order: gin_mlps[0..depth_gin-1], last_mlps[0..depth_gin], output_mlp
    all_mlps = layers_by_type.get("MLP", [])
    idx = 0
    for i in range(depth_gin):
        _set_mlp(torch_model.gin_mlps[i], all_mlps[idx]); idx += 1
    for i in range(depth_gin + 1):
        _set_mlp(torch_model.last_mlps[i], all_mlps[idx]); idx += 1
    _set_mlp(torch_model.output_mlp, all_mlps[idx])


def _transfer_rgcn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Per layer: Dense (self_loop) + RelationalDense
    denses = layers_by_type["Dense"]
    rel_denses = layers_by_type["RelationalDense"]
    for i in range(depth):
        # Self-loop Dense
        _set_dense(torch_model.convs[i].self_loop, denses[i])
        # RelationalDense: weight (num_rel, in, out) + bias (out)
        ws = [torch_model.convs[i].weight.detach().cpu().numpy()]
        if torch_model.convs[i].rel_bias is not None:
            ws.append(torch_model.convs[i].rel_bias.detach().cpu().numpy())
        rel_denses[i].set_weights(ws)
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_gnnfilm(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Per layer: 3 RelationalDense in Keras topological order: h_j, gamma, beta
    rel_denses = layers_by_type["RelationalDense"]
    for i in range(depth):
        layer = torch_model.film_layers[i]
        idx = i * 3
        _set_relational_dense(layer.rel_dense_hj, rel_denses[idx])
        _set_relational_dense(layer.rel_dense_gamma, rel_denses[idx + 1])
        _set_relational_dense(layer.rel_dense_beta, rel_denses[idx + 2])
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_inorp(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    if len(layers_by_type.get("Embedding", [])) > 1:
        _set_embedding(torch_model.edge_embedding, layers_by_type["Embedding"][1])
    # All edge_mlp and node_mlp are class "MLP" in Keras.
    # Order: edge_mlp[0], node_mlp[0], edge_mlp[1], node_mlp[1], ..., output_mlp
    all_mlps = layers_by_type.get("MLP", [])
    for i in range(depth):
        _set_mlp(torch_model.blocks[i]["edge_mlp"], all_mlps[i * 2])
        _set_mlp(torch_model.blocks[i]["node_mlp"], all_mlps[i * 2 + 1])
    _set_mlp(torch_model.output_mlp, all_mlps[depth * 2])


def _transfer_megnet(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # All GraphMLP/MLP appear as "MLP" (GraphMLP = MLP alias in kgcnn).
    # Order: node_ff_init(0), edge_ff_init(1), state_ff_init(2),
    #   [node_ff(3i), edge_ff(3i+1), state_ff(3i+2)] × (depth-1),
    #   output_mlp(last)
    mlps = layers_by_type.get("MLP", [])
    _set_mlp(torch_model.node_ff_init, mlps[0])
    _set_mlp(torch_model.edge_ff_init, mlps[1])
    _set_mlp(torch_model.state_ff_init, mlps[2])
    for i in range(depth - 1):
        _set_mlp(torch_model.node_ffs[i], mlps[3 + i * 3])
        _set_mlp(torch_model.edge_ffs[i], mlps[4 + i * 3])
        _set_mlp(torch_model.state_ffs[i], mlps[5 + i * 3])
    _set_mlp(torch_model.output_mlp, mlps[-1])
    # MEGnetBlock: 9 internal Dense layers per block
    blocks = layers_by_type.get("MEGnetBlock", [])
    for i in range(depth):
        t_block = torch_model.blocks[i]
        k_block = blocks[i]
        _set_dense(t_block.edge_dense_layers[0], k_block.lay_phi_e)
        _set_dense(t_block.edge_dense_layers[1], k_block.lay_phi_e_1)
        _set_dense(t_block.edge_dense_layers[2], k_block.lay_phi_e_2)
        _set_dense(t_block.node_dense_layers[0], k_block.lay_phi_n)
        _set_dense(t_block.node_dense_layers[1], k_block.lay_phi_n_1)
        _set_dense(t_block.node_dense_layers[2], k_block.lay_phi_n_2)
        _set_dense(t_block.state_dense_layers[0], k_block.lay_phi_u)
        _set_dense(t_block.state_dense_layers[1], k_block.lay_phi_u_1)
        _set_dense(t_block.state_dense_layers[2], k_block.lay_phi_u_2)


def _transfer_cgcnn(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # CGCNNLayer: each has f (gate Dense) and s (filter Dense)
    cgcnn_layers = layers_by_type.get("CGCNNLayer", [])
    for i in range(depth):
        _set_dense(torch_model.convs[i].linear_gate, cgcnn_layers[i].f)
        _set_dense(torch_model.convs[i].linear_filter, cgcnn_layers[i].s)
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_hamnet(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Dense[0]: node_init (tanh), Dense[1]: edge_init (tanh)
    denses = layers_by_type.get("Dense", [])
    _set_dense(torch_model.node_init[0], denses[0])
    _set_dense(torch_model.edge_init[0], denses[1])
    # HamNaiveDynMessage: align_dense, attend_dense, edge_dense
    msg_layers = layers_by_type.get("HamNaiveDynMessage", [])
    for i in range(depth):
        t_msg = torch_model.message_layers[i]
        k_msg = msg_layers[i]
        _set_dense(t_msg.align_dense, k_msg.dense_align)
        _set_dense(t_msg.attend_dense[0], k_msg.dense_attend)
        _set_dense(t_msg.edge_dense, k_msg.dense_e)
    # Node updates: GRU (layer class is GRUUpdate in make_model)
    gru_layers = layers_by_type.get("GRUUpdate", layers_by_type.get("HamNetGRUUnion", []))
    for i in range(depth):
        _copy_gru_cell(torch_model.node_update_layers[i], gru_layers[i])
    # HamNetFingerprintGenerator
    fp_layers = layers_by_type.get("HamNetFingerprintGenerator", [])
    if fp_layers:
        t_fp = torch_model.fingerprint
        k_fp = fp_layers[0]
        _set_dense(t_fp.init_dense[0], k_fp.vertex2mol)
        fp_depth = len(t_fp.grus)
        for i in range(fp_depth):
            _set_dense(t_fp.attend_denses[i][0], k_fp.readouts[i].dense_attend)
            _set_dense(t_fp.align_denses[i], k_fp.readouts[i].dense_align)
            _copy_gru_cell_raw(t_fp.grus[i], k_fp.unions[i])
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _transfer_dimenetpp(torch_model, layers_by_type, depth):
    # EmbeddingDimeBlock: nn.Embedding weight
    emb_layers = layers_by_type.get("EmbeddingDimeBlock", [])
    if emb_layers:
        w = torch_model.node_embedding.embedding.weight.detach().cpu().numpy()
        emb_layers[0].set_weights([w])
    # BesselBasisLayer: trainable frequencies
    bessel_layers = layers_by_type.get("BesselBasisLayer", [])
    if bessel_layers:
        freq = torch_model.bessel_basis.frequencies.detach().cpu().numpy()
        bessel_layers[0].set_weights([freq])
    # Dense[0]: rbf_emb, Dense[1]: edge_emb
    denses = layers_by_type.get("Dense", [])
    _set_dense(torch_model.rbf_emb[0], denses[0])
    _set_dense(torch_model.edge_emb[0], denses[1])
    # Output blocks: [output_block_0, output_block_1, ..., output_block_num_blocks]
    out_blocks = layers_by_type.get("DimNetOutputBlock", [])
    num_blocks = len(torch_model.interaction_blocks)
    _copy_dimenet_output_block(torch_model.output_block_0, out_blocks[0])
    for i in range(num_blocks):
        _copy_dimenet_output_block(torch_model.output_blocks[i], out_blocks[1 + i])
    # Interaction blocks
    int_blocks = layers_by_type.get("DimNetInteractionPPBlock", [])
    for i in range(num_blocks):
        _copy_dimenet_interaction_block(torch_model.interaction_blocks[i], int_blocks[i])
    # Output MLP (optional)
    mlps = layers_by_type.get("MLP", [])
    if mlps:
        _set_mlp(torch_model.output_mlp, mlps[0])


def _copy_dimenet_output_block(torch_block, keras_block):
    """Copy DimNetOutputBlock weights."""
    _set_dense(torch_block.dense_rbf, keras_block.dense_rbf)
    _set_dense(torch_block.up_projection, keras_block.up_projection)
    # dense_mlp: Torch nn.Sequential with (Linear, activation) pairs
    # Keras GraphMLP with mlp_dense_layer_list
    n_dense = len(keras_block.dense_mlp.mlp_dense_layer_list)
    for i in range(n_dense):
        _set_dense(torch_block.dense_mlp[2 * i],
                   keras_block.dense_mlp.mlp_dense_layer_list[i])
    _set_dense(torch_block.dense_final, keras_block.dense_final)


def _copy_dimenet_interaction_block(torch_block, keras_block):
    """Copy DimNetInteractionPPBlock weights."""
    _set_dense(torch_block.dense_rbf1, keras_block.dense_rbf1)
    _set_dense(torch_block.dense_rbf2, keras_block.dense_rbf2)
    _set_dense(torch_block.dense_sbf1, keras_block.dense_sbf1)
    _set_dense(torch_block.dense_sbf2, keras_block.dense_sbf2)
    # Sequential([Linear, activation]) → Dense
    _set_dense(torch_block.dense_ji[0], keras_block.dense_ji)
    _set_dense(torch_block.dense_kj[0], keras_block.dense_kj)
    _set_dense(torch_block.down_projection[0], keras_block.down_projection)
    _set_dense(torch_block.up_projection[0], keras_block.up_projection)
    # Residual layers before skip
    for j in range(len(torch_block.layers_before_skip)):
        _set_dense(torch_block.layers_before_skip[j].dense_1,
                   keras_block.layers_before_skip[j].dense_1)
        _set_dense(torch_block.layers_before_skip[j].dense_2,
                   keras_block.layers_before_skip[j].dense_2)
    # Final before skip
    _set_dense(torch_block.final_before_skip[0], keras_block.final_before_skip)
    # Residual layers after skip
    for j in range(len(torch_block.layers_after_skip)):
        _set_dense(torch_block.layers_after_skip[j].dense_1,
                   keras_block.layers_after_skip[j].dense_1)
        _set_dense(torch_block.layers_after_skip[j].dense_2,
                   keras_block.layers_after_skip[j].dense_2)


def _transfer_hdnnp2nd(torch_model, layers_by_type, depth):
    # wACSFRad and wACSFAng are non-trainable (params set at construction)
    # RelationalMLP is the main trainable block
    rel_mlps = layers_by_type.get("RelationalMLP", [])
    if rel_mlps:
        _set_relational_mlp(torch_model.relational_mlp, rel_mlps[0])
    # Output MLP
    mlps = layers_by_type.get("MLP", [])
    if mlps:
        _set_mlp(torch_model.output_mlp, mlps[0])
    # Dipole: charge_dense → Keras Dense(units=1)
    if getattr(torch_model, "predict_dipole", False):
        dense_layers = layers_by_type.get("Dense", [])
        if dense_layers:
            _set_dense(torch_model.charge_dense, dense_layers[0])


def _transfer_mat(torch_model, layers_by_type, depth):
    # Embedding
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    # Dense layers in MAT make_model topological order:
    # Dense[0]: input_projection (64→32, no bias)
    # Dense[1]: adj_projection (1→1, no bias)
    # Then per depth: Dense[2+i*2]: attention_projection, Dense[3+i*2]: ff_projection
    denses = layers_by_type.get("Dense", [])
    _set_dense(torch_model.input_projection, denses[0])
    _set_dense(torch_model.adj_projection, denses[1])
    # Remaining Dense layers: per depth, attention_proj then ff_proj
    for i in range(depth):
        _set_dense(torch_model.attention_projections[i], denses[2 + i * 2])
        _set_dense(torch_model.ff_projections[i], denses[3 + i * 2])
    # LayerNormalization: [attn_norm_0, ff_norm_0, ..., attn_norm_{d-1}, ff_norm_{d-1}, final_norm]
    norms = layers_by_type.get("LayerNormalization", [])
    for i in range(depth):
        _set_layernorm(torch_model.attention_norms[i], norms[i * 2])
        _set_layernorm(torch_model.ff_norms[i], norms[i * 2 + 1])
    _set_layernorm(torch_model.final_norm, norms[depth * 2])
    # MATAttentionHead: per depth, heads heads
    attn_heads = layers_by_type.get("MATAttentionHead", [])
    heads_per_layer = len(torch_model.attention_heads[0])
    for i in range(depth):
        for j in range(heads_per_layer):
            t_head = torch_model.attention_heads[i][j]
            k_head = attn_heads[i * heads_per_layer + j]
            _set_dense(t_head.dense_q, k_head.dense_q)
            _set_dense(t_head.dense_k, k_head.dense_k)
            _set_dense(t_head.dense_v, k_head.dense_v)
    # FF MLPs: per depth, one MLP
    mlps = layers_by_type.get("MLP", [])
    # MLP order: ff_mlp_0, ..., ff_mlp_{d-1}, output_mlp
    for i in range(depth):
        # FF MLPs are Sequential in Torch → MLP in Keras
        _copy_seq_to_keras_mlp(torch_model.ff_mlps[i], mlps[i])
    _copy_seq_to_keras_mlp(torch_model.output_mlp, mlps[depth])


def _copy_seq_to_keras_mlp(torch_seq, keras_mlp):
    """Copy nn.Sequential([Linear, act, ...]) -> Keras MLP weights."""
    keras_idx = 0
    for module in torch_seq:
        if isinstance(module, torch.nn.Linear):
            _set_dense(module, keras_mlp.mlp_dense_layer_list[keras_idx])
            keras_idx += 1


def _copy_residual_layer(torch_res, keras_res):
    """Copy Torch ResidualLayer → Keras ResidualLayer."""
    _set_dense(torch_res.dense_1, keras_res.dense_1)
    _set_dense(torch_res.dense_2, keras_res.dense_2)


def _transfer_mogat(torch_model, layers_by_type, depth):
    _set_embedding(torch_model.node_embedding, layers_by_type["Embedding"][0])
    _set_dense(torch_model.dense_in, layers_by_type["Dense"][0])
    # AttentiveHeadFP_ (note underscore — MoGAT variant)
    heads = layers_by_type.get("AttentiveHeadFP_", [])
    for i in range(depth):
        t_head = torch_model.attention_layers[i]
        k_head = heads[i]
        _set_dense(t_head.linear_trafo, k_head.lay_linear_trafo)
        _set_dense(t_head.alpha_activation[0], k_head.lay_alpha_activation)
        _set_dense(t_head.linear_alpha, k_head.lay_alpha)
        if i == 0 and t_head.use_edge_features:
            _set_dense(t_head.fc1[0], k_head.lay_fc1)
            _set_dense(t_head.fc2[0], k_head.lay_fc2)
    # GRU updates
    grus = layers_by_type.get("GRUUpdate", [])
    for i in range(depth):
        _copy_gru_cell(torch_model.gru_layers[i], grus[i])
    # Per-layer PoolingNodesAttentive
    pools = layers_by_type.get("PoolingNodesAttentive", [])
    for i in range(depth):
        t_pool = torch_model.layer_pools[i]
        k_pool = pools[i]
        _set_dense(t_pool.linear_trafo, k_pool.lay_linear_trafo)
        _set_dense(t_pool.lay_alpha[0], k_pool.lay_alpha)
        _copy_gru_cell_raw(t_pool.gru, k_pool.lay_gru)
    # Self-attention scale
    attn_layers = layers_by_type.get("Attention", [])
    if attn_layers:
        scale = torch_model.attn_scale.detach().cpu().numpy()
        attn_layers[0].set_weights([scale.reshape(())])
    # Output MLP
    _set_mlp(torch_model.output_mlp, layers_by_type["MLP"][0])


def _sort_layers_by_creation(layers):
    """Sort Keras layers by name suffix number to recover creation order.

    Keras functional-API ``model.layers`` uses topological order, which may
    differ from code creation order.  Layer names follow the pattern
    ``class_name``, ``class_name_1``, ``class_name_2``, ... where the suffix
    number reflects creation order.
    """
    import re

    def _key(layer):
        m = re.search(r'_(\d+)$', layer.name)
        return int(m.group(1)) if m else -1
    return sorted(layers, key=_key)


def _transfer_mxmnet(torch_model, layers_by_type, depth):
    # EmbeddingDimeBlock: node embedding
    emb_layers = layers_by_type.get("EmbeddingDimeBlock", [])
    if emb_layers:
        w = torch_model.node_embedding.weight.detach().cpu().numpy()
        emb_layers[0].set_weights([w])
    # BesselBasisLayer: [local, global] — sorted by creation order
    bessel_layers = _sort_layers_by_creation(
        layers_by_type.get("BesselBasisLayer", []))
    if len(bessel_layers) >= 2:
        bessel_layers[0].set_weights(
            [torch_model.bessel_basis_local.frequencies.detach().cpu().numpy()])
        bessel_layers[1].set_weights(
            [torch_model.bessel_basis_global.frequencies.detach().cpu().numpy()])
    # MLP: [mlp_rbf, mlp_sbf_1, mlp_sbf_2, mlp_rbf_global, output_mlp]
    # GraphMLP = MLP (alias), so __class__.__name__ is "MLP" for all.
    # Sort by creation order since topological order differs.
    mlps = _sort_layers_by_creation(layers_by_type.get("MLP", []))
    _copy_seq_to_keras_mlp(torch_model.mlp_rbf, mlps[0])
    _copy_seq_to_keras_mlp(torch_model.mlp_sbf_1, mlps[1])
    _copy_seq_to_keras_mlp(torch_model.mlp_sbf_2, mlps[2])
    _copy_seq_to_keras_mlp(torch_model.mlp_rbf_global, mlps[3])
    # MXMGlobalMP blocks — sorted by creation order
    global_blocks = _sort_layers_by_creation(
        layers_by_type.get("MXMGlobalMP", []))
    for i in range(depth):
        t_gmp = torch_model.global_mp_blocks[i]
        k_gmp = global_blocks[i]
        _copy_seq_to_keras_mlp(t_gmp.h_mlp, k_gmp.h_mlp)
        _copy_residual_layer(t_gmp.res1, k_gmp.res1)
        _copy_residual_layer(t_gmp.res2, k_gmp.res2)
        _copy_residual_layer(t_gmp.res3, k_gmp.res3)
        _copy_seq_to_keras_mlp(t_gmp.mlp, k_gmp.mlp)
        _copy_seq_to_keras_mlp(t_gmp.x_edge_mlp, k_gmp.x_edge_mlp)
        _set_dense(t_gmp.linear, k_gmp.linear)
    # MXMLocalMP blocks — sorted by creation order
    local_blocks = _sort_layers_by_creation(
        layers_by_type.get("MXMLocalMP", []))
    for i in range(depth):
        t_lmp = torch_model.local_mp_blocks[i]
        k_lmp = local_blocks[i]
        _copy_seq_to_keras_mlp(t_lmp.h_mlp, k_lmp.h_mlp)
        _copy_seq_to_keras_mlp(t_lmp.mlp_kj, k_lmp.mlp_kj)
        _copy_seq_to_keras_mlp(t_lmp.mlp_ji_1, k_lmp.mlp_ji_1)
        _copy_seq_to_keras_mlp(t_lmp.mlp_ji_2, k_lmp.mlp_ji_2)
        _copy_seq_to_keras_mlp(t_lmp.mlp_jj, k_lmp.mlp_jj)
        _copy_seq_to_keras_mlp(t_lmp.mlp_sbf1, k_lmp.mlp_sbf1)
        _copy_seq_to_keras_mlp(t_lmp.mlp_sbf2, k_lmp.mlp_sbf2)
        _set_dense(t_lmp.lin_rbf1, k_lmp.lin_rbf1)
        _set_dense(t_lmp.lin_rbf2, k_lmp.lin_rbf2)
        _set_dense(t_lmp.lin_rbf_out, k_lmp.lin_rbf_out)
        _copy_residual_layer(t_lmp.res1, k_lmp.res1)
        _copy_residual_layer(t_lmp.res2, k_lmp.res2)
        _copy_residual_layer(t_lmp.res3, k_lmp.res3)
        _set_mlp(t_lmp.y_mlp, k_lmp.y_mlp)
        _set_dense(t_lmp.y_W, k_lmp.y_W)
    # Output MLP (index 4 — after the 4 GraphMLP projection layers)
    _set_mlp(torch_model.output_mlp, mlps[4])


# ---- Training divergence ----

def run_training(name, torch_fwd, keras_fwd, torch_params, keras_params,
                 target, n_steps=N_STEPS, lr=LR, grad_clip=None,
                 keras_model=None, keras_inputs=None):
    """Run n_steps of independent SGD training on both models.

    When ``keras_model`` and ``keras_inputs`` are provided and
    KERAS_BACKEND is tensorflow, uses native TF GradientTape +
    keras.optimizers.SGD for the Keras side.  Otherwise falls back to
    the torch-only path (torch.optim.SGD for both sides).
    """
    use_native_tf = (_BACKEND == "tensorflow" and keras_model is not None
                     and keras_inputs is not None)

    # Cross-backend: gradient clipping amplifies tiny float differences
    # (TF vs PyTorch scatter/reduce accumulation order) at the clip boundary.
    # Drop grad_clip for TF backend and reduce lr to keep training stable
    # (models like PAiNN/CMPNN have large gradients that NaN without clip).
    if use_native_tf and grad_clip is not None:
        lr = lr * 0.1  # compensate for removed clipping
        grad_clip = None

    torch_opt = torch.optim.SGD(torch_params, lr=lr)
    target_np = target.detach().cpu().numpy()

    if use_native_tf:
        import tensorflow as tf
        keras_opt = _keras.optimizers.SGD(learning_rate=lr)
        target_tf = tf.constant(target_np, dtype=tf.float32)
        n_keras_params = len(keras_model.trainable_variables)
    else:
        keras_opt = torch.optim.SGD(keras_params, lr=lr)
        if keras_params:
            target = target.to(keras_params[0].device)
        n_keras_params = len(keras_params)

    torch_losses, keras_losses, loss_diffs, output_diffs = [], [], [], []

    print(f"\n=== {name}: {n_steps} steps, lr={lr} ===")
    print(f"  params: torch={len(torch_params)}, keras={n_keras_params}")

    for step in range(n_steps):
        # --- Torch forward + loss ---
        torch_out = torch_fwd()
        torch_loss = F.mse_loss(torch_out, target.to(torch_out.device))
        tl = float(torch_loss.item())
        torch_out_np = torch_out.detach().cpu().numpy()

        if use_native_tf:
            # --- Keras/TF forward + loss + grads (inside GradientTape) ---
            with tf.GradientTape() as tape:
                keras_out_raw = keras_model(keras_inputs, training=True)
                if isinstance(keras_out_raw, (list, tuple)):
                    keras_out_tf = tf.concat(keras_out_raw, axis=-1)
                else:
                    keras_out_tf = keras_out_raw
                # Truncate extra outputs to match target shape
                # (e.g. HDNNP2nd+dipole returns energy+dipole+charge
                #  but target only has energy+dipole)
                if keras_out_tf.shape[-1] != target_tf.shape[-1]:
                    keras_out_tf = keras_out_tf[:, :target_tf.shape[-1]]
                keras_loss_tf = tf.reduce_mean(
                    tf.square(keras_out_tf - target_tf))
            grads = tape.gradient(keras_loss_tf,
                                  keras_model.trainable_variables)
            kl = float(keras_loss_tf.numpy())
            keras_out_np = keras_out_tf.numpy()

            # --- Backward + step (both sides) ---
            torch_opt.zero_grad()
            torch_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(torch_params, grad_clip)
                grads = [tf.clip_by_norm(g, grad_clip)
                         if g is not None else g for g in grads]
            torch_opt.step()
            keras_opt.apply(grads, keras_model.trainable_variables)
        else:
            # --- Existing torch-only path ---
            keras_out = keras_fwd()
            keras_out = keras_out.to(target.device)
            keras_loss = F.mse_loss(keras_out, target)
            kl = float(keras_loss.item())
            keras_out_np = keras_out.detach().cpu().numpy()

            torch_opt.zero_grad()
            torch_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(torch_params, grad_clip)
            torch_opt.step()

            keras_opt.zero_grad()
            keras_loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(keras_params, grad_clip)
            keras_opt.step()

        # --- Compare ---
        torch_losses.append(tl)
        keras_losses.append(kl)
        loss_diffs.append(abs(tl - kl))
        out_mae = float(np.abs(torch_out_np - keras_out_np).mean())
        output_diffs.append(out_mae)

        if step % 10 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: loss_t={tl:.6e} loss_k={kl:.6e} "
                  f"loss_diff={loss_diffs[-1]:.3e} out_mae={out_mae:.3e}")

        if math.isnan(tl) or math.isnan(kl):
            print(f"  NaN at step {step}, stopping.")
            break

    return {
        "torch_losses": torch_losses, "keras_losses": keras_losses,
        "loss_diffs": loss_diffs, "output_diffs": output_diffs,
    }


# ---- Keras model forward wrapper ----

def keras_model_forward(keras_model, padded_inputs):
    """Run Keras make_model forward, return torch tensor with grad_fn."""
    out = keras_model(padded_inputs, training=True)
    # On torch backend, out is already a torch tensor with grad_fn.
    # Do NOT use keras_to_torch (goes through numpy, loses grad).
    if isinstance(out, torch.Tensor):
        return out
    return keras_to_torch(out)


# ---- Model tests ----

_mutag_cache = None
_clintox_cache = None


def get_mutag_batch():
    global _mutag_cache
    if _mutag_cache is None:
        _mutag_cache = load_pickle_as_pyg("MUTAG.kgcnn.pickle",
                                           edge_attr_keys=["edge_attributes"])
    return make_batch(_mutag_cache, BATCH_SIZE)


def get_clintox_batch():
    global _clintox_cache
    if _clintox_cache is None:
        pyg_list = load_pickle_as_pyg("ClinTox.kgcnn.pickle",
                                       edge_attr_keys=["edge_attributes"])
        _clintox_cache = [d for d in pyg_list if 'z' in d.keys() and 'edge_attr' in d.keys()]
    return make_batch(_clintox_cache, BATCH_SIZE)


def test_gcn():
    from kgcnn.literature.GCN import make_model
    from kgcnn_torch.models.gcn import GCNModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    input_names = ["node_number", "edge_weights", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    gcn_units = 32

    torch_model = GCNModel(
        node_dim=32, depth=depth, gcn_units=gcn_units,
        gcn_activation="relu", gcn_pooling="sum", node_pooling="sum",
        output_units=[32, 16], output_activation="relu",
        output_final_activation="linear",
        output_use_bias=[True, True, False],
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        gcn_args={"units": gcn_units, "use_bias": True, "activation": "relu",
                  "pooling_method": "scatter_sum"},
        depth=depth,
        node_pooling_args={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, False], "units": [32, 16, 1],
                     "activation": ["relu", "relu", "linear"]},
    )

    # Dry run to initialize
    keras_model(padded, training=False)

    # Transfer weights
    transfer_weights_to_make_model(torch_model, keras_model, "GCN", depth)

    # Verify forward pass
    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    max_abs = float((torch_out - keras_out).abs().max())
    print(f"GCN forward: MAE={mae:.2e}, MAX={max_abs:.2e}")

    # Training divergence
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GCN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_gin():
    from kgcnn.literature.GIN import make_model
    from kgcnn_torch.models.gin import GINModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    input_names = ["node_number", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3

    torch_model = GINModel(
        node_dim=32, depth=depth, units=32,
        gin_mlp_units=[32, 32], gin_mlp_activation="relu",
        gin_mlp_use_normalization=False, gin_pooling="sum",
        epsilon_learnable=False, use_edge_features=False,
        node_pooling="sum", last_mlp_units=[32, 32, 32],
        last_mlp_activation="relu", dropout_rate=0.0,
        output_units=[], output_activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        depth=depth,
        gin_args={"pooling_method": "scatter_sum", "epsilon_learnable": False},
        gin_mlp={"units": [32, 32], "use_bias": True,
                 "activation": ["relu", "linear"]},
        last_mlp={"units": [32, 32, 32], "use_bias": True,
                  "activation": ["relu", "relu", "linear"]},
        dropout=0.0,
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [1],
                     "activation": ["linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "GIN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"GIN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GIN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_gin_edge():
    from kgcnn.literature.GIN import make_model_edge
    from kgcnn_torch.models.gin import GINModel

    batch = get_mutag_batch()
    # MUTAG edge_attr is float (0., 1., 2.) — convert to integer edge_type
    # so pyg_batch_to_padded_kgcnn creates edge_number and Torch model embeds them.
    batch.edge_type = batch.edge_attr.squeeze(-1).long()
    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_type = batch.edge_type
    input_names = ["node_number", "edge_number", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3

    torch_model = GINModel(
        node_dim=32, depth=depth, units=32,
        gin_mlp_units=[32, 32], gin_mlp_activation="relu",
        gin_mlp_use_normalization=False, gin_pooling="sum",
        epsilon_learnable=False,
        use_edge_embedding=True, num_edge_embeddings=10, edge_embedding_dim=32,
        gine_activation="relu",
        node_pooling="sum", last_mlp_units=[32, 32, 32],
        last_mlp_activation="relu", dropout_rate=0.0,
        output_units=[], output_activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model_edge(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None,), "name": "edge_number", "dtype": "int64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding={"input_dim": 10, "output_dim": 32},
        depth=depth,
        gin_args={"epsilon_learnable": False},
        gin_mlp={"units": [32, 32], "use_bias": True,
                 "activation": ["relu", "linear"]},
        last_mlp={"units": [32, 32, 32], "use_bias": True,
                  "activation": ["relu", "relu", "linear"]},
        dropout=0.0,
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [1],
                     "activation": ["linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "GIN_edge", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"GIN-edge forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GIN-edge (MUTAG) [make_model_edge]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_gat():
    from kgcnn.literature.GAT import make_model
    from kgcnn_torch.models.gat import GATModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))  # 11 for ClinTox
    input_names = ["node_number", "edge_attributes", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    heads = 3
    att_units = 16

    torch_model = GATModel(
        node_dim=32, depth=depth, attention_units=att_units,
        attention_heads_num=heads, attention_heads_concat=False,
        attention_activation="leaky_relu2", use_edge_features=True,
        edge_dim=edge_dim, node_pooling="mean",
        output_units=[32, 16], output_activation="relu",
        output_use_bias=[True, True, False],
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        attention_args={"units": att_units, "use_bias": True,
                        "use_edge_features": True, "use_final_activation": False,
                        "has_self_loops": False,
                        "activation": {"class_name": "function", "config": "kgcnn>leaky_relu2"}},
        attention_heads_num=heads,
        attention_heads_concat=False,
        depth=depth,
        pooling_nodes_args={"pooling_method": "scatter_mean"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, False], "units": [32, 16, 1],
                     "activation": ["relu", "relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "GAT", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"GAT forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GAT (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_gatv2():
    from kgcnn.literature.GATv2 import make_model
    from kgcnn_torch.models.gatv2 import GATv2Model

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))
    input_names = ["node_number", "edge_attributes", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    heads = 3
    att_units = 16

    torch_model = GATv2Model(
        node_dim=32, depth=depth, attention_units=att_units,
        attention_heads_num=heads, attention_heads_concat=False,
        attention_activation="leaky_relu2", use_edge_features=True,
        edge_dim=edge_dim, node_pooling="mean",
        output_units=[32, 16], output_activation="relu",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        attention_args={"units": att_units, "use_bias": True,
                        "use_edge_features": True, "use_final_activation": False,
                        "has_self_loops": False,
                        "activation": {"class_name": "function", "config": "kgcnn>leaky_relu2"}},
        attention_heads_num=heads,
        attention_heads_concat=False,
        depth=depth,
        pooling_nodes_args={"pooling_method": "scatter_mean"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, False], "units": [32, 16, 1],
                     "activation": ["relu", "relu", "sigmoid"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "GATv2", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"GATv2 forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GATv2 (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_graphsage():
    from kgcnn.literature.GraphSAGE import make_model
    from kgcnn_torch.models.graphsage import GraphSAGEModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))  # 1 for MUTAG
    input_names = ["node_number", "edge_attributes", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3

    torch_model = GraphSAGEModel(
        node_dim=32, depth=depth, units=32,
        node_mlp_units=[64, 32], edge_mlp_units=[64, 32],
        edge_dim=edge_dim, use_edge_features=True,
        pooling_method="mean", node_pooling="mean",
        output_units=[32, 16], activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        use_edge_features=True,
        node_mlp_args={"units": [64, 32], "use_bias": True,
                       "activation": ["relu", "linear"]},
        edge_mlp_args={"units": [64, 32], "use_bias": True,
                       "activation": ["relu", "linear"]},
        pooling_args={"pooling_method": "scatter_mean"},
        pooling_nodes_args={"pooling_method": "scatter_mean"},
        depth=depth,
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, False], "units": [32, 16, 1],
                     "activation": ["relu", "relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "GraphSAGE", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"GraphSAGE forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GraphSAGE (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_attentivefp():
    from kgcnn.literature.AttentiveFP import make_model
    from kgcnn_torch.models.attentivefp import AttentiveFPModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))  # 1 for MUTAG
    input_names = ["node_number", "edge_attributes", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth_ato = 2
    units = 32

    torch_model = AttentiveFPModel(
        node_dim=32, depth_ato=depth_ato, depth_mol=2, units=units,
        use_edge_features=True, edge_dim=edge_dim,
        attention_activation="leaky_relu2",
        attention_activation_context="elu",
        pooling_activation="leaky_relu2",
        pooling_activation_context="elu",
        node_pooling="sum", dropout=0.0,
        output_units=[32], output_activation="relu",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        depthato=depth_ato,
        depthmol=2,
        dropout=0.0,
        attention_args={"units": units, "use_bias": True,
                        "activation": {"class_name": "function", "config": "kgcnn>leaky_relu2"},
                        "activation_context": "elu"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [32, 1],
                     "activation": ["relu", "sigmoid"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "AttentiveFP", depth_ato)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"AttentiveFP forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("AttentiveFP (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_rgin():
    from kgcnn.literature.rGIN import make_model
    from kgcnn_torch.models.rgin import rGINModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    input_names = ["node_number", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3

    torch_model = rGINModel(
        node_dim=32, depth=depth, units=32,
        gin_mlp_units=[32, 32], gin_mlp_activation=["relu", "linear"],
        gin_mlp_use_normalization=False, gin_pooling="sum",
        epsilon_learnable=False, random_range=100,
        dropout=0.0, node_pooling="sum",
        last_mlp_units=[32, 32, 32], last_mlp_activation="relu",
        output_units=[], output_activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        depth=depth,
        rgin_args={"pooling_method": "scatter_sum", "epsilon_learnable": False,
                   "random_range": 100},
        gin_mlp={"units": [32, 32], "use_bias": True,
                 "activation": ["relu", "linear"]},
        last_mlp={"units": [32, 32, 32], "use_bias": True,
                  "activation": ["relu", "relu", "linear"]},
        dropout=0.0,
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [1],
                     "activation": ["linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "rGIN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"rGIN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("rGIN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_egnn():
    from kgcnn.literature.EGNN import make_model
    from kgcnn_torch.models.egnn import EGNNModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))  # 11 for ClinTox
    input_names = ["node_number", "node_coordinates", "edge_attributes",
                   "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    units = 32

    torch_model = EGNNModel(
        node_dim=32, depth=depth, units=units,
        edge_mlp_units=[units, units], edge_mlp_activation="swish",
        coord_mlp_units=[units, 1], coord_mlp_activation="swish",
        node_mlp_units=[units, units], node_mlp_activation="swish",
        use_edge_attr=True, edge_attr_dim=edge_dim,
        use_attention=False, use_normalize=False, use_skip=True,
        use_node_attributes=False, use_node_normalization=False,
        layer_pooling="sum", coord_pooling="mean",
        node_pooling="sum",
        output_units=[units], output_activation="swish",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        node_mlp_initialize=[units])
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        depth=depth,
        node_mlp_initialize={"units": [units], "activation": ["linear"]},
        use_edge_attributes=True,
        edge_mlp_kwargs={"units": [units, units], "activation": ["swish", "linear"]},
        coord_mlp_kwargs={"units": [units, 1], "activation": ["swish", "linear"]},
        node_mlp_kwargs={"units": [units, units], "activation": ["swish", "linear"]},
        use_skip=True,
        pooling_edge_kwargs={"pooling_method": "scatter_sum"},
        pooling_coord_kwargs={"pooling_method": "scatter_mean"},
        node_pooling_kwargs={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [units, 1],
                     "activation": ["swish", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "EGNN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"EGNN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("EGNN (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_painn():
    from kgcnn.literature.PAiNN import make_model
    from kgcnn_torch.models.painn import PAiNNModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    input_names = ["node_number", "node_coordinates", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    units = 32
    num_radial = 20
    cutoff = 5.0

    torch_model = PAiNNModel(
        node_dim=units, depth=depth, units=units,
        num_radial=num_radial, cutoff=cutoff, envelope_exponent=5,
        conv_activation="swish", conv_pooling="sum",
        update_activation="swish", update_add_eps=False,
        equiv_normalization=False, node_normalization=False,
        node_pooling="sum",
        output_units=[units], output_activation="swish",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": units},
        equiv_initialize_kwargs={"dim": 3, "method": "zeros", "units": units},
        bessel_basis={"num_radial": num_radial, "cutoff": cutoff, "envelope_exponent": 5},
        depth=depth,
        conv_args={"units": units, "cutoff": None, "conv_pool": "scatter_sum"},
        update_args={"units": units},
        pooling_args={"pooling_method": "scatter_sum"},
        equiv_normalization=False,
        node_normalization=False,
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [units, 1],
                     "activation": ["swish", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "PAiNN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"PAiNN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("PAiNN (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_nmpn():
    from kgcnn.literature.NMPN import make_model
    from kgcnn_torch.models.nmpn import NMPNModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))  # 1 for MUTAG
    input_names = ["node_number", "edge_attributes", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    units = 32

    torch_model = NMPNModel(
        node_dim=32, depth=depth, units=units,
        edge_dim=edge_dim, edge_mlp_units=[32, 32, 32],
        edge_mlp_activation="swish",
        message_pooling="sum", use_set2set=False,
        node_pooling="sum",
        output_units=[32], output_activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        node_dim=units,
        depth=depth,
        edge_mlp={"units": [32, 32, 32], "use_bias": True,
                   "activation": ["swish", "swish", "swish"]},
        use_set2set=False,
        pooling_args={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [32, 1],
                     "activation": ["relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "NMPN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"NMPN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("NMPN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_schnet():
    from kgcnn.literature.Schnet import make_model
    from kgcnn_torch.models.schnet import SchNetModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    input_names = ["node_number", "node_coordinates", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3

    torch_model = SchNetModel(
        node_dim=32, depth=depth, units=32,
        gauss_bins=20, gauss_distance=4.0, gauss_sigma=0.4, gauss_offset=0.0,
        interaction_activation="shifted_softplus", interaction_pooling="sum",
        node_pooling="sum", last_mlp_units=[32, 32],
        last_mlp_activation="shifted_softplus",
        output_units=[32], output_activation="shifted_softplus",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        make_distance=True, expand_distance=True, use_output_mlp=True)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        interaction_args={"units": 32, "use_bias": True,
                          "activation": "kgcnn>shifted_softplus",
                          "cfconv_pool": "scatter_sum"},
        node_pooling_args={"pooling_method": "scatter_sum"},
        depth=depth,
        gauss_args={"bins": 20, "distance": 4.0, "offset": 0.0, "sigma": 0.4},
        last_mlp={"use_bias": True, "units": [32, 32],
                   "activation": ["kgcnn>shifted_softplus", "kgcnn>shifted_softplus"]},
        output_embedding="graph",
        output_mlp={"use_bias": [True, True], "units": [32, 1],
                     "activation": ["kgcnn>shifted_softplus", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "SchNet", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"SchNet forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("SchNet (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_dmpnn():
    from kgcnn.literature.DMPNN import make_model
    from kgcnn_torch.models.dmpnn import DMPNNModel

    batch = get_mutag_batch()
    edge_dim = int(batch.edge_attr.size(-1))  # 1 for MUTAG

    # Compute edge_pair_index for Torch
    edge_pair_index = compute_batch_edge_pair_index(batch, BATCH_SIZE)
    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_pair_index = edge_pair_index

    input_names = ["node_number", "edge_attributes", "edge_indices",
                   "edge_indices_reverse", "total_nodes", "total_edges", "total_reverse"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    units = 32

    torch_model = DMPNNModel(
        node_dim=32, edge_dim=edge_dim, depth=depth, units=units,
        message_activation="relu", init_activation=None, node_activation=None,
        message_pooling="sum", node_pooling="sum", dropout_rate=0.0,
        output_units=[32], output_activation="relu",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        use_edge_embedding=False)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_reverse", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        depth=depth,
        edge_initialize={"units": units, "activation": "relu"},
        edge_dense={"units": units, "activation": "linear"},
        edge_activation={"activation": "relu"},
        node_dense={"units": units, "activation": "relu"},
        dropout=None,
        pooling_args={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [32, 1],
                     "activation": ["relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "DMPNN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"DMPNN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("DMPNN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_cmpnn():
    from kgcnn.literature.CMPNN import make_model
    from kgcnn_torch.models.cmpnn import CMPNNModel

    batch = get_mutag_batch()
    edge_dim = int(batch.edge_attr.size(-1))

    edge_pair_index = compute_batch_edge_pair_index(batch, BATCH_SIZE)
    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_pair_index = edge_pair_index

    input_names = ["node_number", "edge_attributes", "edge_indices",
                   "edge_indices_reverse", "total_nodes", "total_edges", "total_reverse"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    # depth=2: exactly 1 edge_dense in both Torch ModuleList and Keras shared Dense
    depth = 2
    units = 32

    torch_model = CMPNNModel(
        node_dim=32, edge_dim=edge_dim, depth=depth, units=units,
        dropout=0.0, activation="relu", node_dense_activation="linear",
        use_final_gru=False, node_pooling="sum",
        output_units=[32], output_activation="relu",
        output_use_bias=[True, False],
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_reverse", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        depth=depth,
        node_initialize={"units": units, "activation": "relu"},
        edge_initialize={"units": units, "activation": "relu"},
        edge_dense={"units": units, "activation": "linear"},
        edge_activation={"activation": "relu"},
        node_dense={"units": units, "activation": "linear"},
        dropout=None,
        use_final_gru=False,
        pooling_kwargs={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [32, 1],
                     "activation": ["relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "CMPNN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"CMPNN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("CMPNN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_dgin():
    from kgcnn.literature.DGIN import make_model
    from kgcnn_torch.models.dgin import DGINModel

    batch = get_mutag_batch()
    edge_dim = int(batch.edge_attr.size(-1))

    edge_pair_index = compute_batch_edge_pair_index(batch, BATCH_SIZE)
    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_pair_index = edge_pair_index

    input_names = ["node_number", "edge_attributes", "edge_indices",
                   "edge_indices_reverse", "total_nodes", "total_edges", "total_reverse"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth_dmpnn = 2
    depth_gin = 2
    units = 32
    gin_mlp_units = [32, 32]
    last_mlp_units = [32, 32]

    torch_model = DGINModel(
        node_dim=32, edge_dim=edge_dim, depth_dmpnn=depth_dmpnn,
        depth_gin=depth_gin, units=units,
        dropout_dmpnn=0.0, dropout_gin=0.0, activation="relu",
        gin_mlp_units=gin_mlp_units, gin_mlp_activation=["relu", "linear"],
        gin_mlp_use_normalization=False,
        last_mlp_units=last_mlp_units,
        node_pooling="mean",
        output_units=[], output_activation="relu",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 1), "name": "edge_indices_reverse", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_reverse", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        depthDMPNN=depth_dmpnn,
        depthGIN=depth_gin,
        edge_initialize={"units": units, "activation": "relu"},
        edge_dense={"units": units, "activation": "linear"},
        edge_activation={"activation": "relu"},
        node_dense={"units": gin_mlp_units[-1], "activation": "linear"},
        dropoutDMPNN=None,
        dropoutGIN=None,
        gin_args={"pooling_method": "scatter_sum", "epsilon_learnable": False},
        gin_mlp={"units": gin_mlp_units, "use_bias": True,
                 "activation": ["relu", "linear"], "use_normalization": False},
        last_mlp={"units": last_mlp_units, "use_bias": True,
                  "activation": ["relu", "relu"]},
        node_pooling_kwargs={"pooling_method": "scatter_mean"},
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [1],
                     "activation": ["linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "DGIN", depth_dmpnn)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"DGIN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("DGIN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_rgcn():
    from kgcnn.literature.RGCN import make_model
    from kgcnn_torch.models.rgcn import RGCNModel

    batch = get_mutag_batch()
    num_edges = batch.edge_index.size(1)
    num_relations = 3

    # Create deterministic edge types from edge_attr
    batch.edge_type = torch.arange(num_edges, dtype=torch.long) % num_relations

    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_type = batch.edge_type
    torch_data.edge_attr = None  # RGCN uses relational transforms, not edge_attr weights

    input_names = ["node_number", "edge_weights", "edge_relations",
                   "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    units = 32

    torch_model = RGCNModel(
        node_dim=32, depth=depth, units=units,
        num_relations=num_relations,
        rgcn_activation="swish", rgcn_pooling="sum",
        use_residual=False, node_pooling="sum",
        output_units=[32], output_activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 1), "name": "edge_weights", "dtype": "float32"},
            {"shape": (None,), "name": "edge_relations", "dtype": "int64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        depth=depth,
        dense_relation_kwargs={"units": units, "num_relations": num_relations,
                               "activation": "linear", "use_bias": True},
        dense_kwargs={"units": units, "use_bias": True},
        activation_kwargs={"activation": "swish"},
        node_pooling_kwargs={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [32, 1],
                     "activation": ["relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "RGCN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"RGCN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("RGCN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_gnnfilm():
    from kgcnn.literature.GNNFilm import make_model
    from kgcnn_torch.models.gnnfilm import GNNFilmModel

    batch = get_mutag_batch()
    num_edges = batch.edge_index.size(1)
    num_relations = 3

    batch.edge_type = torch.arange(num_edges, dtype=torch.long) % num_relations

    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_type = batch.edge_type

    input_names = ["node_attributes", "edge_relations",
                   "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    units = 32  # same as node_dim so no projection needed

    torch_model = GNNFilmModel(
        node_dim=units, depth=depth, units=units,
        num_relations=num_relations,
        activation="swish", modulation_activation="sigmoid",
        film_pooling="sum", node_pooling="sum",
        output_units=[32], output_activation="relu",
        output_final_activation="linear",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_attributes", "dtype": "int64"},
            {"shape": (None,), "name": "edge_relations", "dtype": "int64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": units},
        depth=depth,
        dense_modulation_kwargs={"units": units, "num_relations": num_relations,
                                 "activation": "sigmoid", "use_bias": True},
        dense_relation_kwargs={"units": units, "num_relations": num_relations,
                               "activation": "linear", "use_bias": True},
        activation_kwargs={"activation": "swish"},
        node_pooling_kwargs={"pooling_method": "scatter_sum"},
        output_embedding="graph",
        output_mlp={"use_bias": True, "units": [32, 1],
                     "activation": ["relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "GNNFilm", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"GNNFilm forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("GNNFilm (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_inorp():
    from kgcnn.literature.INorp import make_model
    from kgcnn_torch.models.inorp import INorpModel

    batch = get_mutag_batch()
    num_edges = batch.edge_index.size(1)
    num_edge_embeddings = 5
    edge_embedding_dim = 32
    graph_state_dim = 64

    # Create integer edge types for embedding
    batch.edge_type = torch.arange(num_edges, dtype=torch.long) % num_edge_embeddings

    torch_data = pyg_batch_to_torch_data(batch)
    torch_data.edge_type = batch.edge_type
    # Graph state: zeros for reproducibility
    torch.manual_seed(42)
    torch_data.graph_state = torch.randn(BATCH_SIZE, graph_state_dim)

    # Also attach graph_state to batch for padded conversion
    batch.graph_state = torch_data.graph_state

    input_names = ["node_number", "edge_number", "edge_indices",
                   "graph_attributes", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    node_dim = 32
    edge_mlp_units = [64, 32]
    node_mlp_units = [64, 32]

    torch_model = INorpModel(
        node_dim=node_dim, depth=depth, edge_dim=1,
        edge_mlp_units=edge_mlp_units, edge_mlp_activation="relu",
        node_mlp_units=node_mlp_units, node_mlp_activation="relu",
        message_pooling="mean", use_set2set=False,
        use_graph_state=True, graph_state_dim=graph_state_dim,
        node_pooling="mean",
        output_units=[25, 10], output_activation="relu",
        output_final_activation="sigmoid",
        output_use_bias=[True, True, False],
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        use_edge_embedding=True, num_edge_embeddings=num_edge_embeddings,
        edge_embedding_dim=edge_embedding_dim)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None,), "name": "edge_number", "dtype": "int64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (graph_state_dim,), "name": "graph_attributes", "dtype": "float32"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": node_dim},
        input_edge_embedding={"input_dim": num_edge_embeddings, "output_dim": edge_embedding_dim},
        depth=depth,
        edge_mlp_args={"units": edge_mlp_units, "use_bias": True,
                       "activation": ["relu", "linear"]},
        node_mlp_args={"units": node_mlp_units, "use_bias": True,
                       "activation": ["relu", "linear"]},
        pooling_args={"pooling_method": "scatter_mean"},
        use_set2set=False,
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, False], "units": [25, 10, 1],
                     "activation": ["relu", "relu", "sigmoid"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "INorp", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"INorp forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("INorp (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_megan():
    from kgcnn.literature.MEGAN import make_model
    from kgcnn_torch.models.megan import MEGANModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))
    input_names = ["node_attributes", "edge_attributes", "edge_indices",
                   "graph_labels", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    node_dim = 32
    importance_channels = 2

    torch_model = MEGANModel(
        node_dim=node_dim, units=[node_dim] * depth, num_heads=importance_channels,
        depth=depth, attention_activation="leaky_relu2",
        use_edge_features=True, edge_dim=edge_dim,
        concat_heads=True, importance_channels=importance_channels,
        importance_units=[], importance_activation="relu",
        final_units=[1], final_activation="linear",
        use_bias=True, final_pooling="sum",
        dropout_rate=0.0, final_dropout_rate=0.0,
        regression_reference=None, num_targets=1,
        output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_attributes", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (1,), "name": "graph_labels", "dtype": "float32"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": node_dim},
        units=[node_dim] * depth,
        activation={"class_name": "function", "config": "kgcnn>leaky_relu2"},
        use_bias=True,
        dropout_rate=0.0,
        use_edge_features=True,
        importance_units=[],
        importance_channels=importance_channels,
        importance_activation="sigmoid",
        concat_heads=True,
        final_units=[1],
        final_dropout_rate=0.0,
        final_activation="linear",
        final_pooling="sum",
        regression_reference=None,
        return_importances=False,
        output_embedding="graph",
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "MEGAN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"MEGAN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("MEGAN (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_megnet():
    from kgcnn.literature.Megnet import make_model
    from kgcnn_torch.models.megnet import MEGNetModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    # Use edge_attr directly (no make_distance), graph_state for state input
    torch_data.graph_state = torch.zeros(BATCH_SIZE, 1)
    edge_dim_in = batch.edge_attr.shape[-1]  # 11
    input_names = ["node_number", "edge_attributes", "edge_indices",
                   "charge", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 3
    node_dim = 32
    edge_dim = edge_dim_in
    state_dim = 1

    torch_model = MEGNetModel(
        node_dim=node_dim, edge_dim=edge_dim, state_dim=state_dim,
        edge_input_dim=edge_dim_in, state_input_dim=0,
        depth=depth,
        block_units_edge=[edge_dim, edge_dim, edge_dim],
        block_units_node=[node_dim, node_dim, node_dim],
        block_units_state=[node_dim, node_dim, node_dim],
        node_ff_units=[node_dim, node_dim],
        edge_ff_units=[edge_dim, edge_dim],
        state_ff_units=[state_dim, node_dim],
        activation="softplus2",
        has_ff=True, dropout=None,
        use_set2set=False, node_pooling="sum",
        output_units=[node_dim, 16], output_activation="softplus2",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        use_graph_embedding=False,
    )
    torch_model.eval()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim_in), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (1,), "name": "charge", "dtype": "float32"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": node_dim},
        input_graph_embedding=None,
        make_distance=False,
        expand_distance=False,
        meg_block_args={"node_embed": [node_dim, node_dim, node_dim],
                        "edge_embed": [edge_dim, edge_dim, edge_dim],
                        "env_embed": [node_dim, node_dim, node_dim],
                        "activation": "kgcnn>softplus2",
                        "pooling_method": "scatter_mean"},
        node_ff_args={"units": [node_dim, node_dim], "activation": "kgcnn>softplus2"},
        edge_ff_args={"units": [edge_dim, edge_dim], "activation": "kgcnn>softplus2"},
        state_ff_args={"units": [state_dim, node_dim], "activation": "kgcnn>softplus2"},
        nblocks=depth,
        has_ff=True,
        dropout=None,
        use_set2set=False,
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, True], "units": [node_dim, 16, 1],
                     "activation": ["kgcnn>softplus2", "kgcnn>softplus2", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "MEGNet", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"MEGNet forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("MEGNet (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001,
        keras_model=keras_model, keras_inputs=padded)


def test_cgcnn():
    from kgcnn.literature.CGCNN import make_crystal_model
    from kgcnn_torch.models.cgcnn import CGCNNModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)

    depth = 3
    conv_units = 32
    gauss_bins = 40
    cutoff = 5.0

    # Compute edge distances from pos for Torch model
    src = batch.edge_index[0]
    dst = batch.edge_index[1]
    edge_dist = (batch.pos[src] - batch.pos[dst]).norm(dim=-1, keepdim=True)
    torch_data.edge_attr = edge_dist

    torch_model = CGCNNModel(
        node_dim=conv_units, depth=depth, conv_units=conv_units,
        conv_activation="softplus",
        expand_distance=True, gauss_bins=gauss_bins,
        gauss_distance=cutoff, gauss_offset=0.0, gauss_sigma=0.5,
        node_pooling="mean",
        output_units=[32, 16], output_activation="softplus",
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        batch_normalization=False)
    torch_model.eval()

    # Crystal model: use identity lattice, Cartesian = fractional coords, zero cell_translations
    # Compute per-graph data for crystal format
    nodes_per_graph = []
    edges_per_graph = []
    for i in range(BATCH_SIZE):
        node_mask = batch.batch == i
        nodes_per_graph.append(int(node_mask.sum()))
        src_mask = batch.batch[batch.edge_index[0]] == i
        edges_per_graph.append(int(src_mask.sum()))
    max_nodes = max(nodes_per_graph)
    max_edges = max(edges_per_graph)
    node_offsets = [0] + list(np.cumsum(nodes_per_graph[:-1]))
    edge_offsets = [0] + list(np.cumsum(edges_per_graph[:-1]))
    ei = batch.edge_index.numpy()

    # node_frac_coordinates = pos (identity lattice)
    pos_padded = np.zeros((BATCH_SIZE, max_nodes, 3), dtype='float64')
    node_number = np.zeros((BATCH_SIZE, max_nodes), dtype='int64')
    for i in range(BATCH_SIZE):
        s = node_offsets[i]; n = nodes_per_graph[i]
        pos_padded[i, :n] = batch.pos[s:s + n].numpy()
        node_number[i, :n] = batch.z[s:s + n].numpy()

    edge_indices = np.zeros((BATCH_SIZE, max_edges, 2), dtype='int64')
    cell_trans = np.zeros((BATCH_SIZE, max_edges, 3), dtype='float32')
    for i in range(BATCH_SIZE):
        s = edge_offsets[i]; e = edges_per_graph[i]; no = node_offsets[i]
        edge_indices[i, :e, 0] = ei[1, s:s + e] - no  # dst
        edge_indices[i, :e, 1] = ei[0, s:s + e] - no  # src

    lattice = np.tile(np.eye(3, dtype='float64'), (BATCH_SIZE, 1, 1))
    total_nodes = np.array(nodes_per_graph, dtype='int64')
    total_edges = np.array(edges_per_graph, dtype='int64')

    padded = [node_number, pos_padded, edge_indices, cell_trans, lattice,
              total_nodes, total_edges]

    keras_model = make_crystal_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_frac_coordinates", "dtype": "float64"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 3), "name": "cell_translations", "dtype": "float32"},
            {"shape": (3, 3), "name": "lattice_matrix", "dtype": "float64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": conv_units},
        make_distances=True,
        expand_distance=True,
        gauss_args={"bins": gauss_bins, "distance": cutoff, "offset": 0.0, "sigma": 0.5},
        conv_layer_args={"units": conv_units, "activation_s": "softplus",
                         "activation_out": "softplus",
                         "pooling_method": "scatter_mean"},
        depth=depth,
        node_pooling_args={"pooling_method": "scatter_mean"},
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, False], "units": [32, 16, 1],
                     "activation": ["softplus", "softplus", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "CGCNN", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"CGCNN forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("CGCNN (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_hamnet():
    from kgcnn.literature.HamNet import make_model
    from kgcnn_torch.models.hamnet import HamNetModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))
    input_names = ["node_number", "node_coordinates", "edge_attributes",
                   "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depth = 2
    units = 32
    fp_depth = 2

    torch_model = HamNetModel(
        node_dim=32, edge_dim=edge_dim, depth=depth, units=units,
        fingerprint_dim=units, fingerprint_depth=fp_depth,
        activation="leaky_relu2", activation_last="elu",
        fingerprint_activation="leaky_relu2",
        fingerprint_activation_context="leaky_relu2",
        use_gru_update=True, use_gru_update_edge=False,
        output_units=[32], output_activation="relu",
        output_use_bias=[True, False],
        num_targets=1, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.eval()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        given_coordinates=True,
        gru_kwargs={"units": units},
        message_kwargs={"units": units, "units_edge": units},
        fingerprint_kwargs={"units": units, "units_attend": units,
                            "depth": fp_depth,
                            "pooling_method": "mean"},
        union_type_node="gru",
        union_type_edge=None,
        depth=depth,
        output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [32, 1],
                     "activation": ["relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "HamNet", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"HamNet forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("HamNet (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_dimenetpp():
    from kgcnn.literature.DimeNetPP import make_model
    from kgcnn_torch.models.dimenetpp import DimeNetPPModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)

    # Generate angle indices from edge_index
    angle_idx = generate_angle_index(batch.edge_index)
    torch_data.angle_index = angle_idx

    # Compute padded angle indices per graph
    nodes_per_graph, edges_per_graph = [], []
    for i in range(BATCH_SIZE):
        node_mask = batch.batch == i
        nodes_per_graph.append(int(node_mask.sum()))
        src_mask = batch.batch[batch.edge_index[0]] == i
        edges_per_graph.append(int(src_mask.sum()))
    edge_offsets = [0] + list(np.cumsum(edges_per_graph[:-1]))

    # Compute angle indices per graph
    angles_per_graph = []
    all_angle_indices = []
    for g in range(BATCH_SIZE):
        e_start = edge_offsets[g]
        e_count = edges_per_graph[g]
        # Extract local edge_index for this graph
        local_ei = batch.edge_index[:, e_start:e_start + e_count]
        # Generate angle indices for local edges
        local_angles = generate_angle_index(local_ei)
        # These are local edge indices [0..e_count-1]
        angles_per_graph.append(local_angles.shape[1])
        all_angle_indices.append(local_angles)

    max_angles = max(angles_per_graph) if angles_per_graph else 0
    angle_padded = np.zeros((BATCH_SIZE, max_angles, 2), dtype='int64')
    for g in range(BATCH_SIZE):
        k = angles_per_graph[g]
        if k > 0:
            angle_padded[g, :k] = all_angle_indices[g].numpy().T
    total_angles = np.array(angles_per_graph, dtype='int64')

    input_names = ["node_number", "node_coordinates", "edge_indices",
                   "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)
    # Insert angle_indices at position 3, total_angles at end
    padded.insert(3, angle_padded)
    padded.append(total_angles)

    emb_size = 32
    out_emb_size = 16
    int_emb_size = 16
    basis_emb_size = 8
    num_blocks = 2
    num_spherical = 3
    num_radial = 4
    cutoff = 5.0
    num_before_skip = 1
    num_after_skip = 1
    num_dense_output = 2

    torch_model = DimeNetPPModel(
        emb_size=emb_size, out_emb_size=out_emb_size,
        int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
        num_blocks=num_blocks, num_spherical=num_spherical,
        num_radial=num_radial, cutoff=cutoff, envelope_exponent=5,
        num_before_skip=num_before_skip, num_after_skip=num_after_skip,
        num_dense_output=num_dense_output, num_targets=1,
        activation="swish", extensive=True, output_init="zeros",
        output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        use_output_mlp=True,
        output_mlp_units=[16, 1], output_mlp_activation="swish")
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 2), "name": "angle_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_angles", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": emb_size},
        emb_size=emb_size, out_emb_size=out_emb_size,
        int_emb_size=int_emb_size, basis_emb_size=basis_emb_size,
        num_blocks=num_blocks, num_spherical=num_spherical,
        num_radial=num_radial, cutoff=cutoff, envelope_exponent=5,
        num_before_skip=num_before_skip, num_after_skip=num_after_skip,
        num_dense_output=num_dense_output, num_targets=1,
        activation="swish", extensive=True, output_init="zeros",
        use_output_mlp=True,
        output_embedding="graph",
        output_mlp={"use_bias": [True, False], "units": [16, 1],
                     "activation": ["swish", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "DimeNetPP", num_blocks)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"DimeNetPP forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("DimeNetPP (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_hdnnp2nd():
    from kgcnn.literature.HDNNP2nd import make_model_weighted
    from kgcnn_torch.models.hdnnp2nd import HDNNP2ndModel

    batch = get_clintox_batch()

    # Map atomic numbers to 0-indexed type indices for HDNNP2nd.
    # Both Torch wACSF (uses type_map) and Keras wACSF (uses z directly as
    # index into parameter table) need consistent indexing. Using 0-indexed
    # type indices with element_types=list(range(n_types)) makes both models
    # use the same compact parameter tables.
    unique_z = sorted(batch.z.unique().tolist())
    n_types = len(unique_z)
    z_map = {int(z): idx for idx, z in enumerate(unique_z)}
    batch.z = torch.tensor([z_map[int(z)] for z in batch.z], dtype=torch.long)

    torch_data = pyg_batch_to_torch_data(batch)
    num_nodes = int(batch.z.size(0))

    # Generate angle indices (atom triplets) from edge_index
    angle_idx_nodes = generate_angle_index_nodes(batch.edge_index, num_nodes)
    torch_data.angle_index = angle_idx_nodes

    # Compute padded angle indices per graph
    nodes_per_graph, edges_per_graph = [], []
    for i in range(BATCH_SIZE):
        node_mask = batch.batch == i
        nodes_per_graph.append(int(node_mask.sum()))
        src_mask = batch.batch[batch.edge_index[0]] == i
        edges_per_graph.append(int(src_mask.sum()))
    node_offsets = [0] + list(np.cumsum(nodes_per_graph[:-1]))
    edge_offsets = [0] + list(np.cumsum(edges_per_graph[:-1]))

    angles_per_graph = []
    all_angle_nodes = []
    for g in range(BATCH_SIZE):
        n_off = node_offsets[g]
        e_start = edge_offsets[g]
        e_count = edges_per_graph[g]
        local_ei = batch.edge_index[:, e_start:e_start + e_count] - n_off
        n_g = nodes_per_graph[g]
        local_angles = generate_angle_index_nodes(local_ei, n_g)
        angles_per_graph.append(local_angles.shape[1])
        all_angle_nodes.append(local_angles)

    max_angles = max(angles_per_graph) if angles_per_graph else 0
    angle_nodes_padded = np.zeros((BATCH_SIZE, max_angles, 3), dtype='int64')
    for g in range(BATCH_SIZE):
        k = angles_per_graph[g]
        if k > 0:
            angle_nodes_padded[g, :k] = all_angle_nodes[g].numpy().T
    total_angles = np.array(angles_per_graph, dtype='int64')

    input_names = ["node_number", "node_coordinates", "edge_indices",
                   "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)
    # Insert angle_indices_nodes at position 3, total_angles at end
    padded.insert(3, angle_nodes_padded)
    padded.append(total_angles)

    n_rad_features = 8
    n_ang_features = 6
    cutoff_val = 5.0
    num_relations = n_types
    relational_units = [64, 64, 64]
    relational_activation = ["swish", "swish", "linear"]

    torch_model = HDNNP2ndModel(
        element_types=list(range(n_types)),
        n_rad_features=n_rad_features, n_ang_features=n_ang_features,
        cutoff=cutoff_val, num_relations=num_relations,
        relational_units=relational_units,
        relational_activation=relational_activation,
        use_batch_norm=False, node_pooling="sum",
        use_output_mlp=True, output_units=[32],
        output_activation="swish", num_targets=1,
        output_embedding="graph")
    torch_model.eval()

    # Extract descriptor params from Torch model buffers for Keras
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

    keras_model = make_model_weighted(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_angles", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        has_charge_input=False,
        w_acsf_rad_kwargs={"eta_mu": eta_mu, "cutoff": cutoff_val, "add_eps": True},
        w_acsf_ang_kwargs={"eta_mu_lambda_zeta": eta_mu_lz, "cutoff": cutoff_val, "add_eps": True},
        mlp_kwargs={"units": relational_units, "num_relations": num_relations,
                     "activation": relational_activation},
        node_pooling_args={"pooling_method": "scatter_sum"},
        use_output_mlp=True,
        output_embedding="graph",
        output_mlp={"use_bias": [True, True], "units": [32, 1],
                     "activation": ["swish", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "HDNNP2nd", 0)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"HDNNP2nd forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    return run_training("HDNNP2nd (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_hdnnp2nd_dipole():
    """Test HDNNP2nd with predict_dipole=True alignment (forward-only)."""
    from kgcnn.literature.HDNNP2nd import make_model_weighted
    from kgcnn_torch.models.hdnnp2nd import HDNNP2ndModel

    batch = get_clintox_batch()

    # Map atomic numbers to 0-indexed type indices (same as test_hdnnp2nd).
    unique_z = sorted(batch.z.unique().tolist())
    n_types = len(unique_z)
    z_map = {int(z): idx for idx, z in enumerate(unique_z)}
    batch.z = torch.tensor([z_map[int(z)] for z in batch.z], dtype=torch.long)

    torch_data = pyg_batch_to_torch_data(batch)
    num_nodes = int(batch.z.size(0))

    # Generate angle indices (atom triplets) from edge_index
    angle_idx_nodes = generate_angle_index_nodes(batch.edge_index, num_nodes)
    torch_data.angle_index = angle_idx_nodes

    # Compute padded angle indices per graph
    nodes_per_graph, edges_per_graph = [], []
    for i in range(BATCH_SIZE):
        node_mask = batch.batch == i
        nodes_per_graph.append(int(node_mask.sum()))
        src_mask = batch.batch[batch.edge_index[0]] == i
        edges_per_graph.append(int(src_mask.sum()))
    node_offsets = [0] + list(np.cumsum(nodes_per_graph[:-1]))
    edge_offsets = [0] + list(np.cumsum(edges_per_graph[:-1]))

    angles_per_graph = []
    all_angle_nodes = []
    for g in range(BATCH_SIZE):
        n_off = node_offsets[g]
        e_start = edge_offsets[g]
        e_count = edges_per_graph[g]
        local_ei = batch.edge_index[:, e_start:e_start + e_count] - n_off
        n_g = nodes_per_graph[g]
        local_angles = generate_angle_index_nodes(local_ei, n_g)
        angles_per_graph.append(local_angles.shape[1])
        all_angle_nodes.append(local_angles)

    max_angles = max(angles_per_graph) if angles_per_graph else 0
    angle_nodes_padded = np.zeros((BATCH_SIZE, max_angles, 3), dtype='int64')
    for g in range(BATCH_SIZE):
        k = angles_per_graph[g]
        if k > 0:
            angle_nodes_padded[g, :k] = all_angle_nodes[g].numpy().T
    total_angles = np.array(angles_per_graph, dtype='int64')

    input_names = ["node_number", "node_coordinates", "edge_indices",
                   "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)
    # Insert angle_indices_nodes at position 3, total_angles at end
    padded.insert(3, angle_nodes_padded)
    padded.append(total_angles)

    n_rad_features = 8
    n_ang_features = 6
    cutoff_val = 5.0
    num_relations = n_types
    relational_units = [64, 64, 64]
    relational_activation = ["swish", "swish", "linear"]

    # Build Torch model with predict_dipole=True
    torch_model = HDNNP2ndModel(
        element_types=list(range(n_types)),
        n_rad_features=n_rad_features, n_ang_features=n_ang_features,
        cutoff=cutoff_val, num_relations=num_relations,
        relational_units=relational_units,
        relational_activation=relational_activation,
        use_batch_norm=False, node_pooling="sum",
        use_output_mlp=True, output_units=[32],
        output_activation="swish", num_targets=1,
        output_embedding="graph",
        predict_dipole=True, has_charge_input=False)
    torch_model.eval()

    # Extract descriptor params from Torch model buffers for Keras
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

    # Build Keras model with predict_dipole=True
    keras_model = make_model_weighted(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 3), "name": "angle_indices_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_angles", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        has_charge_input=False,
        w_acsf_rad_kwargs={"eta_mu": eta_mu, "cutoff": cutoff_val, "add_eps": True},
        w_acsf_ang_kwargs={"eta_mu_lambda_zeta": eta_mu_lz, "cutoff": cutoff_val, "add_eps": True},
        mlp_kwargs={"units": relational_units, "num_relations": num_relations,
                     "activation": relational_activation},
        node_pooling_args={"pooling_method": "scatter_sum"},
        use_output_mlp=True,
        output_embedding="graph",
        output_mlp={"use_bias": [True, True], "units": [32, 1],
                     "activation": ["swish", "linear"]},
        predict_dipole=True,
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "HDNNP2nd", 0)

    # Forward comparison: Torch returns (energy, dipole, total_charge)
    # Keras returns list of 3 tensors [energy, dipole, total_charge]
    with torch.no_grad():
        torch_out = torch_model(torch_data)
    torch_energy = torch_out[0].detach().cpu()
    torch_dipole = torch_out[1].detach().cpu()
    torch_charge = torch_out[2].detach().cpu()

    keras_out = keras_model(padded, training=False)
    keras_energy = keras_to_torch(keras_out[0])
    keras_dipole = keras_to_torch(keras_out[1])
    keras_charge = keras_to_torch(keras_out[2])

    mae_energy = float((torch_energy - keras_energy).abs().mean())
    mae_dipole = float((torch_dipole - keras_dipole).abs().mean())
    mae_charge = float((torch_charge - keras_charge).abs().mean())
    print(f"HDNNP2nd dipole forward: energy MAE={mae_energy:.2e}, "
          f"dipole MAE={mae_dipole:.2e}, charge MAE={mae_charge:.2e}")

    # Training alignment: use combined loss on energy + dipole.
    # Keras on torch backend may place outputs on CUDA; detect device from
    # Keras params so the target and Torch forward results live on the same
    # device as the Keras forward results.
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    device = keras_params[0].device if keras_params else torch.device("cpu")

    target_energy = torch.randn(BATCH_SIZE, 1, device=device)
    target_dipole = torch.randn(BATCH_SIZE, 3, device=device)
    target = torch.cat([target_energy, target_dipole], dim=-1)  # (B, 4)
    target.requires_grad_(False)

    def torch_fwd():
        out = torch_model(torch_data)
        return torch.cat([out[0].to(device), out[1].to(device)], dim=-1)  # (B, 4)

    def keras_fwd():
        out = keras_model(padded, training=True)
        # Keras on torch backend returns CUDA tensors; cat and keep on same device.
        e = out[0] if isinstance(out[0], torch.Tensor) else keras_to_torch(out[0])
        d = out[1] if isinstance(out[1], torch.Tensor) else keras_to_torch(out[1])
        return torch.cat([e, d], dim=-1)

    return run_training("HDNNP2nd+dipole (ClinTox) [make_model]",
        torch_fwd, keras_fwd,
        torch_params, keras_params, target, lr=1e-4, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


def test_mat():
    from kgcnn.literature.MAT import make_model
    from kgcnn_torch.models.mat import MATModel

    batch = get_clintox_batch()

    # MAT uses padded format: (B, N, ...) with masks
    # Build padded data from PyG batch
    nodes_per_graph = []
    for i in range(BATCH_SIZE):
        nodes_per_graph.append(int((batch.batch == i).sum()))
    max_n = max(nodes_per_graph)

    node_offsets = [0] + list(np.cumsum(nodes_per_graph[:-1]))
    edges_per_graph = []
    edge_offsets_list = []
    eo = 0
    for i in range(BATCH_SIZE):
        src_mask = batch.batch[batch.edge_index[0]] == i
        e = int(src_mask.sum())
        edges_per_graph.append(e)
        edge_offsets_list.append(eo)
        eo += e

    # Padded node_input (B, N) int64
    node_input = np.zeros((BATCH_SIZE, max_n), dtype='int64')
    # Padded xyz (B, N, 3)
    xyz_input = np.zeros((BATCH_SIZE, max_n, 3), dtype='float32')
    # Node mask (B, N) bool
    node_mask = np.zeros((BATCH_SIZE, max_n), dtype='bool')
    for i in range(BATCH_SIZE):
        s = node_offsets[i]; n = nodes_per_graph[i]
        node_input[i, :n] = batch.z[s:s + n].numpy()
        xyz_input[i, :n] = batch.pos[s:s + n].numpy()
        node_mask[i, :n] = True

    # Adjacency matrix (B, N, N, 1) from edge_index
    adjacency = np.zeros((BATCH_SIZE, max_n, max_n, 1), dtype='float32')
    adj_mask = np.zeros((BATCH_SIZE, max_n, max_n), dtype='bool')
    ei = batch.edge_index.numpy()
    for i in range(BATCH_SIZE):
        n = nodes_per_graph[i]
        adj_mask[i, :n, :n] = True
        s = edge_offsets_list[i]; e = edges_per_graph[i]
        no = node_offsets[i]
        for j in range(e):
            src = ei[0, s + j] - no
            dst = ei[1, s + j] - no
            adjacency[i, dst, src, 0] = 1.0

    padded = [node_input, xyz_input, adjacency, node_mask, adj_mask]

    # MAT torch model: pass float masks
    torch_node_input = torch.from_numpy(node_input)
    torch_xyz = torch.from_numpy(xyz_input)
    torch_adj = torch.from_numpy(adjacency)
    torch_node_mask = torch.from_numpy(node_mask)
    torch_adj_mask = torch.from_numpy(adj_mask)

    depth = 2
    num_heads = 4
    att_units = 8
    emb_units = 32

    torch_model = MATModel(
        embedding_units=emb_units, depth=depth,
        num_heads=num_heads, attention_units=att_units,
        merge_heads="concat", lambda_attention=0.3,
        lambda_distance=0.3, add_identity=False,
        attention_dropout=None, distance_trafo="exp",
        units_ff=[32, 32, 32], ff_activations=["relu", "relu", "linear"],
        output_units=[32, 16], output_activations=["relu", "relu", "linear"],
        num_targets=1,
        use_node_embedding=True, num_embeddings=95,
        input_node_dim=64, output_embedding="graph")
    torch_model.eval()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, None, 1), "name": "adjacency_matrix", "dtype": "float32"},
            {"shape": (None,), "name": "node_mask", "dtype": "bool"},
            {"shape": (None, None), "name": "adjacency_mask", "dtype": "bool"},
        ],
        input_tensor_type="padded",
        input_node_embedding={"input_dim": 95, "output_dim": 64},
        input_edge_embedding=None,
        distance_matrix_kwargs={"trafo": "exp"},
        attention_kwargs={"units": att_units, "lambda_attention": 0.3,
                          "lambda_distance": 0.3, "dropout": None,
                          "add_identity": False},
        feed_forward_kwargs={"units": [32, 32, 32],
                              "activation": ["relu", "relu", "linear"]},
        embedding_units=emb_units,
        depth=depth, heads=num_heads, merge_heads="concat",
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, True], "units": [32, 16, 1],
                     "activation": ["relu", "relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "MAT", depth)

    with torch.no_grad():
        torch_out = torch_model(
            torch_node_input, torch_xyz, torch_adj,
            torch_node_mask.float(), torch_adj_mask.float(),
        ).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"MAT forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 1); target.requires_grad_(False)

    def torch_fwd():
        return torch_model(
            torch_node_input, torch_xyz, torch_adj,
            torch_node_mask.float(), torch_adj_mask.float())

    return run_training("MAT (ClinTox) [make_model]",
        torch_fwd,
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001,
        keras_model=keras_model, keras_inputs=padded)


def test_mogat():
    from kgcnn.literature.MoGAT import make_model
    from kgcnn_torch.models.mogat import MoGATModel

    batch = get_mutag_batch()
    torch_data = pyg_batch_to_torch_data(batch)
    edge_dim = int(batch.edge_attr.size(-1))
    input_names = ["node_number", "edge_attributes", "edge_indices", "total_nodes", "total_edges"]
    padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)

    depthato = 2
    units = 32

    torch_model = MoGATModel(
        node_dim=32, depthato=depthato, depthmol=2, units=units,
        edge_dim=edge_dim, use_edge_features=True,
        activation="leaky_relu2", dropout=0.0,
        output_units=[32, 16], output_activation="relu",
        num_targets=2, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, edge_dim), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": 32},
        input_edge_embedding=None,
        depthato=depthato,
        depthmol=2,
        dropout=0.0,
        attention_args={
            "units": units,
            "activation": {"class_name": "function", "config": "kgcnn>leaky_relu2"},
            "activation_context": "elu",
        },
        output_embedding="graph",
        output_mlp={"use_bias": [True, True, True], "units": [32, 16, 2],
                     "activation": ["relu", "relu", "linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "MoGAT", depthato)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"MoGAT forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, 2); target.requires_grad_(False)

    return run_training("MoGAT (MUTAG) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target,
        keras_model=keras_model, keras_inputs=padded)


def test_mxmnet():
    from kgcnn.literature.MXMNet import make_model
    from kgcnn_torch.models.mxmnet import MXMNetModel

    batch = get_clintox_batch()
    torch_data = pyg_batch_to_torch_data(batch)

    # Generate angle indices [ji, kj] convention
    angle_idx = generate_angle_index(batch.edge_index)
    torch_data.angle_index_1 = angle_idx
    torch_data.angle_index_2 = angle_idx
    # range_index not set → Torch model falls back to edge_index for global MP

    # Compute per-graph info
    nodes_per_graph, edges_per_graph = [], []
    for i in range(BATCH_SIZE):
        node_mask = batch.batch == i
        nodes_per_graph.append(int(node_mask.sum()))
        src_mask = batch.batch[batch.edge_index[0]] == i
        edges_per_graph.append(int(src_mask.sum()))
    edge_offsets = [0] + list(np.cumsum(edges_per_graph[:-1]))

    # Per-graph angle indices (local edge-pair indices)
    angles_per_graph = []
    all_angle_indices = []
    for g in range(BATCH_SIZE):
        e_start = edge_offsets[g]
        e_count = edges_per_graph[g]
        local_ei = batch.edge_index[:, e_start:e_start + e_count]
        local_angles = generate_angle_index(local_ei)
        angles_per_graph.append(local_angles.shape[1])
        all_angle_indices.append(local_angles)

    max_angles = max(angles_per_graph) if angles_per_graph else 0
    angle_padded = np.zeros((BATCH_SIZE, max_angles, 2), dtype='int64')
    for g in range(BATCH_SIZE):
        k = angles_per_graph[g]
        if k > 0:
            angle_padded[g, :k] = all_angle_indices[g].numpy().T
    total_angles = np.array(angles_per_graph, dtype='int64')

    # Build base padded inputs
    input_names = ["node_number", "node_coordinates", "edge_indices",
                   "total_nodes", "total_edges"]
    base_padded = pyg_batch_to_padded_kgcnn(batch, BATCH_SIZE, input_names)
    # base_padded: [node_number, node_coordinates, edge_indices, total_nodes, total_edges]

    max_edges = base_padded[2].shape[1]
    edge_attr_dummy = np.zeros((BATCH_SIZE, max_edges, 1), dtype='float32')

    # MXMNet expects 12 inputs:
    # [node_number, node_coordinates, edge_attributes, edge_indices, range_indices,
    #  angle_indices_1, angle_indices_2, total_nodes, total_edges, total_ranges,
    #  total_angles_1, total_angles_2]
    padded = [
        base_padded[0],       # node_number
        base_padded[1],       # node_coordinates
        edge_attr_dummy,      # edge_attributes (dummy, use_edge_attributes=False)
        base_padded[2],       # edge_indices
        base_padded[2],       # range_indices (= edge_indices)
        angle_padded,         # angle_indices_1
        angle_padded,         # angle_indices_2
        base_padded[3],       # total_nodes
        base_padded[4],       # total_edges
        base_padded[4],       # total_ranges (= total_edges)
        total_angles,         # total_angles_1
        total_angles,         # total_angles_2
    ]

    depth = 2
    node_dim = 32
    units = 32
    num_radial = 6
    num_spherical = 3
    num_radial_spherical = 4
    cutoff = 5.0
    num_targets = 2

    torch_model = MXMNetModel(
        node_dim=node_dim, depth=depth, units=units,
        num_radial=num_radial, num_spherical=num_spherical,
        num_radial_spherical=num_radial_spherical,
        cutoff=cutoff, envelope_exponent=5,
        activation="swish", mp_pooling="sum",
        global_mp_pooling="mean", use_local_mp=True,
        node_pooling="sum", output_units=[],
        num_targets=num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=95,
        use_output_mlp=True)
    torch_model.train()

    keras_model = make_model(
        inputs=[
            {"shape": (None,), "name": "node_number", "dtype": "int64"},
            {"shape": (None, 3), "name": "node_coordinates", "dtype": "float32"},
            {"shape": (None, 1), "name": "edge_attributes", "dtype": "float32"},
            {"shape": (None, 2), "name": "edge_indices", "dtype": "int64"},
            {"shape": (None, 2), "name": "range_indices", "dtype": "int64"},
            {"shape": (None, 2), "name": "angle_indices_1", "dtype": "int64"},
            {"shape": (None, 2), "name": "angle_indices_2", "dtype": "int64"},
            {"shape": (), "name": "total_nodes", "dtype": "int64"},
            {"shape": (), "name": "total_edges", "dtype": "int64"},
            {"shape": (), "name": "total_ranges", "dtype": "int64"},
            {"shape": (), "name": "total_angles_1", "dtype": "int64"},
            {"shape": (), "name": "total_angles_2", "dtype": "int64"},
        ],
        input_tensor_type="padded",
        cast_disjoint_kwargs={},
        input_node_embedding={"input_dim": 95, "output_dim": node_dim},
        input_edge_embedding=None,
        use_edge_attributes=False,
        depth=depth,
        bessel_basis_local={"num_radial": num_radial, "cutoff": cutoff,
                            "envelope_exponent": 5},
        bessel_basis_global={"num_radial": num_radial, "cutoff": cutoff,
                             "envelope_exponent": 5},
        spherical_basis_local={"num_spherical": num_spherical,
                               "num_radial": num_radial_spherical,
                               "cutoff": cutoff, "envelope_exponent": 5},
        mlp_rbf_kwargs={"units": units, "activation": "swish"},
        mlp_sbf_kwargs={"units": units, "activation": "swish"},
        global_mp_kwargs={"units": units, "pooling_method": "mean"},
        local_mp_kwargs={"units": units, "output_units": num_targets,
                         "output_kernel_initializer": "zeros"},
        node_pooling_args={"pooling_method": "scatter_sum"},
        use_output_mlp=True,
        output_embedding="graph",
        output_mlp={"use_bias": [True], "units": [num_targets],
                     "activation": ["linear"]},
    )

    keras_model(padded, training=False)
    transfer_weights_to_make_model(torch_model, keras_model, "MXMNet", depth)

    with torch.no_grad():
        torch_out = torch_model(torch_data).detach().cpu()
    keras_out = keras_to_torch(keras_model(padded, training=False))
    mae = float((torch_out - keras_out).abs().mean())
    print(f"MXMNet forward: MAE={mae:.2e}")

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_model)
    target = torch.randn(BATCH_SIZE, num_targets); target.requires_grad_(False)

    return run_training("MXMNet (ClinTox) [make_model]",
        lambda: torch_model(torch_data),
        lambda: keras_model_forward(keras_model, padded),
        torch_params, keras_params, target, lr=0.001, grad_clip=1.0,
        keras_model=keras_model, keras_inputs=padded)


# ---- Plot ----

def plot_results(all_results, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx, (name, res) in enumerate(all_results.items()):
        tl = res["torch_losses"]
        kl = res["keras_losses"]
        ld = res["loss_diffs"]
        od = res["output_diffs"]
        steps = list(range(len(tl)))

        ax = axes[idx, 0]
        ax.plot(steps, tl, "-", color="#2196F3", label="Torch", linewidth=1.5)
        ax.plot(steps, kl, "--", color="#FF5722", label="Keras make_model", linewidth=1.5)
        ax.set_title(f"{name}: Loss Curves", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        ax = axes[idx, 1]
        ax.semilogy(steps, [max(d, 1e-15) for d in ld], "-", color="#4CAF50", linewidth=1.5)
        ax.set_title(f"{name}: |Loss Diff|", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("Absolute Diff")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1e-6, color="gray", linestyle=":", alpha=0.5, label="1e-6")
        ax.legend(fontsize=9)

        ax = axes[idx, 2]
        ax.semilogy(steps, [max(d, 1e-15) for d in od], "-", color="#9C27B0", linewidth=1.5)
        ax.set_title(f"{name}: Output MAE", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("|out_torch - out_keras|")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Divergence: kgcnn_torch vs kgcnn make_model() — Real Data",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)


# ---- Main ----

def main():
    print("Alignment Test: kgcnn_torch vs kgcnn official make_model()")
    print(f"Datasets: MUTAG, ClinTox")
    print(f"Batch size: {BATCH_SIZE}, Steps: {N_STEPS}, LR: {LR}")
    print("=" * 70)

    tests = [
        ("GCN (MUTAG)", test_gcn),
        ("GIN (MUTAG)", test_gin),
        ("GIN-edge (MUTAG)", test_gin_edge),
        ("GAT (ClinTox)", test_gat),
        ("GATv2 (ClinTox)", test_gatv2),
        ("SchNet (ClinTox)", test_schnet),
        ("GraphSAGE (MUTAG)", test_graphsage),
        ("AttentiveFP (MUTAG)", test_attentivefp),
        ("rGIN (MUTAG)", test_rgin),
        ("EGNN (ClinTox)", test_egnn),
        ("PAiNN (ClinTox)", test_painn),
        ("NMPN (MUTAG)", test_nmpn),
        ("DMPNN (MUTAG)", test_dmpnn),
        ("CMPNN (MUTAG)", test_cmpnn),
        ("DGIN (MUTAG)", test_dgin),
        ("RGCN (MUTAG)", test_rgcn),
        ("GNNFilm (MUTAG)", test_gnnfilm),
        ("INorp (MUTAG)", test_inorp),
        ("MEGAN (MUTAG)", test_megan),
        ("MEGNet (ClinTox)", test_megnet),
        ("CGCNN (ClinTox)", test_cgcnn),
        ("HamNet (ClinTox)", test_hamnet),
        ("DimeNetPP (ClinTox)", test_dimenetpp),
        ("HDNNP2nd (ClinTox)", test_hdnnp2nd),
        ("HDNNP2nd+dipole (ClinTox)", test_hdnnp2nd_dipole),
        ("MAT (ClinTox)", test_mat),
        ("MoGAT (MUTAG)", test_mogat),
        ("MXMNet (ClinTox)", test_mxmnet),
    ]

    all_results = {}
    for name, test_fn in tests:
        try:
            res = test_fn()
            all_results[name] = res
        except Exception as e:
            import traceback
            print(f"\n  {name} ERROR: {e}")
            traceback.print_exc()

    if all_results:
        save_path = os.path.join(SCRIPT_DIR, "training_divergence_make_model.png")
        plot_results(all_results, save_path)

    print(f"\n{'='*90}")
    print(f"{'Model':<30} {'Steps':>5} {'Final Loss Diff':>18} {'Final Out MAE':>18}")
    print(f"{'='*90}")
    for name, res in all_results.items():
        n = len(res["loss_diffs"])
        ld = res["loss_diffs"][-1]
        od = res["output_diffs"][-1]
        print(f"{name:<30} {n:>5} {ld:>18.6e} {od:>18.6e}")

    print(f"\nCompleted: {len(all_results)}/{len(tests)}")


if __name__ == "__main__":
    main()
