#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras MEGnetBlock vs kgcnn-torch MEGNetBlock."""
import os
import sys
from dataclasses import dataclass
from typing import Dict

import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.literature.Megnet._layers import MEGnetBlock as KerasMEGnetBlock
from kgcnn_torch.models.megnet import MEGNetBlock


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    batch_size: int = 2
    node_dim: int = 16
    edge_dim: int = 16
    state_dim: int = 8
    seed: int = 42


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_random_graph(cfg: Config):
    torch.manual_seed(cfg.seed)
    n = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    e = torch.randn(cfg.n_edges, cfg.edge_dim, dtype=torch.float32)
    state = torch.randn(cfg.batch_size, cfg.state_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    batch = torch.cat([
        torch.zeros(cfg.n_nodes // 2, dtype=torch.long),
        torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long),
    ], dim=0)
    batch_edge = batch[src]
    count_nodes = torch.bincount(batch, minlength=cfg.batch_size)
    count_edges = torch.bincount(batch_edge, minlength=cfg.batch_size)
    return n, e, state, edge_index_torch, edge_index_keras, batch, batch_edge, count_nodes, count_edges


def torch_forward(block: MEGNetBlock, n, e, s, edge_index, batch, batch_edge, batch_size):
    n2, e2, s2 = block(n, e, s, edge_index, batch, batch_edge, batch_size)
    return {"node": n2.detach().cpu(), "edge": e2.detach().cpu(), "state": s2.detach().cpu()}


def keras_forward(block: KerasMEGnetBlock, n, e, s, edge_index, batch, batch_edge, count_nodes, count_edges):
    n2, e2, s2 = block([n, e, edge_index, s, batch, batch_edge, count_nodes, count_edges])
    return {
        "node": torch.as_tensor(ops.convert_to_numpy(n2)),
        "edge": torch.as_tensor(ops.convert_to_numpy(e2)),
        "state": torch.as_tensor(ops.convert_to_numpy(s2)),
    }


MAX_MAE, MAX_ABS = get_thresholds(__file__)


def compare_stage_dicts(ref: Dict[str, torch.Tensor], got: Dict[str, torch.Tensor]):
    print("Stage alignment report (Keras vs Torch):")
    worst_mae = 0.0
    worst_abs = 0.0
    for key in ref.keys():
        r, g = ref[key], got[key]
        diff = (r - g).abs()
        mae = float(diff.mean().item())
        rmse = float(torch.sqrt(((r - g) ** 2).mean()).item())
        max_abs = float(diff.max().item())
        worst_mae = max(worst_mae, mae)
        worst_abs = max(worst_abs, max_abs)
        print(f"- {key:6s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    n, e, s, edge_index_torch, edge_index_keras, batch, batch_edge, count_nodes, count_edges = build_random_graph(cfg)

    torch_block = MEGNetBlock(
        edge_dim=cfg.edge_dim,
        node_dim=cfg.node_dim,
        state_dim=cfg.state_dim,
        units_edge=[cfg.edge_dim, cfg.edge_dim, cfg.edge_dim],
        units_node=[cfg.node_dim, cfg.node_dim, cfg.node_dim],
        units_state=[cfg.state_dim, cfg.state_dim, cfg.state_dim],
        activation="softplus2",
    )
    keras_block = KerasMEGnetBlock(
        node_embed=[cfg.node_dim, cfg.node_dim, cfg.node_dim],
        edge_embed=[cfg.edge_dim, cfg.edge_dim, cfg.edge_dim],
        env_embed=[cfg.state_dim, cfg.state_dim, cfg.state_dim],
        pooling_method="mean",
        activation="kgcnn>softplus2",
        use_bias=True,
    )

    _ = keras_forward(
        keras_block,
        ops.convert_to_tensor(n.numpy()),
        ops.convert_to_tensor(e.numpy()),
        ops.convert_to_tensor(s.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(batch_edge.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(count_edges.numpy()),
    )

    # Edge MLP
    copy_dense_torch_to_keras(torch_block.edge_dense_layers[0], keras_block.lay_phi_e)
    copy_dense_torch_to_keras(torch_block.edge_dense_layers[1], keras_block.lay_phi_e_1)
    copy_dense_torch_to_keras(torch_block.edge_dense_layers[2], keras_block.lay_phi_e_2)
    # Node MLP
    copy_dense_torch_to_keras(torch_block.node_dense_layers[0], keras_block.lay_phi_n)
    copy_dense_torch_to_keras(torch_block.node_dense_layers[1], keras_block.lay_phi_n_1)
    copy_dense_torch_to_keras(torch_block.node_dense_layers[2], keras_block.lay_phi_n_2)
    # State MLP
    copy_dense_torch_to_keras(torch_block.state_dense_layers[0], keras_block.lay_phi_u)
    copy_dense_torch_to_keras(torch_block.state_dense_layers[1], keras_block.lay_phi_u_1)
    copy_dense_torch_to_keras(torch_block.state_dense_layers[2], keras_block.lay_phi_u_2)

    torch_out = torch_forward(
        torch_block, n, e, s, edge_index_torch, batch, batch_edge, cfg.batch_size
    )
    keras_out = keras_forward(
        keras_block,
        ops.convert_to_tensor(n.numpy()),
        ops.convert_to_tensor(e.numpy()),
        ops.convert_to_tensor(s.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(batch_edge.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(count_edges.numpy()),
    )
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
