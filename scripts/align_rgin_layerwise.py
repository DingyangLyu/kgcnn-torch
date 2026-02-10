#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras rGIN stack vs kgcnn-torch rGIN."""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch
from types import MethodType

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops
import keras

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.literature.rGIN._layers import rGIN as KerasrGIN
from kgcnn.layers.mlp import GraphMLP
from kgcnn_torch.models.rgin import rGINModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    units: int = 16
    depth: int = 3
    seed: int = 42
    deterministic_random: bool = True


class KerasrGINStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dense_in = keras.layers.Dense(cfg.units, activation="linear", use_bias=True)
        self.convs: List[KerasrGIN] = [
            KerasrGIN(pooling_method="sum", epsilon_learnable=False, random_range=100) for _ in range(cfg.depth)
        ]
        self.mlps: List[GraphMLP] = [
            GraphMLP(units=[cfg.units, cfg.units], activation=["relu", "relu"])
            for _ in range(cfg.depth)
        ]
        if cfg.deterministic_random:
            self._patch_convs_zero_random()

    def _patch_convs_zero_random(self):
        def _call_zero_random(this, inputs, **kwargs):
            node, edge_index = inputs
            random_values = ops.zeros((ops.shape(node)[0], 1), dtype=node.dtype)
            node = this.lay_concat([node, random_values])
            ed = this.lay_gather([node, edge_index], **kwargs)
            nu = this.lay_pool([node, ed, edge_index], **kwargs)
            out = this.lay_add([node, nu], **kwargs)
            return out

        for conv in self.convs:
            conv.call = MethodType(_call_zero_random, conv)

    def forward(self, x, edge_index, batch, count_nodes, seed: int) -> Dict[str, torch.Tensor]:
        torch.manual_seed(seed)
        out: Dict[str, torch.Tensor] = {}
        h = self.dense_in(x)
        out["dense_in"] = torch.as_tensor(ops.convert_to_numpy(h))
        for i in range(self.cfg.depth):
            h = self.convs[i]([h, edge_index])
            h = self.mlps[i]([h, batch, count_nodes])
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))
        return out


def copy_dense_torch_to_keras(torch_linear: torch.nn.Linear, keras_dense):
    kernel = torch_linear.weight.detach().cpu().numpy().T
    if torch_linear.bias is None:
        keras_dense.set_weights([kernel])
    else:
        bias = torch_linear.bias.detach().cpu().numpy()
        keras_dense.set_weights([kernel, bias])


def build_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.node_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    batch = torch.cat([
        torch.zeros(cfg.n_nodes // 2, dtype=torch.long),
        torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long),
    ], dim=0)
    count_nodes = torch.bincount(batch, minlength=2)
    return x, edge_index_torch, edge_index_keras, batch, count_nodes


def torch_forward(model: rGINModel, x, edge_index, batch, seed: int) -> Dict[str, torch.Tensor]:
    torch.manual_seed(seed)
    out: Dict[str, torch.Tensor] = {}
    h = model.dense_in(x)
    out["dense_in"] = h.detach().cpu()
    for i in range(model.depth):
        h = model.convs[i](h, edge_index)
        h = model.mlps[i](h)
        out[f"layer_{i+1}"] = h.detach().cpu()
    return out


def patch_torch_convs_zero_random(model: rGINModel):
    def _forward_zero_random(this, x, edge_index):
        from kgcnn_torch.layers.gather import gather_nodes_outgoing
        num_nodes = x.size(0)
        random_values = torch.zeros(num_nodes, 1, device=x.device, dtype=x.dtype)
        x_aug = torch.cat([x, random_values], dim=-1)
        x_j = gather_nodes_outgoing(x_aug, edge_index)
        agg = this.aggr(x_j, edge_index, num_nodes)
        return (1.0 + this.eps) * x_aug + agg

    for conv in model.convs:
        conv.forward = MethodType(_forward_zero_random, conv)


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
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    x, edge_index_torch, edge_index_keras, batch, count_nodes = build_inputs(cfg)

    torch_model = rGINModel(
        node_dim=cfg.node_dim,
        depth=cfg.depth,
        units=cfg.units,
        gin_mlp_units=[cfg.units, cfg.units],
        gin_mlp_activation="relu",
        gin_mlp_use_normalization=False,
        random_range=100,
        output_units=[],
        output_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_stack = KerasrGINStack(cfg)
    if cfg.deterministic_random:
        patch_torch_convs_zero_random(torch_model)

    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        seed=cfg.seed,
    )

    copy_dense_torch_to_keras(torch_model.dense_in, keras_stack.dense_in)
    for i in range(cfg.depth):
        # epsilon
        keras_stack.convs[i].eps_k.assign(float(torch_model.convs[i].eps.detach().cpu().item()))
        # graph mlp weights
        for j, kd in enumerate(keras_stack.mlps[i].mlp_dense_layer_list):
            copy_dense_torch_to_keras(torch_model.mlps[i].linears[j], kd)

    torch_out = torch_forward(torch_model, x, edge_index_torch, batch, seed=cfg.seed)
    keras_out = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        seed=cfg.seed,
    )
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
