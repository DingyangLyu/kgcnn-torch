#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras MEGAN vs kgcnn-torch MEGAN."""
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

from kgcnn.literature.MEGAN._model import MEGAN as KerasMEGAN
from kgcnn_torch.models.megan import MEGANModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    node_dim: int = 16
    edge_dim: int = 8
    units: tuple = (8, 8)
    importance_channels: int = 2
    seed: int = 42


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
    edge_attr = torch.randn(cfg.n_edges, cfg.edge_dim, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)

    batch = torch.cat([
        torch.zeros(cfg.n_nodes // 2, dtype=torch.long),
        torch.ones(cfg.n_nodes - cfg.n_nodes // 2, dtype=torch.long),
    ], dim=0)
    count_nodes = torch.bincount(batch, minlength=2)
    out_true = torch.zeros((2, 1), dtype=torch.float32)
    return x, edge_attr, edge_index_torch, edge_index_keras, batch, count_nodes, out_true


def torch_forward(model: MEGANModel, x, edge_attr, edge_index, batch) -> Dict[str, torch.Tensor]:
    out = model.forward_explanations(type("Obj", (), {
        "x": x,
        "z": None,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "batch": batch,
    })())
    return {
        "output": out["output"].detach().cpu(),
        "node_importance": out["node_importance"].detach().cpu(),
        "edge_importance": out["edge_importance"].detach().cpu(),
    }


def keras_forward(model: KerasMEGAN, x, edge_attr, edge_index, batch, count_nodes, out_true) -> Dict[str, torch.Tensor]:
    out, node_imp, edge_imp = model([
        x, edge_attr, edge_index, out_true, batch, count_nodes
    ], training=False, return_importances=True)
    return {
        "output": torch.as_tensor(ops.convert_to_numpy(out)),
        "node_importance": torch.as_tensor(ops.convert_to_numpy(node_imp)),
        "edge_importance": torch.as_tensor(ops.convert_to_numpy(edge_imp)),
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
        print(f"- {key:16s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")

    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    x, edge_attr, edge_index_torch, edge_index_keras, batch, count_nodes, out_true = build_inputs(cfg)

    torch_model = MEGANModel(
        node_dim=cfg.node_dim,
        units=list(cfg.units),
        num_heads=cfg.importance_channels,
        depth=len(cfg.units),
        attention_activation="leaky_relu2",
        use_edge_features=True,
        edge_dim=cfg.edge_dim,
        concat_heads=True,
        importance_channels=cfg.importance_channels,
        importance_units=[],
        importance_activation="relu",
        final_units=[1],
        final_activation="linear",
        final_pooling="sum",
        num_targets=1,
        output_embedding="graph",
        use_node_embedding=False,
        node_input_dim=cfg.node_dim,
    )
    # Keras MEGAN with input_node_embedding=None has no projection layer;
    # initialize torch node_projection to identity so it's a no-op (if it exists).
    if torch_model.node_projection is not None:
        with torch.no_grad():
            torch.nn.init.eye_(torch_model.node_projection.weight)
            torch.nn.init.zeros_(torch_model.node_projection.bias)

    keras_model = KerasMEGAN(
        units=list(cfg.units),
        activation="kgcnn>leaky_relu2",
        use_bias=True,
        dropout_rate=0.0,
        use_edge_features=True,
        input_node_embedding=None,
        importance_units=[],
        importance_channels=cfg.importance_channels,
        importance_activation="sigmoid",
        importance_dropout_rate=0.0,
        importance_factor=0.0,
        importance_multiplier=10.0,
        sparsity_factor=0.0,
        concat_heads=True,
        final_units=[1],
        final_dropout_rate=0.0,
        final_activation="linear",
        final_pooling="sum",
        regression_limits=None,
        regression_reference=None,
        return_importances=True,
    )

    _ = keras_forward(
        keras_model,
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(out_true.numpy()),
    )

    # Attention stacks
    for i, (k_layer, t_layer) in enumerate(zip(keras_model.attention_layers, torch_model.attention_layers)):
        for k in range(cfg.importance_channels):
            k_linear, k_alpha_act, k_alpha = k_layer.head_layers[k]
            copy_dense_torch_to_keras(t_layer.head_linears[k][0], k_linear)
            copy_dense_torch_to_keras(t_layer.head_alpha_acts[k][0], k_alpha_act)
            copy_dense_torch_to_keras(t_layer.head_alphas[k], k_alpha)

    # Importance MLP
    copy_dense_torch_to_keras(torch_model.importance_mlp.linears[0], keras_model.node_importance_layers[0])

    # Final output layer
    copy_dense_torch_to_keras(torch_model.output_mlp.linears[0], keras_model.final_layers[0])

    torch_out = torch_forward(torch_model, x, edge_attr, edge_index_torch, batch)
    keras_out = keras_forward(
        keras_model,
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_attr.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
        ops.convert_to_tensor(batch.numpy()),
        ops.convert_to_tensor(count_nodes.numpy()),
        ops.convert_to_tensor(out_true.numpy()),
    )
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
