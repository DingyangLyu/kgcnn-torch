#!/usr/bin/env python3
"""Layerwise numeric alignment check: Keras GNNFilm vs kgcnn-torch GNNFilm."""
import os
import sys
from dataclasses import dataclass
from typing import Dict, List

import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops
import keras

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.layers.aggr import AggregateLocalEdges as KerasAggregateLocalEdges
from kgcnn.layers.gather import GatherNodes as KerasGatherNodes
from kgcnn.layers.relational import RelationalDense as KerasRelationalDense
from kgcnn_torch.models.gnnfilm import GNNFilmModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    units: int = 16
    depth: int = 3
    num_relations: int = 5
    seed: int = 42


class KerasGNNFilmStack:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.layers: List[dict] = []
        self.gather = KerasGatherNodes(split_indices=[0, 1], concat_axis=None)
        self.aggr = KerasAggregateLocalEdges(pooling_method="scatter_sum")
        self.act = keras.layers.Activation("swish")
        self.mul = keras.layers.Multiply()
        self.add = keras.layers.Add()
        for _ in range(cfg.depth):
            self.layers.append({
                "gamma": KerasRelationalDense(cfg.units, cfg.num_relations, activation="sigmoid", use_bias=True),
                "beta": KerasRelationalDense(cfg.units, cfg.num_relations, activation="sigmoid", use_bias=True),
                "hj": KerasRelationalDense(cfg.units, cfg.num_relations, activation="linear", use_bias=True),
            })

    def forward(self, x, edge_type, edge_index) -> Dict[str, torch.Tensor]:
        out: Dict[str, torch.Tensor] = {}
        h = x
        for i in range(self.cfg.depth):
            n_i, n_j = self.gather([h, edge_index])
            gamma = self.layers[i]["gamma"]([n_i, edge_type])
            beta = self.layers[i]["beta"]([n_i, edge_type])
            h_j = self.layers[i]["hj"]([n_j, edge_type])
            m = self.add([self.mul([h_j, gamma]), beta])
            h = self.aggr([h, m, edge_index])
            h = self.act(h)
            out[f"layer_{i+1}"] = torch.as_tensor(ops.convert_to_numpy(h))
        return out


def build_inputs(cfg: Config):
    torch.manual_seed(cfg.seed)
    x = torch.randn(cfg.n_nodes, cfg.units, dtype=torch.float32)
    src = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, size=(cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    edge_type = torch.randint(0, cfg.num_relations, (cfg.n_edges,), dtype=torch.long)
    return x, edge_type, edge_index_torch, edge_index_keras


def torch_forward(model: GNNFilmModel, x, edge_type, edge_index) -> Dict[str, torch.Tensor]:
    out: Dict[str, torch.Tensor] = {}
    h = x
    for i in range(model.depth):
        h = model.film_layers[i](h, edge_index, edge_type)
        out[f"layer_{i+1}"] = h.detach().cpu()
    return out


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
    x, edge_type, edge_index_torch, edge_index_keras = build_inputs(cfg)

    torch_model = GNNFilmModel(
        node_dim=cfg.units,
        depth=cfg.depth,
        units=cfg.units,
        num_relations=cfg.num_relations,
        activation="swish",
        modulation_activation="sigmoid",
        film_pooling="sum",
        output_units=[],
        output_activation="linear",
        output_final_activation="linear",
        num_targets=1,
        output_embedding="node",
        use_node_embedding=False,
    )
    keras_stack = KerasGNNFilmStack(cfg)

    _ = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_type.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )

    for i in range(cfg.depth):
        # Keras RelationalDense expects kernel shape (R, in, out)
        gamma = torch_model.film_layers[i].rel_dense_gamma
        beta = torch_model.film_layers[i].rel_dense_beta
        hj = torch_model.film_layers[i].rel_dense_hj

        keras_stack.layers[i]["gamma"].set_weights([
            gamma.weight.detach().cpu().numpy(),
            gamma.bias.detach().cpu().numpy()
        ])
        keras_stack.layers[i]["beta"].set_weights([
            beta.weight.detach().cpu().numpy(),
            beta.bias.detach().cpu().numpy()
        ])
        keras_stack.layers[i]["hj"].set_weights([
            hj.weight.detach().cpu().numpy(),
            hj.bias.detach().cpu().numpy()
        ])

    torch_out = torch_forward(torch_model, x, edge_type, edge_index_torch)
    keras_out = keras_stack.forward(
        ops.convert_to_tensor(x.numpy()),
        ops.convert_to_tensor(edge_type.numpy()),
        ops.convert_to_tensor(edge_index_keras.numpy()),
    )
    compare_stage_dicts(keras_out, torch_out)


if __name__ == "__main__":
    main()
