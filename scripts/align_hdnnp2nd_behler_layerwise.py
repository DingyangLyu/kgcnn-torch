#!/usr/bin/env python3
"""Layerwise numeric alignment check: HDNNP2nd Behler ACSF descriptors."""
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

from kgcnn.literature.HDNNP2nd._acsf import ACSFG2, ACSFG4
from kgcnn_torch.models.hdnnp2nd import HDNNP2ndBehlerModel


@dataclass
class Config:
    n_nodes: int = 12
    n_edges: int = 36
    n_angles: int = 30
    seed: int = 42


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
        print(f"- {key:10s} | shape={tuple(r.shape)} | MAE={mae:.6e} | RMSE={rmse:.6e} | MAX={max_abs:.6e}")
        worst_mae = max(worst_mae, mae)
        worst_abs = max(worst_abs, max_abs)
    if worst_mae > MAX_MAE or worst_abs > MAX_ABS:
        raise SystemExit(
            f"Alignment assertion failed: worst MAE={worst_mae:.3e}, "
            f"worst MAX={worst_abs:.3e}, thresholds MAE<={MAX_MAE:.1e}, MAX<={MAX_ABS:.1e}"
        )


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    elements = [1, 6]
    z = torch.where(torch.rand(cfg.n_nodes) > 0.5, torch.tensor(1), torch.tensor(6)).long()
    pos = torch.randn(cfg.n_nodes, 3)

    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    e_t = torch.stack([src, dst], dim=0)
    e_k = torch.stack([dst, src], dim=0)

    center = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr2 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    # Avoid degenerate triplets that can cause undefined angles.
    nbr1 = torch.where(nbr1 == center, (nbr1 + 1) % cfg.n_nodes, nbr1)
    nbr2 = torch.where(nbr2 == center, (nbr2 + 2) % cfg.n_nodes, nbr2)
    nbr2 = torch.where(nbr2 == nbr1, (nbr2 + 1) % cfg.n_nodes, nbr2)
    a_idx = torch.stack([center, nbr1, nbr2], dim=0)

    g2_eta = [0.0, 0.3]
    g2_rs = [0.0, 3.0]
    g4_eta = [0.0, 0.3]
    g4_zeta = [1.0, 8.0]
    g4_lamda = [-1.0, 1.0]

    t_model = HDNNP2ndBehlerModel(
        element_types=elements,
        g2_eta=g2_eta, g2_rs=g2_rs, g2_rc=6.0,
        g4_eta=g4_eta, g4_zeta=g4_zeta, g4_lamda=g4_lamda, g4_rc=6.0,
        g4_multiplicity=2.0,
        relational_units=[8, 8], relational_activation=["swish", "linear"],
        output_embedding="node",
    )

    k_g2 = ACSFG2(**ACSFG2.make_param_table(eta=g2_eta, rs=g2_rs, rc=6.0, elements=elements))
    k_g4 = ACSFG4(**ACSFG4.make_param_table(
        eta=g4_eta, zeta=g4_zeta, lamda=g4_lamda, rc=6.0, elements=elements, multiplicity=2.0
    ))

    rep_g2_t = t_model._compute_g2(z, pos, e_t).detach().cpu()
    rep_g4_t = t_model._compute_g4(z, pos, a_idx).detach().cpu()

    rep_g2_k = k_g2([
        ops.convert_to_tensor(z.numpy()),
        ops.convert_to_tensor(pos.numpy()),
        ops.convert_to_tensor(e_k.numpy()),
    ])
    rep_g4_k = k_g4([
        ops.convert_to_tensor(z.numpy()),
        ops.convert_to_tensor(pos.numpy()),
        ops.convert_to_tensor(a_idx.numpy()),
    ])

    ref = {
        "rep_g2": torch.as_tensor(ops.convert_to_numpy(rep_g2_k)),
        "rep_g4": torch.as_tensor(ops.convert_to_numpy(rep_g4_k)),
    }
    got = {
        "rep_g2": rep_g2_t,
        "rep_g4": rep_g4_t,
    }
    compare_stage_dicts(ref, got)


if __name__ == "__main__":
    main()
