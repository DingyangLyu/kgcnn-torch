#!/usr/bin/env python3
"""Layerwise numeric alignment check: HDNNP2nd weighted ACSF descriptors."""
import os
import sys
from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch

from alignment_thresholds import get_thresholds

os.environ.setdefault("KERAS_BACKEND", "torch")
from keras import ops

ROOT = "/home/yuanbai/Downloads/MLIPs"
TORCH_REPO = os.path.join(ROOT, "kgcnn-torch")
KERAS_REPO = os.path.join(ROOT, "gcnn_keras-master")
sys.path.insert(0, TORCH_REPO)
sys.path.insert(0, KERAS_REPO)

from kgcnn.literature.HDNNP2nd._wacsf import wACSFRad as KeraswACSFRad, wACSFAng as KeraswACSFAng
from kgcnn_torch.models.hdnnp2nd import wACSFRad, wACSFAng


@dataclass
class Config:
    n_nodes: int = 14
    n_edges: int = 40
    n_angles: int = 36
    n_types: int = 4
    n_rad_features: int = 8
    n_ang_features: int = 6
    cutoff: float = 5.0
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
    torch.manual_seed(cfg.seed)

    # Use type ids directly: 0..n_types-1
    z = torch.randint(0, cfg.n_types, (cfg.n_nodes,), dtype=torch.long)
    pos = torch.randn(cfg.n_nodes, 3)

    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    e_t = torch.stack([src, dst], dim=0)
    e_k = torch.stack([dst, src], dim=0)

    center = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr2 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    angle_index = torch.stack([center, nbr1, nbr2], dim=0)

    type_map = z.clone()

    t_rad = wACSFRad(cfg.n_types, cfg.n_rad_features, cfg.cutoff)
    t_ang = wACSFAng(cfg.n_types, cfg.n_ang_features, cfg.cutoff)

    eta_mu = np.stack([t_rad.eta.detach().cpu().numpy(), t_rad.mu.detach().cpu().numpy()], axis=-1)
    eta_mu_lz = np.stack([
        t_ang.eta.detach().cpu().numpy(),
        t_ang.mu.detach().cpu().numpy(),
        t_ang.lam.detach().cpu().numpy(),
        t_ang.zeta.detach().cpu().numpy(),
    ], axis=-1)

    k_rad = KeraswACSFRad(eta_mu=eta_mu, cutoff=cfg.cutoff, add_eps=True)
    k_ang = KeraswACSFAng(eta_mu_lambda_zeta=eta_mu_lz, cutoff=cfg.cutoff, add_eps=True)

    rep_rad_t = t_rad(z, pos, e_t, type_map).detach().cpu()
    rep_ang_t = t_ang(z, pos, angle_index, type_map).detach().cpu()

    rep_rad_k = k_rad([
        ops.convert_to_tensor(z.numpy()),
        ops.convert_to_tensor(pos.numpy()),
        ops.convert_to_tensor(e_k.numpy()),
    ])
    rep_ang_k = k_ang([
        ops.convert_to_tensor(z.numpy()),
        ops.convert_to_tensor(pos.numpy()),
        ops.convert_to_tensor(angle_index.numpy()),
    ])

    ref = {
        "rep_rad": torch.as_tensor(ops.convert_to_numpy(rep_rad_k)),
        "rep_ang": torch.as_tensor(ops.convert_to_numpy(rep_ang_k)),
    }
    got = {
        "rep_rad": rep_rad_t,
        "rep_ang": rep_ang_t,
    }
    compare_stage_dicts(ref, got)


if __name__ == "__main__":
    main()
