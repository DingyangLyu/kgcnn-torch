#!/usr/bin/env python3
"""Deep investigation: DimeNetPP & MXMNet training divergence.

Questions to answer:
1. With proper optimizer (Adam + grad clip), do they train stably?
2. If stable, does the Torch vs Keras divergence grow or stay bounded?
3. Where exactly does the divergence originate? (basis layers vs MLP layers)

Usage:
    KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES="" python scripts/investigate_basis_divergence.py
"""
import os
import sys
import math
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.functional as F

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from model_alignment_utils import make_disjoint_graph
from train_alignment_utils import collect_keras_params


def run_experiment(name, torch_fwd, keras_fwd, torch_params, keras_params, target,
                   n_steps, optimizer_type, lr, grad_clip=None):
    """Run training with configurable optimizer."""
    if optimizer_type == "sgd":
        torch_opt = torch.optim.SGD(torch_params, lr=lr)
        keras_opt = torch.optim.SGD(keras_params, lr=lr)
    elif optimizer_type == "adam":
        torch_opt = torch.optim.Adam(torch_params, lr=lr)
        keras_opt = torch.optim.Adam(keras_params, lr=lr)

    torch_losses, keras_losses, loss_diffs, output_diffs = [], [], [], []
    grad_norms_torch, grad_norms_keras = [], []

    print(f"\n--- {name}: {optimizer_type.upper()}, lr={lr}, "
          f"clip={grad_clip}, {n_steps} steps ---")

    for step in range(n_steps):
        torch_out = torch_fwd()
        keras_out = keras_fwd()

        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)

        tl = float(torch_loss.item())
        kl = float(keras_loss.item())
        torch_losses.append(tl)
        keras_losses.append(kl)
        loss_diffs.append(abs(tl - kl))

        with torch.no_grad():
            out_mae = float((torch_out.detach() - keras_out.detach()).abs().mean().item())
        output_diffs.append(out_mae)

        if math.isnan(tl) or math.isnan(kl) or math.isinf(tl) or math.isinf(kl):
            print(f"  Step {step}: NaN/Inf — loss_t={tl:.3e} loss_k={kl:.3e}")
            break

        # Backward
        torch_opt.zero_grad()
        torch_loss.backward()
        # Measure grad norm BEFORE clipping
        t_gnorm = torch.nn.utils.clip_grad_norm_(torch_params, float('inf'))
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(torch_params, grad_clip)
        torch_opt.step()

        keras_opt.zero_grad()
        keras_loss.backward()
        k_gnorm = torch.nn.utils.clip_grad_norm_(keras_params, float('inf'))
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(keras_params, grad_clip)
        keras_opt.step()

        grad_norms_torch.append(float(t_gnorm))
        grad_norms_keras.append(float(k_gnorm))

        if step % 5 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: loss_t={tl:.4e} loss_k={kl:.4e} "
                  f"diff={loss_diffs[-1]:.3e} out_mae={out_mae:.3e} "
                  f"gnorm_t={t_gnorm:.2e} gnorm_k={k_gnorm:.2e}")

    return {
        "torch_losses": torch_losses, "keras_losses": keras_losses,
        "loss_diffs": loss_diffs, "output_diffs": output_diffs,
        "grad_norms_torch": grad_norms_torch, "grad_norms_keras": grad_norms_keras,
    }


def setup_dimenetpp():
    from align_dimenetpp_model import (
        Config, KerasDimeNetPPFullStack, DimeNetPPModel,
        transfer_all_weights, generate_angle_index)
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_index = generate_angle_index(torch_data.edge_index)
    torch_data.angle_index = angle_index

    def build_models():
        torch_model = DimeNetPPModel(
            emb_size=cfg.emb_size, out_emb_size=cfg.out_emb_size,
            int_emb_size=cfg.int_emb_size, basis_emb_size=cfg.basis_emb_size,
            num_blocks=cfg.num_blocks, num_spherical=cfg.num_spherical,
            num_radial=cfg.num_radial, cutoff=cfg.cutoff,
            envelope_exponent=cfg.envelope_exponent,
            num_before_skip=cfg.num_before_skip, num_after_skip=cfg.num_after_skip,
            num_dense_output=cfg.num_dense_output,
            num_targets=cfg.num_targets, activation=cfg.activation,
            extensive=cfg.extensive, output_init=cfg.output_init, output_embedding="graph",
            use_node_embedding=True, num_embeddings=cfg.num_embeddings,
            use_output_mlp=cfg.use_output_mlp, output_mlp_units=cfg.output_mlp_units,
            output_mlp_activation=cfg.output_mlp_activation)
        torch_model.train()

        keras_stack = KerasDimeNetPPFullStack(cfg)
        z_k, pos_k = keras_data["z"], keras_data["pos"]
        ei_k = keras_data["edge_index"]
        bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
        _ = keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k)
        transfer_all_weights(torch_model, keras_stack, cfg)

        torch_params = list(torch_model.parameters())
        keras_params = collect_keras_params(keras_stack)
        target = torch.randn(cfg.batch_size, cfg.num_targets)
        target.requires_grad_(False)

        def torch_fwd():
            return torch_model(torch_data)
        def keras_fwd():
            return keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k)

        return torch_fwd, keras_fwd, torch_params, keras_params, target

    return build_models


def setup_mxmnet():
    from align_mxmnet_model import (
        Config, KerasMXMNetFullStack, MXMNetModel,
        transfer_all_weights, generate_angle_index_mxmnet)
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_idx = generate_angle_index_mxmnet(torch_data.edge_index)
    torch_data.angle_index_1 = angle_idx
    torch_data.angle_index_2 = angle_idx

    def build_models():
        torch_model = MXMNetModel(
            node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
            num_radial=cfg.num_radial, num_spherical=cfg.num_spherical,
            num_radial_spherical=cfg.num_radial_spherical,
            cutoff=cfg.cutoff, envelope_exponent=cfg.envelope_exponent,
            activation=cfg.activation, mp_pooling=cfg.mp_pooling,
            global_mp_pooling=cfg.global_mp_pooling,
            use_local_mp=True, node_pooling=cfg.node_pooling,
            output_units=cfg.output_units, output_activation=cfg.output_activation,
            num_targets=cfg.num_targets, output_embedding="graph",
            use_node_embedding=True, num_embeddings=cfg.num_embeddings, use_output_mlp=True)
        torch_model.train()

        keras_stack = KerasMXMNetFullStack(cfg)
        z_k, pos_k = keras_data["z"], keras_data["pos"]
        ei_k = keras_data["edge_index"]
        bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
        _ = keras_stack.forward(z_k, pos_k, ei_k, angle_idx, angle_idx, bid_k, cn_k)
        transfer_all_weights(torch_model, keras_stack, cfg)

        torch_params = list(torch_model.parameters())
        keras_params = collect_keras_params(keras_stack)
        target = torch.randn(cfg.batch_size, cfg.num_targets)
        target.requires_grad_(False)

        def torch_fwd():
            return torch_model(torch_data)
        def keras_fwd():
            return keras_stack.forward(z_k, pos_k, ei_k, angle_idx, angle_idx, bid_k, cn_k)

        return torch_fwd, keras_fwd, torch_params, keras_params, target

    return build_models


def plot_results(all_results, save_path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(n, 4, figsize=(20, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx, (label, res) in enumerate(all_results.items()):
        tl = res["torch_losses"]
        kl = res["keras_losses"]
        ld = res["loss_diffs"]
        od = res["output_diffs"]
        gnt = res.get("grad_norms_torch", [])
        gnk = res.get("grad_norms_keras", [])
        steps = list(range(len(tl)))

        # Col 1: Loss curves
        ax = axes[idx, 0]
        ax.plot(steps, tl, "-", color="#2196F3", label="Torch", linewidth=1.5)
        ax.plot(steps, kl, "--", color="#FF5722", label="Keras", linewidth=1.5)
        ax.set_title(f"{label}: Loss", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Col 2: Loss diff
        ax = axes[idx, 1]
        ax.semilogy(steps, [max(d, 1e-15) for d in ld], "-", color="#4CAF50", linewidth=1.5)
        ax.set_title(f"{label}: |Loss Diff|", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("Abs Diff")
        ax.grid(True, alpha=0.3)

        # Col 3: Output MAE
        ax = axes[idx, 2]
        ax.semilogy(steps, [max(d, 1e-15) for d in od], "-", color="#9C27B0", linewidth=1.5)
        ax.set_title(f"{label}: Output MAE", fontsize=10, fontweight="bold")
        ax.set_xlabel("Step"); ax.set_ylabel("MAE")
        ax.grid(True, alpha=0.3)

        # Col 4: Grad norms
        ax = axes[idx, 3]
        if gnt:
            gsteps = list(range(len(gnt)))
            ax.semilogy(gsteps, gnt, "-", color="#2196F3", label="Torch", linewidth=1)
            ax.semilogy(gsteps, gnk, "--", color="#FF5722", label="Keras", linewidth=1)
            ax.set_title(f"{label}: Grad Norm", fontsize=10, fontweight="bold")
            ax.legend(fontsize=8)
        ax.set_xlabel("Step"); ax.set_ylabel("||grad||")
        ax.grid(True, alpha=0.3)

    fig.suptitle("DimeNetPP & MXMNet: Training Stability Investigation",
                 fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)


def main():
    all_results = {}

    # ---- DimeNetPP experiments ----
    print("=" * 70)
    print("DimeNetPP Investigation")
    print("=" * 70)
    build_dpp = setup_dimenetpp()

    # Exp 1: Adam + aggressive grad clip
    tf, kf, tp, kp, tgt = build_dpp()
    all_results["DimeNetPP: Adam lr=1e-4 clip=0.5"] = run_experiment(
        "DimeNetPP", tf, kf, tp, kp, tgt,
        n_steps=50, optimizer_type="adam", lr=1e-4, grad_clip=0.5)

    # Exp 2: Adam + smaller lr + clip
    tf, kf, tp, kp, tgt = build_dpp()
    all_results["DimeNetPP: Adam lr=1e-5 clip=0.1"] = run_experiment(
        "DimeNetPP", tf, kf, tp, kp, tgt,
        n_steps=50, optimizer_type="adam", lr=1e-5, grad_clip=0.1)

    # Exp 3: SGD + very small lr + tight clip
    tf, kf, tp, kp, tgt = build_dpp()
    all_results["DimeNetPP: SGD lr=1e-9 clip=0.01"] = run_experiment(
        "DimeNetPP", tf, kf, tp, kp, tgt,
        n_steps=50, optimizer_type="sgd", lr=1e-9, grad_clip=0.01)

    # ---- MXMNet experiments ----
    print("\n" + "=" * 70)
    print("MXMNet Investigation")
    print("=" * 70)
    build_mxm = setup_mxmnet()

    # Exp 1: Adam + aggressive grad clip
    tf, kf, tp, kp, tgt = build_mxm()
    all_results["MXMNet: Adam lr=1e-4 clip=0.5"] = run_experiment(
        "MXMNet", tf, kf, tp, kp, tgt,
        n_steps=50, optimizer_type="adam", lr=1e-4, grad_clip=0.5)

    # Exp 2: Adam + smaller lr + clip
    tf, kf, tp, kp, tgt = build_mxm()
    all_results["MXMNet: Adam lr=1e-5 clip=0.1"] = run_experiment(
        "MXMNet", tf, kf, tp, kp, tgt,
        n_steps=50, optimizer_type="adam", lr=1e-5, grad_clip=0.1)

    # Exp 3: SGD + very small lr + tight clip
    tf, kf, tp, kp, tgt = build_mxm()
    all_results["MXMNet: SGD lr=1e-9 clip=0.01"] = run_experiment(
        "MXMNet", tf, kf, tp, kp, tgt,
        n_steps=50, optimizer_type="sgd", lr=1e-9, grad_clip=0.01)

    # Plot
    save_path = os.path.join(os.path.dirname(__file__), "basis_model_investigation.png")
    plot_results(all_results, save_path)

    # Summary
    print(f"\n{'='*90}")
    print(f"{'Experiment':<40} {'Final Loss Diff':>15} {'Final Out MAE':>15} {'Stable?':>10}")
    print(f"{'='*90}")
    for label, res in all_results.items():
        ld = res["loss_diffs"][-1] if res["loss_diffs"] else float('nan')
        od = res["output_diffs"][-1] if res["output_diffs"] else float('nan')
        tl = res["torch_losses"][-1] if res["torch_losses"] else float('nan')
        stable = "YES" if (not math.isnan(tl) and not math.isinf(tl) and tl < 1e6) else "NO"
        print(f"{label:<40} {ld:>15.3e} {od:>15.3e} {stable:>10}")


if __name__ == "__main__":
    main()
