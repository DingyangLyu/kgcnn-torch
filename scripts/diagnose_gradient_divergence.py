#!/usr/bin/env python3
"""Diagnose DimeNetPP gradient divergence: which parameters have different gradients?

This script does ONE forward+backward pass from identical weights and compares
per-parameter gradient norms between Torch and Keras implementations.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.functional as F

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from model_alignment_utils import make_disjoint_graph
from train_alignment_utils import collect_keras_params
from align_dimenetpp_model import (
    Config, KerasDimeNetPPFullStack, DimeNetPPModel,
    transfer_all_weights, generate_angle_index)


def named_keras_params(keras_stack):
    """Collect keras params with human-readable names."""
    import keras
    named = []
    visited_layers = set()
    visited_params = set()

    def collect_from_layer(prefix, layer):
        lid = id(layer)
        if lid in visited_layers:
            return
        visited_layers.add(lid)
        for var in layer.trainable_weights:
            t = var.value
            if id(t) not in visited_params:
                visited_params.add(id(t))
                # Use keras variable path as name
                named.append((f"{prefix}/{var.path}", t))

    def visit(prefix, obj):
        if isinstance(obj, keras.Layer):
            collect_from_layer(prefix, obj)
            return
        if not hasattr(obj, '__dict__'):
            return
        for attr_name, attr_val in obj.__dict__.items():
            if isinstance(attr_val, keras.Layer):
                collect_from_layer(f"{prefix}.{attr_name}", attr_val)
            elif isinstance(attr_val, (list, tuple)):
                for i, item in enumerate(attr_val):
                    if isinstance(item, keras.Layer):
                        collect_from_layer(f"{prefix}.{attr_name}[{i}]", item)

    visit("stack", keras_stack)
    return named


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_index = generate_angle_index(torch_data.edge_index)
    torch_data.angle_index = angle_index

    # Build models
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

    # Same target
    target = torch.randn(cfg.batch_size, cfg.num_targets)
    target.requires_grad_(False)

    # === Forward + backward on BOTH ===
    torch_out = torch_model(torch_data)
    keras_out = keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k)

    torch_loss = F.mse_loss(torch_out, target)
    keras_loss = F.mse_loss(keras_out, target)

    print(f"Forward outputs identical: "
          f"MAE={float((torch_out - keras_out).detach().abs().mean()):.2e}")
    print(f"Losses: torch={torch_loss.item():.6e}, keras={keras_loss.item():.6e}")

    torch_loss.backward()
    keras_loss.backward()

    # === Compare per-parameter gradient norms ===
    print("\n" + "=" * 100)
    print(f"{'Torch param':<50} {'Keras param':<50} {'T gnorm':>12} {'K gnorm':>12} {'ratio':>8}")
    print("=" * 100)

    torch_params_named = list(torch_model.named_parameters())
    keras_params_named = named_keras_params(keras_stack)

    print(f"\nTotal: torch={len(torch_params_named)}, keras={len(keras_params_named)}")

    # Compute total grad norms
    t_total_sq = sum(p.grad.norm().item()**2 for _, p in torch_params_named if p.grad is not None)
    k_total_sq = sum(p.grad.norm().item()**2 for _, p in keras_params_named if p.grad is not None)
    print(f"\nTotal grad norm: torch={t_total_sq**0.5:.4e}, keras={k_total_sq**0.5:.4e}, "
          f"ratio={k_total_sq**0.5 / t_total_sq**0.5:.4f}")

    # Print per-param comparison
    print(f"\n--- Torch parameters (sorted by grad norm, descending) ---")
    torch_gnorms = [(name, p, p.grad.norm().item() if p.grad is not None else 0.0)
                    for name, p in torch_params_named]
    torch_gnorms.sort(key=lambda x: x[2], reverse=True)
    for name, p, gn in torch_gnorms[:30]:
        print(f"  {name:<55} shape={str(list(p.shape)):<20} gnorm={gn:.4e}")

    print(f"\n--- Keras parameters (sorted by grad norm, descending) ---")
    keras_gnorms = [(name, p, p.grad.norm().item() if p.grad is not None else 0.0)
                    for name, p in keras_params_named]
    keras_gnorms.sort(key=lambda x: x[2], reverse=True)
    for name, p, gn in keras_gnorms[:30]:
        print(f"  {name:<55} shape={str(list(p.shape)):<20} gnorm={gn:.4e}")

    # Find parameters where grad values differ
    print(f"\n--- Parameters with matching shape (checking if grad values match) ---")
    torch_params_flat = list(torch_model.parameters())
    keras_params_flat = collect_keras_params(keras_stack)

    # Match by shape and check value similarity
    # The params should be in the same order if transfer_all_weights maps them 1:1
    # But shapes differ (transposed), so let's match by grad norm rank
    t_by_norm = sorted([(i, n, p) for i, (n, p) in enumerate(torch_params_named)],
                       key=lambda x: -x[2].grad.norm().item() if x[2].grad is not None else 0)
    k_by_norm = sorted([(i, n, p) for i, (n, p) in enumerate(keras_params_named)],
                       key=lambda x: -x[2].grad.norm().item() if x[2].grad is not None else 0)

    print(f"\n--- Top gradient contributors (paired by rank) ---")
    print(f"{'Rank':>4} {'Torch name':<45} {'T gnorm':>12} {'Keras name':<45} {'K gnorm':>12} {'ratio':>8}")
    for rank in range(min(20, len(t_by_norm), len(k_by_norm))):
        ti, tn, tp = t_by_norm[rank]
        ki, kn, kp = k_by_norm[rank]
        tg = tp.grad.norm().item() if tp.grad is not None else 0
        kg = kp.grad.norm().item() if kp.grad is not None else 0
        ratio = kg / tg if tg > 0 else float('inf')
        print(f"{rank:4d} {tn:<45} {tg:12.4e} {kn:<45} {kg:12.4e} {ratio:8.4f}")


if __name__ == "__main__":
    main()
