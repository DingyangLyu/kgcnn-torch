#!/usr/bin/env python3
"""Diagnose MAT training divergence: compare intermediate values and gradients.

The 50-step divergence test shows MAT accumulates ~3.4e-2 loss difference.
This script traces where the difference originates by comparing:
1. Forward outputs at each step
2. Per-parameter gradient norms
3. Weight updates
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

from align_mat_model import (
    Config, KerasMATFullStack, MATModel, transfer_all_weights, make_padded_data)
from train_alignment_utils import collect_keras_params


def compare(name, t, k):
    if t.shape != k.shape:
        print(f"  {name}: SHAPE MISMATCH {list(t.shape)} vs {list(k.shape)}")
        return
    diff = (t - k).abs()
    mae = diff.mean().item()
    maxabs = diff.max().item()
    status = "OK" if mae < 1e-6 else ("MINOR" if mae < 1e-4 else "DIFFER")
    print(f"  {name:<40} MAE={mae:.3e} MAX={maxabs:.3e} [{status}]")


def compare_intermediates(torch_model, keras_stack, cfg,
                          node_input, xyz_input, adjacency, node_mask, adj_mask):
    """Compare intermediate values in one forward pass."""
    import keras
    from keras import ops

    with torch.no_grad():
        # === TORCH forward ===
        n_t = torch_model.node_embedding(node_input.long())
        n_mask_t = node_mask.float().unsqueeze(-1)
        a_mask_t = adj_mask.float().unsqueeze(-1)
        dist_t, dist_mask_t = torch_model.distance_layer(xyz_input, n_mask_t)
        adj_t = torch_model.adj_projection(adjacency.float())
        h_t = torch_model.input_projection(n_t)

        # === KERAS forward ===
        n_k = keras_stack.node_embedding(node_input)
        n_mask_k = ops.expand_dims(ops.cast(node_mask, dtype="float32"), axis=-1)
        a_mask_k = ops.expand_dims(ops.cast(adj_mask, dtype="float32"), axis=-1)
        dist_k, dist_mask_k = keras_stack.distance_layer(xyz_input, mask=n_mask_k)
        adj_k = keras_stack.adj_projection(adjacency.float())
        h_k = keras_stack.input_projection(n_k)

        compare("node_embedding", n_t, n_k)
        compare("n_mask", n_mask_t, n_mask_k)
        compare("distance_matrix", dist_t, dist_k)
        compare("dist_mask", dist_mask_t, dist_mask_k)
        compare("adj_projection", adj_t, adj_k)
        compare("input_projection (h0)", h_t, h_k)

        h_mask_t = n_mask_t
        h_mask_k = n_mask_k

        for i in range(cfg.depth):
            print(f"\n  --- Transformer block {i} ---")

            # LayerNorm
            hn_t = torch_model.attention_norms[i](h_t)
            hn_k = keras_stack.attention_norms[i](h_k)
            compare(f"  attn_norm[{i}]", hn_t, hn_k)

            # Attention heads
            head_outs_t = []
            head_outs_k = []
            for j, (th, kh) in enumerate(zip(
                    torch_model.attention_heads[i], keras_stack.attention_heads[i])):
                ho_t = th(hn_t, dist_t, adj_t, h_mask_t, dist_mask_t, a_mask_t)
                ho_k = kh([hn_k, dist_k, adj_k],
                          mask=[n_mask_k, dist_mask_k, a_mask_k])
                compare(f"  head[{i}][{j}]", ho_t, ho_k)
                head_outs_t.append(ho_t)
                head_outs_k.append(ho_k)

            # Merge heads
            if cfg.merge_heads in ("add", "sum", "reduce_sum"):
                hu_t = sum(head_outs_t)
                hu_k = keras.layers.Add()(head_outs_k)
            else:
                hu_t = torch.cat(head_outs_t, dim=-1)
                hu_k = keras.layers.Concatenate(axis=-1)(head_outs_k)
            compare(f"  merged_heads[{i}]", hu_t, hu_k)

            # Attention projection
            hu_t = torch_model.attention_projections[i](hu_t)
            hu_k = keras_stack.attention_projections[i](hu_k)
            compare(f"  attn_proj[{i}]", hu_t, hu_k)

            # Residual
            h_t = h_t + hu_t
            h_k = h_k + hu_k
            compare(f"  h_after_attn[{i}]", h_t, h_k)

            # FF norm
            hn_t = torch_model.ff_norms[i](h_t)
            hn_k = keras_stack.ff_norms[i](h_k)
            compare(f"  ff_norm[{i}]", hn_t, hn_k)

            # FF MLP
            hu_t = torch_model.ff_mlps[i](hn_t)
            hu_k = keras_stack.ff_mlps[i](hn_k)
            compare(f"  ff_mlp[{i}]", hu_t, hu_k)

            # FF projection
            hu_t = torch_model.ff_projections[i](hu_t)
            hu_k = keras_stack.ff_projections[i](hu_k)
            compare(f"  ff_proj[{i}]", hu_t, hu_k)

            # Mask + residual
            hu_t = hu_t * h_mask_t
            hu_k = hu_k * h_mask_k
            h_t = h_t + hu_t
            h_k = h_k + hu_k
            compare(f"  h_after_ff[{i}]", h_t, h_k)

        # Final norm
        out_t = torch_model.final_norm(h_t)
        out_k = keras_stack.final_norm(h_k)
        compare("final_norm", out_t, out_k)

        # Mask + pool
        out_t = out_t * h_mask_t
        out_k = out_k * h_mask_k
        out_t = torch_model.pool(out_t)
        out_k = keras_stack.pool(out_k, mask=h_mask_k)
        compare("pooled", out_t, out_k)

        # Output MLP
        out_t = torch_model.output_mlp(out_t)
        out_k = keras_stack.output_mlp(out_k)
        compare("output", out_t, out_k)


def main():
    cfg = Config()
    node_input, xyz_input, adjacency, node_mask, adj_mask = make_padded_data(cfg)

    # Build models
    torch_model = MATModel(
        embedding_units=cfg.embedding_units, depth=cfg.depth,
        num_heads=cfg.num_heads, attention_units=cfg.attention_units,
        merge_heads=cfg.merge_heads, lambda_attention=cfg.lambda_attention,
        lambda_distance=cfg.lambda_distance, add_identity=cfg.add_identity,
        attention_dropout=None, distance_trafo=cfg.distance_trafo,
        units_ff=cfg.units_ff, ff_activations=cfg.ff_activations,
        output_units=cfg.output_units, output_activations=cfg.output_activations,
        num_targets=cfg.num_targets, use_node_embedding=True,
        num_embeddings=cfg.num_embeddings, input_node_dim=cfg.input_node_dim,
        output_embedding="graph")
    torch_model.train()

    keras_stack = KerasMATFullStack(cfg)
    _ = keras_stack.forward(node_input, xyz_input, adjacency, node_mask, adj_mask)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets)
    target.requires_grad_(False)

    torch_opt = torch.optim.SGD(torch_params, lr=1e-3)
    keras_opt = torch.optim.SGD(keras_params, lr=1e-3)

    print(f"params: torch={len(torch_params)}, keras={len(keras_params)}")

    # === Step 0: verify initial weight alignment ===
    print("\n" + "=" * 80)
    print("Step 0: Before any training (weights should be identical)")
    print("=" * 80)
    compare_intermediates(torch_model, keras_stack, cfg,
                          node_input, xyz_input, adjacency, node_mask, adj_mask)

    # === Training steps with detailed comparison ===
    n_steps = 10
    for step in range(n_steps):
        print(f"\n{'=' * 80}")
        print(f"Training step {step}")
        print(f"{'=' * 80}")

        # Forward
        torch_out = torch_model(
            node_input, xyz_input, adjacency,
            node_mask.float(), adj_mask.float())
        keras_out = keras_stack.forward(
            node_input, xyz_input, adjacency, node_mask, adj_mask)

        compare("forward_output", torch_out, keras_out)

        # Loss
        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)
        loss_diff = abs(torch_loss.item() - keras_loss.item())
        print(f"  loss_torch={torch_loss.item():.8e}")
        print(f"  loss_keras={keras_loss.item():.8e}")
        print(f"  loss_diff ={loss_diff:.3e}")

        # Backward
        torch_opt.zero_grad()
        torch_loss.backward()
        torch.nn.utils.clip_grad_norm_(torch_params, 1.0)

        keras_opt.zero_grad()
        keras_loss.backward()
        torch.nn.utils.clip_grad_norm_(keras_params, 1.0)

        # Compare gradient norms
        t_gnorm = sum(p.grad.norm().item()**2 for p in torch_params if p.grad is not None)**0.5
        k_gnorm = sum(p.grad.norm().item()**2 for p in keras_params if p.grad is not None)**0.5
        print(f"  grad_norm_torch={t_gnorm:.6e}")
        print(f"  grad_norm_keras={k_gnorm:.6e}")
        print(f"  grad_norm_diff ={abs(t_gnorm - k_gnorm):.3e}")

        # Compare per-parameter gradient values (top differences)
        if step == 0:
            print("\n  --- Per-parameter gradient comparison (step 0) ---")
            t_named = list(torch_model.named_parameters())
            for name, tp in t_named:
                if tp.grad is not None:
                    gn = tp.grad.norm().item()
                    if gn > 1e-8:
                        print(f"    {name:<50} grad_norm={gn:.4e}")

        # Check if any grad values differ
        max_grad_diff = 0.0
        for tp, kp in zip(torch_params, keras_params):
            if tp.grad is not None and kp.grad is not None and tp.shape == kp.shape:
                gd = (tp.grad - kp.grad).abs().max().item()
                max_grad_diff = max(max_grad_diff, gd)
        print(f"  max_grad_value_diff={max_grad_diff:.3e}")

        # Step
        torch_opt.step()
        keras_opt.step()

        # Compare weights after step
        max_weight_diff = 0.0
        for tp, kp in zip(torch_params, keras_params):
            if tp.shape == kp.shape:
                wd = (tp.data - kp.data).abs().max().item()
                max_weight_diff = max(max_weight_diff, wd)
        print(f"  max_weight_diff_after_step={max_weight_diff:.3e}")

    # Final intermediate comparison
    print(f"\n{'=' * 80}")
    print(f"Final intermediate values (after {n_steps} steps)")
    print(f"{'=' * 80}")
    compare_intermediates(torch_model, keras_stack, cfg,
                          node_input, xyz_input, adjacency, node_mask, adj_mask)


if __name__ == "__main__":
    main()
