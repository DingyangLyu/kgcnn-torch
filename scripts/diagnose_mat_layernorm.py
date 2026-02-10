#!/usr/bin/env python3
"""Test whether LayerNorm implementation difference causes MAT divergence.

Finding: torch.nn.LayerNorm and keras.layers.LayerNormalization produce nearly
identical forward values (~5e-8), but their INPUT GRADIENTS differ by ~2e-2.
This is because torch uses a fused C++ kernel for backward while Keras decomposes
into separate ops. The gradient difference accumulates over training steps.
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
    Config, KerasMATFullStack, MATModel, transfer_all_weights, make_padded_data,
    copy_dense, copy_embedding, copy_attention_head, copy_seq_to_keras_mlp)
from train_alignment_utils import collect_keras_params
from model_alignment_utils import copy_layernorm

import keras


# === Test 1: Compare LayerNorm forward+backward directly ===
def test_layernorm_gradient():
    """Compare gradients through torch.nn.LayerNorm vs keras.layers.LayerNormalization."""
    print("=" * 80)
    print("Test 1: LayerNorm gradient comparison")
    print("=" * 80)

    dim = 32
    torch_ln = torch.nn.LayerNorm(dim, eps=1e-3)
    keras_ln = keras.layers.LayerNormalization(epsilon=1e-3)

    # Build keras LN
    dummy = torch.randn(4, 8, dim)
    _ = keras_ln(dummy)

    # Copy weights
    copy_layernorm(torch_ln, keras_ln)

    # Same input
    x = torch.randn(4, 8, dim, requires_grad=False)
    x_t = x.clone().requires_grad_(True)
    x_k = x.clone().requires_grad_(True)

    # Forward
    y_t = torch_ln(x_t)
    y_k = keras_ln(x_k)

    fwd_diff = (y_t - y_k).abs()
    print(f"Forward diff: MAE={fwd_diff.mean():.3e}, MAX={fwd_diff.max():.3e}")

    # Backward with same upstream gradient
    upstream = torch.randn_like(y_t)
    y_t.backward(upstream)
    y_k.backward(upstream)

    grad_diff_x = (x_t.grad - x_k.grad).abs()
    print(f"Input grad diff: MAE={grad_diff_x.mean():.3e}, MAX={grad_diff_x.max():.3e}")

    # Parameter gradients
    gamma_diff = (torch_ln.weight.grad - keras_ln.gamma.value.grad).abs()
    beta_diff = (torch_ln.bias.grad - keras_ln.beta.value.grad).abs()
    print(f"Gamma grad diff: MAE={gamma_diff.mean():.3e}, MAX={gamma_diff.max():.3e}")
    print(f"Beta grad diff: MAE={beta_diff.mean():.3e}, MAX={beta_diff.max():.3e}")

    print(f"\nGrad norms: x_t={x_t.grad.norm():.6e}, x_k={x_k.grad.norm():.6e}")
    print(f"Gamma norms: t={torch_ln.weight.grad.norm():.6e}, k={keras_ln.gamma.value.grad.norm():.6e}")


# === Test 2: MAT training with torch.nn.LayerNorm replacing Keras LayerNorm ===
def test_training_with_torch_layernorm():
    """Replace Keras LayerNorm with separate torch.nn.LayerNorm instances."""
    print("\n" + "=" * 80)
    print("Test 2: MAT training with torch.nn.LayerNorm in Keras stack")
    print("=" * 80)

    cfg = Config()
    node_input, xyz_input, adjacency, node_mask, adj_mask = make_padded_data(cfg)

    # Build both models normally
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

    # Transfer all weights first (while keras LayerNorms still exist)
    transfer_all_weights(torch_model, keras_stack, cfg)

    # Create NEW torch.nn.LayerNorm instances for the Keras stack, copy weights from torch model
    for i in range(cfg.depth):
        new_attn_norm = torch.nn.LayerNorm(cfg.embedding_units, eps=1e-3)
        new_attn_norm.weight.data.copy_(torch_model.attention_norms[i].weight.data)
        new_attn_norm.bias.data.copy_(torch_model.attention_norms[i].bias.data)
        keras_stack.attention_norms[i] = new_attn_norm

        new_ff_norm = torch.nn.LayerNorm(cfg.embedding_units, eps=1e-3)
        new_ff_norm.weight.data.copy_(torch_model.ff_norms[i].weight.data)
        new_ff_norm.bias.data.copy_(torch_model.ff_norms[i].bias.data)
        keras_stack.ff_norms[i] = new_ff_norm

    new_final_norm = torch.nn.LayerNorm(cfg.embedding_units, eps=1e-3)
    new_final_norm.weight.data.copy_(torch_model.final_norm.weight.data)
    new_final_norm.bias.data.copy_(torch_model.final_norm.bias.data)
    keras_stack.final_norm = new_final_norm

    # Collect keras params manually (torch.nn modules + keras layers)
    keras_params = []
    visited = set()

    def add_keras_layer(layer):
        for var in layer.trainable_weights:
            t = var.value
            if id(t) not in visited:
                visited.add(id(t))
                keras_params.append(t)

    def add_torch_module(module):
        for p in module.parameters():
            if id(p) not in visited:
                visited.add(id(p))
                keras_params.append(p)

    add_keras_layer(keras_stack.node_embedding)
    add_keras_layer(keras_stack.input_projection)
    add_keras_layer(keras_stack.adj_projection)
    for i in range(cfg.depth):
        add_torch_module(keras_stack.attention_norms[i])
        for head in keras_stack.attention_heads[i]:
            add_keras_layer(head)
        add_keras_layer(keras_stack.attention_projections[i])
        add_torch_module(keras_stack.ff_norms[i])
        add_keras_layer(keras_stack.ff_mlps[i])
        add_keras_layer(keras_stack.ff_projections[i])
    add_torch_module(keras_stack.final_norm)
    add_keras_layer(keras_stack.output_mlp)

    torch_params = list(torch_model.parameters())
    print(f"params: torch={len(torch_params)}, keras={len(keras_params)}")

    target = torch.randn(cfg.batch_size, cfg.num_targets)
    target.requires_grad_(False)

    torch_opt = torch.optim.SGD(torch_params, lr=1e-3)
    keras_opt = torch.optim.SGD(keras_params, lr=1e-3)

    for step in range(20):
        torch_out = torch_model(
            node_input, xyz_input, adjacency,
            node_mask.float(), adj_mask.float())
        keras_out = keras_stack.forward(
            node_input, xyz_input, adjacency, node_mask, adj_mask)

        out_diff = (torch_out - keras_out).abs()
        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)
        loss_diff = abs(torch_loss.item() - keras_loss.item())

        if step % 5 == 0 or step < 3:
            print(f"  Step {step:3d}: out_MAE={out_diff.mean().item():.3e} "
                  f"loss_diff={loss_diff:.3e}")

        torch_opt.zero_grad()
        torch_loss.backward()
        torch.nn.utils.clip_grad_norm_(torch_params, 1.0)
        torch_opt.step()

        keras_opt.zero_grad()
        keras_loss.backward()
        torch.nn.utils.clip_grad_norm_(keras_params, 1.0)
        keras_opt.step()


# === Test 3: Original (Keras LayerNorm) for comparison ===
def test_training_with_keras_layernorm():
    """Original MAT training (with Keras LayerNorm) for comparison."""
    print("\n" + "=" * 80)
    print("Test 3: MAT training with keras.layers.LayerNormalization (original)")
    print("=" * 80)

    cfg = Config()
    node_input, xyz_input, adjacency, node_mask, adj_mask = make_padded_data(cfg)

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

    for step in range(20):
        torch_out = torch_model(
            node_input, xyz_input, adjacency,
            node_mask.float(), adj_mask.float())
        keras_out = keras_stack.forward(
            node_input, xyz_input, adjacency, node_mask, adj_mask)

        out_diff = (torch_out - keras_out).abs()
        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)
        loss_diff = abs(torch_loss.item() - keras_loss.item())

        if step % 5 == 0 or step < 3:
            print(f"  Step {step:3d}: out_MAE={out_diff.mean().item():.3e} "
                  f"loss_diff={loss_diff:.3e}")

        torch_opt.zero_grad()
        torch_loss.backward()
        torch.nn.utils.clip_grad_norm_(torch_params, 1.0)
        torch_opt.step()

        keras_opt.zero_grad()
        keras_loss.backward()
        torch.nn.utils.clip_grad_norm_(keras_params, 1.0)
        keras_opt.step()


def main():
    test_layernorm_gradient()
    test_training_with_keras_layernorm()
    test_training_with_torch_layernorm()


if __name__ == "__main__":
    main()
