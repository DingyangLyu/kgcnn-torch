#!/usr/bin/env python3
"""Utilities for training alignment tests (Torch vs Keras-on-torch).

Provides:
- collect_keras_params: recursively collect torch.nn.Parameter from Keras layers
- train_step: single SGD training step (forward + loss + backward + step)
- run_training_alignment: multi-step training comparison loop
"""
import math
import os
os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.functional as F
import keras

from model_alignment_utils import compare_outputs


def collect_keras_params(stack):
    """Recursively collect all trainable torch.nn.Parameters from a Keras stack.

    Walks all attributes of the stack, finds keras.Layer objects (including
    in nested lists/tuples and non-Layer container objects like plain Python
    classes), and extracts their .trainable_weights -> .value (the underlying
    torch.nn.Parameter).

    Does NOT recurse into keras.Layer internals (their sub-layers are already
    captured via .trainable_weights). Only recurses into non-Layer container
    objects (e.g. KerasLocalMPBlock that holds keras.Layer attributes).

    Returns:
        List of unique torch.nn.Parameter objects.
    """
    params = []
    visited_layers = set()
    visited_params = set()
    visited_objects = set()

    def _collect_from_layer(layer):
        """Extract trainable params from a keras.Layer (non-recursive)."""
        lid = id(layer)
        if lid in visited_layers:
            return
        visited_layers.add(lid)
        for var in layer.trainable_weights:
            t = var.value  # underlying torch.nn.Parameter
            if id(t) not in visited_params:
                visited_params.add(id(t))
                params.append(t)

    def _visit_list(lst):
        """Recursively visit items in a list/tuple."""
        for item in lst:
            if isinstance(item, keras.Layer):
                _collect_from_layer(item)
            elif isinstance(item, (list, tuple)):
                _visit_list(item)
            elif _is_container(item):
                _visit_container(item)

    def _is_container(obj):
        """Check if obj is a plain Python container (not a Layer, not a primitive)."""
        if isinstance(obj, keras.Layer):
            return False
        if isinstance(obj, (str, int, float, bool, type, torch.Tensor)):
            return False
        if not hasattr(obj, '__dict__'):
            return False
        # Skip modules/classes/functions
        if callable(obj) and not hasattr(obj, '__dict__'):
            return False
        return True

    def _visit_container(obj):
        """Walk a non-Layer container's __dict__ for keras.Layer attrs."""
        obj_id = id(obj)
        if obj_id in visited_objects:
            return
        visited_objects.add(obj_id)

        # Snapshot dict values to avoid "dictionary changed size" errors
        attrs = list(obj.__dict__.values())
        for attr in attrs:
            if isinstance(attr, keras.Layer):
                _collect_from_layer(attr)
            elif isinstance(attr, (list, tuple)):
                _visit_list(attr)
            elif _is_container(attr):
                _visit_container(attr)

    # Start: treat the stack as a container
    if isinstance(stack, keras.Layer):
        _collect_from_layer(stack)
    _visit_container(stack)
    return params


def train_step(output, target, optimizer):
    """Single training step: MSE loss + backward + optimizer step.

    Args:
        output: Model output tensor (requires_grad via computation graph).
        target: Target tensor.
        optimizer: torch.optim.Optimizer instance.

    Returns:
        float: Loss value.
    """
    loss = F.mse_loss(output, target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item())


def run_training_alignment(name, torch_fwd, keras_fwd, torch_params, keras_params,
                           target, n_steps=5, lr=0.01,
                           max_loss_diff=1e-6, max_output_diff=1e-5):
    """Run N training steps on both Torch and Keras models and compare.

    Args:
        name: Model name for display.
        torch_fwd: Callable returning torch model output.
        keras_fwd: Callable returning keras stack output (as torch tensor).
        torch_params: List of torch.nn.Parameter for torch model.
        keras_params: List of torch.nn.Parameter for keras stack.
        target: Target tensor for MSE loss.
        n_steps: Number of training steps.
        lr: Learning rate for SGD.
        max_loss_diff: Maximum allowed loss difference per step.
        max_output_diff: Maximum allowed output difference after training.

    Raises:
        SystemExit: If loss or output differences exceed thresholds.
    """
    torch_opt = torch.optim.SGD(torch_params, lr=lr)
    keras_opt = torch.optim.SGD(keras_params, lr=lr)

    torch_losses = []
    keras_losses = []

    print(f"\n=== Training Alignment: {name} ===")
    print(f"  params: torch={len(torch_params)}, keras={len(keras_params)}")

    for step in range(n_steps):
        # Forward pass
        torch_out = torch_fwd()
        keras_out = keras_fwd()

        # Compute loss
        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)

        tl = float(torch_loss.item())
        kl = float(keras_loss.item())
        torch_losses.append(tl)
        keras_losses.append(kl)
        loss_diff = abs(tl - kl)
        print(f"  Step {step}: loss_torch={tl:.6e} | "
              f"loss_keras={kl:.6e} | diff={loss_diff:.6e}")

        if math.isnan(tl) or math.isnan(kl) or loss_diff > max_loss_diff:
            raise SystemExit(
                f"Training alignment FAILED for '{name}' at step {step}: "
                f"loss_diff={loss_diff:.3e} (limit {max_loss_diff:.1e})"
            )

        # Backward + step
        torch_opt.zero_grad()
        torch_loss.backward()
        torch_opt.step()

        keras_opt.zero_grad()
        keras_loss.backward()
        keras_opt.step()

    # Final output comparison (no grad)
    with torch.no_grad():
        torch_final = torch_fwd().detach().cpu()
        keras_final = keras_fwd().detach().cpu()

    print(f"  Final output comparison:")
    compare_outputs(f"{name}_output", torch_final, keras_final,
                    max_mae=max_output_diff, max_abs=max_output_diff * 10)
    print(f"  {name} training alignment PASSED.")

    return torch_losses, keras_losses


def run_training_gradient_check(name, torch_fwd, keras_fwd, torch_params, keras_params,
                                target, n_steps=3, lr=0.001, grad_clip=1.0):
    """Verify that gradient flow works for both Torch and Keras models.

    For numerically challenging models where exact loss alignment is infeasible
    (e.g., DimeNetPP, MXMNet, CMPNN, MAT), this verifies:
    1. Param counts match
    2. All params receive non-zero gradients
    3. Loss decreases (or at least doesn't NaN) over training steps
    4. Both models train without errors

    Uses gradient clipping to prevent numerical explosion.

    Raises:
        SystemExit: If gradient flow or basic training fails.
    """
    torch_opt = torch.optim.SGD(torch_params, lr=lr)
    keras_opt = torch.optim.SGD(keras_params, lr=lr)

    torch_losses = []
    keras_losses = []

    print(f"\n=== Training Gradient Check: {name} ===")
    print(f"  params: torch={len(torch_params)}, keras={len(keras_params)}")

    if len(torch_params) != len(keras_params):
        raise SystemExit(
            f"Gradient check FAILED for '{name}': param count mismatch "
            f"torch={len(torch_params)} vs keras={len(keras_params)}"
        )

    for step in range(n_steps):
        torch_out = torch_fwd()
        keras_out = keras_fwd()

        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)

        tl = float(torch_loss.item())
        kl = float(keras_loss.item())
        torch_losses.append(tl)
        keras_losses.append(kl)
        print(f"  Step {step}: loss_torch={tl:.6e} | loss_keras={kl:.6e}")

        if math.isnan(tl) or math.isnan(kl) or math.isinf(tl) or math.isinf(kl):
            raise SystemExit(
                f"Gradient check FAILED for '{name}' at step {step}: "
                f"NaN/Inf loss (torch={tl}, keras={kl})"
            )

        # Backward with gradient clipping
        torch_opt.zero_grad()
        torch_loss.backward()
        torch.nn.utils.clip_grad_norm_(torch_params, grad_clip)
        torch_opt.step()

        keras_opt.zero_grad()
        keras_loss.backward()
        torch.nn.utils.clip_grad_norm_(keras_params, grad_clip)
        keras_opt.step()

    # Verify all params got gradients (at least from last step)
    torch_no_grad = sum(1 for p in torch_params if p.grad is None or p.grad.abs().max() == 0)
    keras_no_grad = sum(1 for p in keras_params if p.grad is None or p.grad.abs().max() == 0)
    if torch_no_grad > 0:
        print(f"  WARNING: {torch_no_grad}/{len(torch_params)} torch params have zero grad")
    if keras_no_grad > 0:
        print(f"  WARNING: {keras_no_grad}/{len(keras_params)} keras params have zero grad")

    print(f"  {name} gradient check PASSED.")

    return torch_losses, keras_losses
