"""PyTorch training loop for graph neural networks."""
import logging
import os
import time
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader
from typing import Optional, Callable, List

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def _model_needs_grad(model: nn.Module) -> bool:
    """Check if a model requires gradient computation during eval.

    Force models (EnergyForceModel) need torch.autograd.grad even during
    evaluation, so we cannot wrap the forward pass in torch.no_grad().
    """
    from kgcnn_torch.models.force import EnergyForceModel
    if isinstance(model, EnergyForceModel):
        return True
    # Check if any submodule is an EnergyForceModel (e.g. wrapped models).
    for m in model.modules():
        if isinstance(m, EnergyForceModel):
            return True
    return False


def train_epoch(model: nn.Module, loader: DataLoader,
                optimizer: torch.optim.Optimizer,
                loss_fn: Callable,
                device: torch.device = None,
                scaler: Optional[object] = None,
                callbacks: list = None) -> float:
    """Run one training epoch.

    Args:
        model: The GNN model.
        loader: PyG DataLoader for training data.
        optimizer: PyTorch optimizer.
        loss_fn: Loss function(pred, target) -> scalar. For dict predictions
            (e.g. force models), receives (pred_dict, batch) instead.
        device: Device to use.
        scaler: Optional label scaler with inverse_transform.
        callbacks: Optional list of TrainingCallback instances.

    Returns:
        Average training loss for this epoch.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if callbacks:
            for cb in callbacks:
                cb.on_batch_begin(batch_idx)

        if device is not None:
            batch = batch.to(device)

        optimizer.zero_grad()
        pred = model(batch)

        # Handle dict predictions (e.g. from EnergyForceModel)
        if isinstance(pred, dict):
            loss = loss_fn(pred, batch)
        else:
            target = batch.y
            if target.dim() == 1:
                target = target.unsqueeze(-1)
            loss = loss_fn(pred, target)

        loss.backward()
        optimizer.step()

        batch_loss = loss.item()
        total_loss += batch_loss
        n_batches += 1

        if callbacks:
            for cb in callbacks:
                cb.on_batch_end(batch_idx, {"batch_loss": batch_loss})

    return total_loss / max(n_batches, 1)


def eval_epoch(model: nn.Module, loader: DataLoader,
               loss_fn: Callable,
               device: torch.device = None,
               metrics: dict = None,
               scaler=None) -> dict:
    """Run one evaluation epoch.

    Note: Does NOT use @torch.no_grad() decorator because force models
    (EnergyForceModel) require gradient computation via torch.autograd.grad
    even during evaluation. For non-force models the computation graph is
    not retained so overhead is minimal.

    Args:
        model: The GNN model.
        loader: PyG DataLoader for validation/test data.
        loss_fn: Loss function. For dict predictions (e.g. force models),
            receives (pred_dict, batch) instead of (pred, target).
        device: Device to use.
        metrics: Dict of {name: metric_fn(pred, target)} to compute.
        scaler: Optional scaler with inverse_transform(X) method. If provided,
            predictions and targets are inverse-transformed before computing
            metrics (but loss is computed in scaled space).

    Returns:
        Dict with 'loss' and any additional metric values.
    """
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
    all_targets = []

    # Check if this is a force model that needs gradients during eval.
    _needs_grad = _model_needs_grad(model)

    for batch in loader:
        if device is not None:
            batch = batch.to(device)

        with torch.set_grad_enabled(_needs_grad):
            pred = model(batch)

        # Handle dict predictions (e.g. from EnergyForceModel)
        if isinstance(pred, dict):
            loss = loss_fn(pred, batch)
        else:
            target = batch.y
            if target is not None and target.dim() == 1:
                target = target.unsqueeze(-1)
            loss = loss_fn(pred, target)

        total_loss += loss.item()
        n_batches += 1

        if isinstance(pred, dict):
            all_preds.append({k: v.detach().cpu() for k, v in pred.items()})
        else:
            all_preds.append(pred.detach().cpu())
            target = batch.y
            if target is not None:
                if target.dim() == 1:
                    target = target.unsqueeze(-1)
                all_targets.append(target.detach().cpu())

    results = {"loss": total_loss / max(n_batches, 1)}

    if metrics and all_preds:
        # Handle dict predictions (e.g. from EnergyForceModel)
        if isinstance(all_preds[0], dict):
            # Compute per-key metrics if metric functions accept dicts
            # Otherwise skip metrics for dict outputs
            return results
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        # Inverse transform to original space for metrics
        if scaler is not None:
            pred_np = all_preds.numpy()
            target_np = all_targets.numpy()
            try:
                pred_np = scaler.inverse_transform(pred_np)
                target_np = scaler.inverse_transform(target_np)
            except (TypeError, ValueError, AttributeError) as e:
                module_logger.warning(
                    "Scaler inverse_transform failed: %s. "
                    "Metrics will be computed in scaled space.", e
                )
            all_preds = torch.from_numpy(pred_np)
            all_targets = torch.from_numpy(target_np)
        for name, metric_fn in metrics.items():
            results[name] = metric_fn(all_preds, all_targets).item()

    return results


def fit(model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader = None,
        optimizer: torch.optim.Optimizer = None,
        loss_fn: Callable = None,
        scheduler=None,
        epochs: int = 100,
        device: torch.device = None,
        metrics: dict = None,
        verbose: int = 1,
        early_stopping_patience: int = 0,
        checkpoint_path: str = None,
        scaler=None,
        callbacks: list = None,
        compute_train_metrics: bool = False) -> dict:
    """Full training loop.

    Args:
        model: The GNN model.
        train_loader: Training DataLoader.
        val_loader: Validation DataLoader.
        optimizer: Optimizer. Defaults to Adam(lr=1e-3).
        loss_fn: Loss function. Defaults to MSELoss.
        scheduler: LR scheduler.
        epochs: Number of training epochs.
        device: Device to use.
        metrics: Validation metrics dict.
        verbose: Verbosity level (0=silent, 1=progress, 2=detailed).
        early_stopping_patience: Stop after this many epochs without val loss
            improvement. 0=disabled.
        checkpoint_path: Path to save best model checkpoint.
        scaler: Optional label scaler with inverse_transform. If provided,
            predictions and targets are inverse-transformed before computing
            metrics (but loss is computed in scaled space).
        callbacks: Optional list of TrainingCallback instances. Callbacks are
            invoked at training start/end, epoch start/end, and batch start/end.
        compute_train_metrics: If True and metrics is not None, also compute
            metrics on the training set each epoch (via an extra eval pass).

    Returns:
        Dict with training history.
    """
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    if loss_fn is None:
        loss_fn = nn.MSELoss()
    if device is not None:
        model = model.to(device)
    if callbacks is None:
        callbacks = []

    # Initialize callbacks
    for cb in callbacks:
        cb.set_model(model)
        cb.set_optimizer(optimizer)

    history = {"train_loss": [], "val_loss": [], "lr": []}
    if metrics:
        for name in metrics:
            history[f"val_{name}"] = []
        if compute_train_metrics:
            for name in metrics:
                history[f"train_{name}"] = []

    best_val_loss = float('inf')
    patience_counter = 0

    # Notify callbacks of training start
    for cb in callbacks:
        cb.on_train_begin({"epochs": epochs})

    for epoch in range(epochs):
        t0 = time.time()

        # Notify callbacks of epoch start
        for cb in callbacks:
            cb.on_epoch_begin(epoch)

        # Train
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn,
                                 device, callbacks=callbacks)
        history["train_loss"].append(train_loss)

        # Compute training metrics via an extra eval pass
        if compute_train_metrics and metrics:
            train_results = eval_epoch(model, train_loader, loss_fn, device,
                                       metrics, scaler=scaler)
            for name in metrics:
                history[f"train_{name}"].append(train_results.get(name, 0.0))

        # Get current lr
        current_lr = optimizer.param_groups[0]['lr']
        history["lr"].append(current_lr)

        # Validate
        val_results = None
        if val_loader is not None:
            val_results = eval_epoch(model, val_loader, loss_fn, device,
                                     metrics, scaler=scaler)
            history["val_loss"].append(val_results["loss"])
            if metrics:
                for name in metrics:
                    history[f"val_{name}"].append(val_results.get(name, 0.0))

            # Early stopping
            if early_stopping_patience > 0:
                if val_results["loss"] < best_val_loss:
                    best_val_loss = val_results["loss"]
                    patience_counter = 0
                    if checkpoint_path:
                        torch.save(model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        if verbose >= 1:
                            module_logger.info(f"Early stopping at epoch {epoch+1}")
                        # Notify callbacks of epoch end before breaking
                        epoch_logs = {
                            "train_loss": train_loss, "lr": current_lr
                        }
                        if val_results:
                            epoch_logs.update({f"val_{k}": v for k, v in val_results.items()})
                            epoch_logs["val_loss"] = val_results["loss"]
                        for cb in callbacks:
                            cb.on_epoch_end(epoch, epoch_logs)
                        break

        # Scheduler step
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                val_loss = val_results["loss"] if val_results else train_loss
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # Logging
        dt = time.time() - t0
        if verbose >= 1 and (epoch + 1) % max(1, epochs // 20) == 0:
            msg = f"Epoch {epoch+1}/{epochs} - {dt:.1f}s - loss: {train_loss:.4f}"
            if val_results:
                msg += f" - val_loss: {val_results['loss']:.4f}"
            msg += f" - lr: {current_lr:.2e}"
            module_logger.info(msg)

        # Notify callbacks of epoch end
        epoch_logs = {"train_loss": train_loss, "lr": current_lr}
        if val_results:
            epoch_logs.update({f"val_{k}": v for k, v in val_results.items()})
            epoch_logs["val_loss"] = val_results["loss"]
        for cb in callbacks:
            cb.on_epoch_end(epoch, epoch_logs)

        # Check if any callback requests stopping (e.g. EarlyStoppingCallback)
        for cb in callbacks:
            if getattr(cb, 'stop_training', False):
                if verbose >= 1:
                    module_logger.info(
                        f"Training stopped by callback at epoch {epoch+1}")
                break
        else:
            continue
        break  # Break outer loop if inner loop broke

    # Notify callbacks of training end
    for cb in callbacks:
        cb.on_train_end({"history": history})

    # Load best model if early stopping was used
    if checkpoint_path and os.path.exists(checkpoint_path) and early_stopping_patience > 0:
        model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

    return history
