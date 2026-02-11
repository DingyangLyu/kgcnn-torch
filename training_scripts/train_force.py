"""Training script for energy/force prediction (MLIPs).

Supports models that predict both energy and forces, such as SchNet, PAiNN, DimeNetPP.

Usage:
    python train_force.py --hyper hyper/hyper_md17.json --category SchNet --output results/
"""
import argparse
import copy
import json
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from datetime import timedelta
from torch_geometric.loader import DataLoader
from sklearn.model_selection import KFold

from kgcnn_torch.training.hyper import HyperParameter
from kgcnn_torch.training.scheduler import get_scheduler
from kgcnn_torch.losses.losses import EnergyForceLoss
try:
    from training_scripts.train_graph import translate_model_config, adapt_model_config_from_data
except ImportError:
    from train_graph import translate_model_config, adapt_model_config_from_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models that support force prediction (energy + forces output)
_FORCE_MODEL_REGISTRY = {
    "SchNet": "kgcnn_torch.models.schnet.SchNetModel",
    "PAiNN": "kgcnn_torch.models.painn.PAiNNModel",
    "DimeNetPP": "kgcnn_torch.models.dimenetpp.DimeNetPPModel",
    "EGNN": "kgcnn_torch.models.egnn.EGNNModel",
}


def get_model_class(name: str):
    """Import and return model class by name."""
    if name in _FORCE_MODEL_REGISTRY:
        module_path, class_name = _FORCE_MODEL_REGISTRY[name].rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise ValueError(f"Unknown force model '{name}'. Available: {list(_FORCE_MODEL_REGISTRY.keys())}")


def train_epoch_force(model, loader, optimizer, loss_fn, device,
                      energy_key='y', force_key='force',
                      use_autograd_forces=True):
    """Train one epoch for energy/force prediction.

    Args:
        model: GNN model that outputs energy.
        loader: DataLoader with energy and force targets.
        optimizer: Optimizer.
        loss_fn: EnergyForceLoss or similar.
        device: Device.
        energy_key: Key for energy target in batch.
        force_key: Key for force target in batch.
        use_autograd_forces: If True, compute forces as -dE/dpos via autograd.
    """
    model.train()
    total_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        if use_autograd_forces:
            # Enable gradient on positions for force computation
            batch.pos.requires_grad_(True)

        # Forward pass - model predicts energy
        energy_pred = model(batch)
        if energy_pred.dim() > 1:
            energy_pred = energy_pred.squeeze(-1)

        # Compute forces via autograd: F = -dE/dpos
        if use_autograd_forces:
            energy_sum = energy_pred.sum()
            grad_result = torch.autograd.grad(
                energy_sum, batch.pos,
                create_graph=True, retain_graph=True,
                allow_unused=True
            )[0]
            if grad_result is None:
                raise RuntimeError(
                    "autograd.grad returned None for positions. "
                    "The model's computation graph does not depend on batch.pos. "
                    "Ensure the model uses positions (e.g. for distance computation)."
                )
            force_pred = -grad_result
        else:
            # Model directly outputs forces (if supported)
            force_pred = getattr(batch, '_force_pred', None)
            if force_pred is None:
                raise ValueError("Model must output forces or use_autograd_forces=True")

        # Targets
        energy_target = getattr(batch, energy_key)
        if energy_target.dim() > 1:
            energy_target = energy_target.squeeze(-1)
        force_target = getattr(batch, force_key)

        loss = loss_fn(energy_pred, energy_target, force_pred, force_target)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def eval_epoch_force(model, loader, loss_fn, device,
                     energy_key='y', force_key='force'):
    """Evaluate one epoch for energy/force prediction.

    Note: For autograd forces, we need gradients enabled.
    """
    model.eval()
    total_loss = 0.0
    total_energy_mae = 0.0
    total_force_mae = 0.0
    n_batches = 0
    n_atoms = 0
    n_energy_samples = 0

    for batch in loader:
        batch = batch.to(device)

        with torch.enable_grad():
            batch.pos.requires_grad_(True)
            energy_pred = model(batch)
            if energy_pred.dim() > 1:
                energy_pred = energy_pred.squeeze(-1)

            energy_sum = energy_pred.sum()
            grad_result = torch.autograd.grad(
                energy_sum, batch.pos, create_graph=False,
                allow_unused=True
            )[0]
            if grad_result is None:
                force_pred = torch.zeros_like(batch.pos)
            else:
                force_pred = -grad_result

        energy_target = getattr(batch, energy_key)
        if energy_target.dim() > 1:
            energy_target = energy_target.squeeze(-1)
        force_target = getattr(batch, force_key)

        loss = loss_fn(energy_pred, energy_target, force_pred, force_target)
        total_loss += loss.item()

        total_energy_mae += (energy_pred - energy_target).abs().sum().item()
        total_force_mae += (force_pred - force_target).abs().sum().item()
        n_energy_samples += energy_target.size(0)
        n_atoms += force_target.size(0)
        n_batches += 1

    return {
        "loss": total_loss / max(n_batches, 1),
        "energy_mae": total_energy_mae / max(n_energy_samples, 1),
        "force_mae": total_force_mae / max(n_atoms * 3, 1),
    }


def _transform_force_data(data_list, scaler, energy_key, force_key):
    """Apply ForceStandardScaler transform to a list of PyG Data objects.

    Modifies data in-place: scales energy and force targets.
    """
    energies = []
    forces = []
    atoms = []
    for d in data_list:
        e = getattr(d, energy_key, None)
        if e is None:
            e = getattr(d, 'energy', None)
        energies.append(e.numpy().reshape(-1) if e is not None else np.zeros(1))
        f = getattr(d, force_key, None)
        forces.append(f.numpy() if f is not None else np.zeros((1, 3)))
        z = d.z if hasattr(d, 'z') else None
        atoms.append(z.numpy() if z is not None else np.zeros(1))

    energy_arr = np.array(energies)
    if energy_arr.ndim == 1:
        energy_arr = energy_arr[:, None]

    scaled_e, scaled_f = scaler.transform(y=(energy_arr, forces), X=atoms)

    for i, d in enumerate(data_list):
        e_val = scaled_e[i] if scaled_e.ndim > 1 else scaled_e[i:i + 1]
        if hasattr(d, energy_key) and getattr(d, energy_key) is not None:
            setattr(d, energy_key, torch.tensor(e_val, dtype=torch.float))
        elif hasattr(d, 'energy') and d.energy is not None:
            d.energy = torch.tensor(e_val, dtype=torch.float)
        if hasattr(d, force_key) and getattr(d, force_key) is not None:
            setattr(d, force_key, torch.tensor(scaled_f[i], dtype=torch.float))


def main():
    parser = argparse.ArgumentParser(description="Train energy/force GNN model")
    parser.add_argument("--hyper", type=str, required=True, help="Path to hyperparameter file")
    parser.add_argument("--category", type=str, default=None, help="Model category in hyper file")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=int, nargs="+", default=None, help="Specific folds to run")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load hyperparameters
    hyper = HyperParameter(args.hyper, model_name=args.category)
    model_config = hyper.model_config
    fit_config = hyper.fit_config
    compile_config = hyper.compile_config
    scheduler_config = hyper.scheduler_config
    scaler_config = hyper.scaler_config
    cv_config = hyper.cross_validation_config

    model_name = args.model or model_config.pop("model_name", args.category or "SchNet")
    model_config = translate_model_config(model_name, model_config)
    logger.info(f"Model: {model_name}")

    # Training parameters
    epochs = fit_config.get("epochs", 500)
    batch_size = fit_config.get("batch_size", 32)
    early_stopping = fit_config.get("early_stopping_patience", 50)
    energy_weight = compile_config.get("energy_weight", 1.0)
    force_weight = compile_config.get("force_weight", 100.0)
    energy_key = fit_config.get("energy_key", "y")
    force_key = fit_config.get("force_key", "force")

    # Load dataset
    data_config = hyper.data_config
    dataset_config = data_config.get("dataset", {})
    pickle_path = dataset_config.get("config", {}).get("file_path", None)

    if pickle_path and os.path.exists(pickle_path):
        from kgcnn_torch.data.base import MemoryGraphList
        gl = MemoryGraphList()
        gl.load(pickle_path)
        pyg_data_list = gl.to_pyg_list()
    else:
        # Try class-based dataset loading (same logic as train_graph.py)
        class_name = dataset_config.get("class_name", "")
        config = dataset_config.get("config", {})
        loaded = False
        if class_name:
            try:
                import importlib
                module = importlib.import_module("kgcnn_torch.data.datasets")
                try:
                    DatasetClass = getattr(module, class_name)
                except AttributeError:
                    DatasetClass = getattr(module, class_name + "Dataset")
                dataset = DatasetClass(**config)
                if hasattr(dataset, 'to_pyg_list'):
                    pyg_data_list = dataset.to_pyg_list()
                else:
                    pyg_data_list = list(dataset)
                loaded = True
            except (ImportError, AttributeError) as e:
                logger.warning(f"Could not load dataset class '{class_name}': {e}")
        if not loaded:
            raise ValueError(
                f"Could not load force dataset. Provide a valid 'file_path' to a pickle "
                f"file, or a valid 'class_name' in data config. Got: {dataset_config}"
            )

    data_length = len(pyg_data_list)
    logger.info(f"Dataset loaded with {data_length} structures")
    model_config = adapt_model_config_from_data(model_name, model_config, pyg_data_list)

    # Cross-validation
    if cv_config:
        n_splits = cv_config.get("n_splits", cv_config.get("config", {}).get("n_splits", 5))
        kfold = KFold(n_splits=n_splits, shuffle=True, random_state=args.seed)
        train_test_indices = list(kfold.split(np.zeros(data_length)))
    else:
        indices = np.arange(data_length)
        np.random.shuffle(indices)
        split_point = int(0.8 * data_length)
        train_test_indices = [(indices[:split_point], indices[split_point:])]

    # Output directory
    output_dir = os.path.join(args.output, f"{model_name}_force")
    os.makedirs(output_dir, exist_ok=True)
    hyper.save(os.path.join(output_dir, "hyper_used.json"))

    # Loss
    loss_fn = EnergyForceLoss(energy_weight=energy_weight, force_weight=force_weight)

    execute_folds = args.fold

    for fold_idx, (train_index, test_index) in enumerate(train_test_indices):
        if execute_folds is not None and fold_idx not in execute_folds:
            continue

        logger.info(f"\nFold {fold_idx+1}/{len(train_test_indices)} - "
                     f"Train: {len(train_index)}, Test: {len(test_index)}")

        # Deep copy to avoid cross-fold mutation from scaler transforms
        train_data = [copy.deepcopy(pyg_data_list[i]) for i in train_index]
        test_data = [copy.deepcopy(pyg_data_list[i]) for i in test_index]

        # Apply scaler if configured
        scaler = None
        if scaler_config:
            scaler_class = scaler_config.get("class_name", "")
            if scaler_class in ("EnergyForceExtensiveLabelScaler", "ForceStandardScaler"):
                from kgcnn_torch.data.transform import ForceStandardScaler
                scaler = ForceStandardScaler()

                # Collect energy, forces, atomic numbers from training set
                train_energies = []
                train_forces = []
                train_atoms = []
                for d in train_data:
                    e = getattr(d, energy_key, None)
                    if e is None:
                        e = getattr(d, 'energy', None)
                    if e is not None:
                        train_energies.append(e.numpy().reshape(-1))
                    f = getattr(d, force_key, None)
                    if f is not None:
                        train_forces.append(f.numpy())
                    z = d.z if hasattr(d, 'z') else None
                    if z is not None:
                        train_atoms.append(z.numpy())

                if train_energies and train_forces and train_atoms:
                    energy_arr = np.array(train_energies)
                    if energy_arr.ndim == 1:
                        energy_arr = energy_arr[:, None]

                    scaler.fit(y=(energy_arr, train_forces), X=train_atoms)

                    # Transform training data
                    _transform_force_data(train_data, scaler, energy_key, force_key)
                    # Transform test data
                    _transform_force_data(test_data, scaler, energy_key, force_key)

                    scaler.save(os.path.join(output_dir, f"scaler_fold_{fold_idx}.json"))
                    logger.info("ForceStandardScaler fitted and applied")
                else:
                    logger.warning("Scaler configured but data keys missing; skipping scaler")
                    scaler = None

        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Create model
        ModelClass = get_model_class(model_name)
        model = ModelClass(**model_config).to(device)
        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        # Optimizer
        opt_config = compile_config.get("optimizer", {}).get("config", {"lr": 5e-4})
        optimizer = torch.optim.Adam(model.parameters(), **opt_config)

        # Scheduler
        scheduler = None
        if scheduler_config:
            sched_config = dict(scheduler_config)
            sched_name = sched_config.pop("class_name", None)
            if sched_name:
                sched_config.setdefault("steps_per_epoch", len(train_loader))
                scheduler = get_scheduler(sched_name, optimizer, **sched_config)

        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        history = {"train_loss": [], "val_loss": [], "val_energy_mae": [], "val_force_mae": []}
        checkpoint_path = os.path.join(output_dir, f"best_model_fold_{fold_idx}.pt")

        start_time = time.time()
        for epoch in range(epochs):
            t0 = time.time()

            train_loss = train_epoch_force(
                model, train_loader, optimizer, loss_fn, device,
                energy_key=energy_key, force_key=force_key
            )
            val_results = eval_epoch_force(
                model, val_loader, loss_fn, device,
                energy_key=energy_key, force_key=force_key
            )

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_results["loss"])
            history["val_energy_mae"].append(val_results["energy_mae"])
            history["val_force_mae"].append(val_results["force_mae"])

            # Scheduler step
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(val_results["loss"])
                else:
                    scheduler.step()

            # Early stopping
            if early_stopping > 0:
                if val_results["loss"] < best_val_loss:
                    best_val_loss = val_results["loss"]
                    patience_counter = 0
                    torch.save(model.state_dict(), checkpoint_path)
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break

            dt = time.time() - t0
            if (epoch + 1) % max(1, epochs // 20) == 0:
                lr = optimizer.param_groups[0]['lr']
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - {dt:.1f}s - "
                    f"loss: {train_loss:.4f} - val_loss: {val_results['loss']:.4f} - "
                    f"E_mae: {val_results['energy_mae']:.4f} - "
                    f"F_mae: {val_results['force_mae']:.6f} - lr: {lr:.2e}"
                )

        elapsed = str(timedelta(seconds=time.time() - start_time))
        logger.info(f"Fold {fold_idx+1} time: {elapsed}")

        # Load best model
        if os.path.exists(checkpoint_path) and early_stopping > 0:
            model.load_state_dict(torch.load(checkpoint_path, weights_only=True))

        # Save history
        with open(os.path.join(output_dir, f"history_fold_{fold_idx}.json"), 'w') as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

        torch.save(model.state_dict(), os.path.join(output_dir, f"model_fold_{fold_idx}.pt"))

    logger.info(f"\nForce training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
