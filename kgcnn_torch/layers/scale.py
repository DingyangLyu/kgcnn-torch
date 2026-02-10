"""Label scaling utilities for graph neural network training."""
import importlib
import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StandardLabelScaler(nn.Module):
    """Standard scaling: (y - mean) / std.

    Stores mean and std as buffers for use during training and inference.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("mean_", torch.tensor(0.0))
        self.register_buffer("std_", torch.tensor(1.0))

    def fit(self, y: np.ndarray):
        """Fit scaler to data.

        Args:
            y: Target values of shape (N,) or (N, D).
        """
        y = np.asarray(y, dtype=np.float32)
        self.mean_ = torch.tensor(np.mean(y, axis=0), dtype=torch.float32)
        self.std_ = torch.tensor(np.std(y, axis=0), dtype=torch.float32)
        self.std_[self.std_ < 1e-7] = 1.0

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        """Scale target values."""
        return (y - self.mean_.to(y.device)) / self.std_.to(y.device)

    def inverse_transform(self, y_scaled: torch.Tensor) -> torch.Tensor:
        """Inverse scale predictions."""
        return y_scaled * self.std_.to(y_scaled.device) + self.mean_.to(y_scaled.device)


class ExtensiveMolecularLabelScaler(nn.Module):
    """Extensive property scaling using per-element ridge regression kernel.

    Matches Keras ExtensiveMolecularLabelScaler: stores a ridge_kernel_ of shape
    (max_atomic_number, D) with per-element energy offsets, and scale_ for residual
    scaling. At inference: y_scaled = (y - sum_atoms(ridge_kernel[z])) / scale_.

    For extensive properties like total energy that scale with system size.
    """

    max_atomic_number = 95

    def __init__(self):
        super().__init__()
        self.register_buffer("scale_", torch.tensor(1.0))
        self.register_buffer("ridge_kernel_", torch.zeros(self.max_atomic_number, 1))

    def set_scale(self, scaler):
        """Set weights from a fitted ExtensiveMolecularScaler (numpy-based).

        Args:
            scaler: Fitted ExtensiveMolecularScaler from kgcnn_torch.data.transform.
        """
        ridge_kernel = np.transpose(np.array(scaler.ridge.coef_))  # (num_atoms_used, D)
        atom_mask = np.array(scaler._fit_atom_mask, dtype=bool)
        scale = np.array(scaler.scale_)

        # Build full kernel indexed by atomic number
        D = ridge_kernel.shape[1] if ridge_kernel.ndim > 1 else 1
        full_kernel = np.zeros((self.max_atomic_number, D), dtype=np.float64)
        positions = np.where(atom_mask)[0]
        full_kernel[positions] = ridge_kernel
        full_kernel[0] = 0.0  # Ensure atomic number 0 is always zero

        self.ridge_kernel_ = torch.tensor(full_kernel, dtype=torch.float32)
        self.scale_ = torch.tensor(scale, dtype=torch.float32)

    def fit(self, y: np.ndarray, atomic_number: list):
        """Fit scaler using ridge regression on atom count matrix.

        Args:
            y: Target values of shape (N,) or (N, D).
            atomic_number: List of arrays, each containing atomic numbers per molecule.
        """
        from sklearn.linear_model import Ridge

        y = np.asarray(y, dtype=np.float64)
        if y.ndim == 1:
            y = y[:, None]

        # Build atom count matrix
        max_z = self.max_atomic_number
        counts = np.zeros((len(atomic_number), max_z), dtype=np.float64)
        for i, z_arr in enumerate(atomic_number):
            z_arr = np.asarray(z_arr).ravel()
            for z in z_arr:
                zi = int(z)
                if zi < max_z:
                    counts[i, zi] += 1

        atom_mask = counts.sum(axis=0) > 0

        ridge = Ridge(alpha=1e-9, fit_intercept=False)
        ridge.fit(counts[:, atom_mask], y)

        ridge_kernel = np.transpose(np.array(ridge.coef_))  # (num_used, D)
        D = y.shape[1]
        full_kernel = np.zeros((max_z, D), dtype=np.float64)
        full_kernel[np.where(atom_mask)[0]] = ridge_kernel
        full_kernel[0] = 0.0

        residual = y - ridge.predict(counts[:, atom_mask])
        scale = np.std(residual, axis=0)
        scale[scale < 1e-7] = 1.0

        self.ridge_kernel_ = torch.tensor(full_kernel, dtype=torch.float32)
        self.scale_ = torch.tensor(scale, dtype=torch.float32)

    def transform(self, y: torch.Tensor, atomic_number: torch.Tensor,
                  batch: torch.Tensor) -> torch.Tensor:
        """Scale target values.

        Args:
            y: Target values (B, D).
            atomic_number: Atomic numbers of all nodes (N,).
            batch: Batch assignment (N,).
        """
        device = y.device
        kernel = self.ridge_kernel_.to(device)
        scale = self.scale_.to(device)
        # Per-atom offset lookup and sum per graph
        energy_per_node = kernel[atomic_number.long()]  # (N, D)
        batch_size = y.size(0)
        extensive = torch.zeros(batch_size, energy_per_node.size(-1),
                                device=device, dtype=y.dtype)
        extensive.scatter_add_(0, batch.unsqueeze(-1).expand_as(energy_per_node),
                               energy_per_node.to(y.dtype))
        return (y - extensive) / scale

    def inverse_transform(self, y_scaled: torch.Tensor,
                          atomic_number: torch.Tensor,
                          batch: torch.Tensor) -> torch.Tensor:
        """Inverse scale predictions.

        Args:
            y_scaled: Scaled predictions (B, D).
            atomic_number: Atomic numbers of all nodes (N,).
            batch: Batch assignment (N,).
        """
        device = y_scaled.device
        kernel = self.ridge_kernel_.to(device)
        scale = self.scale_.to(device)
        energy_per_node = kernel[atomic_number.long()]
        batch_size = y_scaled.size(0)
        extensive = torch.zeros(batch_size, energy_per_node.size(-1),
                                device=device, dtype=y_scaled.dtype)
        extensive.scatter_add_(0, batch.unsqueeze(-1).expand_as(energy_per_node),
                               energy_per_node.to(y_scaled.dtype))
        return y_scaled * scale + extensive


class QMGraphLabelScaler(nn.Module):
    """Container scaler for QM properties that may mix extensive and intensive targets.

    Wraps a list of scalers (StandardLabelScaler or ExtensiveMolecularLabelScaler)
    and applies each to its corresponding target column.
    """

    def __init__(self, scaler_list: list = None):
        super().__init__()
        self.scaler_list = nn.ModuleList(scaler_list or [])

    def transform(self, y: torch.Tensor, atomic_number: torch.Tensor = None,
                  batch: torch.Tensor = None) -> torch.Tensor:
        """Scale targets column by column.

        Args:
            y: Targets of shape (B, D) where D = len(scaler_list).
            atomic_number: Atomic numbers of all nodes (N,).
            batch: Batch assignment (N,).

        Returns:
            Scaled targets of shape (B, D).
        """
        out_cols = []
        for i, scaler in enumerate(self.scaler_list):
            col = y[:, i:i+1]
            if isinstance(scaler, ExtensiveMolecularLabelScaler):
                out_cols.append(scaler.transform(col, atomic_number, batch))
            else:
                out_cols.append(scaler.transform(col))
        return torch.cat(out_cols, dim=-1)

    def inverse_transform(self, y_scaled: torch.Tensor,
                          atomic_number: torch.Tensor = None,
                          batch: torch.Tensor = None) -> torch.Tensor:
        """Inverse scale predictions column by column.

        Args:
            y_scaled: Scaled predictions of shape (B, D).
            atomic_number: Atomic numbers of all nodes (N,).
            batch: Batch assignment (N,).

        Returns:
            Original-scale targets of shape (B, D).
        """
        out_cols = []
        for i, scaler in enumerate(self.scaler_list):
            col = y_scaled[:, i:i+1]
            if isinstance(scaler, ExtensiveMolecularLabelScaler):
                out_cols.append(scaler.inverse_transform(col, atomic_number, batch))
            else:
                out_cols.append(scaler.inverse_transform(col))
        return torch.cat(out_cols, dim=-1)


# Registry mapping class names to their modules (mirrors Keras scaler/serial.py).
# Includes both numpy-based scalers (data.transform) and Keras-compatible aliases.
_scaler_module_list = {
    "StandardScaler": "kgcnn_torch.data.transform",
    "StandardLabelScaler": "kgcnn_torch.data.transform",
    "ExtensiveMolecularScaler": "kgcnn_torch.data.transform",
    "ExtensiveMolecularLabelScaler": "kgcnn_torch.data.transform",
    "ForceStandardScaler": "kgcnn_torch.data.transform",
    "EnergyForceExtensiveLabelScaler": "kgcnn_torch.data.transform",
    "QMGraphLabelScaler": "kgcnn_torch.data.transform",
}

# Keras name → torch name alias (EnergyForceExtensiveLabelScaler is the Keras name
# for what torch calls ForceStandardScaler).
_scaler_class_aliases = {
    "EnergyForceExtensiveLabelScaler": "ForceStandardScaler",
}

# Short name aliases for convenience.
_scaler_short_aliases = {
    "standard": "StandardLabelScaler",
    "extensive": "ExtensiveMolecularScaler",
    "force": "ForceStandardScaler",
    "qm": "QMGraphLabelScaler",
}


def deserialize(name: Union[str, dict], **kwargs):
    """Deserialize a scaler class from a string name or serialization dict.

    Supports three input forms:

    1. **String name**: Looks up in the registry (class names or short aliases).
       ``deserialize("StandardLabelScaler")`` or ``deserialize("standard")``.
    2. **Dict with class_name**: Auto-fills ``module_name`` from registry,
       instantiates with ``config``, and restores ``weights`` if present.
       ``deserialize({"class_name": "EnergyForceExtensiveLabelScaler", "config": {...}})``.
    3. **Dict with module_name**: Delegates to ``kgcnn_torch.utils.serial.deserialize()``.

    Args:
        name: Scaler class name (str) or serialization dict with at least ``class_name``.
        **kwargs: Extra keyword arguments passed to the constructor when ``name`` is str.

    Returns:
        Scaler instance.

    Raises:
        ValueError: If the scaler name or class_name is unknown.
        TypeError: If ``name`` is neither str nor dict.
    """
    if isinstance(name, dict):
        if "class_name" not in name:
            raise ValueError("Require 'class_name' for scaler deserialization.")

        class_name = name["class_name"]
        # Resolve Keras aliases
        class_name = _scaler_class_aliases.get(class_name, class_name)

        if "module_name" not in name:
            if class_name in _scaler_module_list:
                name = dict(name)  # copy to avoid mutating caller's dict
                name["module_name"] = _scaler_module_list[class_name]
                name["class_name"] = class_name
            else:
                raise ValueError(
                    f"Unknown scaler class '{name['class_name']}'. "
                    f"Available: {list(_scaler_module_list.keys())}")

        if "config" not in name:
            name = dict(name)
            name["config"] = {}

        # Delegate to general deserializer (handles config + weights + methods)
        from kgcnn_torch.utils.serial import deserialize as deserialize_general
        return deserialize_general(name)

    if isinstance(name, str):
        # Resolve short aliases first
        resolved = _scaler_short_aliases.get(name, name)
        # Resolve Keras class aliases
        resolved = _scaler_class_aliases.get(resolved, resolved)

        if resolved not in _scaler_module_list:
            raise ValueError(
                f"Unknown scaler '{name}'. Available: "
                f"{list(_scaler_short_aliases.keys())} or {list(_scaler_module_list.keys())}")

        module_name = _scaler_module_list[resolved]
        obj_class = getattr(importlib.import_module(module_name), resolved)
        return obj_class(**kwargs)

    raise TypeError(
        f"Expected str or dict for scaler deserialization, got {type(name).__name__}.")


def get_scaler(name: str, **kwargs):
    """Factory function to get scaler by name (backward-compatible wrapper).

    Args:
        name: Scaler name or class name.
        **kwargs: Keyword arguments passed to the constructor.

    Returns:
        Scaler instance.
    """
    return deserialize(name, **kwargs)
