"""Hyperparameter configuration management for kgcnn-torch.

Simplified from KGCNN's HyperParameter class, removing Keras serialization.
"""
import json
import logging
import os
from copy import deepcopy
from typing import Union

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def load_hyper_file(filepath: str) -> dict:
    """Load hyperparameter from JSON or Python file.

    Args:
        filepath: Path to .json or .py hyperparameter file.

    Returns:
        Hyperparameter dictionary.
    """
    if filepath.endswith('.json'):
        with open(filepath, 'r') as f:
            return json.load(f)
    elif filepath.endswith('.py'):
        import importlib.util
        spec = importlib.util.spec_from_file_location("hyper_module", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if hasattr(module, 'hyper'):
            return module.hyper
        raise ValueError(f"Python hyper file {filepath} must define a 'hyper' dict.")
    else:
        raise ValueError(f"Unsupported hyper file format: {filepath}")


class HyperParameter:
    """Hyperparameter manager for training configuration.

    Stores model config, training config, and data config.
    Provides methods to extract specific sections.

    Expected structure:
        {
            "model": {"config": {...}},
            "training": {
                "fit": {"epochs": 200, "batch_size": 32, ...},
                "compile": {"optimizer": {...}, "loss": "mse"},
                "scheduler": {...},
                "scaler": {...},
                "cross_validation": {"n_splits": 5}
            },
            "data": {
                "dataset": {"class_name": "...", "config": {...}},
                "data_unit": "..."
            }
        }
    """

    def __init__(self, hyper_info: Union[str, dict],
                 model_name: str = None,
                 dataset_name: str = None):
        if isinstance(hyper_info, str):
            self._hyper_all = load_hyper_file(hyper_info)
        elif isinstance(hyper_info, dict):
            self._hyper_all = hyper_info
        else:
            raise TypeError("HyperParameter requires dict or path to file.")

        self._model_name = model_name
        self._dataset_name = dataset_name

        # Find the relevant hyper section
        if "model" in self._hyper_all and "training" in self._hyper_all:
            self._hyper = self._hyper_all
        elif model_name and model_name in self._hyper_all:
            self._hyper = self._hyper_all[model_name]
        else:
            self._hyper = self._hyper_all

    @property
    def model_config(self) -> dict:
        """Get model configuration."""
        return deepcopy(self._hyper.get("model", {}).get("config", {}))

    @property
    def training_config(self) -> dict:
        """Get training configuration."""
        return deepcopy(self._hyper.get("training", {}))

    @property
    def fit_config(self) -> dict:
        """Get fit/training loop parameters."""
        return deepcopy(self._hyper.get("training", {}).get("fit", {}))

    @property
    def compile_config(self) -> dict:
        """Get optimizer/loss configuration."""
        return deepcopy(self._hyper.get("training", {}).get("compile", {}))

    @property
    def scheduler_config(self) -> dict:
        """Get learning rate scheduler configuration."""
        return deepcopy(self._hyper.get("training", {}).get("scheduler", {}))

    @property
    def scaler_config(self) -> dict:
        """Get data scaler configuration."""
        return deepcopy(self._hyper.get("training", {}).get("scaler", {}))

    @property
    def cross_validation_config(self) -> dict:
        """Get cross-validation configuration."""
        return deepcopy(self._hyper.get("training", {}).get("cross_validation", {}))

    @property
    def data_config(self) -> dict:
        """Get data/dataset configuration."""
        return deepcopy(self._hyper.get("data", {}))

    @property
    def dataset_config(self) -> dict:
        """Get dataset-specific configuration."""
        return deepcopy(self._hyper.get("data", {}).get("dataset", {}))

    def save(self, filepath: str):
        """Save hyperparameters to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self._hyper, f, indent=2)

    def __repr__(self):
        return f"HyperParameter(model={self._model_name}, dataset={self._dataset_name})"
