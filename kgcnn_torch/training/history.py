"""Utilities for saving and loading training history in PyTorch training.

Provides functions for recording training results (loss curves, metrics) across
cross-validation folds and persisting them to disk as YAML files. Also supports
loading history from saved pickle files.
"""
import numpy as np
import os
import sys
import pickle
import logging
from typing import Union, Optional
from datetime import datetime

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def _save_yaml_file(data: dict, filepath: str):
    """Save a dictionary to a YAML file.

    Args:
        data: Dictionary to save.
        filepath: Full path to the output file.
    """
    if not HAS_YAML:
        # Fallback: write a simple representation if PyYAML is not installed
        module_logger.warning("PyYAML not installed. Saving as text instead of YAML.")
        with open(filepath, 'w') as f:
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        return

    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    with open(filepath, 'w') as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def _load_pickle_file(filepath: str):
    """Load a pickle file.

    Args:
        filepath: Full path to the pickle file.

    Returns:
        The deserialized object.
    """
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def _check_device() -> dict:
    """Check available compute devices and return info dict.

    Returns:
        Dictionary with device information.
    """
    device_info = {}
    if HAS_TORCH:
        device_info["torch_version"] = torch.__version__
        device_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            device_info["cuda_device_count"] = torch.cuda.device_count()
            device_info["cuda_device_name"] = torch.cuda.get_device_name(0)
    return device_info


def load_history_list(file_path: str, folds: int) -> list:
    """Load a list of training histories from individual fold pickle files.

    This function expects pickled history dictionaries saved with filenames
    containing '(i)' as a placeholder for the fold index. For example,
    'history_(i).pkl' would load 'history_0.pkl', 'history_1.pkl', etc.

    Args:
        file_path: Path template with '(i)' placeholder for fold index.
        folds: Number of folds to load.

    Returns:
        List of history dictionaries, one per fold that was found on disk.
    """
    history_list = []
    for i in range(folds):
        file_path_i = str(file_path).replace("(i)", str(i))
        if os.path.exists(file_path_i):
            history_list.append(_load_pickle_file(file_path_i))
    return history_list


# Alias for backward compatibility
load_time_list = load_history_list


def save_history_score(
        histories: list,
        filepath: Optional[str] = None,
        loss_name: Union[str, list, None] = None,
        val_loss_name: Union[str, list, None] = None,
        data_unit: str = "",
        model_name: str = "",
        file_name: str = "score.yaml",
        model_version: str = "",
        dataset_name: str = "",
        model_class: str = "",
        execute_folds: Union[list, int, None] = None,
        multi_target_indices: Union[list, int, None] = None,
        trajectory_name: Optional[str] = None,
        seed: Optional[int] = None,
        time_list: Optional[list] = None
) -> dict:
    """Save fit results from training histories to a YAML file.

    This function is designed for use in training scripts that perform
    K-fold cross-validation. It extracts final and best metrics from each
    fold's history and saves a summary to a YAML file.

    Each entry in ``histories`` should be a dictionary mapping metric names
    to lists of per-epoch values (e.g., ``{'train_loss': [0.5, 0.3, ...],
    'val_loss': [0.6, 0.4, ...]}``, as returned by the ``fit()`` function
    in ``trainer.py``).

    Args:
        histories: List of history dicts, one per fold. Each dict maps
            metric names to lists of per-epoch values.
        filepath: Directory where to save the score file. Default is None (no save).
        loss_name: Name(s) of training loss metric(s) in the history dict.
            If None, all keys not starting with 'val_' are used.
        val_loss_name: Name(s) of validation metric(s) in the history dict.
            If None, all keys starting with 'val_' are used.
        data_unit: Physical unit of the loss/metric values (e.g. 'eV', 'kcal/mol').
        model_name: Name of the model architecture.
        file_name: Name of the output file. Default is 'score.yaml'.
        model_version: Version string of the model.
        dataset_name: Name of the dataset used for training.
        model_class: Class name or type of the model.
        execute_folds: Which folds were executed (list of indices or count).
        multi_target_indices: Indices of targets in multi-target training.
        trajectory_name: Name of the MD trajectory if applicable.
        seed: Random seed used for reproducibility.
        time_list: List of per-fold training time information.

    Returns:
        Dictionary containing the aggregated score information.
    """
    # Normalize histories: accept dict-like objects
    normalized = []
    for hist in histories:
        if isinstance(hist, dict):
            normalized.append(hist)
        elif hasattr(hist, 'history'):
            # Support keras-style History objects that wrap a .history dict
            normalized.append(hist.history)
        else:
            normalized.append(dict(hist))
    histories = normalized

    if data_unit is None:
        data_unit = ""

    # Auto-detect loss and validation metric names
    if loss_name is None and len(histories) > 0:
        loss_name = [x for x in list(histories[0].keys()) if "val_" not in x and x != "lr"]
    if val_loss_name is None and len(histories) > 0:
        val_loss_name = [x for x in list(histories[0].keys()) if "val_" in x]

    if not isinstance(loss_name, list):
        loss_name = [loss_name] if loss_name is not None else []
    if not isinstance(val_loss_name, list):
        val_loss_name = [val_loss_name] if val_loss_name is not None else []

    if isinstance(multi_target_indices, list):
        multi_target_indices = [int(x) for x in multi_target_indices]
    elif multi_target_indices is not None:
        multi_target_indices = int(multi_target_indices)

    # Extract final, min, and max values for each metric across folds
    train_loss = []
    for x in loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories if x in hist])
        train_loss.append(loss)

    val_loss = []
    for x in val_loss_name:
        loss = np.array([np.array(hist[x]) for hist in histories if x in hist])
        val_loss.append(loss)

    result_dict = {}

    # Record train metrics
    for name, values in zip(loss_name, train_loss):
        if len(values) > 0:
            result_dict[name] = [float(x[-1]) for x in values]
            result_dict["max_%s" % name] = [float(np.amax(x)) for x in values]
            result_dict["min_%s" % name] = [float(np.amin(x)) for x in values]

    # Record validation metrics
    for name, values in zip(val_loss_name, val_loss):
        if len(values) > 0:
            result_dict[name] = [float(x[-1]) for x in values]
            result_dict["max_%s" % name] = [float(np.amax(x)) for x in values]
            result_dict["min_%s" % name] = [float(np.amin(x)) for x in values]

    # Record metadata
    result_dict["data_unit"] = str(data_unit)
    if len(train_loss) > 0 and len(train_loss[0]) > 0:
        result_dict["epochs"] = [int(len(x)) for x in train_loss[0]]

    result_dict["date_time"] = str(datetime.today().strftime('%Y-%m-%d %H:%M:%S'))
    result_dict["model_class"] = str(model_class)
    result_dict["model_version"] = str(model_version)
    result_dict["model_name"] = str(model_name)
    result_dict["number_histories"] = len(histories)
    result_dict["multi_target_indices"] = multi_target_indices
    result_dict["execute_folds"] = execute_folds
    result_dict["time_list"] = time_list
    result_dict["seed"] = seed
    result_dict["backend"] = "torch"
    result_dict["OS"] = "%s_%s" % (os.name, sys.platform)
    result_dict.update(_check_device())

    if trajectory_name:
        result_dict["trajectory_name"] = trajectory_name

    # Save to file
    if filepath is not None:
        output_path = os.path.join(filepath, "%s_%s_%s" % (model_name, dataset_name, file_name))
        _save_yaml_file(result_dict, output_path)
        module_logger.info("Saved training score to %s" % output_path)

    return result_dict
