"""File I/O utilities for saving and loading data in various formats."""
import os
import pickle
import json
import logging
import numpy as np
from typing import Any
from importlib.machinery import SourceFileLoader

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def save_pickle_file(obj: Any, file_path: str, **kwargs):
    """Save a Python object to a pickle file.

    Args:
        obj: Python object to serialize.
        file_path (str): Path to the output file.
        **kwargs: Additional keyword arguments passed to ``pickle.dump``.
    """
    with open(file_path, "wb") as f:
        pickle.dump(obj, f, **kwargs)


def load_pickle_file(file_path: str, **kwargs) -> Any:
    """Load a Python object from a pickle file.

    Args:
        file_path (str): Path to the pickle file.
        **kwargs: Additional keyword arguments passed to ``pickle.load``.

    Returns:
        The deserialized Python object.
    """
    with open(file_path, "rb") as f:
        obj = pickle.load(f, **kwargs)
    return obj


def save_json_file(obj: Any, file_path: str, **kwargs):
    """Save a Python object to a JSON file.

    Args:
        obj: Python object to serialize (must be JSON-serializable).
        file_path (str): Path to the output file.
        **kwargs: Additional keyword arguments passed to ``json.dump``.
    """
    with open(file_path, "w") as json_file:
        json.dump(obj, json_file, **kwargs)


def load_json_file(file_path: str, **kwargs) -> Any:
    """Load a Python object from a JSON file.

    Args:
        file_path (str): Path to the JSON file.
        **kwargs: Additional keyword arguments passed to ``json.load``.

    Returns:
        The deserialized Python object.
    """
    with open(file_path, "r") as json_file:
        file_read = json.load(json_file, **kwargs)
    return file_read


def load_yaml_file(file_path: str) -> Any:
    """Load a Python object from a YAML file.

    Args:
        file_path (str): Path to the YAML file.

    Returns:
        The deserialized Python object.
    """
    import yaml
    with open(file_path, "r") as stream:
        obj = yaml.safe_load(stream)
    return obj


def save_yaml_file(obj: Any, file_path: str, default_flow_style: bool = False, **kwargs):
    """Save a Python object to a YAML file.

    Args:
        obj: Python object to serialize.
        file_path (str): Path to the output file.
        default_flow_style (bool): YAML flow style flag. Default is False.
        **kwargs: Additional keyword arguments passed to ``yaml.dump``.
    """
    import yaml
    with open(file_path, "w") as yaml_file:
        yaml.dump(obj, yaml_file, default_flow_style=default_flow_style, **kwargs)


def load_hyper_file(file_name: str, **kwargs) -> dict:
    """Load hyperparameters from a file.

    Supports '.yaml', '.json', '.pickle', and '.py' file formats.
    For '.py' files, the module must contain a ``hyper`` variable.

    Args:
        file_name (str): Path to the hyperparameter file.
        **kwargs: Additional keyword arguments for the loader.

    Returns:
        dict: Dictionary of hyperparameters.
    """
    if "." not in file_name:
        module_logger.error("Cannot determine file type for '%s'." % file_name)
        return {}
    type_ending = file_name.split(".")[-1]
    if type_ending == "json":
        return load_json_file(file_name, **kwargs)
    elif type_ending == "yaml":
        return load_yaml_file(file_name)
    elif type_ending == "pickle":
        return load_pickle_file(file_name, **kwargs)
    elif type_ending == "py":
        path = os.path.realpath(file_name)
        hyper = getattr(
            SourceFileLoader(os.path.basename(path).replace(".py", ""), path).load_module(),
            "hyper"
        )
        return hyper
    else:
        module_logger.error("Unsupported file type '%s'." % type_ending)
    return {}


class RaggedNumpyFile:
    """Store and load ragged (variable-length) arrays as NumPy npz files.

    Supports ragged tensors with ragged rank of one, stored as flattened values
    and row splits.
    """

    def __init__(self, file_path: str, compressed: bool = False):
        """Initialize with file path.

        Args:
            file_path (str): Path to the npz file.
            compressed (bool): Whether to use compression. Default is False.
        """
        self.file_path = file_path
        self.compressed = compressed

    def write(self, arrays: list):
        """Write a list of arrays (possibly ragged) to file.

        Args:
            arrays (list): List of numpy arrays with potentially different
                lengths along axis 0.
        """
        row_lengths = np.array([len(x) for x in arrays], dtype=np.int64)
        row_splits = np.concatenate([[0], np.cumsum(row_lengths)]).astype(np.int64)
        values = np.concatenate(arrays, axis=0)
        out = {"values": values, "row_splits": row_splits}
        if self.compressed:
            np.savez_compressed(self.file_path, **out)
        else:
            np.savez(self.file_path, **out)

    def read(self) -> list:
        """Read the ragged arrays from file.

        Returns:
            list: List of numpy arrays.
        """
        data = np.load(self.file_path)
        values = data["values"]
        row_splits = data["row_splits"]
        return np.split(values, row_splits[1:-1])

    def __getitem__(self, item: int) -> np.ndarray:
        """Get a single item by index.

        Args:
            item (int): Index of the item.

        Returns:
            np.ndarray: The array at the given index.
        """
        assert isinstance(item, int), "Only single integer index is supported."
        data = np.load(self.file_path)
        row_splits = data["row_splits"]
        return np.array(data["values"][row_splits[item]:row_splits[item + 1]])

    def __len__(self) -> int:
        """Return the number of arrays stored."""
        data = np.load(self.file_path)
        row_splits = data["row_splits"]
        return int(row_splits.shape[0]) - 1

    def exists(self) -> bool:
        """Check if the file exists on disk."""
        return os.path.exists(self.file_path)
