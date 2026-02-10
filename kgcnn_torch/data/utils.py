"""Data utility functions for graph dataset processing."""
import numpy as np
from typing import List, Optional, Union


def pad_np_array_list_batch_dim(values: list, dtype: str = None):
    """Pad a list of numpy arrays along the first dimension to create a uniform batch.

    All arrays must have the same rank. They are padded with zeros to match the
    maximum size along each dimension.

    Args:
        values (list): List of numpy arrays with potentially different shapes.
        dtype (str): Optional data type override. Default is None (keep original).

    Returns:
        tuple: (padded, mask) where:
            - padded (np.ndarray): Padded array of shape (B, max_d0, max_d1, ...).
            - mask (np.ndarray): Boolean mask of shape (B, max_d0, max_d1, ...) indicating
              valid (non-padded) entries.
    """
    max_shape = np.amax([x.shape for x in values], axis=0)
    final_shape = np.concatenate(
        [np.array([len(values)], dtype="int64"), np.array(max_shape, dtype="int64")])
    padded = np.zeros(final_shape, dtype=values[0].dtype)
    mask = np.zeros(final_shape, dtype="bool")
    for i, x in enumerate(values):
        index = [i] + [slice(0, int(j)) for j in x.shape]
        padded[tuple(index)] = x
        mask[tuple(index)] = True
    if dtype is not None:
        padded = padded.astype(dtype=dtype)
    return padded, mask


def ragged_values_to_list(values: np.ndarray, row_splits: np.ndarray) -> list:
    """Split flat values array into a list of arrays using row splits.

    Inverse of concatenation with cumulative counts.

    Args:
        values (np.ndarray): Flattened values of shape (total, ...).
        row_splits (np.ndarray): Row splits of shape (n+1,), where
            row_splits[i]:row_splits[i+1] defines the i-th element.

    Returns:
        list: List of numpy arrays.
    """
    return np.split(values, row_splits[1:-1])


def list_to_ragged_values(array_list: list):
    """Convert a list of arrays to flat values with row splits.

    Args:
        array_list (list): List of numpy arrays with the same rank but
            potentially different first-dimension sizes.

    Returns:
        tuple: (values, row_splits) where:
            - values (np.ndarray): Concatenated array.
            - row_splits (np.ndarray): Row splits of shape (n+1,).
    """
    values = np.concatenate(array_list, axis=0)
    lengths = np.array([len(x) for x in array_list], dtype=np.int64)
    row_splits = np.concatenate([[0], np.cumsum(lengths)]).astype(np.int64)
    return values, row_splits


def pad_at_axis(x: np.ndarray, pad_width: tuple, axis: int = 0, **kwargs) -> np.ndarray:
    """Pad a numpy array at a specific axis.

    Args:
        x (np.ndarray): Input array.
        pad_width (tuple): Padding widths as (before, after) for the given axis.
        axis (int): Axis to pad. Default is 0.
        **kwargs: Additional keyword arguments for ``np.pad``.

    Returns:
        np.ndarray: Padded array.
    """
    pads = [(0, 0) for _ in range(len(x.shape))]
    pads[axis] = pad_width
    return np.pad(x, pad_width=pads, **kwargs)


def check_inner_shape(array_list: List[np.ndarray]) -> Optional[tuple]:
    """Check if all arrays in a list have the same inner shape (all dims except first).

    Args:
        array_list (list): List of numpy arrays.

    Returns:
        tuple or None: The common inner shape, or None if shapes differ or list is empty.
    """
    if len(array_list) == 0:
        return None
    if not all(isinstance(x, np.ndarray) for x in array_list):
        return None
    shapes = [x.shape for x in array_list]
    if not all(len(x) == len(shapes[0]) for x in shapes):
        return None
    if len(shapes[0]) == 0:
        return None
    if len(shapes[0]) <= 1:
        return tuple([])
    if all(x[1:] == shapes[0][1:] for x in shapes):
        return shapes[0][1:]
    return None


def train_test_indices(n_samples: int, train_ratio: float = 0.8,
                       shuffle: bool = True, seed: int = 42):
    """Generate train/test index splits.

    Args:
        n_samples (int): Total number of samples.
        train_ratio (float): Fraction of samples for training. Default is 0.8.
        shuffle (bool): Whether to shuffle indices. Default is True.
        seed (int): Random seed. Default is 42.

    Returns:
        tuple: (train_indices, test_indices) as numpy arrays.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)
    split = int(n_samples * train_ratio)
    return indices[:split], indices[split:]


def kfold_indices(n_samples: int, n_splits: int = 5,
                  shuffle: bool = True, seed: int = 42):
    """Generate k-fold cross-validation index splits.

    Args:
        n_samples (int): Total number of samples.
        n_splits (int): Number of folds. Default is 5.
        shuffle (bool): Whether to shuffle before splitting. Default is True.
        seed (int): Random seed. Default is 42.

    Returns:
        list: List of (train_indices, val_indices) tuples for each fold.
    """
    rng = np.random.default_rng(seed)
    indices = np.arange(n_samples)
    if shuffle:
        rng.shuffle(indices)

    fold_sizes = np.full(n_splits, n_samples // n_splits, dtype=int)
    fold_sizes[:n_samples % n_splits] += 1

    folds = []
    current = 0
    for fold_size in fold_sizes:
        val_idx = indices[current:current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size:]])
        folds.append((train_idx, val_idx))
        current += fold_size

    return folds
