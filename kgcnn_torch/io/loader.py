"""DataLoader wrappers for PyG-based graph datasets with convenience functions."""
import logging
from typing import Optional, Union, List

import torch
import numpy as np

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


def get_dataloader(dataset, batch_size: int = 32, shuffle: bool = True,
                   num_workers: int = 0, **kwargs):
    """Create a PyG DataLoader from a dataset with sensible defaults.

    Args:
        dataset: A PyG InMemoryDataset or list of Data objects.
        batch_size (int): Batch size. Default is 32.
        shuffle (bool): Whether to shuffle. Default is True.
        num_workers (int): Number of data loading workers. Default is 0.
        **kwargs: Additional keyword arguments for the DataLoader.

    Returns:
        torch_geometric.loader.DataLoader: The configured data loader.
    """
    from torch_geometric.loader import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=num_workers, **kwargs)


def get_train_val_test_loaders(dataset, train_indices, val_indices, test_indices=None,
                                batch_size: int = 32, num_workers: int = 0, **kwargs):
    """Create train, validation, and optionally test DataLoaders from index splits.

    Args:
        dataset: A PyG dataset or list of Data objects.
        train_indices: Array or list of training indices.
        val_indices: Array or list of validation indices.
        test_indices: Array or list of test indices. Optional.
        batch_size (int): Batch size. Default is 32.
        num_workers (int): Number of data loading workers. Default is 0.
        **kwargs: Additional keyword arguments for the DataLoader.

    Returns:
        tuple: (train_loader, val_loader) or (train_loader, val_loader, test_loader)
            if test_indices is provided.
    """
    from torch_geometric.loader import DataLoader

    train_dataset = [dataset[int(i)] for i in train_indices]
    val_dataset = [dataset[int(i)] for i in val_indices]

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, **kwargs)

    if test_indices is not None:
        test_dataset = [dataset[int(i)] for i in test_indices]
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, **kwargs)
        return train_loader, val_loader, test_loader

    return train_loader, val_loader


def pad_collate_graph_batch(data_list: list, pad_value: float = 0.0):
    """Collate a list of graph Data objects into padded batch tensors.

    This is useful when a fixed-size batch tensor is needed instead of the
    default disjoint (variable-size) batching from PyG.

    Args:
        data_list (list): List of PyG Data objects. Each must have 'x' and
            'edge_index' attributes at minimum.
        pad_value (float): Value for padding. Default is 0.0.

    Returns:
        dict: Dictionary with padded batch tensors:
            - 'x': (B, max_nodes, F) padded node features.
            - 'edge_index': (B, 2, max_edges) padded edge indices.
            - 'node_mask': (B, max_nodes) boolean mask for valid nodes.
            - 'edge_mask': (B, max_edges) boolean mask for valid edges.
            - Plus any other float attributes found in the first Data object.
    """
    max_nodes = max(d.x.size(0) for d in data_list)
    max_edges = max(d.edge_index.size(1) for d in data_list)
    batch_size = len(data_list)
    node_dim = data_list[0].x.size(1)

    x_padded = torch.full((batch_size, max_nodes, node_dim), pad_value)
    edge_index_padded = torch.zeros(batch_size, 2, max_edges, dtype=torch.long)
    node_mask = torch.zeros(batch_size, max_nodes, dtype=torch.bool)
    edge_mask = torch.zeros(batch_size, max_edges, dtype=torch.bool)

    extra_keys = {}
    for key in data_list[0].keys():
        if key in ("x", "edge_index", "batch", "ptr"):
            continue
        val = data_list[0][key]
        if isinstance(val, torch.Tensor) and val.dtype.is_floating_point:
            extra_keys[key] = val

    extra_padded = {}
    for key, val in extra_keys.items():
        if val.dim() == 0:
            extra_padded[key] = torch.zeros(batch_size, dtype=val.dtype)
        elif val.dim() == 1:
            extra_padded[key] = torch.full((batch_size, val.size(0)), pad_value, dtype=val.dtype)

    for i, data in enumerate(data_list):
        n_nodes = data.x.size(0)
        n_edges = data.edge_index.size(1)
        x_padded[i, :n_nodes] = data.x
        edge_index_padded[i, :, :n_edges] = data.edge_index
        node_mask[i, :n_nodes] = True
        edge_mask[i, :n_edges] = True
        for key in extra_padded:
            val = data[key]
            if val.dim() == 0:
                extra_padded[key][i] = val
            elif val.dim() == 1:
                extra_padded[key][i, :val.size(0)] = val

    result = {
        "x": x_padded,
        "edge_index": edge_index_padded,
        "node_mask": node_mask,
        "edge_mask": edge_mask,
    }
    result.update(extra_padded)
    return result
