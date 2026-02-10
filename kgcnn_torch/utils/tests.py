"""Test utilities for comparing tensor outputs and generating test data."""
import unittest
from typing import Union

import numpy as np
import torch


def compare_static_shapes(found: Union[tuple, None], expected: Union[tuple, None]) -> bool:
    """Compare two shapes allowing None (dynamic) dimensions.

    Args:
        found: The shape that was found (e.g. from model output).
        expected: The expected shape.

    Returns:
        bool: True if shapes are compatible.
    """
    if found is None and expected is None:
        return True
    elif found is None and expected is not None:
        return False
    elif found is not None and expected is None:
        return True
    elif len(found) != len(expected):
        return False
    shapes_okay = []
    for f, e in zip(found, expected):
        if f is None and e is not None:
            shapes_okay.append(False)
        elif f is not None and e is None:
            shapes_okay.append(True)
        elif f is None and e is None:
            shapes_okay.append(True)
        elif f == e:
            shapes_okay.append(True)
        else:
            shapes_okay.append(False)
    return all(shapes_okay)


def tensor_to_numpy(x) -> np.ndarray:
    """Convert a tensor (torch.Tensor or np.ndarray) to numpy array.

    Args:
        x: Input tensor or array.

    Returns:
        np.ndarray: Numpy array.
    """
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class TestCase(unittest.TestCase):
    """Extended TestCase with tensor comparison methods for PyTorch."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @staticmethod
    def assertAllClose(x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        """Assert that two tensors are element-wise close.

        Args:
            x1: First tensor (torch.Tensor or np.ndarray).
            x2: Second tensor (torch.Tensor or np.ndarray).
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            msg (str): Optional message on failure.
        """
        x1 = tensor_to_numpy(x1)
        x2 = tensor_to_numpy(x2)
        np.testing.assert_allclose(x1, x2, atol=atol, rtol=rtol, err_msg=msg or "")

    def assertNotAllClose(self, x1, x2, atol=1e-6, rtol=1e-6, msg=None):
        """Assert that two tensors are NOT element-wise close.

        Args:
            x1: First tensor.
            x2: Second tensor.
            atol (float): Absolute tolerance.
            rtol (float): Relative tolerance.
            msg (str): Optional message on failure.
        """
        try:
            self.assertAllClose(x1, x2, atol=atol, rtol=rtol, msg=msg)
        except AssertionError:
            return
        msg = msg or ""
        raise AssertionError(
            "The two values are close at all elements. \n"
            "%s.\n"
            "Values: %s" % (msg, str(tensor_to_numpy(x1)))
        )

    def assertAllEqual(self, x1, x2, msg=None):
        """Assert that two sequences of tensors are element-wise equal.

        Args:
            x1: First sequence of tensors.
            x2: Second sequence of tensors.
            msg (str): Optional message on failure.
        """
        self.assertEqual(len(x1), len(x2), msg=msg)
        for e1, e2 in zip(x1, x2):
            if isinstance(e1, (list, tuple)) or isinstance(e2, (list, tuple)):
                self.assertAllEqual(e1, e2, msg=msg)
            else:
                e1 = tensor_to_numpy(e1)
                e2 = tensor_to_numpy(e2)
                self.assertEqual(e1, e2, msg=msg)

    def assertLen(self, iterable, expected_len, msg=None):
        """Assert that an iterable has the expected length.

        Args:
            iterable: Object with __len__.
            expected_len (int): Expected length.
            msg (str): Optional message on failure.
        """
        self.assertEqual(len(iterable), expected_len, msg=msg)


def generate_test_graph_data(num_nodes: int = 10, num_edges: int = 20,
                             node_features: int = 16, edge_features: int = 8,
                             dtype: torch.dtype = torch.float32,
                             device: str = "cpu") -> dict:
    """Generate random graph data for testing.

    Args:
        num_nodes (int): Number of nodes.
        num_edges (int): Number of edges.
        node_features (int): Dimension of node features.
        edge_features (int): Dimension of edge features.
        dtype (torch.dtype): Data type for feature tensors.
        device (str): Device to place tensors on.

    Returns:
        dict: Dictionary with 'x' (node features), 'edge_attr' (edge features),
            'edge_index' (2, num_edges), 'batch' (node batch assignment).
    """
    x = torch.randn(num_nodes, node_features, dtype=dtype, device=device)
    edge_attr = torch.randn(num_edges, edge_features, dtype=dtype, device=device)
    edge_index = torch.randint(0, num_nodes, (2, num_edges), device=device)
    batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
    return {
        "x": x,
        "edge_attr": edge_attr,
        "edge_index": edge_index,
        "batch": batch,
    }
