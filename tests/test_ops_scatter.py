"""Tests for scatter operations."""
import torch
import unittest
from kgcnn_torch.ops.scatter import (
    scatter_reduce_sum, scatter_reduce_mean, scatter_reduce_max,
    scatter_reduce_min, scatter_reduce_softmax
)


class TestScatterOps(unittest.TestCase):

    def test_scatter_reduce_sum(self):
        indices = torch.tensor([0, 0, 1, 1, 2])
        values = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0], [9.0, 10.0]])
        result = scatter_reduce_sum(indices, values, dim_size=3)
        expected = torch.tensor([[4.0, 6.0], [12.0, 14.0], [9.0, 10.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_scatter_reduce_mean(self):
        indices = torch.tensor([0, 0, 1, 1])
        values = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
        result = scatter_reduce_mean(indices, values, dim_size=2)
        expected = torch.tensor([[2.0], [6.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_scatter_reduce_max(self):
        indices = torch.tensor([0, 0, 1, 1])
        values = torch.tensor([[1.0], [3.0], [2.0], [7.0]])
        result = scatter_reduce_max(indices, values, dim_size=2)
        expected = torch.tensor([[3.0], [7.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_scatter_reduce_min(self):
        indices = torch.tensor([0, 0, 1, 1])
        values = torch.tensor([[1.0], [3.0], [2.0], [7.0]])
        result = scatter_reduce_min(indices, values, dim_size=2)
        expected = torch.tensor([[1.0], [2.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_scatter_reduce_softmax(self):
        indices = torch.tensor([0, 0, 1, 1])
        values = torch.tensor([[1.0], [1.0], [2.0], [2.0]])
        result = scatter_reduce_softmax(indices, values, dim_size=2)
        # Equal logits should give equal softmax
        self.assertTrue(torch.allclose(result[0], result[1], atol=1e-5))
        self.assertTrue(torch.allclose(result[2], result[3], atol=1e-5))
        # Sum per group should be ~1
        sum0 = result[0] + result[1]
        self.assertTrue(torch.allclose(sum0, torch.tensor([1.0]), atol=1e-5))

    def test_empty_scatter(self):
        indices = torch.tensor([], dtype=torch.long)
        values = torch.zeros(0, 3)
        result = scatter_reduce_sum(indices, values, dim_size=2)
        self.assertEqual(result.shape, (2, 3))

    def test_gradient_flow(self):
        indices = torch.tensor([0, 0, 1])
        values = torch.randn(3, 2, requires_grad=True)
        result = scatter_reduce_sum(indices, values, dim_size=2)
        result.sum().backward()
        self.assertIsNotNone(values.grad)


if __name__ == '__main__':
    unittest.main()
