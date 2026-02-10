"""Tests for data pipeline."""
import numpy as np
import torch
import unittest
from kgcnn_torch.data.base import GraphDict, MemoryGraphList


class TestGraphDict(unittest.TestCase):

    def test_assign_and_obtain(self):
        g = GraphDict()
        g.assign_property("node_number", np.array([6, 1, 1]))
        result = g.obtain_property("node_number")
        self.assertTrue(np.array_equal(result, np.array([6, 1, 1])))

    def test_missing_property(self):
        g = GraphDict()
        self.assertIsNone(g.obtain_property("nonexistent"))


class TestMemoryGraphList(unittest.TestCase):

    def test_empty_and_set(self):
        graphs = MemoryGraphList()
        graphs.empty(3)
        self.assertEqual(len(graphs), 3)

        graphs.set("node_number", [np.array([1, 2]), np.array([3]), np.array([4, 5, 6])])
        result = graphs.get("node_number")
        self.assertEqual(len(result), 3)

    def test_clean(self):
        graphs = MemoryGraphList()
        graphs.empty(3)
        graphs.set("node_number", [np.array([1]), None, np.array([3])])
        kept = graphs.clean("node_number")
        self.assertEqual(len(graphs), 2)

    def test_to_pyg_list(self):
        graphs = MemoryGraphList()
        graphs.empty(2)
        graphs.set("node_number", [np.array([6, 1, 1]), np.array([8, 1])])
        graphs.set("node_coordinates", [np.random.randn(3, 3), np.random.randn(2, 3)])
        # KGCNN convention: [target, source]
        graphs.set("edge_indices", [
            np.array([[0, 1], [1, 0], [0, 2], [2, 0]]),
            np.array([[0, 1], [1, 0]])
        ])
        graphs.set("graph_labels", [np.array([1.0]), np.array([2.0])])

        pyg_list = graphs.to_pyg_list()
        self.assertEqual(len(pyg_list), 2)

        # Check edge index swap
        d = pyg_list[0]
        self.assertEqual(d.z.shape, (3,))
        self.assertEqual(d.edge_index.shape[0], 2)
        self.assertEqual(d.edge_index.shape[1], 4)
        # Source and target should be swapped from KGCNN
        # Original [0,1] (target=0, source=1) -> PyG source=1, target=0
        self.assertEqual(d.edge_index[0, 0].item(), 1)  # source = old column 1
        self.assertEqual(d.edge_index[1, 0].item(), 0)  # target = old column 0

    def test_to_pyg_list_with_dataloader(self):
        from torch_geometric.loader import DataLoader
        graphs = MemoryGraphList()
        graphs.empty(4)
        graphs.set("node_number", [np.array([1, 2, 3]) for _ in range(4)])
        graphs.set("edge_indices", [np.array([[0, 1], [1, 2]]) for _ in range(4)])
        graphs.set("graph_labels", [np.array([float(i)]) for i in range(4)])

        pyg_list = graphs.to_pyg_list()
        loader = DataLoader(pyg_list, batch_size=2)
        for batch in loader:
            self.assertEqual(batch.z.shape[0], 6)  # 3 nodes * 2 graphs
            break


class TestTransform(unittest.TestCase):

    def test_standard_scaler(self):
        from kgcnn_torch.data.transform import StandardScaler
        scaler = StandardScaler()
        X = np.random.randn(50, 3) * 5 + 10
        scaler.fit(X)
        X_t = scaler.transform(X)
        X_back = scaler.inverse_transform(X_t)
        np.testing.assert_allclose(X, X_back, atol=1e-10)

    def test_standard_scaler_save_load(self):
        import tempfile
        import os
        from kgcnn_torch.data.transform import StandardScaler
        scaler = StandardScaler()
        scaler.fit(np.random.randn(50, 2))
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            path = f.name
        try:
            scaler.save(path)
            scaler2 = StandardScaler()
            scaler2.load(path)
            np.testing.assert_allclose(scaler.mean_, scaler2.mean_)
            np.testing.assert_allclose(scaler.scale_, scaler2.scale_)
        finally:
            os.unlink(path)


if __name__ == '__main__':
    unittest.main()
