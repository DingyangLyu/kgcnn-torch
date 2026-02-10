"""Tests for core layers."""
import torch
import unittest


class TestGatherFunctions(unittest.TestCase):

    def test_gather_nodes_outgoing(self):
        from kgcnn_torch.layers.gather import gather_nodes_outgoing
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])  # src -> tgt
        result = gather_nodes_outgoing(x, edge_index)
        expected = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_gather_nodes_ingoing(self):
        from kgcnn_torch.layers.gather import gather_nodes_ingoing
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        result = gather_nodes_ingoing(x, edge_index)
        expected = torch.tensor([[3.0, 4.0], [5.0, 6.0], [1.0, 2.0]])
        self.assertTrue(torch.allclose(result, expected))

    def test_gather_nodes(self):
        from kgcnn_torch.layers.gather import gather_nodes
        x = torch.tensor([[1.0], [2.0], [3.0]])
        edge_index = torch.tensor([[0, 1], [2, 0]])
        result = gather_nodes(x, edge_index)
        self.assertEqual(result.shape, (2, 2))


class TestAggregation(unittest.TestCase):

    def test_aggregate_local_edges(self):
        from kgcnn_torch.layers.aggr import AggregateLocalEdges
        aggr = AggregateLocalEdges(pooling_method="sum")
        edges = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        edge_index = torch.tensor([[0, 1, 2], [0, 0, 1]])  # all aggregate to node 0 and 1
        result = aggr(edges, edge_index, num_nodes=3)
        self.assertEqual(result.shape, (3, 2))
        # Node 0 gets edges 0 and 1: [4.0, 6.0]
        self.assertTrue(torch.allclose(result[0], torch.tensor([4.0, 6.0])))

    def test_aggregate_attention(self):
        from kgcnn_torch.layers.aggr import AggregateLocalEdgesAttention
        aggr = AggregateLocalEdgesAttention()
        edges = torch.ones(4, 2)
        attention = torch.zeros(4, 1)
        edge_index = torch.tensor([[0, 1, 2, 3], [0, 0, 0, 0]])
        result = aggr(edges, attention, edge_index, num_nodes=4)
        self.assertEqual(result.shape, (4, 2))


class TestGeom(unittest.TestCase):

    def test_gauss_basis(self):
        from kgcnn_torch.layers.geom import GaussBasisLayer
        layer = GaussBasisLayer(bins=10, distance=5.0)
        dist = torch.randn(5, 1).abs()
        result = layer(dist)
        self.assertEqual(result.shape, (5, 10))

    def test_bessel_basis(self):
        from kgcnn_torch.layers.geom import BesselBasisLayer
        layer = BesselBasisLayer(num_radial=6, cutoff=5.0)
        dist = torch.randn(5, 1).abs()
        result = layer(dist)
        self.assertEqual(result.shape, (5, 6))

    def test_cos_cutoff(self):
        from kgcnn_torch.layers.geom import CosCutOffEnvelope
        layer = CosCutOffEnvelope(cutoff=5.0)
        dist = torch.tensor([[0.0], [2.5], [5.0], [7.0]])
        result = layer(dist)
        self.assertTrue(torch.allclose(result[0], torch.tensor([1.0])))
        self.assertTrue(result[2].item() < 0.01)  # At cutoff, should be ~0

    def test_compute_edge_distances(self):
        from kgcnn_torch.layers.geom import compute_edge_distances
        pos = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        edge_index = torch.tensor([[0, 1], [1, 2]])
        dist = compute_edge_distances(pos, edge_index)
        self.assertEqual(dist.shape, (2, 1))
        self.assertTrue(torch.allclose(dist[0], torch.tensor([1.0]), atol=1e-3))

    def test_gradient_through_geom(self):
        from kgcnn_torch.layers.geom import GaussBasisLayer
        layer = GaussBasisLayer(bins=5, distance=3.0)
        dist = torch.tensor([[0.5], [1.5], [2.5]], requires_grad=True)
        result = layer(dist)
        result.sum().backward()
        self.assertIsNotNone(dist.grad)


class TestConv(unittest.TestCase):

    def test_gcn_conv(self):
        from kgcnn_torch.layers.conv import GCNConv
        conv = GCNConv(in_features=8, out_features=8)
        x = torch.randn(5, 8)
        edge_index = torch.randint(0, 5, (2, 10))
        edge_weight = torch.ones(10, 1)
        out = conv(x, edge_index, edge_weight)
        self.assertEqual(out.shape, (5, 8))

    def test_schnet_interaction(self):
        from kgcnn_torch.layers.conv import SchNetInteraction
        layer = SchNetInteraction(units=16, edge_dim=10)
        x = torch.randn(5, 16)
        edge_attr = torch.randn(8, 10)
        edge_index = torch.randint(0, 5, (2, 8))
        out = layer(x, edge_attr, edge_index)
        self.assertEqual(out.shape, (5, 16))


class TestAttention(unittest.TestCase):

    def test_gat_head(self):
        from kgcnn_torch.layers.attention import AttentionHeadGAT
        head = AttentionHeadGAT(in_features=8, units=4)
        x = torch.randn(5, 8)
        edge_index = torch.randint(0, 5, (2, 10))
        out = head(x, edge_index)
        self.assertEqual(out.shape, (5, 4))

    def test_gatv2_head(self):
        from kgcnn_torch.layers.attention import AttentionHeadGATV2
        head = AttentionHeadGATV2(in_features=8, units=4)
        x = torch.randn(5, 8)
        edge_index = torch.randint(0, 5, (2, 10))
        out = head(x, edge_index)
        self.assertEqual(out.shape, (5, 4))


class TestPooling(unittest.TestCase):

    def test_pooling_nodes(self):
        from kgcnn_torch.layers.pooling import PoolingNodes
        pool = PoolingNodes(pooling_method="sum")
        x = torch.randn(10, 4)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        out = pool(x, batch, batch_size=2)
        self.assertEqual(out.shape, (2, 4))


class TestMLP(unittest.TestCase):

    def test_mlp_forward(self):
        from kgcnn_torch.layers.mlp import MLP
        mlp = MLP(units=[16, 8], input_dim=4)
        x = torch.randn(5, 4)
        out = mlp(x)
        self.assertEqual(out.shape, (5, 8))

    def test_mlp_with_normalization(self):
        from kgcnn_torch.layers.mlp import MLP
        mlp = MLP(units=[16], input_dim=4, use_normalization=True)
        x = torch.randn(5, 4)
        out = mlp(x)
        self.assertEqual(out.shape, (5, 16))


class TestNorm(unittest.TestCase):

    def test_graph_batch_norm(self):
        from kgcnn_torch.layers.norm import GraphBatchNorm
        norm = GraphBatchNorm(num_features=8)
        x = torch.randn(10, 8)
        out = norm(x)
        self.assertEqual(out.shape, (10, 8))

    def test_graph_normalization(self):
        from kgcnn_torch.layers.norm import GraphNormalization
        norm = GraphNormalization(num_features=8)
        x = torch.randn(10, 8)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        out = norm(x, batch)
        self.assertEqual(out.shape, (10, 8))


class TestUpdate(unittest.TestCase):

    def test_gru_update(self):
        from kgcnn_torch.layers.update import GRUUpdate
        gru = GRUUpdate(input_dim=8, hidden_dim=16)
        msg = torch.randn(5, 8)
        hidden = torch.randn(5, 16)
        out = gru(msg, hidden)
        self.assertEqual(out.shape, (5, 16))

    def test_residual_layer(self):
        from kgcnn_torch.layers.update import ResidualLayer
        res = ResidualLayer(units=8)
        x = torch.randn(5, 8)
        out = res(x)
        self.assertEqual(out.shape, (5, 8))


class TestScale(unittest.TestCase):

    def test_standard_scaler(self):
        import numpy as np
        from kgcnn_torch.layers.scale import StandardLabelScaler
        scaler = StandardLabelScaler()
        y = np.random.randn(100) * 5 + 10
        scaler.fit(y)
        y_t = torch.tensor(y, dtype=torch.float32)
        scaled = scaler.transform(y_t)
        back = scaler.inverse_transform(scaled)
        self.assertTrue(torch.allclose(y_t, back, atol=1e-4))


class TestNewAggregation(unittest.TestCase):

    def test_aggregate_lstm(self):
        from kgcnn_torch.layers.aggr import AggregateLocalEdgesLSTM
        aggr = AggregateLocalEdgesLSTM(units=8, max_edges_per_node=5)
        edges = torch.randn(10, 4)
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                                   [0, 0, 0, 1, 1, 2, 2, 2, 2, 3]])
        out = aggr(edges, edge_index, num_nodes=5)
        self.assertEqual(out.shape, (5, 8))

    def test_relational_aggregate(self):
        from kgcnn_torch.layers.aggr import RelationalAggregateLocalEdges
        aggr = RelationalAggregateLocalEdges(num_relations=3, pooling_method="sum")
        edges = torch.randn(8, 4)
        edge_index = torch.randint(0, 5, (2, 8))
        edge_relation = torch.randint(0, 3, (8,))
        out = aggr(edges, edge_index, edge_relation, num_nodes=5)
        self.assertEqual(out.shape, (5, 3, 4))


class TestNewGeom(unittest.TestCase):

    def test_node_distance_euclidean(self):
        from kgcnn_torch.layers.geom import NodeDistanceEuclidean
        layer = NodeDistanceEuclidean()
        pos1 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        pos2 = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        out = layer(pos1, pos2)
        self.assertEqual(out.shape, (2, 1))
        self.assertTrue(torch.allclose(out[0], torch.tensor([1.0]), atol=1e-3))

    def test_edge_direction_normalized(self):
        from kgcnn_torch.layers.geom import EdgeDirectionNormalized
        layer = EdgeDirectionNormalized()
        pos1 = torch.tensor([[3.0, 0.0, 0.0]])
        pos2 = torch.tensor([[0.0, 0.0, 0.0]])
        out = layer(pos1, pos2)
        self.assertEqual(out.shape, (1, 3))
        self.assertTrue(torch.allclose(out, torch.tensor([[1.0, 0.0, 0.0]]), atol=1e-3))

    def test_vector_angle(self):
        from kgcnn_torch.layers.geom import VectorAngle
        layer = VectorAngle()
        v1 = torch.tensor([[1.0, 0.0, 0.0]])
        v2 = torch.tensor([[0.0, 1.0, 0.0]])
        out = layer(v1, v2)
        self.assertEqual(out.shape, (1, 1))
        self.assertTrue(torch.allclose(out, torch.tensor([[torch.pi / 2]]), atol=1e-3))

    def test_cos_cutoff_applied(self):
        from kgcnn_torch.layers.geom import CosCutOff
        layer = CosCutOff(cutoff=5.0)
        x = torch.tensor([[1.0], [3.0], [5.0], [7.0]])
        out = layer(x)
        self.assertEqual(out.shape, (4, 1))
        # At cutoff distance, envelope should be ~0, so output ~0
        self.assertTrue(out[2].abs().item() < 0.01)

    def test_displacement_vectors_unit_cell(self):
        from kgcnn_torch.layers.geom import DisplacementVectorsUnitCell
        layer = DisplacementVectorsUnitCell()
        frac_coords = torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]])
        edge_index = torch.tensor([[0, 1], [1, 2]])  # src -> tgt
        cell_trans = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        out = layer(frac_coords, edge_index, cell_trans)
        self.assertEqual(out.shape, (2, 3))

    def test_frac_to_real_coordinates(self):
        from kgcnn_torch.layers.geom import FracToRealCoordinates
        layer = FracToRealCoordinates()
        frac = torch.tensor([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0]])
        lattice = torch.eye(3).unsqueeze(0) * 10.0  # 10 Angstrom cubic cell
        batch = torch.tensor([0, 0])
        out = layer(frac, lattice, batch)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(torch.allclose(out[0], torch.tensor([5.0, 0.0, 0.0]), atol=1e-3))

    def test_real_to_frac_coordinates(self):
        from kgcnn_torch.layers.geom import RealToFracCoordinates
        layer = RealToFracCoordinates()
        real = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
        lattice = torch.eye(3).unsqueeze(0) * 10.0
        batch = torch.tensor([0, 0])
        out = layer(real, lattice, batch)
        self.assertEqual(out.shape, (2, 3))
        self.assertTrue(torch.allclose(out[0], torch.tensor([0.5, 0.0, 0.0]), atol=1e-3))


class TestNewNorm(unittest.TestCase):

    def test_graph_instance_normalization(self):
        from kgcnn_torch.layers.norm import GraphInstanceNormalization
        norm = GraphInstanceNormalization(num_features=8)
        x = torch.randn(10, 8)
        batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        out = norm(x, batch)
        self.assertEqual(out.shape, (10, 8))


class TestNewScale(unittest.TestCase):

    def test_qm_graph_label_scaler(self):
        import numpy as np
        from kgcnn_torch.layers.scale import (
            QMGraphLabelScaler, StandardLabelScaler, ExtensiveMolecularLabelScaler
        )
        s1 = StandardLabelScaler()
        s1.fit(np.random.randn(100))
        s2 = StandardLabelScaler()
        s2.fit(np.random.randn(100) * 3 + 5)
        qm_scaler = QMGraphLabelScaler(scaler_list=[s1, s2])
        y = torch.randn(10, 2)
        scaled = qm_scaler.transform(y)
        back = qm_scaler.inverse_transform(scaled)
        self.assertTrue(torch.allclose(y, back, atol=1e-4))

    def test_get_scaler(self):
        from kgcnn_torch.layers.scale import get_scaler
        s = get_scaler("standard")
        self.assertIsNotNone(s)


class TestRelationalDense(unittest.TestCase):

    def test_basic(self):
        from kgcnn_torch.layers.relational import RelationalDense
        layer = RelationalDense(8, 4, num_relations=3)
        x = torch.randn(10, 8)
        r = torch.randint(0, 3, (10,))
        out = layer(x, r)
        self.assertEqual(out.shape, (10, 4))

    def test_basis_decomposition(self):
        from kgcnn_torch.layers.relational import RelationalDense
        layer = RelationalDense(8, 4, num_relations=5, num_bases=2)
        x = torch.randn(10, 8)
        r = torch.randint(0, 5, (10,))
        out = layer(x, r)
        self.assertEqual(out.shape, (10, 4))

    def test_block_diagonal(self):
        from kgcnn_torch.layers.relational import RelationalDense
        layer = RelationalDense(8, 4, num_relations=3, num_blocks=2)
        x = torch.randn(10, 8)
        r = torch.randint(0, 3, (10,))
        out = layer(x, r)
        self.assertEqual(out.shape, (10, 4))


class TestRelationalMLP(unittest.TestCase):

    def test_forward(self):
        from kgcnn_torch.layers.mlp import RelationalMLP
        mlp = RelationalMLP(units=[16, 8], input_dim=4, num_relations=3)
        x = torch.randn(10, 4)
        r = torch.randint(0, 3, (10,))
        out = mlp(x, r)
        self.assertEqual(out.shape, (10, 8))


class TestMessagePassing(unittest.TestCase):

    def test_mat_mul_messages(self):
        from kgcnn_torch.layers.message import MatMulMessages
        layer = MatMulMessages()
        # mat: (M, F', F), edges: (M, F) -> output: (M, F')
        mat = torch.randn(10, 8, 4)
        edges = torch.randn(10, 4)
        out = layer(mat, edges)
        self.assertEqual(out.shape, (10, 8))


if __name__ == '__main__':
    unittest.main()
