"""Tests for all models - forward and backward pass."""
import torch
import unittest
from types import SimpleNamespace


def _make_data(N=20, M=40, K=60):
    """Create random test data."""
    return SimpleNamespace(
        z=torch.randint(0, 10, (N,)),
        x=torch.randint(0, 10, (N,)),
        pos=torch.randn(N, 3),
        edge_index=torch.randint(0, N, (2, M)),
        edge_weight=torch.randn(M, 1),
        edge_attr=torch.randn(M, 8),
        angle_index=torch.randint(0, M, (2, K)),
        batch=torch.cat([torch.zeros(N // 2, dtype=torch.long),
                         torch.ones(N - N // 2, dtype=torch.long)])
    )


class TestGCNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.gcn import GCNModel
        model = GCNModel(node_dim=16, depth=2, gcn_units=16, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()
        self.assertIsNotNone(model.dense_in.weight.grad)

    def test_different_depths(self):
        from kgcnn_torch.models.gcn import GCNModel
        for depth in [1, 3, 5]:
            model = GCNModel(node_dim=16, depth=depth, gcn_units=16, num_targets=3)
            data = _make_data()
            out = model(data)
            self.assertEqual(out.shape, (2, 3))

    def test_without_edge_weight(self):
        from kgcnn_torch.models.gcn import GCNModel
        model = GCNModel(node_dim=16, depth=2, gcn_units=16, num_targets=1)
        data = _make_data()
        delattr(data, "edge_weight")
        out = model(data)
        self.assertEqual(out.shape, (2, 1))


class TestGATModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.gat import GATModel
        model = GATModel(node_dim=16, depth=2, attention_units=8,
                         attention_heads_num=2, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_concat_vs_average(self):
        from kgcnn_torch.models.gat import GATModel
        for concat in [True, False]:
            model = GATModel(node_dim=16, depth=1, attention_units=8,
                             attention_heads_num=2, attention_heads_concat=concat,
                             num_targets=1)
            data = _make_data()
            out = model(data)
            self.assertEqual(out.shape, (2, 1))

    def test_without_edge_attr(self):
        from kgcnn_torch.models.gat import GATModel
        model = GATModel(node_dim=16, depth=1, attention_units=8,
                         attention_heads_num=2, num_targets=1,
                         use_edge_features=True, edge_dim=8)
        data = _make_data()
        delattr(data, "edge_attr")
        out = model(data)
        self.assertEqual(out.shape, (2, 1))


class TestSchNetModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.schnet import SchNetModel
        model = SchNetModel(node_dim=32, depth=2, units=32,
                            edge_dim=10, gauss_bins=10, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_no_output_mlp(self):
        from kgcnn_torch.models.schnet import SchNetModel
        model = SchNetModel(node_dim=32, depth=1, units=32,
                            edge_dim=10, gauss_bins=10, num_targets=1,
                            use_output_mlp=False)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape[0], 2)


class TestPAiNNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.painn import PAiNNModel
        model = PAiNNModel(node_dim=32, depth=2, units=32,
                           num_radial=8, cutoff=5.0, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_with_normalization(self):
        from kgcnn_torch.models.painn import PAiNNModel
        model = PAiNNModel(node_dim=32, depth=2, units=32,
                           num_radial=8, cutoff=5.0, num_targets=1,
                           equiv_normalization=True, node_normalization=True)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))


class TestDimeNetPPModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.dimenetpp import DimeNetPPModel
        model = DimeNetPPModel(
            emb_size=16, out_emb_size=16, int_emb_size=8, basis_emb_size=4,
            num_blocks=1, num_spherical=3, num_radial=4,
            cutoff=5.0, num_targets=1, num_before_skip=1, num_after_skip=1,
            num_dense_output=1
        )
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_multiple_targets(self):
        from kgcnn_torch.models.dimenetpp import DimeNetPPModel
        model = DimeNetPPModel(
            emb_size=16, out_emb_size=16, int_emb_size=8, basis_emb_size=4,
            num_blocks=1, num_spherical=3, num_radial=4,
            cutoff=5.0, num_targets=5, num_before_skip=1, num_after_skip=1,
            num_dense_output=1
        )
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 5))

    def test_invalid_output_embedding_raises(self):
        from kgcnn_torch.models.dimenetpp import DimeNetPPModel
        with self.assertRaises(ValueError):
            DimeNetPPModel(output_embedding="node")


class TestTrainingRegistry(unittest.TestCase):

    def test_megnet_registry_resolves(self):
        from training_scripts.train_graph import get_model_class
        from kgcnn_torch.models.megnet import MEGNetModel
        model_cls = get_model_class("Megnet")
        self.assertIs(model_cls, MEGNetModel)

    def test_count_model_parameters_handles_lazy(self):
        from training_scripts.train_graph import count_model_parameters
        from kgcnn_torch.models.gat import GATModel
        model = GATModel(node_dim=16, depth=1, attention_units=8, attention_heads_num=2, num_targets=1)
        n_params, n_uninitialized = count_model_parameters(model)
        self.assertGreaterEqual(n_uninitialized, 0)
        self.assertGreaterEqual(n_params, 0)


def _make_data_with_edge_pair(N=20, M=40):
    """Create test data with edge_pair_index for directed message passing models."""
    # Create symmetric edges: for each (i,j) also add (j,i)
    src = torch.randint(0, N, (M // 2,))
    tgt = torch.randint(0, N, (M // 2,))
    edge_index = torch.stack([
        torch.cat([src, tgt]),
        torch.cat([tgt, src])
    ])
    M_actual = edge_index.shape[1]
    # edge_pair_index: edge i's reverse is at position i + M//2 (or i - M//2)
    half = M_actual // 2
    edge_pair_index = torch.cat([torch.arange(half, M_actual), torch.arange(0, half)])
    return SimpleNamespace(
        z=torch.randint(0, 10, (N,)),
        x=torch.randint(0, 10, (N,)),
        pos=torch.randn(N, 3),
        edge_index=edge_index,
        edge_attr=torch.randn(M_actual, 14),
        edge_pair_index=edge_pair_index,
        batch=torch.cat([torch.zeros(N // 2, dtype=torch.long),
                         torch.ones(N - N // 2, dtype=torch.long)])
    )


def _make_data_with_edge_type(N=20, M=40, num_relations=4):
    """Create test data with edge_type for relational models."""
    data = _make_data(N, M)
    data.edge_type = torch.randint(0, num_relations, (M,))
    return data


def _make_data_with_angle(N=20, M=40, K=60):
    """Create test data with angle_index for angular models."""
    data = _make_data(N, M, K)
    # angle_index should be (3, K) with [center, neighbor1, neighbor2]
    data.angle_index = torch.stack([
        torch.randint(0, N, (K,)),
        torch.randint(0, N, (K,)),
        torch.randint(0, N, (K,)),
    ])
    return data


# =========================================================================
# New model tests (Phase 3 models)
# =========================================================================

class TestGATv2Model(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.gatv2 import GATv2Model
        model = GATv2Model(node_dim=16, depth=2, attention_units=8,
                           attention_heads_num=2, num_targets=1, edge_dim=8)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestGINModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.gin import GINModel
        model = GINModel(node_dim=16, depth=3, units=16, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_with_edge_features(self):
        from kgcnn_torch.models.gin import GINModel
        model = GINModel(node_dim=16, depth=2, units=16,
                         use_edge_features=True, edge_dim=8, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))


class TestEGNNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.egnn import EGNNModel
        model = EGNNModel(node_dim=16, depth=2, units=16, num_targets=1,
                          edge_attr_dim=8)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_without_edge_attr(self):
        from kgcnn_torch.models.egnn import EGNNModel
        model = EGNNModel(node_dim=16, depth=2, units=16, num_targets=1,
                          use_edge_attr=False)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_with_normalization(self):
        from kgcnn_torch.models.egnn import EGNNModel
        model = EGNNModel(node_dim=16, depth=2, units=16, num_targets=1,
                          use_edge_attr=False, use_node_normalization=True)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_with_attention(self):
        from kgcnn_torch.models.egnn import EGNNModel
        model = EGNNModel(node_dim=16, depth=2, units=16, num_targets=1,
                          use_edge_attr=False, use_attention=True)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestDMPNNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.dmpnn import DMPNNModel
        model = DMPNNModel(node_dim=16, edge_dim=14, depth=2, units=16,
                           num_targets=1)
        data = _make_data_with_edge_pair()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestGraphSAGEModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.graphsage import GraphSAGEModel
        model = GraphSAGEModel(node_dim=16, depth=2, units=16, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestMEGNetModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.megnet import MEGNetModel
        model = MEGNetModel(node_dim=16, edge_dim=16, state_dim=8,
                            edge_input_dim=8, depth=2, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestAttentiveFPModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.attentivefp import AttentiveFPModel
        model = AttentiveFPModel(node_dim=16, depth_ato=2, depth_mol=2,
                                 units=16, edge_dim=8, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestCGCNNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.cgcnn import CGCNNModel
        model = CGCNNModel(node_dim=16, depth=2, gauss_bins=10,
                           num_targets=1)
        data = _make_data()
        # CGCNN expects edge_attr as distances (M, 1) for Gaussian expansion
        data.edge_attr = torch.randn(data.edge_index.shape[1], 1).abs()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()

    def test_without_batch_norm(self):
        from kgcnn_torch.models.cgcnn import CGCNNModel
        model = CGCNNModel(node_dim=16, depth=2, gauss_bins=10,
                           num_targets=1, batch_normalization=False)
        data = _make_data()
        data.edge_attr = torch.randn(data.edge_index.shape[1], 1).abs()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestNMPNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.nmpn import NMPNModel
        model = NMPNModel(node_dim=16, depth=2, units=16,
                          edge_dim=8, num_targets=1, use_set2set=False)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestINorpModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.inorp import INorpModel
        model = INorpModel(node_dim=16, depth=2,
                           edge_dim=8, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestMEGANModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.megan import MEGANModel
        model = MEGANModel(node_dim=16, num_heads=2, depth=2,
                           importance_channels=2, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestRGCNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.rgcn import RGCNModel
        model = RGCNModel(node_dim=16, depth=2, units=16,
                          num_relations=4, num_targets=1)
        data = _make_data_with_edge_type(num_relations=4)
        # RGCN uses edge_attr as multiplicative weights on messages;
        # remove to test without edge weights (or match dims)
        del data.edge_attr
        data.edge_attr = None
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestGNNFilmModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.gnnfilm import GNNFilmModel
        model = GNNFilmModel(node_dim=16, depth=2, units=16,
                             num_relations=4, num_targets=1)
        data = _make_data_with_edge_type(num_relations=4)
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestrGINModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.rgin import rGINModel
        model = rGINModel(node_dim=16, depth=2, units=16, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestMXMNetModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.mxmnet import MXMNetModel
        model = MXMNetModel(node_dim=16, depth=2, units=16,
                            num_radial=8, num_spherical=3,
                            cutoff=5.0, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestMoGATModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.mogat import MoGATModel
        model = MoGATModel(node_dim=16, depthato=2, depthmol=2,
                           units=16, edge_dim=8, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestCMPNNModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.cmpnn import CMPNNModel
        model = CMPNNModel(node_dim=16, edge_dim=14, depth=2,
                           units=16, num_targets=1)
        data = _make_data_with_edge_pair()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestDGINModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.dgin import DGINModel
        model = DGINModel(node_dim=16, edge_dim=14, depth_dmpnn=2,
                          depth_gin=2, units=16, num_targets=1)
        data = _make_data_with_edge_pair()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestHamNetModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.hamnet import HamNetModel
        model = HamNetModel(node_dim=16, edge_dim=8, depth=2,
                            units=16, fingerprint_dim=16,
                            fingerprint_depth=2, num_targets=1)
        data = _make_data()
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestMATModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.mat import MATModel
        B, N_max = 2, 10
        model = MATModel(embedding_units=16, depth=2, num_heads=2,
                         units_ff=16, num_targets=1, input_node_dim=16,
                         use_node_embedding=False)
        node_input = torch.randn(B, N_max, 16)
        xyz_input = torch.randn(B, N_max, 3)
        adjacency = torch.ones(B, N_max, N_max)
        node_mask = torch.ones(B, N_max)
        # Mask out last 3 nodes in second graph
        node_mask[1, 7:] = 0
        adj_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
        out = model(node_input, xyz_input, adjacency, node_mask, adj_mask)
        self.assertEqual(out.shape, (B, 1))
        out.sum().backward()


class TestHDNNP2ndModel(unittest.TestCase):

    def test_forward_backward(self):
        from kgcnn_torch.models.hdnnp2nd import HDNNP2ndModel
        model = HDNNP2ndModel(element_types=[1, 6, 7, 8],
                              n_rad_features=8, n_ang_features=4,
                              cutoff=5.0, relational_units=[10, 10, 1],
                              num_targets=1)
        data = _make_data_with_angle(N=20, M=40, K=30)
        # Use only elements from the allowed types
        data.z = torch.tensor([1, 6, 7, 8] * 5)
        out = model(data)
        self.assertEqual(out.shape, (2, 1))
        out.sum().backward()


class TestGNNExplainer(unittest.TestCase):

    def test_explain_with_gcn(self):
        """Test GNNExplainer with a simple GCN model."""
        from kgcnn_torch.models.gcn import GCNModel
        from kgcnn_torch.models.gnnexplain import GNNInterface, GNNExplainer

        # Build a simple GCN
        model = GCNModel(node_dim=16, depth=2, gcn_units=16, num_targets=1)
        model.eval()

        class GCNWrapper(GNNInterface):
            def __init__(self, m):
                self.m = m

            def predict(self, data, **kwargs):
                with torch.no_grad():
                    return self.m(data)

            def masked_predict(self, data, edge_mask, feature_mask, node_mask, **kwargs):
                # Simple masking: scale edge messages by edge_mask
                import copy
                d = copy.copy(data)
                d.edge_weight = edge_mask  # (M, 1) shape expected by GCNConv
                return self.m(d)

            def get_number_of_nodes(self, data):
                return data.z.shape[0]

            def get_number_of_edges(self, data):
                return data.edge_index.shape[1]

            def get_number_of_node_features(self, data):
                return 16  # embedding dim

            def get_explanation(self, data, edge_mask, feature_mask, node_mask, **kwargs):
                return {"edge_mask": edge_mask, "feature_mask": feature_mask, "node_mask": node_mask}

        data = _make_data(N=10, M=20, K=30)
        wrapper = GCNWrapper(model)
        explainer = GNNExplainer(wrapper, lr=0.01, epochs=10)

        # Test explain without error
        explainer.explain(data)

        # Test get_masks
        masks = explainer.get_masks()
        self.assertEqual(masks["edge"].shape, (20, 1))
        self.assertEqual(masks["feature"].shape, (16, 1))
        self.assertEqual(masks["node"].shape, (10, 1))

        # Masks should be in [0, 1]
        for name, mask in masks.items():
            self.assertTrue(torch.all(mask >= 0) and torch.all(mask <= 1),
                            f"{name} mask values out of [0,1] range")

        # Test get_explanation
        explanation = explainer.get_explanation()
        self.assertIn("edge_mask", explanation)
        self.assertIn("feature_mask", explanation)
        self.assertIn("node_mask", explanation)

    def test_explain_with_inspection(self):
        """Test GNNExplainer inspection mode."""
        from kgcnn_torch.models.gcn import GCNModel
        from kgcnn_torch.models.gnnexplain import GNNInterface, GNNExplainer

        model = GCNModel(node_dim=16, depth=1, gcn_units=16, num_targets=1)
        model.eval()

        class GCNWrapper(GNNInterface):
            def __init__(self, m):
                self.m = m

            def predict(self, data, **kwargs):
                with torch.no_grad():
                    return self.m(data)

            def masked_predict(self, data, edge_mask, feature_mask, node_mask, **kwargs):
                return self.m(data)

            def get_number_of_nodes(self, data):
                return data.z.shape[0]

            def get_number_of_edges(self, data):
                return data.edge_index.shape[1]

            def get_number_of_node_features(self, data):
                return 16

            def get_explanation(self, data, edge_mask, feature_mask, node_mask, **kwargs):
                return {}

        data = _make_data(N=8, M=16, K=20)
        wrapper = GCNWrapper(model)
        explainer = GNNExplainer(wrapper, epochs=5)

        history = explainer.explain(data, inspection=True)
        self.assertIsNotNone(history)
        self.assertEqual(len(history["total_loss"]), 5)
        self.assertEqual(len(history["predictions"]), 5)
        # Loss should generally decrease (or at least not crash)
        self.assertGreater(len(history["edge_mask_loss"]), 0)
        self.assertGreater(len(history["feature_mask_loss"]), 0)

    def test_error_before_explain(self):
        """Test that get_explanation raises before explain is called."""
        from kgcnn_torch.models.gnnexplain import GNNExplainer, GNNInterface

        class DummyGNN(GNNInterface):
            pass

        explainer = GNNExplainer(DummyGNN())
        with self.assertRaises(RuntimeError):
            explainer.get_explanation()
        with self.assertRaises(RuntimeError):
            explainer.get_masks()


if __name__ == '__main__':
    unittest.main()
