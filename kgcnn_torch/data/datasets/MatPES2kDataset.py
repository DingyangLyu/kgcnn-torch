"""MatPES2k dataset for kgcnn-torch.

Local dataset wrapper for the converted MatPES 2k subset pickle.
Loads pre-converted graph dicts and applies key aliasing/normalization
for compatibility with all model configs, matching the Keras version.
"""
import os
import pickle
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset
from kgcnn_torch.data.base import MemoryGraphDataset, GraphDict


class MatPES2kDataset(KgcnnGraphDataset):
    """Local dataset wrapper for the converted MatPES 2k subset pickle.

    Args:
        file_path: Path to the pickle file. If None, uses the default path.
        root: Root directory for PyG processed cache.
        reload: If True, reprocess data.
    """

    dataset_name = "MatPES2k"
    label_names = "graph_labels"
    label_units = None

    def __init__(self, file_path: str = None, root=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        if file_path is None:
            file_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
                    os.path.abspath(__file__))))),
                "tmp_matpes_2k", "matpes_pbe_2k.pkl"
            )
        self.file_path_local = file_path
        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds: MemoryGraphDataset):
        """Load MatPES 2k pickle and apply key normalization.

        Mirrors the Keras MatPES2kDataset.read_in_memory() logic:
        aliases keys for compatibility with all model configs, creates
        counters, angle placeholders, and dense adjacency matrices.
        """
        if not os.path.exists(self.file_path_local):
            raise FileNotFoundError(f"Dataset file not found: {self.file_path_local}")

        with open(self.file_path_local, "rb") as f:
            in_list = pickle.load(f)

        kgcnn_ds.clear()
        for x in in_list:
            kgcnn_ds.append(GraphDict(x))

        for g in kgcnn_ds:
            # Aliases used by many literature model configs.
            if "edge_indices" in g and "range_indices" not in g:
                g["range_indices"] = np.array(g["edge_indices"], dtype=np.int64)
            if "edge_weight" in g and "range_attributes" not in g:
                g["range_attributes"] = np.array(g["edge_weight"], dtype=np.float32)
            if "edge_attributes" in g and "edge_weights" not in g:
                ew = g.get("edge_weight", None)
                if ew is not None:
                    g["edge_weights"] = np.array(ew, dtype=np.float32)
            if "edge_pair_index" in g and "edge_indices_reverse" not in g:
                g["edge_indices_reverse"] = np.expand_dims(
                    np.array(g["edge_pair_index"], dtype=np.int64), axis=-1)
            if "node_number" in g and "node_attributes" not in g:
                g["node_attributes"] = np.expand_dims(
                    np.array(g["node_number"], dtype=np.float32), axis=-1)
            if "graph_attributes" in g and "charge" not in g:
                ga = np.array(g["graph_attributes"], dtype=np.float32).reshape(-1)
                g["charge"] = ga[:1] if ga.size > 0 else np.array([0.0], dtype=np.float32)
            if "graph_attributes" in g:
                g["graph_attributes"] = np.array(
                    g["graph_attributes"], dtype=np.float32).reshape(-1)
            if "graph_lattice" in g and "range_image" not in g:
                m = len(g["edge_indices"]) if "edge_indices" in g else 0
                g["range_image"] = np.zeros((m, 3), dtype=np.int64)
            if "graph_labels" in g:
                g["graph_labels"] = np.array(
                    g["graph_labels"], dtype=np.float32).reshape(-1)
            if "node_coordinates" in g and "node_frac_coordinates" not in g:
                g["node_frac_coordinates"] = np.array(
                    g["node_coordinates"], dtype=np.float32)

            # Frequently required counters.
            n_nodes = int(len(g["node_number"])) if "node_number" in g else 0
            n_edges = int(len(g["edge_indices"])) if "edge_indices" in g else 0
            n_ranges = int(len(g["range_indices"])) if "range_indices" in g else 0
            n_angles = int(len(g["angle_indices"])) if "angle_indices" in g else 0
            n_reverse = int(len(g["edge_indices_reverse"])) if "edge_indices_reverse" in g else 0
            g.setdefault("total_nodes", np.array(n_nodes, dtype=np.int64))
            g.setdefault("total_edges", np.array(n_edges, dtype=np.int64))
            g.setdefault("total_ranges", np.array(n_ranges, dtype=np.int64))
            g.setdefault("total_angles", np.array(n_angles, dtype=np.int64))
            g.setdefault("total_reverse", np.array(n_reverse, dtype=np.int64))
            g.setdefault("graph_size", np.array(n_nodes, dtype=np.int64))

            # Placeholder entries for angle-based models.
            if "angle_indices" in g:
                ai = np.array(g["angle_indices"], dtype=np.int64)
                if ai.size == 0:
                    ai = np.array([[0, 0]], dtype=np.int64)
                g["angle_indices"] = ai
                g["total_angles"] = np.array(len(ai), dtype=np.int64)

            # MXMNet expects split angle index lists.
            if "angle_indices" in g and "angle_indices_1" not in g:
                ai = np.array(g["angle_indices"], dtype=np.int64)
                g["angle_indices_1"] = ai
                g["angle_indices_2"] = ai
                g["total_angles_1"] = np.array(len(ai), dtype=np.int64)
                g["total_angles_2"] = np.array(len(ai), dtype=np.int64)

            # HDNNP config alternate key.
            if "angle_indices" in g and "angle_indices_nodes" not in g:
                g["angle_indices_nodes"] = np.array([[0, 0, 0]], dtype=np.int64)

            # MAT expects dense adjacency + masks.
            if "adjacency_matrix" not in g:
                adj = np.zeros((n_nodes, n_nodes, 1), dtype=np.float32)
                if "edge_indices" in g:
                    ei = np.array(g["edge_indices"], dtype=np.int64)
                    if ei.size > 0:
                        adj[ei[:, 0], ei[:, 1], 0] = 1.0
                g["adjacency_matrix"] = adj
            if "node_mask" not in g:
                g["node_mask"] = np.ones((n_nodes,), dtype=np.bool_)
            if "adjacency_mask" not in g:
                g["adjacency_mask"] = np.ones((n_nodes, n_nodes), dtype=np.bool_)
