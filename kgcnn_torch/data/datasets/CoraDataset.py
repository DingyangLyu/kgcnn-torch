"""Cora (full) dataset for kgcnn-torch.

Full Cora dataset loaded from graph2gauss repository. Nodes represent documents,
edges represent citation links. Node classification task.

References:
    (1) Bojchevski, A. and Guennemann, S., Deep Gaussian Embedding of Graphs: Unsupervised
        Inductive Learning via Ranking, arXiv:1707.03815, 2017.
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class CoraDataset(KgcnnGraphDataset):
    """Cora (full) dataset: citation network for node classification."""

    dataset_name = "Cora"
    download_info = {
        "dataset_name": "Cora",
        "data_directory_name": "cora",
        "download_url": "https://github.com/abojchevski/graph2gauss/raw/master/data/cora.npz",
        "download_file_name": "cora.npz",
    }
    label_names = ["node_class"]
    label_units = [""]
    file_name = "cora.npz"

    def kgcnn_prepare(self, kgcnn_ds):
        """Load Cora from sparse NPZ file."""
        from scipy import sparse

        file_path = os.path.join(self.raw_dir, "cora.npz")
        data = np.load(file_path, allow_pickle=True)

        # Reconstruct adjacency matrix
        adj = sparse.csr_matrix(
            (data["adj_data"], data["adj_indices"], data["adj_indptr"]),
            shape=data["adj_shape"])
        # Reconstruct attribute matrix
        attr = sparse.csr_matrix(
            (data["attr_data"], data["attr_indices"], data["attr_indptr"]),
            shape=data["attr_shape"])

        labels = data["labels"]
        node_attributes = np.array(attr.todense(), dtype="float32")
        node_labels = np.eye(max(labels) + 1, dtype="float32")[labels]

        # Convert adjacency to edge list
        adj_coo = adj.tocoo()
        edge_indices = np.stack([adj_coo.row, adj_coo.col], axis=-1).astype("int")
        edge_weights = np.array(adj_coo.data, dtype="float32").reshape(-1, 1)

        # Single graph dataset
        kgcnn_ds.assign_property("node_attributes", [node_attributes])
        kgcnn_ds.assign_property("node_labels", [node_labels])
        kgcnn_ds.assign_property("edge_indices", [edge_indices])
        kgcnn_ds.assign_property("edge_weights", [edge_weights])
