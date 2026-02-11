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
        """Load Cora from sparse NPZ file.

        Uses fixed 70-column one-hot labels and scaled adjacency edge weights
        matching Keras version.
        """
        from scipy import sparse
        from kgcnn_torch.graph.methods import convert_scaled_adjacency_to_list

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

        # Labels: fixed 70 columns matching Keras
        labels = data["labels"]
        labels = np.expand_dims(labels, axis=-1)
        labels = np.array(labels == np.arange(70), dtype="float")

        node_attributes = np.array(attr.todense(), dtype="float32")

        # Edge indices and weights via scaled adjacency (matching Keras)
        edi, ed = convert_scaled_adjacency_to_list(adj)

        # Single graph dataset
        kgcnn_ds.assign_property("node_attributes", [node_attributes])
        kgcnn_ds.assign_property("node_labels", [labels])
        kgcnn_ds.assign_property("edge_indices", [edi])
        kgcnn_ds.assign_property("edge_attributes", [np.expand_dims(ed, axis=-1)])
        kgcnn_ds.assign_property("edge_weights", [np.expand_dims(ed, axis=-1)])
