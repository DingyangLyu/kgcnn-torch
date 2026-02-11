"""Cora Lu dataset for kgcnn-torch.

Cora dataset after Lu et al. 2003. 2708 scientific publications classified into 7 classes.
5429 citation links. Each publication described by a 0/1-valued word vector (1433 unique words).

References:
    (1) McCallum, A.K. et al. Information Retrieval 3, 127-163, 2000.
    (2) Lu, Q. and Getoor, L. Link-Based Classification. ICML, 2003.
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class CoraLuDataset(KgcnnGraphDataset):
    """Cora Lu dataset: 2708 publications, 7 classes, citation network."""

    dataset_name = "cora_lu"
    download_info = {
        "dataset_name": "cora_lu",
        "data_directory_name": "cora_lu",
        "download_url": "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz",
        "download_file_name": "cora.tgz",
        "unpack_tar": True,
        "unpack_directory_name": "cora_lu",
    }
    label_names = ["node_class"]
    label_units = [""]
    class_label_mapping = {
        "Genetic_Algorithms": 0,
        "Reinforcement_Learning": 1,
        "Theory": 2,
        "Rule_Learning": 3,
        "Case_Based": 4,
        "Probabilistic_Methods": 5,
        "Neural_Networks": 6,
    }

    def kgcnn_prepare(self, kgcnn_ds):
        """Load Cora-Lu from text files.

        Includes edge sorting, edge_attributes, edge_weights, and node_number
        matching Keras version.
        """
        base_path = os.path.join(self.raw_dir, "cora_lu")

        # Find the actual directory (may be nested)
        cites_file = None
        for root, dirs, files in os.walk(base_path):
            if "cora.cites" in files:
                cites_file = os.path.join(root, "cora.cites")
                content_file = os.path.join(root, "cora.content")
                break

        if cites_file is None:
            raise FileNotFoundError("Cannot find cora.cites in extracted data.")

        # Read content file (nodes + features + label)
        lines = []
        with open(content_file, "r") as f:
            lines = f.readlines()

        labels_str = [x.strip().split('\t')[-1] for x in lines]
        nodes_raw = [x.strip().split('\t')[0:-1] for x in lines]
        nodes = np.array([[int(y) for y in x] for x in nodes_raw], dtype="int64")

        # Map node IDs to sequential indices (matching Keras)
        node_map = np.zeros(np.max(nodes[:, 0]) + 1, dtype="int64")
        idx_new = np.arange(len(nodes))
        node_map[nodes[:, 0]] = idx_new

        # Read citation edges
        ids = np.loadtxt(cites_file, dtype="int64")
        indexlist = node_map[ids]

        # Sort edges with stable mergesort (matching Keras)
        order1 = np.argsort(indexlist[:, 1], axis=0, kind='mergesort')  # stable!
        ind1 = indexlist[order1]
        order2 = np.argsort(ind1[:, 0], axis=0, kind='mergesort')
        indices = ind1[order2]

        # Class label encoding
        label_id = np.array([self.class_label_mapping[x] for x in labels_str], dtype="int")
        label_onehot = np.expand_dims(label_id, axis=-1)
        label_onehot = np.array(label_onehot == np.arange(7), dtype="float")

        node_attributes = nodes[:, 1:].astype("float32")

        # Single graph dataset (matching Keras properties)
        kgcnn_ds.assign_property("node_attributes", [node_attributes])
        kgcnn_ds.assign_property("edge_indices", [indices])
        kgcnn_ds.assign_property("edge_attributes", [np.ones_like(indices)[:, :1]])
        kgcnn_ds.assign_property("node_labels", [label_onehot])
        kgcnn_ds.assign_property("node_number", [label_id])
        kgcnn_ds.assign_property("edge_weights", [np.ones_like(indices)[:, :1]])
