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
        """Load Cora-Lu from text files."""
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
        node_ids = []
        features = []
        labels = []
        with open(content_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                node_ids.append(int(parts[0]))
                features.append([int(x) for x in parts[1:-1]])
                labels.append(parts[-1])

        # Map node IDs to sequential indices
        id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

        # Read citation edges
        edges = []
        with open(cites_file, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) == 2:
                    src, tgt = int(parts[0]), int(parts[1])
                    if src in id_to_idx and tgt in id_to_idx:
                        edges.append([id_to_idx[tgt], id_to_idx[src]])

        node_attributes = np.array(features, dtype="float32")
        edge_indices = np.array(edges, dtype="int") if edges else np.zeros((0, 2), dtype="int")

        # Encode labels
        num_classes = len(self.class_label_mapping)
        node_labels = np.zeros((len(labels), num_classes), dtype="float32")
        for i, lab in enumerate(labels):
            if lab in self.class_label_mapping:
                node_labels[i, self.class_label_mapping[lab]] = 1.0

        # Single graph dataset
        kgcnn_ds.assign_property("node_attributes", [node_attributes])
        kgcnn_ds.assign_property("node_labels", [node_labels])
        kgcnn_ds.assign_property("edge_indices", [edge_indices])
