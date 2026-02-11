"""PROTEINS dataset for kgcnn-torch.

Proteins classified as enzymes or non-enzymes. Nodes represent amino acids;
edges connect amino acids less than 6 Angstroms apart.

References:
    (1) K.M. Borgwardt et al. Bioinformatics, 21(Suppl 1):i47-i56, 2005.
    (2) P.D. Dobson, A.J. Doig. J. Mol. Biol., 330(4):771-783, 2003.
"""
import numpy as np
from kgcnn_torch.data.datasets.GraphTUDataset2020 import GraphTUDataset2020
from kgcnn_torch.molecule.encoder import OneHotEncoder


class PROTEINSDataset(GraphTUDataset2020):
    """PROTEINS dataset: protein graphs classified as enzyme/non-enzyme."""

    dataset_name = "PROTEINS"
    _tu_name = "PROTEINS"
    label_names = ["enzyme"]
    label_units = [""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(tu_name="PROTEINS", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Read TU data with one-hot encoding and label offset matching Keras version."""
        super().kgcnn_prepare(kgcnn_ds)

        # One-hot encoders matching Keras
        ohe = OneHotEncoder(
            [-538, -345, -344, -134, -125, -96, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
             21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 41, 42, 47, 61, 63, 73, 74, 75,
             82, 104, 353, 355, 360, 558, 797, 798], add_unknown=False)
        ohe2 = OneHotEncoder([0, 1, 2], add_unknown=False)
        ohe3 = OneHotEncoder([i for i in range(0, 17)], add_unknown=False)

        graph_labels = kgcnn_ds.obtain_property("graph_labels")
        node_attributes = kgcnn_ds.obtain_property("node_attributes")
        node_labels = kgcnn_ds.obtain_property("node_labels")
        node_degree = kgcnn_ds.obtain_property("node_degree")

        # graph_labels: -1 offset (convert 1-indexed to 0-indexed)
        kgcnn_ds.assign_property("graph_labels", [x - 1 for x in graph_labels])
        # node_attributes: one-hot encode (59 dims)
        kgcnn_ds.assign_property("node_attributes",
                                 [np.array([ohe(int(y)) for y in x]) for x in node_attributes])
        # node_labels: one-hot encode (3 dims)
        kgcnn_ds.assign_property("node_labels",
                                 [np.array([ohe2(int(y)) for y in x]) for x in node_labels])
        # node_degree: one-hot encode (17 dims)
        kgcnn_ds.assign_property("node_degree",
                                 [np.array([ohe3(int(y)) for y in x]) for x in node_degree])
        # graph_size
        kgcnn_ds.assign_property("graph_size",
                                 [len(x) if x is not None else None for x in node_attributes])
