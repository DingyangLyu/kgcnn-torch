"""MUTAG dataset for kgcnn-torch.

Collection of 188 nitroaromatic compounds with mutagenicity labels on Salmonella typhimurium.
7 discrete node labels representing atom types. Graph classification task.

References:
    (1) Debnath, A.K. et al. J. Med. Chem. 34(2):786-797, 1991.
    (2) N. Kriege, P. Mutzel. ICML 2012.
"""
import numpy as np
from kgcnn_torch.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class MUTAGDataset(GraphTUDataset2020):
    """MUTAG dataset: 188 chemical compounds with mutagenicity labels."""

    dataset_name = "MUTAG"
    _tu_name = "MUTAG"
    label_names = ["mutagenicity"]
    label_units = [""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(tu_name="MUTAG", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Read TU data and translate node labels to atom types, matching Keras version."""
        super().kgcnn_prepare(kgcnn_ds)

        node_translate = np.array([6, 7, 8, 9, 53, 17, 35], dtype="int")
        atoms_translate = ['C', 'N', 'O', 'F', 'I', 'Cl', 'Br']

        node_labels = kgcnn_ds.obtain_property("node_labels")
        edge_labels = kgcnn_ds.obtain_property("edge_labels")
        graph_labels = kgcnn_ds.obtain_property("graph_labels")

        # Node labels -> atomic numbers (1D shape matching Keras)
        node_attributes = [node_translate[np.array(x, dtype="int")][:, 0] for x in node_labels]
        # Node labels -> atom symbols
        atoms = [[atoms_translate[int(y)] for y in x] for x in node_labels]

        # Graph labels: clip negative to 0
        graph_labels = np.array(graph_labels)
        graph_labels[graph_labels < 0] = 0

        kgcnn_ds.assign_property("node_attributes", node_attributes)
        # Edge attributes: take first column only (matching Keras)
        kgcnn_ds.assign_property("edge_attributes", [x[:, 0] for x in edge_labels])
        kgcnn_ds.assign_property("node_symbol", atoms)
        kgcnn_ds.assign_property("node_number", node_attributes)
        kgcnn_ds.assign_property("graph_labels", [x for x in graph_labels])
        kgcnn_ds.assign_property("graph_size", [len(x) for x in node_attributes])
