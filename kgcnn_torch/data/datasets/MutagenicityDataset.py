"""Mutagenicity dataset for kgcnn-torch.

Chemical compound dataset of drugs categorized into mutagen and non-mutagen classes.
From TUDatasets.

References:
    (1) Riesen, K. and Bunke, H. SSPR&SPR 2008, LNCS, vol. 5342, pp. 287-297, 2008.
"""
import numpy as np
from kgcnn_torch.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class MutagenicityDataset(GraphTUDataset2020):
    """Mutagenicity dataset: drug compounds classified as mutagen/non-mutagen."""

    dataset_name = "Mutagenicity"
    _tu_name = "Mutagenicity"
    label_names = ["mutagenicity"]
    label_units = [""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(tu_name="Mutagenicity", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Read TU data with full post-processing matching Keras version.

        Includes node label to atomic number mapping, removal of unconnected atoms
        (except Na/Li/K/Ca ions), edge index remapping, and bond order processing.
        """
        super().kgcnn_prepare(kgcnn_ds)

        # Node label index -> atomic number mapping (14 elements)
        node_translate = np.array([6, 8, 17, 1, 7, 9, 35, 16, 15, 53, 11, 19, 3, 20], dtype="int")
        atoms_translate = ['C', 'O', 'Cl', 'H', 'N', 'F', 'Br', 'S', 'P', 'I', 'Na', 'ksb', 'Li', 'Ca']
        z_translate = {node_translate[i]: atoms_translate[i] for i in range(len(node_translate))}

        edge_indices = kgcnn_ds.obtain_property("edge_indices")
        node_labels = kgcnn_ds.obtain_property("node_labels")
        edge_labels = kgcnn_ds.obtain_property("edge_labels")
        graph_labels = kgcnn_ds.obtain_property("graph_labels")

        # Translate node labels to atomic numbers
        nodes = [node_translate[np.array(x, dtype="int")][:, 0] for x in node_labels]
        atoms = [[atoms_translate[int(y[0])] for y in x] for x in node_labels]
        # Edge labels: take first column + 1 (bond order)
        edges = [x[:, 0] + 1 for x in edge_labels]
        labels = graph_labels

        # Clean: remove unconnected atoms (except Na, Li, K, Ca ions)
        labels_clean = []
        nodes_clean = []
        edge_indices_clean = []
        edges_clean = []
        atoms_clean = []

        for i in range(len(nodes)):
            nats = nodes[i]
            cons = np.arange(len(nodes[i]))
            test_cons = np.sort(np.unique(edge_indices[i].flatten()))
            is_cons = np.zeros_like(cons, dtype="bool")
            is_cons[test_cons] = True
            is_cons[nats == 20] = True  # Ca: allow unconnected
            is_cons[nats == 3] = True   # Li: allow unconnected
            is_cons[nats == 19] = True  # K:  allow unconnected
            is_cons[nats == 11] = True  # Na: allow unconnected
            if np.sum(is_cons) != len(cons):
                # Remove unconnected atoms and remap edge indices
                nodes_clean.append(nats[is_cons])
                atoms_clean.append([atoms[i][j] for j in range(len(is_cons)) if is_cons[j]])
                indices_used = cons[is_cons]
                indices_new = np.arange(len(indices_used))
                indices_old = np.zeros(len(nodes[i]), dtype="int")
                indices_old[indices_used] = indices_new
                edge_idx_new = indices_old[edge_indices[i]]
                edge_indices_clean.append(edge_idx_new)
            else:
                nodes_clean.append(nats)
                atoms_clean.append(atoms[i])
                edge_indices_clean.append(edge_indices[i])
            edges_clean.append(edges[i])
            labels_clean.append(labels[i])

        kgcnn_ds.assign_property("graph_labels", labels_clean)
        kgcnn_ds.assign_property("edge_indices", edge_indices_clean)
        kgcnn_ds.assign_property("node_attributes", nodes_clean)
        kgcnn_ds.assign_property("edge_attributes", edges_clean)
        kgcnn_ds.assign_property("node_labels", nodes_clean)
        kgcnn_ds.assign_property("edge_labels", edges_clean)
        kgcnn_ds.assign_property("node_symbol", atoms_clean)
        kgcnn_ds.assign_property("node_number", nodes_clean)
        kgcnn_ds.assign_property("graph_size", [len(x) for x in nodes_clean])
