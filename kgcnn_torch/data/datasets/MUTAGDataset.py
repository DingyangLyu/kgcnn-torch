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

    # MUTAG node label to proton number mapping
    _node_translate = {6: 6, 7: 7, 8: 8, 9: 9, 53: 53, 17: 17, 35: 35}
    _node_symbol = {6: "C", 7: "N", 8: "O", 9: "F", 53: "I", 17: "Cl", 35: "Br"}

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(tu_name="MUTAG", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)

    def kgcnn_prepare(self, kgcnn_ds):
        """Read TU data and translate node labels to atom types."""
        # First load the TU data via parent
        super().kgcnn_prepare(kgcnn_ds)

        # Translate node labels to proton numbers and atom symbols
        node_translate = [6, 7, 8, 9, 53, 17, 35]
        node_symbol = ["C", "N", "O", "F", "I", "Cl", "Br"]

        node_labels = kgcnn_ds.obtain_property("node_labels")
        if node_labels is not None:
            new_node_attr = []
            new_node_symbol = []
            new_node_number = []
            for nl in node_labels:
                if nl is not None:
                    nl_int = np.array(nl, dtype="int").flatten()
                    new_node_attr.append(np.array([[node_translate[int(x)]] for x in nl_int], dtype="float32"))
                    new_node_symbol.append(np.array([node_symbol[int(x)] for x in nl_int]))
                    new_node_number.append(np.array([node_translate[int(x)] for x in nl_int], dtype="int"))
                else:
                    new_node_attr.append(None)
                    new_node_symbol.append(None)
                    new_node_number.append(None)
            kgcnn_ds.assign_property("node_attributes", new_node_attr)
            kgcnn_ds.assign_property("node_symbol", new_node_symbol)
            kgcnn_ds.assign_property("node_number", new_node_number)

        # Fix edge labels (negative values → 0)
        edge_labels = kgcnn_ds.obtain_property("edge_labels")
        if edge_labels is not None:
            new_edge_attr = []
            for el in edge_labels:
                if el is not None:
                    el_arr = np.array(el, dtype="float32")
                    el_arr[el_arr < 0] = 0
                    new_edge_attr.append(el_arr)
                else:
                    new_edge_attr.append(None)
            kgcnn_ds.assign_property("edge_attributes", new_edge_attr)

        # Graph labels and size
        graph_labels = kgcnn_ds.obtain_property("graph_labels")
        if graph_labels is not None:
            kgcnn_ds.assign_property("graph_labels", [
                np.array(x, dtype="float32") if x is not None else None for x in graph_labels])
