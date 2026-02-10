"""PROTEINS dataset for kgcnn-torch.

Proteins classified as enzymes or non-enzymes. Nodes represent amino acids;
edges connect amino acids less than 6 Angstroms apart.

References:
    (1) K.M. Borgwardt et al. Bioinformatics, 21(Suppl 1):i47-i56, 2005.
    (2) P.D. Dobson, A.J. Doig. J. Mol. Biol., 330(4):771-783, 2003.
"""
from kgcnn_torch.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class PROTEINSDataset(GraphTUDataset2020):
    """PROTEINS dataset: protein graphs classified as enzyme/non-enzyme."""

    dataset_name = "PROTEINS"
    _tu_name = "PROTEINS"
    label_names = ["enzyme"]
    label_units = [""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(tu_name="PROTEINS", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
