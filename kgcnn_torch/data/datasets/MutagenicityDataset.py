"""Mutagenicity dataset for kgcnn-torch.

Chemical compound dataset of drugs categorized into mutagen and non-mutagen classes.
From TUDatasets.

References:
    (1) Riesen, K. and Bunke, H. SSPR&SPR 2008, LNCS, vol. 5342, pp. 287-297, 2008.
"""
from kgcnn_torch.data.datasets.GraphTUDataset2020 import GraphTUDataset2020


class MutagenicityDataset(GraphTUDataset2020):
    """Mutagenicity dataset: drug compounds classified as mutagen/non-mutagen."""

    dataset_name = "Mutagenicity"
    _tu_name = "Mutagenicity"
    label_names = ["mutagenicity"]
    label_units = [""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(tu_name="Mutagenicity", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
