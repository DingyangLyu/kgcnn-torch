"""Lipop (Lipophilicity) dataset for kgcnn-torch.

Experimental results of octanol/water distribution coefficient (logD at pH 7.4) of 4200 compounds
curated from ChEMBL database. Random or Scaffold splitting is recommended.

References:
    (1) Hersey, A. ChEMBL Deposited Data Set - AZ dataset; 2015.
"""
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class LipopDataset(MoleculeNetDataset2018):
    """Lipop dataset: 4200 compounds with lipophilicity (logD)."""

    dataset_name = "Lipop"
    _molnet_name = "Lipop"
    label_names = ["exp"]
    label_units = ["logD"]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(molnet_name="Lipop", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
