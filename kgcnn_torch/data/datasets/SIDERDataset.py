"""SIDER dataset for kgcnn-torch.

Side Effect Resource (SIDER) database of marketed drugs and adverse drug reactions (ADR).
27 system organ classes for 1427 approved drugs. Random splitting is recommended.

References:
    (1) Kuhn, M. et al. The SIDER database of drugs and side effects.
        Nucleic acids research 44.D1 (2015): D1075-D1079.
"""
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class SIDERDataset(MoleculeNetDataset2018):
    """SIDER dataset: 1427 drugs with 27 side effect classes."""

    dataset_name = "SIDER"
    _molnet_name = "SIDER"
    label_names = "side_effects"
    label_units = ""

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(molnet_name="SIDER", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
