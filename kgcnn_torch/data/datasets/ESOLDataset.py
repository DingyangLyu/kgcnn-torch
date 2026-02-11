"""ESOL dataset for kgcnn-torch.

Water solubility data (log solubility in mols per litre) for 1128 common organic small molecules.
Random or Scaffold splitting is recommended.

References:
    (1) Delaney, J.S. ESOL: estimating aqueous solubility directly from molecular structure.
        J. Chem. Inf. Comput. Sci. 44.3 (2004): 1000-1005.
"""
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ESOLDataset(MoleculeNetDataset2018):
    """ESOL dataset: 1128 molecules with aqueous solubility."""

    dataset_name = "ESOL"
    _molnet_name = "ESOL"
    label_names = ["measured log solubility in mols per litre"]
    label_units = ["log mol/L"]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(molnet_name="ESOL", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
