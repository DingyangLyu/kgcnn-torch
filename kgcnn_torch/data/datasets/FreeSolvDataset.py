"""FreeSolv dataset for kgcnn-torch.

Experimental and calculated hydration free energy of 642 small molecules in water.
Random splitting is recommended.

References:
    (1) Mobley DL, Guthrie JP. FreeSolv: a database of experimental and calculated hydration
        free energies. J Comput Aided Mol Des. 2014;28(7):711-720.
"""
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class FreeSolvDataset(MoleculeNetDataset2018):
    """FreeSolv dataset: 642 molecules with hydration free energies."""

    dataset_name = "FreeSolv"
    _molnet_name = "FreeSolv"
    label_names = ["expt"]
    label_units = ["kcal/mol"]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(molnet_name="FreeSolv", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
