"""ClinTox dataset for kgcnn-torch.

Qualitative data of drugs approved by the FDA and those that have failed clinical trials
for toxicity reasons. Two classification tasks for 1491 drug compounds.
Random splitting is recommended.

References:
    (1) Gayvert, K.M. et al. Cell chemical biology 23.10 (2016): 1294-1301.
    (2) Artemov, A.V. et al. bioRxiv (2016): 095653.
"""
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ClinToxDataset(MoleculeNetDataset2018):
    """ClinTox dataset: 1491 drugs with FDA approval and toxicity labels."""

    dataset_name = "ClinTox"
    _molnet_name = "ClinTox"
    label_names = ["FDA_APPROVED", "CT_TOX"]
    label_units = ["", ""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, label_index: int = 0):
        self.label_index = label_index
        super().__init__(molnet_name="ClinTox", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
