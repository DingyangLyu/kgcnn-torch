"""Tox21 MolNet dataset for kgcnn-torch.

Toxicology in the 21st Century (Tox21) dataset measuring toxicity of compounds.
Qualitative toxicity measurements for 8k compounds on 12 different targets,
including nuclear receptors and stress response pathways. Random splitting is recommended.

References:
    (1) Tox21 Challenge. https://tripod.nih.gov/tox21/challenge/
"""
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class Tox21MolNetDataset(MoleculeNetDataset2018):
    """Tox21 MolNet dataset: ~8k compounds with 12 toxicity targets."""

    dataset_name = "Tox21"
    _molnet_name = "Tox21"
    label_names = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase", "NR-ER", "NR-ER-LBD",
        "NR-PPAR-gamma", "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]
    label_units = [""] * 12

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, remove_nan: bool = False):
        self._remove_nan_label = remove_nan
        super().__init__(molnet_name="Tox21", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
