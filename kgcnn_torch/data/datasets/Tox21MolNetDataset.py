"""Tox21 MolNet dataset for kgcnn-torch.

Toxicology in the 21st Century (Tox21) dataset measuring toxicity of compounds.
Qualitative toxicity measurements for 8k compounds on 12 different targets,
including nuclear receptors and stress response pathways. Random splitting is recommended.

References:
    (1) Tox21 Challenge. https://tripod.nih.gov/tox21/challenge/
"""
import numpy as np
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
                 reload: bool = False, remove_nan: bool = False, **kwargs):
        self._remove_nan_label = remove_nan
        super().__init__(molnet_name="Tox21", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Load MoleculeNet data and optionally replace NaN labels with 0."""
        super().kgcnn_prepare(kgcnn_ds)

        graph_labels = kgcnn_ds.obtain_property("graph_labels")
        if graph_labels is not None:
            graph_labels = [np.array(x, dtype="float") for x in graph_labels]
            if self._remove_nan_label:
                graph_labels = [np.nan_to_num(x) for x in graph_labels]
            kgcnn_ds.assign_property("graph_labels", graph_labels)
