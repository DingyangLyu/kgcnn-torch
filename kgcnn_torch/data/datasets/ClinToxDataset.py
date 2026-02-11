"""ClinTox dataset for kgcnn-torch.

Qualitative data of drugs approved by the FDA and those that have failed clinical trials
for toxicity reasons. Two classification tasks for 1491 drug compounds.
Random splitting is recommended.

References:
    (1) Gayvert, K.M. et al. Cell chemical biology 23.10 (2016): 1294-1301.
    (2) Artemov, A.V. et al. bioRxiv (2016): 095653.
"""
import numpy as np
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018


class ClinToxDataset(MoleculeNetDataset2018):
    """ClinTox dataset: 1491 drugs with FDA approval and toxicity labels."""

    dataset_name = "ClinTox"
    _molnet_name = "ClinTox"
    label_names = ["FDA_APPROVED", "CT_TOX"]
    label_units = ["", ""]

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, label_index: int = 0, **kwargs):
        self.label_index = label_index
        super().__init__(molnet_name="ClinTox", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Load MoleculeNet data and select label column by label_index."""
        super().kgcnn_prepare(kgcnn_ds)

        # Select single label column matching Keras behavior
        graph_labels = kgcnn_ds.obtain_property("graph_labels")
        if graph_labels is not None:
            graph_labels = [
                np.array([x[self.label_index]], dtype="float") if x is not None else None
                for x in graph_labels
            ]
            kgcnn_ds.assign_property("graph_labels", graph_labels)
