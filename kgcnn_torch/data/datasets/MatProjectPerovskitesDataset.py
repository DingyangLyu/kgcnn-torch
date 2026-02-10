"""MatProjectPerovskites dataset for kgcnn-torch.

Matbench test dataset for predicting formation energy from crystal structure
(matbench_perovskites). 18928 samples, regression task, adapted from Castelli et al.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectPerovskitesDataset(MatBenchDataset2020):
    """MatProjectPerovskites dataset: 18928 perovskite structures with formation energy."""

    dataset_name = "matbench_perovskites"
    _matbench_name = "matbench_perovskites"
    label_names = "e_form"
    label_units = "eV/unit_cell"

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(matbench_name="matbench_perovskites", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
