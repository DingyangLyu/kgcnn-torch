"""MatProjectEForm dataset for kgcnn-torch.

Matbench test dataset for predicting DFT formation energy from structure (matbench_mp_e_form).
132752 samples, regression task.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectEFormDataset(MatBenchDataset2020):
    """MatProjectEForm dataset: 132752 structures with formation energy."""

    dataset_name = "matbench_mp_e_form"
    _matbench_name = "matbench_mp_e_form"
    label_names = "e_form"
    label_units = "eV/atom"

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(matbench_name="matbench_mp_e_form", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
