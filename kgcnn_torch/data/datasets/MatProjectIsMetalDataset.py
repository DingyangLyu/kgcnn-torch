"""MatProjectIsMetal dataset for kgcnn-torch.

Matbench test dataset for predicting DFT metallicity from structure (matbench_mp_is_metal).
106113 samples, classification task.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectIsMetalDataset(MatBenchDataset2020):
    """MatProjectIsMetal dataset: 106113 structures with metallicity label."""

    dataset_name = "matbench_mp_is_metal"
    _matbench_name = "matbench_mp_is_metal"
    label_names = "is_metal"
    label_units = ""

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(matbench_name="matbench_mp_is_metal", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
