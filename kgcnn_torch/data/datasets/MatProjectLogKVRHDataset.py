"""MatProjectLogKVRH dataset for kgcnn-torch.

Matbench test dataset for predicting DFT log10 VRH-average bulk modulus from structure
(matbench_log_kvrh). 10987 samples, regression task.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectLogKVRHDataset(MatBenchDataset2020):
    """MatProjectLogKVRH dataset: 10987 structures with log10(K_VRH)."""

    dataset_name = "matbench_log_kvrh"
    _matbench_name = "matbench_log_kvrh"
    label_names = "log10(K_VRH)"
    label_units = "GPa"

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(matbench_name="matbench_log_kvrh", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
