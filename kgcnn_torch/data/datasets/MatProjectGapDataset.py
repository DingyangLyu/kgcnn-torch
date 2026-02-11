"""MatProjectGap dataset for kgcnn-torch.

Matbench test dataset for predicting DFT PBE band gap from structure (matbench_mp_gap).
106113 samples, regression task.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectGapDataset(MatBenchDataset2020):
    """MatProjectGap dataset: 106113 structures with PBE band gap."""

    dataset_name = "matbench_mp_gap"
    _matbench_name = "matbench_mp_gap"
    label_names = "gap"
    label_units = "eV"

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(matbench_name="matbench_mp_gap", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
