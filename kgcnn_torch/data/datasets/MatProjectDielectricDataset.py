"""MatProjectDielectric dataset for kgcnn-torch.

Matbench test dataset for predicting refractive index from structure (matbench_dielectric).
4764 samples, regression task.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectDielectricDataset(MatBenchDataset2020):
    """MatProjectDielectric dataset: 4764 structures with refractive index."""

    dataset_name = "matbench_dielectric"
    _matbench_name = "matbench_dielectric"
    label_names = "n_r"
    label_units = ""

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(matbench_name="matbench_dielectric", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
