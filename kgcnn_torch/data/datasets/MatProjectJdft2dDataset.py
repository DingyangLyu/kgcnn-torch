"""MatProjectJdft2d dataset for kgcnn-torch.

Matbench test dataset for predicting exfoliation energies from crystal structure (matbench_jdft2d).
636 samples, regression task, adapted from JARVIS DFT database.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectJdft2dDataset(MatBenchDataset2020):
    """MatProjectJdft2d dataset: 636 structures with exfoliation energy."""

    dataset_name = "matbench_jdft2d"
    _matbench_name = "matbench_jdft2d"
    label_names = "exfoliation_en"
    label_units = "meV/atom"

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        super().__init__(matbench_name="matbench_jdft2d", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload)
