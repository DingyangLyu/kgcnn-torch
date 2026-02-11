"""MatProjectPhonons dataset for kgcnn-torch.

Matbench test dataset for predicting vibration properties from crystal structure
(matbench_phonons). 1265 samples, regression task. Frequency of highest frequency
optical phonon mode peak in units of 1/cm.
"""
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020


class MatProjectPhononsDataset(MatBenchDataset2020):
    """MatProjectPhonons dataset: 1265 structures with phonon frequency."""

    dataset_name = "matbench_phonons"
    _matbench_name = "matbench_phonons"
    label_names = "omega_max"
    label_units = "1/cm"

    def __init__(self, root=None, transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        super().__init__(matbench_name="matbench_phonons", root=root, transform=transform,
                         pre_transform=pre_transform, pre_filter=pre_filter, reload=reload, **kwargs)
