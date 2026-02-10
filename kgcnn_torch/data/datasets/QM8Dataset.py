"""QM8 dataset for kgcnn-torch.

Electronic spectra dataset from MoleculeNet. Over 20000 small organic molecules with up to eight
CONF atoms. Predictions of CC2 excitation energies and TDDFT spectra.

References:
    (1) L. Ruddigkeit et al., J. Chem. Inf. Model. 52, 2864-2875, 2012.
    (2) R. Ramakrishnan et al., J. Chem. Phys. 143 084111, 2015.
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class QM8Dataset(KgcnnGraphDataset):
    """QM8 dataset: ~22k molecules with electronic spectra properties."""

    dataset_name = "QM8"
    download_info = {
        "dataset_name": "QM8",
        "data_directory_name": "qm8",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb8.tar.gz",
        "download_file_name": "gdb8.tar.gz",
        "unpack_tar": True,
        "unpack_directory_name": "gdb8",
    }
    label_names = [
        "E1-CC2", "E2-CC2", "f1-CC2", "f2-CC2", "E1-PBE0", "E2-PBE0", "f1-PBE0",
        "f2-PBE0", "E1-PBE0", "E2-PBE0", "f1-PBE0", "f2-PBE0", "E1-CAM", "E2-CAM",
        "f1-CAM", "f2-CAM"
    ]
    label_units = ["[?]"] * 16
    label_unit_conversion = np.array([1.0] * 14)
    file_name = "qm8.csv"

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.qm import QMDataset
        # Data is in the unpacked gdb8 directory (do NOT pre-create — would block tar extraction)
        data_dir = os.path.join(self.raw_dir, "gdb8")
        return QMDataset(
            data_directory=data_dir,
            dataset_name="QM8",
            file_name="qm8.csv",
        )

    def kgcnn_prepare(self, kgcnn_ds):
        """Rename CSV file if needed and read into memory."""
        data_dir = os.path.join(self.raw_dir, "gdb8")

        # MoleculeNet provides qm8.sdf.csv — rename to qm8.csv if needed
        sdf_csv = os.path.join(data_dir, "qm8.sdf.csv")
        csv_path = os.path.join(data_dir, "qm8.csv")
        if os.path.exists(sdf_csv) and not os.path.exists(csv_path):
            os.rename(sdf_csv, csv_path)

        # SDF already provided by MoleculeNet — no need to generate
        kgcnn_ds.prepare_data(overwrite=False, make_sdf=False)
        kgcnn_ds.read_in_memory(label_column_name=self.label_names)
