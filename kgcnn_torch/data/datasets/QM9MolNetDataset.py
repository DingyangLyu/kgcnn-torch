"""QM9MolNet dataset for kgcnn-torch.

QM9 dataset as preprocessed from MoleculeNet with structure and labels.

References:
    (1) L. Ruddigkeit et al., J. Chem. Inf. Model. 52, 2864-2875, 2012.
    (2) R. Ramakrishnan et al., Scientific Data 1, 140022, 2014.
"""
import os
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class QM9MolNetDataset(KgcnnGraphDataset):
    """QM9MolNet dataset: QM9 preprocessed by MoleculeNet."""

    dataset_name = "QM9MolNet"
    download_info = {
        "dataset_name": "QM9MolNet",
        "data_directory_name": "qm9_mol_net",
        "download_url": "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/gdb9.tar.gz",
        "download_file_name": "gdb9.tar.gz",
        "unpack_tar": True,
        "unpack_directory_name": "gdb9",
    }
    label_names = [
        "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "u0", "u298", "h298", "g298", "cv", "u0_atom", "u298_atom", "h298_atom", "g298_atom"
    ]
    label_units = [
        "GHz", "GHz", "GHz", "D", "a_0^3", "H", "H", "H", "a_0^2", "H",
        "H", "H", "H", "H", "cal/mol K", "kcal/mol", "kcal/mol", "kcal/mol", "kcal/mol"
    ]
    file_name = "gdb9.csv"

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.qm import QMDataset
        # Data is in the unpacked gdb9 directory (do NOT pre-create — would block tar extraction)
        data_dir = os.path.join(self.raw_dir, "gdb9")
        return QMDataset(
            data_directory=data_dir,
            dataset_name="QM9MolNet",
            file_name="gdb9.csv",
        )

    def kgcnn_prepare(self, kgcnn_ds):
        """Rename CSV file if needed and read into memory."""
        data_dir = os.path.join(self.raw_dir, "gdb9")

        # MoleculeNet provides gdb9.sdf.csv — rename to gdb9.csv if needed
        sdf_csv = os.path.join(data_dir, "gdb9.sdf.csv")
        csv_path = os.path.join(data_dir, "gdb9.csv")
        if os.path.exists(sdf_csv) and not os.path.exists(csv_path):
            os.rename(sdf_csv, csv_path)

        # SDF already provided by MoleculeNet — no need to generate
        kgcnn_ds.prepare_data(overwrite=False, make_sdf=False)
        kgcnn_ds.read_in_memory(label_column_name=self.label_names)
