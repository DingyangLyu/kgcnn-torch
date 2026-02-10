"""QM7 dataset for kgcnn-torch.

Store and process QM7 dataset from Quantum Machine (http://quantum-machine.org/datasets/).
Subset of GDB-13 composed of all molecules of up to 23 atoms (including 7 heavy atoms C, N, O, S),
totalling 7165 molecules. Atomization energies in kcal/mol.

References:
    (1) L. C. Blum, J.-L. Reymond, J. Am. Chem. Soc., 131:8732, 2009.
    (2) M. Rupp et al., Phys. Rev. Lett., 108(5):058301, 2012.
"""
import os
import numpy as np
import pandas as pd
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class QM7Dataset(KgcnnGraphDataset):
    """QM7 dataset: 7165 small organic molecules with atomization energies."""

    dataset_name = "QM7"
    download_info = {
        "dataset_name": "QM7",
        "data_directory_name": "qm7",
        "download_url": "http://quantum-machine.org/data/qm7.mat",
        "download_file_name": "qm7.mat",
    }
    label_names = ["u0_atom"]
    label_units = ["kcal/mol"]
    label_unit_conversion = np.array([1.0] * 14)
    file_name = "qm7.csv"

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.qm import QMDataset
        return QMDataset(
            data_directory=self.raw_dir,
            dataset_name="QM7",
            file_name="qm7.csv",
        )

    def kgcnn_prepare(self, kgcnn_ds):
        """Convert QM7 .mat file to XYZ + CSV, then read into memory."""
        from scipy.io import loadmat
        from kgcnn_torch.molecule.methods import inverse_global_proton_dict
        from kgcnn_torch.molecule.io import write_list_to_xyz_file

        path = self.raw_dir
        mat_path = os.path.join(path, "qm7.mat")
        csv_path = os.path.join(path, "qm7.csv")
        xyz_path = os.path.join(path, "qm7.xyz")

        if not (os.path.exists(csv_path) and os.path.exists(xyz_path)):
            mat = loadmat(mat_path)

            # Extract atomic numbers and positions from Coulomb matrix
            z_all = np.array(mat["Z"], dtype="int")  # (N, max_atoms)
            r_all = np.array(mat["R"], dtype="float")  # (N, max_atoms, 3)
            # Convert from Bohr to Angstrom
            r_all = r_all * 0.529177210903

            # Build XYZ list
            xyz_list = []
            for i in range(len(z_all)):
                mask = z_all[i] > 0
                z_i = z_all[i][mask]
                r_i = r_all[i][mask]
                symbols = [inverse_global_proton_dict[int(z)] for z in z_i]
                xyz_list.append([symbols, r_i.tolist()])

            write_list_to_xyz_file(xyz_path, xyz_list)

            # Extract labels
            targets = mat["T"][0]  # (N,) atomization energies
            df = pd.DataFrame({"u0_atom [kcal/mol]": targets})
            df.to_csv(csv_path, index=False)

            # Save splits
            if "P" in mat:
                np.save(os.path.join(path, "qm7_splits.npy"), mat["P"])

        # Use QMDataset pipeline
        kgcnn_ds.prepare_data(overwrite=False, make_sdf=True)
        kgcnn_ds.read_in_memory(label_column_name="u0_atom [kcal/mol]")
