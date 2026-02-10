"""QM7b dataset for kgcnn-torch.

Extension of QM7 for multitask learning with 13 additional properties (polarizability, HOMO/LUMO
eigenvalues, excitation energies) at ZINDO, SCS, PBE0, GW levels. 7211 molecules.

References:
    (1) L. C. Blum, J.-L. Reymond, J. Am. Chem. Soc., 131:8732, 2009.
    (2) G. Montavon et al., New J. Phys. 15 095003, 2013.
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class QM7bDataset(KgcnnGraphDataset):
    """QM7b dataset: 7211 molecules with 14 electronic properties."""

    dataset_name = "QM7b"
    download_info = {
        "dataset_name": "QM7b",
        "data_directory_name": "qm7b",
        "download_url": "http://quantum-machine.org/data/qm7b.mat",
        "download_file_name": "qm7b.mat",
    }
    label_names = [
        "aepbe0", "zindo-excitation-energy-with-the-most-absorption",
        "zindo-highest-absorption", "zindo-homo", "zindo-lumo",
        "zindo-1st-excitation-energy", "zindo-ionization-potential", "zindo-electron-affinity",
        "ks-homo", "ks-lumo", "gw-homo", "gw-lumo", "polarizability-pbe", "polarizability-scs"
    ]
    label_units = ["[?]"] * 14
    label_unit_conversion = np.array([1.0] * 14)
    file_name = "qm7b.csv"

    def kgcnn_prepare(self, kgcnn_ds):
        """Convert QM7b .mat Coulomb matrices directly to graph properties.

        Bypasses SDF conversion since reconstructed coordinates from Coulomb matrices
        are approximate and often fail RDKit validation.
        """
        from scipy.io import loadmat
        from kgcnn_torch.molecule.methods import inverse_global_proton_dict

        mat_path = os.path.join(self.raw_dir, "qm7b.mat")
        mat = loadmat(mat_path)
        coulomb = mat["X"]  # Coulomb matrix (N, max_atoms, max_atoms)
        targets = mat["T"]  # (N, 14) labels
        n_samples = coulomb.shape[0]

        # Extract proton numbers from diagonal: C_ii = 0.5 * Z^2.4
        diag = np.array([np.diag(coulomb[i]) for i in range(n_samples)])
        z_float = (2.0 * diag) ** (1.0 / 2.4)
        z_all = np.round(z_float).astype("int")

        node_numbers = []
        node_coords = []
        node_symbols = []
        graph_labels = []

        for i in range(n_samples):
            mask = z_all[i] > 0
            n_atoms = int(np.sum(mask))
            z_i = z_all[i][:n_atoms]

            symbols = np.array([inverse_global_proton_dict.get(int(z), "X") for z in z_i])

            # Reconstruct distances: C_ij = Z_i * Z_j / |R_i - R_j|
            dist = np.zeros((n_atoms, n_atoms))
            for a in range(n_atoms):
                for b in range(a + 1, n_atoms):
                    c_ab = coulomb[i, a, b]
                    if abs(c_ab) > 1e-10:
                        dist[a, b] = z_i[a] * z_i[b] / abs(c_ab)
                        dist[b, a] = dist[a, b]

            # Coordinate reconstruction via eigendecomposition of centered distance matrix
            if n_atoms > 1:
                d2 = dist ** 2
                h = np.eye(n_atoms) - np.ones((n_atoms, n_atoms)) / n_atoms
                b_mat = -0.5 * h @ d2 @ h
                eigvals, eigvecs = np.linalg.eigh(b_mat)
                idx = np.argsort(eigvals)[::-1][:3]
                coords = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0.0))
                if coords.shape[1] < 3:
                    coords = np.pad(coords, ((0, 0), (0, 3 - coords.shape[1])))
            else:
                coords = np.zeros((n_atoms, 3))

            node_numbers.append(np.array(z_i, dtype="int"))
            node_coords.append(np.array(coords, dtype="float32"))
            node_symbols.append(symbols)
            graph_labels.append(np.array(targets[i], dtype="float32"))

        kgcnn_ds.assign_property("node_number", node_numbers)
        kgcnn_ds.assign_property("node_coordinates", node_coords)
        kgcnn_ds.assign_property("node_symbol", node_symbols)
        kgcnn_ds.assign_property("graph_labels", graph_labels)
