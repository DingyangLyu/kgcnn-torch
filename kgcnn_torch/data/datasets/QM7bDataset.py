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
        """Convert QM7b .mat Coulomb matrices to graph properties.

        Uses the same reconstruction algorithm as Keras:
        coulomb_matrix_to_inverse_distance_proton + coordinates_from_distance_matrix.
        """
        from scipy.io import loadmat
        from kgcnn_torch.molecule.methods import inverse_global_proton_dict
        from kgcnn_torch.graph.methods import (
            coulomb_matrix_to_inverse_distance_proton,
            coordinates_from_distance_matrix,
            invert_distance,
        )

        mat_path = os.path.join(self.raw_dir, "qm7b.mat")
        mat = loadmat(mat_path)
        coulomb = mat["X"]  # Coulomb matrix (N, max_atoms, max_atoms)
        targets = mat["T"]  # (N, 14) labels
        n_samples = coulomb.shape[0]

        # Get number of atoms per molecule from diagonal
        graph_len = [int(np.around(np.sum(np.diag(coulomb[i]) > 0))) for i in range(n_samples)]

        # Extract proton numbers and inverse distances using Keras-compatible function
        proton_inv_dist = [
            coulomb_matrix_to_inverse_distance_proton(
                coulomb[i, :graph_len[i], :graph_len[i]],
                unit_conversion=0.529177210903
            )
            for i in range(n_samples)
        ]
        proton = [x[1] for x in proton_inv_dist]
        inv_dist = [x[0] for x in proton_inv_dist]
        dist = [invert_distance(x) for x in inv_dist]
        pos = [coordinates_from_distance_matrix(x) for x in dist]

        node_numbers = []
        node_coords = []
        node_symbols = []
        graph_labels = []

        for i in range(n_samples):
            z_i = np.array(proton[i], dtype="int")
            symbols = np.array([inverse_global_proton_dict.get(int(z), "X") for z in z_i])

            node_numbers.append(z_i)
            node_coords.append(np.array(pos[i], dtype="float32"))
            node_symbols.append(symbols)
            graph_labels.append(np.array(targets[i], dtype="float32"))

        kgcnn_ds.assign_property("node_number", node_numbers)
        kgcnn_ds.assign_property("node_coordinates", node_coords)
        kgcnn_ds.assign_property("node_symbol", node_symbols)
        kgcnn_ds.assign_property("graph_labels", graph_labels)
