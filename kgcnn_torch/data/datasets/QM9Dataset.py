"""QM9 dataset for kgcnn-torch.

134k stable small organic molecules made up of C, H, O, N, F with geometric, energetic,
electronic, and thermodynamic properties computed at B3LYP/6-31G(2df,p) level.

References:
    (1) L. Ruddigkeit et al., J. Chem. Inf. Model. 52, 2864-2875, 2012.
    (2) R. Ramakrishnan et al., Scientific Data 1, 140022, 2014.
"""
import os
import json
import logging
import numpy as np
import pandas as pd
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset, DownloadDataset

logger = logging.getLogger(__name__)


class QM9Dataset(KgcnnGraphDataset):
    """QM9 dataset: 134k small organic molecules with 15+ properties.

    Molecules that have a different SMILES code after convergence can be
    removed with :meth:`remove_uncharacterized`.
    """

    dataset_name = "QM9"
    _removed_uncharacterized = False
    download_info = {
        "dataset_name": "QM9",
        "data_directory_name": "qm9",
        "download_url": "https://ndownloader.figshare.com/files/3195389",
        "download_file_name": "dsgdb9nsd.xyz.tar.bz2",
        "unpack_tar": True,
        "unpack_directory_name": "dsgdb9nsd.xyz",
    }
    label_names = [
        "A", "B", "C", "mu", "alpha", "homo", "lumo", "gap", "r2", "zpve",
        "U0", "U", "H", "G", "Cv", "U0_atom", "U_atom", "H_atom", "G_atom", "Cv_atom"
    ]
    label_units = [
        "GHz", "GHz", "GHz", "D", "a_0^3", "eV", "eV", "eV", "a_0^2", "eV",
        "eV", "eV", "eV", "eV", "cal/mol K", "eV", "eV", "eV", "eV", "cal/mol K"
    ]
    label_unit_conversion = np.array(
        [[1.0, 1.0, 1.0, 1.0, 1.0, 27.2114, 27.2114, 27.2114, 1.0, 27.2114,
          27.2114, 27.2114, 27.2114, 27.2114, 1.0, 27.2114, 27.2114, 27.2114, 27.2114, 1.0]]
    )
    file_name = "qm9.csv"

    _atom_ref = {
        "U0": {"H": -0.500273, "C": -37.846772, "N": -54.583861, "O": -75.064579, "F": -99.718730},
        "U": {"H": -0.498857, "C": -37.845355, "N": -54.582445, "O": -75.063163, "F": -99.717314},
        "H": {"H": -0.497912, "C": -37.844411, "N": -54.581501, "O": -75.062219, "F": -99.716370},
        "G": {"H": -0.510927, "C": -37.861317, "N": -54.598897, "O": -75.079532, "F": -99.733544},
        "CV": {"H": 2.981, "C": 2.981, "N": 2.981, "O": 2.981, "F": 2.981},
    }

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.qm import QMDataset
        return QMDataset(
            data_directory=self.raw_dir,
            dataset_name="QM9",
            file_name="qm9.csv",
        )

    def kgcnn_prepare(self, kgcnn_ds):
        """Parse individual QM9 xyz files, create CSV + combined XYZ, then read into memory."""
        path = self.raw_dir
        dataset_size = 133885
        csv_path = os.path.join(path, "qm9.csv")
        xyz_path = os.path.join(path, "qm9.xyz")

        # Download additional files
        DownloadDataset.download_database(
            path, "https://figshare.com/ndownloader/files/3195392", "readme.txt", logger=None)
        DownloadDataset.download_database(
            path, "https://figshare.com/ndownloader/files/3195404", "uncharacterized.txt", logger=None)
        DownloadDataset.download_database(
            path, "https://figshare.com/ndownloader/files/3195395", "atomref.txt", logger=None)
        DownloadDataset.download_database(
            path, "https://figshare.com/ndownloader/files/3195401", "validation.txt", logger=None)

        if not (os.path.exists(csv_path) and os.path.exists(xyz_path)):
            xyz_dir = os.path.join(path, "dsgdb9nsd.xyz")
            if not os.path.exists(xyz_dir):
                raise FileNotFoundError("Cannot find extracted dsgdb9nsd.xyz directory.")

            # Read individual files
            qm9 = []
            for i in range(1, dataset_size + 1):
                mol = []
                fname = "dsgdb9nsd_{:06d}.xyz".format(i)
                with open(os.path.join(xyz_dir, fname), "r") as f:
                    lines = f.readlines()
                mol.append(int(lines[0]))
                labels = lines[1].strip().split(" ")[1].split("\t")
                if int(labels[0]) != i:
                    pass  # Index mismatch warning
                labels = ([lines[1].strip().split(" ")[0].strip()] +
                          [int(labels[0])] + [float(x) for x in labels[1:]])
                mol.append(labels)
                cords = []
                for j in range(int(lines[0])):
                    atom_info = lines[2 + j].strip().split("\t")
                    cords.append([atom_info[0]] + [float(x.replace("*^", "e")) for x in atom_info[1:]])
                mol.append(cords)
                freqs = lines[int(lines[0]) + 2].strip().split("\t")
                freqs = [float(x) for x in freqs]
                mol.append(freqs)
                smiles = lines[int(lines[0]) + 3].strip().split("\t")
                mol.append(smiles)
                inchis = lines[int(lines[0]) + 4].strip().split("\t")
                mol.append(inchis)
                qm9.append(mol)

            # Save JSON
            with open(os.path.join(path, "qm9.json"), "w") as f:
                json.dump(qm9, f)

            # Clean up individual files
            for i in range(1, dataset_size + 1):
                fpath = os.path.join(xyz_dir, "dsgdb9nsd_{:06d}.xyz".format(i))
                if os.path.exists(fpath):
                    os.remove(fpath)

            # Create combined XYZ
            from kgcnn_torch.molecule.io import write_list_to_xyz_file
            pos = [[y[1:4] for y in x[2]] for x in qm9]
            atoms = [[y[0] for y in x[2]] for x in qm9]
            atoms_pos = [[x, y] for x, y in zip(atoms, pos)]
            write_list_to_xyz_file(xyz_path, atoms_pos)

            # Create CSV with labels
            labels = np.array([x[1][1:] if len(x[1]) == 17 else x[1] for x in qm9])
            atom_energy = [[sum([self._atom_ref[t][a] for a in x])
                            for t in ["U0", "U", "H", "G", "CV"]] for x in atoms]
            targets_atom = labels[:, 11:] - np.array(atom_energy)
            targets = np.concatenate([labels[:, 1:], targets_atom], axis=-1) * self.label_unit_conversion
            cols = ["%s [%s]" % (a, b) for a, b in zip(self.label_names, self.label_units)]
            df = pd.DataFrame(targets, columns=cols)
            df.insert(targets.shape[1], "ID", labels[:, 0].astype(dtype="int"))
            df.to_csv(csv_path, index=False)

        # Use QMDataset pipeline
        kgcnn_ds.prepare_data(overwrite=False, make_sdf=True)
        kgcnn_ds.read_in_memory(
            label_column_name=["%s [%s]" % (a, b) for a, b in zip(self.label_names, self.label_units)])

        # Add graph_attributes (mean molecular weight)
        mass_dict = {"H": 1.0079, "C": 12.0107, "N": 14.0067, "O": 15.9994, "F": 18.9984,
                     "S": 32.065, "C3": 12.0107}

        def mmw(atoms):
            mass = [mass_dict[x[:1]] for x in atoms]
            return np.array([np.mean(mass), len(mass)])

        node_symbols = kgcnn_ds.obtain_property("node_symbol")
        if node_symbols is not None:
            kgcnn_ds.assign_property("graph_attributes", [
                mmw(x) if x is not None else None for x in node_symbols])

    def remove_uncharacterized(self):
        """Remove 3054 uncharacterized molecules that failed structure test from this dataset.

        Reads the ``uncharacterized.txt`` file (downloaded during processing) and
        filters out the listed molecule indices by setting PyG's ``_indices``.

        Returns:
            numpy.ndarray: Sorted (descending) array of removed 0-based indices.
        """
        if self._removed_uncharacterized:
            logger.warning("Uncharacterized molecules have already been removed.")
            return np.array([], dtype="int")

        unchar_path = os.path.join(self.raw_dir, "uncharacterized.txt")
        if not os.path.exists(unchar_path):
            raise FileNotFoundError(
                f"Cannot find uncharacterized.txt at {unchar_path}. "
                f"Try reloading the dataset with reload=True."
            )

        with open(unchar_path, "r") as f:
            data = f.readlines()[9:-1]
        data = [x.strip().split(" ") for x in data]
        data = [[y for y in x if y != ""] for x in data]
        indices = np.array([x[0] for x in data], dtype="int") - 1
        remove_set = set(indices.tolist())

        # Compute valid indices respecting any existing _indices filter
        if self._indices is not None:
            current_indices = list(self._indices)
        else:
            current_indices = list(range(len(self)))

        valid_indices = [i for i in current_indices if i not in remove_set]
        n_removed = len(current_indices) - len(valid_indices)
        self._indices = valid_indices

        logger.info("Removed %s uncharacterized molecules." % n_removed)
        self._removed_uncharacterized = True
        return np.flip(np.sort(indices))
