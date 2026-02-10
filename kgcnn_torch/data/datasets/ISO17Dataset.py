"""ISO17 dataset for kgcnn-torch.

129 molecules with fixed composition C7O2H10 arranged in different chemically valid structures.
5000 conformational geometries per molecule with energies (eV) and forces (eV/Ang) from DFT.

References:
    (1) R. Ramakrishnan et al., Scientific Data, 1, 2014.
    (2) K. T. Schutt et al., Nature Communications, 8, 13890, 2017.
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class ISO17Dataset(KgcnnGraphDataset):
    """ISO17 dataset: C7O2H10 isomers with energies and forces."""

    dataset_name = "ISO17"
    download_info = {
        "dataset_name": "ISO17",
        "data_directory_name": "ISO17",
        "download_url": "http://quantum-machine.org/datasets/iso17.tar.gz",
        "download_file_name": "iso17.tar.gz",
        "unpack_tar": True,
        "unpack_directory_name": "iso17",
    }
    label_names = ["total_energy"]
    label_units = ["eV"]
    file_name = [
        "reference.db", "reference_eq.db", "test_within.db", "test_other.db", "test_eq.db"
    ]

    def kgcnn_prepare(self, kgcnn_ds):
        """Load ISO17 from ASE SQLite databases."""
        try:
            import ase.db
        except ImportError:
            raise ImportError("ISO17Dataset requires the `ase` package. Install via: pip install ase")

        base_dir = os.path.join(self.raw_dir, "iso17", "iso17")
        if not os.path.isdir(base_dir):
            base_dir = os.path.join(self.raw_dir, "iso17")

        db_files = [
            ("reference.db", "train", np.array([0])),
            ("reference_eq.db", "train", np.array([1])),
            ("test_within.db", "test", np.array([0])),
            ("test_other.db", "test", np.array([1])),
            ("test_eq.db", "test", np.array([2])),
        ]

        node_numbers = []
        node_coords = []
        node_symbols = []
        total_energies = []
        atomic_forces = []
        train_split = []
        test_split = []

        for db_name, split_type, split_val in db_files:
            db_path = os.path.join(base_dir, db_name)
            if not os.path.exists(db_path):
                continue
            db = ase.db.connect(db_path)
            for row in db.select():
                atoms = row.toatoms()
                node_numbers.append(np.array(atoms.get_atomic_numbers(), dtype="int"))
                node_coords.append(np.array(atoms.get_positions(), dtype="float"))
                node_symbols.append(np.array(atoms.get_chemical_symbols()))
                total_energies.append(np.array([row.total_energy], dtype="float"))
                atomic_forces.append(np.array(row.data.get("atomic_forces",
                                              np.zeros_like(atoms.get_positions())), dtype="float"))
                train_split.append(split_val if split_type == "train" else None)
                test_split.append(split_val if split_type == "test" else None)

        kgcnn_ds.assign_property("node_number", node_numbers)
        kgcnn_ds.assign_property("node_coordinates", node_coords)
        kgcnn_ds.assign_property("node_symbol", node_symbols)
        kgcnn_ds.assign_property("energy", total_energies)
        kgcnn_ds.assign_property("force", atomic_forces)
        kgcnn_ds.assign_property("train", train_split)
        kgcnn_ds.assign_property("test", test_split)
