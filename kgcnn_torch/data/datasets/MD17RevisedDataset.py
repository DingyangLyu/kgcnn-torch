"""MD17 Revised dataset for kgcnn-torch.

Revised version of MD17 with energies and forces recalculated at PBE/def2-SVP level with
very tight SCF convergence and dense DFT integration grid. 100,000 structures per molecule.

WARNING: Do NOT train on more than 1000 samples due to autocorrelation in MD trajectories.

References:
    (1) A. Christensen, O.A. von Lilienfeld, Materials Cloud Archive 2020.82, 2020.
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class MD17RevisedDataset(KgcnnGraphDataset):
    """MD17 Revised dataset: recalculated MD trajectories at PBE/def2-SVP level.

    Args:
        trajectory_name: Name of molecule trajectory.
            Options: aspirin, azobenzene, benzene, ethanol, malonaldehyde,
            naphthalene, paracetamol, salicylic, toluene, uracil.
        root: Root directory.
        reload: If True, reprocess data.
    """

    possible_trajectory_names = [
        "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
        "naphthalene", "paracetamol", "salicylic", "toluene", "uracil"
    ]

    download_info = {
        "dataset_name": "MD17Revised",
        "data_directory_name": "MD17Revised",
        "download_url": "https://archive.materialscloud.org/records/pfffs-fff86/files/rmd17.tar.bz2?download=1",
        "download_file_name": "rmd17.tar.bz2",
        "unpack_tar": True,
        "unpack_directory_name": "rmd17",
    }

    label_names = ["energies", "forces"]
    label_units = ["kcal/mol", "kcal/mol/Ang"]

    def __init__(self, trajectory_name: str = None, root=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        if trajectory_name not in self.possible_trajectory_names:
            raise ValueError(
                f"Unknown trajectory '{trajectory_name}'. "
                f"Choose from: {self.possible_trajectory_names}"
            )
        self.trajectory_name = trajectory_name
        self.dataset_name = f"MD17Revised_{trajectory_name}"
        self.file_name = f"rmd17_{trajectory_name}.npz"
        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Load revised MD17 trajectory from NPZ file."""
        # tar extracts to rmd17/rmd17/npz_data/
        npz_dir = os.path.join(self.raw_dir, "rmd17", "rmd17", "npz_data")
        npz_path = os.path.join(npz_dir, f"rmd17_{self.trajectory_name}.npz")
        if not os.path.exists(npz_path):
            # Try alternative paths
            for alt in [
                os.path.join(self.raw_dir, "rmd17", "rmd17", f"rmd17_{self.trajectory_name}.npz"),
                os.path.join(self.raw_dir, "rmd17", f"rmd17_{self.trajectory_name}.npz"),
            ]:
                if os.path.exists(alt):
                    npz_path = alt
                    break

        data = np.load(npz_path, allow_pickle=True)
        num_points = len(data["coords"])

        # Map to standard kgcnn property names for to_pyg_list()
        kgcnn_ds.assign_property("node_coordinates",
                                 [np.array(x, dtype="float") for x in data["coords"]])
        kgcnn_ds.assign_property("energy",
                                 [np.array([e], dtype="float") for e in data["energies"]])
        kgcnn_ds.assign_property("force",
                                 [np.array(x, dtype="float") for x in data["forces"]])

        if "nuclear_charges" in data:
            z = data["nuclear_charges"]
            kgcnn_ds.assign_property("node_number",
                                     [np.array(z, dtype="int") for _ in range(num_points)])

        # Load train/test splits matching Keras convention:
        # Each graph gets a "train" and "test" property containing an array of
        # split IDs (1-5) it belongs to, or None. This is compatible with
        # MemoryGraphDataset.get_train_test_indices(train="train", test="test").
        splits_dir = os.path.join(self.raw_dir, "rmd17", "rmd17", "splits")
        if not os.path.exists(splits_dir):
            splits_dir = os.path.join(npz_dir, "splits")
        if os.path.exists(splits_dir):
            def _read_split_indices(file_path: str) -> set:
                if not os.path.exists(file_path):
                    return set()
                values = np.loadtxt(file_path, dtype="int", delimiter=",")
                values = np.atleast_1d(values).astype(int)
                return set(values.tolist())

            splits_train = []
            splits_test = []
            for split_idx in range(1, 6):
                train_file = os.path.join(splits_dir, f"index_train_0{split_idx}.csv")
                splits_train.append(_read_split_indices(train_file))

                test_file = os.path.join(splits_dir, f"index_test_0{split_idx}.csv")
                splits_test.append(_read_split_indices(test_file))

            property_train = []
            property_test = []
            for i in range(num_points):
                is_train = [j + 1 for j, s in enumerate(splits_train) if i in s]
                is_test = [j + 1 for j, s in enumerate(splits_test) if i in s]
                property_train.append(np.array(is_train, dtype="int") if is_train else None)
                property_test.append(np.array(is_test, dtype="int") if is_test else None)

            kgcnn_ds.assign_property("train", property_train)
            kgcnn_ds.assign_property("test", property_test)
