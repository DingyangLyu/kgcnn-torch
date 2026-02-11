"""MD17 dataset for kgcnn-torch.

Molecular dynamics trajectories with atomic coordinates, total energy (kcal/mol) and
forces (kcal/mol/Angstrom) on each atom. Contains DFT and CCSD(T) level trajectories
for various molecules (aspirin, benzene, ethanol, malonaldehyde, etc.).

References:
    (1) Chmiela et al., Sci. Adv., 2017.
    (2) http://www.sgdml.org/#datasets
"""
import os
import numpy as np
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class MD17Dataset(KgcnnGraphDataset):
    """MD17 dataset: molecular dynamics trajectories with energies and forces.

    Args:
        trajectory_name: Name of the trajectory to load.
            DFT trajectories: aspirin_dft, azobenzene_dft, benzene2017_dft,
            benzene2018_dft, ethanol_dft, malonaldehyde_dft, naphthalene_dft,
            paracetamol_dft, salicylic_dft, toluene_dft, uracil_dft.
            CCSD(T) trajectories: aspirin_ccsd, benzene_ccsd_t, ethanol_ccsd_t,
            malonaldehyde_ccsd_t, toluene_ccsd_t.
        root: Root directory.
        reload: If True, reprocess data.
    """

    datasets_download_info = {
        "CG-CG": {"download_file_name": "CG-CG.npz"},
        "aspirin_dft": {"download_file_name": "aspirin_dft.npz"},
        "azobenzene_dft": {"download_file_name": "azobenzene_dft.npz"},
        "benzene2017_dft": {"download_file_name": "benzene2017_dft.npz"},
        "benzene2018_dft": {"download_file_name": "benzene2018_dft.npz"},
        "ethanol_dft": {"download_file_name": "ethanol_dft.npz"},
        "malonaldehyde_dft": {"download_file_name": "malonaldehyde_dft.npz"},
        "naphthalene_dft": {"download_file_name": "naphthalene_dft.npz"},
        "paracetamol_dft": {"download_file_name": "paracetamol_dft.npz"},
        "salicylic_dft": {"download_file_name": "salicylic_dft.npz"},
        "toluene_dft": {"download_file_name": "toluene_dft.npz"},
        "uracil_dft": {"download_file_name": "uracil_dft.npz"},
        "aspirin_ccsd": {"download_file_name": "aspirin_ccsd.zip", "unpack_zip": True,
                         "unpack_directory_name": "aspirin_ccsd"},
        "benzene_ccsd_t": {"download_file_name": "benzene_ccsd_t.zip", "unpack_zip": True,
                           "unpack_directory_name": "benzene_ccsd_t"},
        "ethanol_ccsd_t": {"download_file_name": "ethanol_ccsd_t.zip", "unpack_zip": True,
                           "unpack_directory_name": "ethanol_ccsd_t"},
        "malonaldehyde_ccsd_t": {"download_file_name": "malonaldehyde_ccsd_t.zip", "unpack_zip": True,
                                 "unpack_directory_name": "malonaldehyde_ccsd_t"},
        "toluene_ccsd_t": {"download_file_name": "toluene_ccsd_t.zip", "unpack_zip": True,
                           "unpack_directory_name": "toluene_ccsd_t"},
    }

    label_names = ["E", "F"]
    label_units = ["kcal/mol", "kcal/mol/Ang"]

    def __init__(self, trajectory_name: str = None, root=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        if trajectory_name not in self.datasets_download_info:
            raise ValueError(
                f"Unknown trajectory '{trajectory_name}'. "
                f"Choose from: {list(self.datasets_download_info.keys())}"
            )
        self.trajectory_name = trajectory_name
        info = self.datasets_download_info[trajectory_name]
        self.dataset_name = f"MD17_{trajectory_name}"
        self.download_info = dict(info)
        self.download_info["download_url"] = (
            f"https://sgdml.org/secure_proxy.php?file={info['download_file_name']}"
        )
        self.download_info["dataset_name"] = "MD17"
        self.download_info["data_directory_name"] = "MD17"
        self.file_name = info["download_file_name"]
        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, reload=reload, **kwargs)

    def kgcnn_prepare(self, kgcnn_ds):
        """Load MD17 trajectory from NPZ file."""
        info = self.datasets_download_info[self.trajectory_name]

        if not info.get("unpack_zip"):
            file_path = os.path.join(self.raw_dir, info["download_file_name"])
            data_loaded = np.load(file_path, allow_pickle=True)
        else:
            base = os.path.splitext(info["download_file_name"])[0]
            dir_path = os.path.join(self.raw_dir, info["unpack_directory_name"])
            train_path = os.path.join(dir_path, base + "-train.npz")
            test_path = os.path.join(dir_path, base + "-test.npz")
            data_loaded = [np.load(train_path, allow_pickle=True),
                           np.load(test_path, allow_pickle=True)]

        def make_dict_from_data(data, is_split=None):
            out_dict = {}
            # Map MD17 keys to standard kgcnn property names
            out_dict["node_coordinates"] = [np.array(x, dtype="float") for x in data["R"]]
            out_dict["energy"] = [np.array([e], dtype="float") for e in data["E"]]
            out_dict["force"] = [np.array(x, dtype="float") for x in data["F"]]
            num_data_points = len(out_dict["node_coordinates"])
            if "z" in data:
                z = data["z"]
                out_dict["node_number"] = [np.array(z, dtype="int") for _ in range(num_data_points)]
            if is_split is not None:
                for key, value in is_split.items():
                    out_dict[key] = [value for _ in range(num_data_points)]
            return out_dict

        if isinstance(data_loaded, (list, tuple)):
            split_assignment = [
                {"train": np.array([1]), "test": None},
                {"train": None, "test": np.array([1])}
            ]
            prop_dicts = [make_dict_from_data(x, is_split=s)
                          for x, s in zip(data_loaded, split_assignment)]
            for key_prop in prop_dicts[0].keys():
                kgcnn_ds.assign_property(key_prop,
                                         prop_dicts[0][key_prop] + prop_dicts[1][key_prop])
        else:
            for key_prop, value_prop in make_dict_from_data(data_loaded).items():
                kgcnn_ds.assign_property(key_prop, value_prop)
