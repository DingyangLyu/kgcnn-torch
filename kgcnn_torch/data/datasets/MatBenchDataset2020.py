"""MatBench 2020 base dataset for kgcnn-torch.

Base class for MatBench crystal structure datasets. Supports structure-based prediction tasks
from the Materials Project via the MatBench benchmark suite.

References:
    (1) Dunn, A. et al. npj Comput Mater 6, 138, 2020.
"""
import os
import pandas as pd
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class MatBenchDataset2020(KgcnnGraphDataset):
    """Base class for MatBench crystal property prediction datasets.

    Subclasses set ``_matbench_name`` to select which dataset to load.
    """

    datasets_download_info = {
        "matbench_steels": {"dataset_name": "matbench_steels",
                            "download_file_name": "matbench_steels.json.gz",
                            "data_directory_name": "matbench_steels", "extract_gz": True,
                            "extract_file_name": "matbench_steels.json"},
        "matbench_jdft2d": {"dataset_name": "matbench_jdft2d",
                            "download_file_name": "matbench_jdft2d.json.gz",
                            "data_directory_name": "matbench_jdft2d", "extract_gz": True,
                            "extract_file_name": "matbench_jdft2d.json"},
        "matbench_phonons": {"dataset_name": "matbench_phonons",
                             "download_file_name": "matbench_phonons.json.gz",
                             "data_directory_name": "matbench_phonons", "extract_gz": True,
                             "extract_file_name": "matbench_phonons.json"},
        "matbench_expt_gap": {"dataset_name": "matbench_expt_gap",
                              "download_file_name": "matbench_expt_gap.json.gz",
                              "data_directory_name": "matbench_expt_gap", "extract_gz": True,
                              "extract_file_name": "matbench_expt_gap.json"},
        "matbench_dielectric": {"dataset_name": "matbench_dielectric",
                                "download_file_name": "matbench_dielectric.json.gz",
                                "data_directory_name": "matbench_dielectric", "extract_gz": True,
                                "extract_file_name": "matbench_dielectric.json"},
        "matbench_expt_is_metal": {"dataset_name": "matbench_expt_is_metal",
                                   "download_file_name": "matbench_expt_is_metal.json.gz",
                                   "data_directory_name": "matbench_expt_is_metal", "extract_gz": True,
                                   "extract_file_name": "matbench_expt_is_metal.json"},
        "matbench_glass": {"dataset_name": "matbench_glass",
                           "download_file_name": "matbench_glass.json.gz",
                           "data_directory_name": "matbench_glass", "extract_gz": True,
                           "extract_file_name": "matbench_glass.json"},
        "matbench_log_gvrh": {"dataset_name": "matbench_log_gvrh",
                              "download_file_name": "matbench_log_gvrh.json.gz",
                              "data_directory_name": "matbench_log_gvrh", "extract_gz": True,
                              "extract_file_name": "matbench_log_gvrh.json"},
        "matbench_log_kvrh": {"dataset_name": "matbench_log_kvrh",
                              "download_file_name": "matbench_log_kvrh.json.gz",
                              "data_directory_name": "matbench_log_kvrh", "extract_gz": True,
                              "extract_file_name": "matbench_log_kvrh.json"},
        "matbench_perovskites": {"dataset_name": "matbench_perovskites",
                                 "download_file_name": "matbench_perovskites.json.gz",
                                 "data_directory_name": "matbench_perovskites", "extract_gz": True,
                                 "extract_file_name": "matbench_perovskites.json"},
        "matbench_mp_gap": {"dataset_name": "matbench_mp_gap",
                            "download_file_name": "matbench_mp_gap.json.gz",
                            "data_directory_name": "matbench_mp_gap", "extract_gz": True,
                            "extract_file_name": "matbench_mp_gap.json"},
        "matbench_mp_is_metal": {"dataset_name": "matbench_mp_is_metal",
                                 "download_file_name": "matbench_mp_is_metal.json.gz",
                                 "data_directory_name": "matbench_mp_is_metal", "extract_gz": True,
                                 "extract_file_name": "matbench_mp_is_metal.json"},
        "matbench_mp_e_form": {"dataset_name": "matbench_mp_e_form",
                               "download_file_name": "matbench_mp_e_form.json.gz",
                               "data_directory_name": "matbench_mp_e_form", "extract_gz": True,
                               "extract_file_name": "matbench_mp_e_form.json"},
    }

    datasets_prepare_data_info = {
        "matbench_steels": {"file_column_name": "composition"},
        "matbench_jdft2d": {"file_column_name": "structure"},
        "matbench_phonons": {"file_column_name": "structure"},
        "matbench_expt_gap": {"file_column_name": "composition"},
        "matbench_dielectric": {"file_column_name": "structure"},
        "matbench_expt_is_metal": {"file_column_name": "composition"},
        "matbench_glass": {"file_column_name": "composition"},
        "matbench_log_gvrh": {"file_column_name": "structure"},
        "matbench_log_kvrh": {"file_column_name": "structure"},
        "matbench_perovskites": {"file_column_name": "structure"},
        "matbench_mp_gap": {"file_column_name": "structure"},
        "matbench_mp_is_metal": {"file_column_name": "structure"},
        "matbench_mp_e_form": {"file_column_name": "structure"},
    }

    datasets_read_in_memory_info = {
        "matbench_steels": {"label_column_name": "yield strength"},
        "matbench_jdft2d": {"label_column_name": "exfoliation_en"},
        "matbench_phonons": {"label_column_name": "last phdos peak"},
        "matbench_expt_gap": {"label_column_name": "gap expt"},
        "matbench_dielectric": {"label_column_name": "n"},
        "matbench_expt_is_metal": {"label_column_name": "is_metal"},
        "matbench_glass": {"label_column_name": "gfa"},
        "matbench_log_gvrh": {"label_column_name": "log10(G_VRH)"},
        "matbench_log_kvrh": {"label_column_name": "log10(K_VRH)"},
        "matbench_perovskites": {"label_column_name": "e_form"},
        "matbench_mp_gap": {"label_column_name": "gap pbe"},
        "matbench_mp_is_metal": {"label_column_name": "is_metal"},
        "matbench_mp_e_form": {"label_column_name": "e_form"},
    }

    _matbench_name: str = None

    def __init__(self, matbench_name: str = None, root=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False, **kwargs):
        name = matbench_name or self._matbench_name
        if name is None:
            raise ValueError("Must provide matbench_name or set _matbench_name on subclass.")
        if name not in self.datasets_download_info:
            raise ValueError(
                f"Unknown MatBench dataset '{name}'. "
                f"Choose from: {list(self.datasets_download_info.keys())}"
            )
        self._matbench_name = name
        info = self.datasets_download_info[name]
        self.dataset_name = info["dataset_name"]
        self.download_info = dict(info)
        self.download_info["download_url"] = (
            "https://ml.materialsproject.org/projects/" + info["download_file_name"]
        )
        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, reload=reload, **kwargs)

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.crystal import CrystalDataset
        name = self._matbench_name
        info = self.datasets_download_info[name]
        file_name_download = info.get("extract_file_name") or info["download_file_name"]
        csv_name = os.path.splitext(file_name_download)[0] + ".csv"
        return CrystalDataset(
            data_directory=self.raw_dir,
            dataset_name=name,
            file_name=csv_name,
        )

    def kgcnn_prepare(self, kgcnn_ds):
        """Convert MatBench JSON to pymatgen JSON + CSV, then read structures."""
        from kgcnn_torch.data.base import load_json_file, save_json_file

        name = self._matbench_name
        info = self.datasets_download_info[name]
        file_name_download = info.get("extract_file_name") or info["download_file_name"]
        json_path = os.path.join(self.raw_dir, file_name_download)

        # Extract structure/composition column and build CSV
        file_col = self.datasets_prepare_data_info[name].get("file_column_name", "structure")
        data = load_json_file(json_path)
        data_columns = data["columns"]

        index_structure = 0
        for i, col in enumerate(data_columns):
            if col == file_col:
                index_structure = i
                break

        py_mat_list = [x[index_structure] for x in data["data"]]
        save_json_file(py_mat_list, kgcnn_ds.pymatgen_json_file_path)

        df_dict = {"index": data["index"]}
        for i, col in enumerate(data_columns):
            if i != index_structure:
                df_dict[col] = [x[i] for x in data["data"]]
        df = pd.DataFrame(df_dict)
        df.to_csv(kgcnn_ds.file_path)

        # Read structures into memory
        label_col = self.datasets_read_in_memory_info[name].get("label_column_name")
        kgcnn_ds.read_in_memory(label_column_name=label_col)
