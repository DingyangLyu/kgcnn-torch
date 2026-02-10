"""MoleculeNet 2018 base dataset for kgcnn-torch.

Base class for MoleculeNet benchmark datasets that loads SMILES-based molecular data.
Supports: ESOL, FreeSolv, Lipop, PCBA, MUV, HIV, BACE, BBBP, Tox21, ToxCast, SIDER, ClinTox.

References:
    (1) Z. Wu et al., MoleculeNet: A Benchmark for Molecular Machine Learning,
        arXiv: 1703.00564, 2017.
"""
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class MoleculeNetDataset2018(KgcnnGraphDataset):
    """Base class for MoleculeNet 2018 datasets.

    Subclasses only need to set ``_molnet_name`` to select which dataset to load.
    All download URLs, file names, and label column names are configured here.
    """

    datasets_download_info = {
        "ESOL": {"dataset_name": "ESOL", "download_file_name": "delaney-processed.csv",
                 "data_directory_name": "ESOL"},
        "FreeSolv": {"dataset_name": "FreeSolv", "data_directory_name": "FreeSolv",
                     "download_file_name": "SAMPL.csv"},
        "Lipop": {"dataset_name": "Lipop", "data_directory_name": "Lipop",
                  "download_file_name": "Lipophilicity.csv"},
        "PCBA": {"dataset_name": "PCBA", "data_directory_name": "PCBA",
                 "download_file_name": "pcba.csv.gz", "extract_gz": True,
                 "extract_file_name": "pcba.csv"},
        "MUV": {"dataset_name": "MUV", "data_directory_name": "MUV",
                "download_file_name": "muv.csv.gz", "extract_gz": True,
                "extract_file_name": "muv.csv"},
        "HIV": {"dataset_name": "HIV", "data_directory_name": "HIV",
                "download_file_name": "HIV.csv"},
        "BACE": {"dataset_name": "BACE", "data_directory_name": "BACE",
                 "download_file_name": "bace.csv"},
        "BBBP": {"dataset_name": "BBBP", "data_directory_name": "BBBP",
                 "download_file_name": "BBBP.csv"},
        "Tox21": {"dataset_name": "Tox21", "data_directory_name": "Tox21",
                  "download_file_name": "tox21.csv.gz", "extract_gz": True,
                  "extract_file_name": "tox21.csv"},
        "ToxCast": {"dataset_name": "ToxCast", "data_directory_name": "ToxCast",
                    "download_file_name": "toxcast_data.csv.gz", "extract_gz": True,
                    "extract_file_name": "toxcast_data.csv"},
        "SIDER": {"dataset_name": "SIDER", "data_directory_name": "SIDER",
                  "download_file_name": "sider.csv.gz", "extract_gz": True,
                  "extract_file_name": "sider.csv"},
        "ClinTox": {"dataset_name": "ClinTox", "data_directory_name": "ClinTox",
                    "download_file_name": "clintox.csv.gz", "extract_gz": True,
                    "extract_file_name": "clintox.csv"},
    }

    datasets_prepare_data_info = {
        "ESOL": {"make_conformers": True, "add_hydrogen": True},
        "FreeSolv": {"make_conformers": True, "add_hydrogen": True},
        "Lipop": {"make_conformers": True, "add_hydrogen": True},
        "PCBA": {"make_conformers": True, "add_hydrogen": True},
        "MUV": {"make_conformers": True, "add_hydrogen": True},
        "HIV": {"make_conformers": True, "add_hydrogen": True},
        "BACE": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "mol"},
        "BBBP": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "Tox21": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "ToxCast": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "SIDER": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
        "ClinTox": {"make_conformers": True, "add_hydrogen": True, "smiles_column_name": "smiles"},
    }

    datasets_read_in_memory_info = {
        "ESOL": {"add_hydrogen": False, "has_conformers": True,
                 "label_column_name": "measured log solubility in mols per litre"},
        "FreeSolv": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "expt"},
        "Lipop": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "exp"},
        "PCBA": {"add_hydrogen": False, "has_conformers": False, "label_column_name": slice(0, 128)},
        "MUV": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(0, 17)},
        "HIV": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "HIV_active"},
        "BACE": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "Class"},
        "BBBP": {"add_hydrogen": False, "has_conformers": True, "label_column_name": "p_np"},
        "Tox21": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(0, 12)},
        "ToxCast": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(1, 618)},
        "SIDER": {"add_hydrogen": False, "has_conformers": True, "label_column_name": slice(1, 28)},
        "ClinTox": {"add_hydrogen": False, "has_conformers": True, "label_column_name": [1, 2]},
    }

    _molnet_name: str = None

    def __init__(self, molnet_name: str = None, root=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        name = molnet_name or self._molnet_name
        if name is None:
            raise ValueError("Must provide molnet_name or set _molnet_name on subclass.")
        if name not in self.datasets_download_info:
            raise ValueError(
                f"Unknown MoleculeNet dataset '{name}'. "
                f"Choose from: {list(self.datasets_download_info.keys())}"
            )
        self._molnet_name = name
        info = self.datasets_download_info[name]
        self.dataset_name = info["dataset_name"]
        self.download_info = dict(info)
        self.download_info["download_url"] = (
            "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/"
            + info["download_file_name"]
        )
        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, reload=reload)

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.moleculenet import MoleculeNetDataset
        name = self._molnet_name
        info = self.datasets_download_info[name]
        file_name = info.get("extract_file_name") or info["download_file_name"]
        return MoleculeNetDataset(
            data_directory=self.raw_dir,
            dataset_name=self.dataset_name,
            file_name=file_name,
        )

    def kgcnn_prepare(self, kgcnn_ds):
        name = self._molnet_name
        kgcnn_ds.prepare_data(overwrite=False, **self.datasets_prepare_data_info[name])
        kgcnn_ds.read_in_memory(**self.datasets_read_in_memory_info[name])
