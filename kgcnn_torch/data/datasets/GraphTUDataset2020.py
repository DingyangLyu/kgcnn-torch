"""GraphTU Dataset 2020 base class for kgcnn-torch.

Base class for loading graph datasets published by TU Dortmund University.
Downloads and loads TUDatasets in a generic way.

References:
    (1) C. Morris et al., TUDataset: A collection of benchmark datasets for learning with graphs.
        ICML 2020 Workshop on Graph Representation Learning and Beyond (GRL+ 2020).
"""
import os
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset


class GraphTUDataset2020(KgcnnGraphDataset):
    """Base class for TU Dortmund graph classification datasets.

    Subclasses set ``_tu_name`` to select which dataset to load.
    """

    tudataset_ids = [
        "AIDS", "alchemy_full", "aspirin", "benzene", "BZR", "BZR_MD", "COX2", "COX2_MD",
        "DHFR", "DHFR_MD", "ER_MD", "ethanol", "FRANKENSTEIN", "malonaldehyde", "MCF-7",
        "MCF-7H", "MOLT-4", "MOLT-4H", "Mutagenicity", "MUTAG", "naphthalene", "NCI1",
        "NCI109", "NCI-H23", "NCI-H23H", "OVCAR-8", "OVCAR-8H", "P388", "P388H", "PC-3",
        "PC-3H", "PTC_FM", "PTC_FR", "PTC_MM", "PTC_MR", "QM9", "salicylic_acid", "SF-295",
        "SF-295H", "SN12C", "SN12CH", "SW-620", "SW-620H", "toluene",
        "COLLAB", "dblp_ct1", "dblp_ct2", "DBLP_v1", "deezer_ego_nets", "facebook_ct1",
        "facebook_ct2", "github_stargazers", "highschool_ct1", "highschool_ct2",
        "IMDB-BINARY", "IMDB-MULTI", "infectious_ct1", "infectious_ct2", "mit_ct1",
        "mit_ct2", "REDDIT-BINARY", "REDDIT-MULTI-5K", "REDDIT-MULTI-12K", "reddit_threads",
        "tumblr_ct1", "tumblr_ct2", "twitch_egos", "TWITTER-Real-Graph-Partial",
        "DD", "ENZYMES", "KKI", "OHSU", "Peking_1", "PROTEINS", "PROTEINS_full",
        "COIL-DEL", "COIL-RAG", "Cuneiform", "Fingerprint", "FIRSTMM_DB",
        "Letter-high", "Letter-low", "Letter-med", "MSRC_9", "MSRC_21", "MSRC_21C",
        "COLORS-3", "SYNTHETIC", "SYNTHETICnew", "Synthie", "TRIANGLES",
    ]

    _tu_name: str = None

    def __init__(self, tu_name: str = None, root=None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        name = tu_name or self._tu_name
        if name is None:
            raise ValueError("Must provide tu_name or set _tu_name on subclass.")
        if name not in self.tudataset_ids:
            raise ValueError(
                f"Unknown TUDataset '{name}'. "
                f"Not in known tudataset_ids list."
            )
        self._tu_name = name
        self.dataset_name = name
        self.download_info = {
            "data_directory_name": name,
            "download_url": f"https://www.chrsmrrs.com/graphkerneldatasets/{name}.zip",
            "download_file_name": f"{name}.zip",
            "unpack_zip": True,
            "unpack_directory_name": name,
            "dataset_name": name,
        }
        super().__init__(root=root, transform=transform, pre_transform=pre_transform,
                         pre_filter=pre_filter, reload=reload)

    def _create_kgcnn_dataset(self):
        from kgcnn_torch.data.tudataset import GraphTUDataset
        name = self._tu_name
        return GraphTUDataset(
            data_directory=self.raw_dir,
            dataset_name=name,
            file_directory=os.path.join(name, name),
        )

    def kgcnn_prepare(self, kgcnn_ds):
        kgcnn_ds.read_in_memory()
