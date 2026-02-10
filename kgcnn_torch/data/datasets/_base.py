"""Base helpers for kgcnn-torch dataset wrappers.

Each dataset class is a thin wrapper around PyG's InMemoryDataset that stores
the KGCNN dataset metadata (dataset_name, download_url, file_name, label_names,
label_units, etc.) and delegates data loading to the KGCNN MemoryGraphDataset
pipeline via a ``kgcnn_prepare`` callback.

Typical usage pattern:

    class MyDataset(KgcnnGraphDataset):
        dataset_name = "MyDataset"
        download_info = { ... }

        def kgcnn_prepare(self, kgcnn_ds):
            # Load raw data into kgcnn_ds (a MemoryGraphDataset)
            ...
"""
import os
import logging
from typing import Optional

from torch_geometric.data import InMemoryDataset
from kgcnn_torch.data.base import MemoryGraphDataset, DownloadDataset

logger = logging.getLogger(__name__)


class KgcnnGraphDataset(InMemoryDataset):
    """Base PyG InMemoryDataset for all kgcnn-torch dataset wrappers.

    Subclasses should set:
        - ``dataset_name`` (str): unique name.
        - ``download_info`` (dict): download metadata (url, file names, unpack flags).
        - ``label_names`` (str or list): label column name(s).
        - ``label_units`` (str or list): label unit(s).

    And override ``kgcnn_prepare(self, kgcnn_ds)`` to populate the KGCNN dataset,
    and optionally ``_create_kgcnn_dataset()`` to return a specialized KGCNN dataset type.

    Args:
        root: Root directory where the dataset should be saved.
        transform: PyG transform applied to each Data object.
        pre_transform: PyG pre-transform applied once during processing.
        pre_filter: PyG pre-filter applied once during processing.
        reload: If True, forces reprocessing (deletes processed files).
    """

    # Override in subclasses
    dataset_name: str = "KgcnnGraphDataset"
    download_info: dict = {}
    label_names = None
    label_units = None

    def __init__(self, root: Optional[str] = None,
                 transform=None, pre_transform=None, pre_filter=None,
                 reload: bool = False):
        if root is None:
            root = os.path.join(os.path.expanduser("~"), "kgcnn_datasets", self.dataset_name)
        self._reload = reload

        if reload:
            # Remove processed files to trigger reprocessing
            proc_dir = os.path.join(root, "processed")
            if os.path.isdir(proc_dir):
                for fname in self.processed_file_names:
                    fpath = os.path.join(proc_dir, fname)
                    if os.path.exists(fpath):
                        os.remove(fpath)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return [f"{self.dataset_name}_pyg.pt"]

    @property
    def raw_file_names(self):
        return []

    def download(self):
        """Download is handled by KGCNN pipeline inside process()."""
        pass

    def _create_kgcnn_dataset(self):
        """Create the KGCNN dataset instance. Override for specialized dataset types."""
        return MemoryGraphDataset(
            data_directory=self.raw_dir,
            dataset_name=self.dataset_name,
        )

    def _download_to_raw_dir(self):
        """Download and extract raw data to the PyG raw_dir using download_info."""
        info = self.download_info
        if not info:
            return
        raw = self.raw_dir
        os.makedirs(raw, exist_ok=True)

        if info.get("download_url"):
            DownloadDataset.download_database(
                raw, info["download_url"], info["download_file_name"], logger=logger)

        if info.get("unpack_tar"):
            DownloadDataset.unpack_tar_file(
                raw, info["download_file_name"],
                info.get("unpack_directory_name", ""), logger=logger)

        if info.get("unpack_zip"):
            DownloadDataset.unpack_zip_file(
                raw, info["download_file_name"],
                info.get("unpack_directory_name", ""), logger=logger)

        if info.get("extract_gz"):
            DownloadDataset.extract_gz_file(
                raw, info["download_file_name"],
                info.get("extract_file_name"), logger=logger)

    def process(self):
        """Load data via KGCNN pipeline and convert to PyG Data objects."""
        kgcnn_ds = self._create_kgcnn_dataset()
        pickle_path = os.path.join(self.raw_dir, self.dataset_name + ".kgcnn.pickle")

        if os.path.exists(pickle_path) and not self._reload:
            kgcnn_ds.load(pickle_path)
        else:
            self._download_to_raw_dir()
            self.kgcnn_prepare(kgcnn_ds)
            os.makedirs(os.path.dirname(pickle_path), exist_ok=True)
            kgcnn_ds.save(pickle_path)

        pyg_list = kgcnn_ds.to_pyg_list()

        # Filter out incomplete graphs (e.g. failed RDKit conversions)
        n_before = len(pyg_list)
        pyg_list = [d for d in pyg_list if d.num_nodes is not None and d.num_nodes > 0]
        n_after = len(pyg_list)
        if n_after < n_before:
            logger.info(f"Filtered out {n_before - n_after} incomplete graphs "
                        f"({n_after}/{n_before} remaining).")

        # Ensure uniform attributes across all Data objects for PyG collate
        if pyg_list:
            all_keys = set()
            for d in pyg_list:
                all_keys.update(d.keys())
            # Keep only attributes present in ALL graphs
            common_keys = set(all_keys)
            for d in pyg_list:
                common_keys &= set(d.keys())
            drop_keys = all_keys - common_keys
            if drop_keys:
                logger.info(f"Dropping non-uniform attributes: {drop_keys}")
                for d in pyg_list:
                    for k in drop_keys:
                        if k in d:
                            delattr(d, k)

        if self.pre_filter is not None:
            pyg_list = [d for d in pyg_list if self.pre_filter(d)]
        if self.pre_transform is not None:
            pyg_list = [self.pre_transform(d) for d in pyg_list]

        self.save(pyg_list, self.processed_paths[0])

    def kgcnn_prepare(self, kgcnn_ds: MemoryGraphDataset):
        """Prepare data in the KGCNN MemoryGraphDataset.

        Override in subclasses to load raw data files. The default
        implementation raises an error asking the user to place a
        pre-built pickle cache.
        """
        raise RuntimeError(
            f"No cached dataset found for '{self.dataset_name}'. "
            f"Place a '{self.dataset_name}.kgcnn.pickle' file in '{self.raw_dir}' "
            f"or override kgcnn_prepare() in your dataset subclass."
        )

    def __repr__(self):
        return f"{self.__class__.__name__}({len(self)})"
