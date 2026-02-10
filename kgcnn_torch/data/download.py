"""Download utilities for dataset management."""
import gzip
import os
import shutil
import tarfile
import zipfile
import logging

import requests

logging.basicConfig()
module_logger = logging.getLogger(__name__)
module_logger.setLevel(logging.INFO)


class DownloadDataset:
    """Download and extraction utility for datasets.

    Provides methods for downloading files and extracting archives (tar, zip, gz).
    Downloads are stored in a main directory (default: ``~/.kgcnn/datasets``).

    .. warning::
        Downloads are not checked for safety or malware. Use with caution!

    Example:

    .. code-block:: python

        from kgcnn_torch.data.download import DownloadDataset
        dl = DownloadDataset(
            download_url="https://example.com/data.zip",
            data_directory_name="my_dataset",
            download_file_name="data.zip",
            unpack_zip=True,
            unpack_directory_name="data",
        )
    """

    def __init__(self,
                 dataset_name: str = None,
                 download_file_name: str = None,
                 data_directory_name: str = None,
                 unpack_directory_name: str = None,
                 extract_file_name: str = None,
                 download_url: str = None,
                 unpack_tar: bool = False,
                 unpack_zip: bool = False,
                 extract_gz: bool = False,
                 reload: bool = False,
                 verbose: int = 10,
                 data_main_dir: str = os.path.join(os.path.expanduser("~"), ".kgcnn", "datasets"),
                 execute_download_dataset_on_init: bool = True):
        """Initialize download configuration.

        Args:
            dataset_name (str): Name of the dataset. Default is None.
            download_file_name (str): Name of the file to download to. Default is None.
            data_directory_name (str): Name of the dataset directory. Default is None.
            unpack_directory_name (str): Name of subdirectory for extraction. Default is None.
            extract_file_name (str): Name of a gz-file to extract. Default is None.
            download_url (str): URL to download from. Default is None.
            unpack_tar (bool): Whether to unpack a tar archive. Default is False.
            unpack_zip (bool): Whether to unpack a zip archive. Default is False.
            extract_gz (bool): Whether to extract a gz file. Default is False.
            reload (bool): Whether to re-download existing files. Default is False.
            verbose (int): Logging level. Default is 10.
            data_main_dir (str): Main directory for all datasets.
            execute_download_dataset_on_init (bool): Whether to download on construction.
        """
        self.data_main_dir = data_main_dir
        self.dataset_name = dataset_name
        self.download_file_name = download_file_name
        self.data_directory_name = data_directory_name
        self.unpack_directory_name = unpack_directory_name
        self.extract_file_name = extract_file_name
        self.download_url = download_url
        self.unpack_tar = unpack_tar
        self.unpack_zip = unpack_zip
        self.extract_gz = extract_gz
        self.download_reload = reload
        self.logger_download = module_logger
        self.logger_download.setLevel(verbose)
        self.execute_download_dataset_on_init = bool(execute_download_dataset_on_init)

        if self.execute_download_dataset_on_init:
            self.download_dataset_to_disk()

    def download_dataset_to_disk(self):
        """Run the full download and extraction pipeline."""
        self.logger_download.info(
            "Checking and possibly downloading dataset with name %s" % str(self.dataset_name))

        if self.data_directory_name is not None:
            self.setup_dataset_main(self.data_main_dir, logger=self.logger_download)
            self.setup_dataset_dir(self.data_main_dir, self.data_directory_name,
                                   logger=self.logger_download)

        if self.download_url is not None:
            self.download_database(
                os.path.join(self.data_main_dir, self.data_directory_name),
                self.download_url, self.download_file_name,
                overwrite=self.download_reload, logger=self.logger_download)

        if self.unpack_tar:
            self.unpack_tar_file(
                os.path.join(self.data_main_dir, self.data_directory_name),
                self.download_file_name, self.unpack_directory_name,
                overwrite=self.download_reload, logger=self.logger_download)

        if self.unpack_zip:
            self.unpack_zip_file(
                os.path.join(self.data_main_dir, self.data_directory_name),
                self.download_file_name, self.unpack_directory_name,
                overwrite=self.download_reload, logger=self.logger_download)

        if self.extract_gz:
            self.extract_gz_file(
                os.path.join(self.data_main_dir, self.data_directory_name),
                self.download_file_name, self.extract_file_name,
                overwrite=self.download_reload, logger=self.logger_download)

    @staticmethod
    def setup_dataset_main(data_main_dir: str, logger=None):
        """Create the main directory for all datasets.

        Args:
            data_main_dir (str): Path to create.
            logger: Optional logger.
        """
        os.makedirs(data_main_dir, exist_ok=True)
        if logger is not None:
            logger.info("Dataset directory located at %s" % data_main_dir)

    @staticmethod
    def setup_dataset_dir(data_main_dir: str, data_directory: str, logger=None):
        """Create a directory for a specific dataset.

        Args:
            data_main_dir (str): Parent directory for all datasets.
            data_directory (str): Name of the dataset subdirectory.
            logger: Optional logger.
        """
        path = os.path.join(data_main_dir, data_directory)
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            if logger is not None:
                logger.info("Dataset directory found. Done.")

    @staticmethod
    def download_database(path: str, download_url: str, filename: str,
                          overwrite: bool = False, logger=None) -> str:
        """Download a file from a URL.

        Args:
            path (str): Target directory.
            download_url (str): URL to download from.
            filename (str): Target filename.
            overwrite (bool): Overwrite existing file. Default is False.
            logger: Optional logger.

        Returns:
            str: Full path of the downloaded file.
        """
        def _log(msg):
            if logger is not None:
                logger.info(msg)

        filepath = os.path.join(path, filename)
        if not os.path.exists(filepath) or overwrite:
            _log("Downloading dataset...")
            r = requests.get(download_url, allow_redirects=True)
            with open(filepath, "wb") as f:
                f.write(r.content)
        else:
            _log("Dataset found. Done.")
        return filepath

    @staticmethod
    def unpack_tar_file(path: str, filename: str, unpack_directory: str,
                        overwrite: bool = False, logger=None) -> str:
        """Extract a tar archive.

        Args:
            path (str): Directory containing the tar file.
            filename (str): Name of the tar file.
            unpack_directory (str): Subdirectory to extract to.
            overwrite (bool): Overwrite existing extraction. Default is False.
            logger: Optional logger.

        Returns:
            str: Path to the extracted directory.
        """
        def _log(msg):
            if logger is not None:
                logger.info(msg)

        extract_path = os.path.join(path, unpack_directory)
        if not os.path.exists(extract_path):
            _log("Creating directory...")
            os.mkdir(extract_path)
        else:
            _log("Directory for extraction exists. Done.")
            if not overwrite:
                _log("Not extracting tar file. Stopped.")
                return extract_path

        _log("Read tar file...")
        archive = tarfile.open(os.path.join(path, filename), "r")
        _log("Extracting tar file...")
        # Validate member paths to prevent path traversal attacks.
        for member in archive.getmembers():
            member_path = os.path.join(extract_path, member.name)
            abs_extract = os.path.realpath(extract_path)
            abs_member = os.path.realpath(member_path)
            if not abs_member.startswith(abs_extract + os.sep) and abs_member != abs_extract:
                raise RuntimeError(
                    f"Tar archive contains path traversal attempt: {member.name!r}"
                )
        archive.extractall(extract_path)
        archive.close()
        return extract_path

    @staticmethod
    def unpack_zip_file(path: str, filename: str, unpack_directory: str,
                        overwrite: bool = False, logger=None) -> str:
        """Extract a zip archive.

        Args:
            path (str): Directory containing the zip file.
            filename (str): Name of the zip file.
            unpack_directory (str): Subdirectory to extract to.
            overwrite (bool): Overwrite existing extraction. Default is False.
            logger: Optional logger.

        Returns:
            str: Path to the extracted directory.
        """
        def _log(msg):
            if logger is not None:
                logger.info(msg)

        extract_path = os.path.join(path, unpack_directory)
        if os.path.exists(extract_path):
            _log("Directory for extraction exists. Done.")
            if not overwrite:
                _log("Not extracting zip file. Stopped.")
                return extract_path

        _log("Read zip file...")
        archive = zipfile.ZipFile(os.path.join(path, filename), "r")
        _log("Extracting zip file...")
        archive.extractall(extract_path)
        archive.close()
        return extract_path

    @staticmethod
    def extract_gz_file(path: str, filename: str, out_filename: str = None,
                        overwrite: bool = False, logger=None) -> str:
        """Extract a gz-compressed file.

        Args:
            path (str): Directory containing the gz file.
            filename (str): Name of the gz file.
            out_filename (str): Name for the extracted file. If None, derived from filename.
            overwrite (bool): Overwrite existing extraction. Default is False.
            logger: Optional logger.

        Returns:
            str: Path to the extracted file.
        """
        def _log(msg):
            if logger is not None:
                logger.info(msg)

        if out_filename is None:
            out_filename = filename.replace(".gz", "")

        out_path = os.path.join(path, out_filename)
        if os.path.exists(out_path):
            _log("Extracted file exists. Done.")
            if not overwrite:
                _log("Not extracting gz-file. Stopped.")
                return out_path

        _log("Extract gz-file...")
        with gzip.open(os.path.join(path, filename), "rb") as f_in:
            with open(out_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

        return out_path
