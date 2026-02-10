"""Dataset classes for kgcnn-torch.

All 32 dataset classes ported from kgcnn (Keras) to PyTorch/PyG.

Each dataset inherits from PyG's InMemoryDataset (via KgcnnGraphDataset base)
and preserves the same dataset_name, download_url, and file_name as the Keras
originals so the same data files can be loaded.

Dataset categories:
    - QM datasets: QM7, QM7b, QM8, QM9, QM9MolNet, ISO17
    - MoleculeNet: ESOL, FreeSolv, Lipop, SIDER, ClinTox, Tox21, MoleculeNet2018
    - Force/MD: MD17, MD17Revised
    - Node classification: Cora, CoraLu
    - Graph classification (TUDatasets): MUTAG, Mutagenicity, PROTEINS, GraphTU2020
    - Materials/MatBench: MatBench2020, MatProjectDielectric, MatProjectEForm,
      MatProjectGap, MatProjectIsMetal, MatProjectJdft2d, MatProjectLogGVRH,
      MatProjectLogKVRH, MatProjectPerovskites, MatProjectPhonons
    - Custom: MatPES2k
"""

# Base classes
from kgcnn_torch.data.datasets._base import KgcnnGraphDataset

# QM datasets
from kgcnn_torch.data.datasets.QM7Dataset import QM7Dataset
from kgcnn_torch.data.datasets.QM7bDataset import QM7bDataset
from kgcnn_torch.data.datasets.QM8Dataset import QM8Dataset
from kgcnn_torch.data.datasets.QM9Dataset import QM9Dataset
from kgcnn_torch.data.datasets.QM9MolNetDataset import QM9MolNetDataset
from kgcnn_torch.data.datasets.ISO17Dataset import ISO17Dataset

# MoleculeNet datasets
from kgcnn_torch.data.datasets.MoleculeNetDataset2018 import MoleculeNetDataset2018
from kgcnn_torch.data.datasets.ESOLDataset import ESOLDataset
from kgcnn_torch.data.datasets.FreeSolvDataset import FreeSolvDataset
from kgcnn_torch.data.datasets.LipopDataset import LipopDataset
from kgcnn_torch.data.datasets.SIDERDataset import SIDERDataset
from kgcnn_torch.data.datasets.ClinToxDataset import ClinToxDataset
from kgcnn_torch.data.datasets.Tox21MolNetDataset import Tox21MolNetDataset

# Force/MD datasets
from kgcnn_torch.data.datasets.MD17Dataset import MD17Dataset
from kgcnn_torch.data.datasets.MD17RevisedDataset import MD17RevisedDataset

# Node classification datasets
from kgcnn_torch.data.datasets.CoraDataset import CoraDataset
from kgcnn_torch.data.datasets.CoraLuDataset import CoraLuDataset

# Graph classification (TUDatasets)
from kgcnn_torch.data.datasets.GraphTUDataset2020 import GraphTUDataset2020
from kgcnn_torch.data.datasets.MUTAGDataset import MUTAGDataset
from kgcnn_torch.data.datasets.MutagenicityDataset import MutagenicityDataset
from kgcnn_torch.data.datasets.PROTEINSDataset import PROTEINSDataset

# MatBench / Materials Project datasets
from kgcnn_torch.data.datasets.MatBenchDataset2020 import MatBenchDataset2020
from kgcnn_torch.data.datasets.MatProjectDielectricDataset import MatProjectDielectricDataset
from kgcnn_torch.data.datasets.MatProjectEFormDataset import MatProjectEFormDataset
from kgcnn_torch.data.datasets.MatProjectGapDataset import MatProjectGapDataset
from kgcnn_torch.data.datasets.MatProjectIsMetalDataset import MatProjectIsMetalDataset
from kgcnn_torch.data.datasets.MatProjectJdft2dDataset import MatProjectJdft2dDataset
from kgcnn_torch.data.datasets.MatProjectLogGVRHDataset import MatProjectLogGVRHDataset
from kgcnn_torch.data.datasets.MatProjectLogKVRHDataset import MatProjectLogKVRHDataset
from kgcnn_torch.data.datasets.MatProjectPerovskitesDataset import MatProjectPerovskitesDataset
from kgcnn_torch.data.datasets.MatProjectPhononsDataset import MatProjectPhononsDataset

# Custom datasets
from kgcnn_torch.data.datasets.MatPES2kDataset import MatPES2kDataset

__all__ = [
    # Base
    "KgcnnGraphDataset",
    # QM
    "QM7Dataset",
    "QM7bDataset",
    "QM8Dataset",
    "QM9Dataset",
    "QM9MolNetDataset",
    "ISO17Dataset",
    # MoleculeNet
    "MoleculeNetDataset2018",
    "ESOLDataset",
    "FreeSolvDataset",
    "LipopDataset",
    "SIDERDataset",
    "ClinToxDataset",
    "Tox21MolNetDataset",
    # Force/MD
    "MD17Dataset",
    "MD17RevisedDataset",
    # Node classification
    "CoraDataset",
    "CoraLuDataset",
    # Graph classification (TU)
    "GraphTUDataset2020",
    "MUTAGDataset",
    "MutagenicityDataset",
    "PROTEINSDataset",
    # MatBench / Materials
    "MatBenchDataset2020",
    "MatProjectDielectricDataset",
    "MatProjectEFormDataset",
    "MatProjectGapDataset",
    "MatProjectIsMetalDataset",
    "MatProjectJdft2dDataset",
    "MatProjectLogGVRHDataset",
    "MatProjectLogKVRHDataset",
    "MatProjectPerovskitesDataset",
    "MatProjectPhononsDataset",
    # Custom
    "MatPES2kDataset",
]
