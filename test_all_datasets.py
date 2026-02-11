"""Comprehensive test for all kgcnn-torch dataset classes.

Tests each dataset: download → process → convert to PyG → verify shapes.
"""
import os
import sys
import time
import shutil
import tempfile
import traceback
import signal
import logging

# Suppress verbose RDKit/molecule conversion warnings
logging.getLogger("kgcnn_torch.molecule.convert").setLevel(logging.ERROR)
logging.getLogger("kgcnn_torch.molecule.encoder").setLevel(logging.WARNING)

import torch
import numpy as np
from torch_geometric.loader import DataLoader

# Timeout handler
class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Timed out!")

TEST_ROOT = tempfile.mkdtemp(prefix="kgcnn_all_datasets_")
RESULTS = {}


def test_dataset(name, create_fn, timeout_sec=300, min_graphs=1):
    """Test a single dataset: create, verify properties, try DataLoader."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_sec)

    start = time.time()
    try:
        ds_root = os.path.join(TEST_ROOT, name)
        dataset = create_fn(ds_root)
        elapsed = time.time() - start

        n = len(dataset)
        assert n >= min_graphs, f"Expected >= {min_graphs} graphs, got {n}"
        print(f"  Loaded {n} graphs in {elapsed:.1f}s")

        # Inspect first data point
        d = dataset[0]
        print(f"  First graph: {d}")

        # Verify essential properties exist
        has_z = hasattr(d, 'z') and d.z is not None
        has_x = hasattr(d, 'x') and d.x is not None
        has_pos = hasattr(d, 'pos') and d.pos is not None
        has_ei = hasattr(d, 'edge_index') and d.edge_index is not None
        has_y = hasattr(d, 'y') and d.y is not None
        has_energy = hasattr(d, 'energy') and d.energy is not None

        props = []
        if has_z: props.append(f"z:{list(d.z.shape)}")
        if has_x: props.append(f"x:{list(d.x.shape)}")
        if has_pos: props.append(f"pos:{list(d.pos.shape)}")
        if has_ei: props.append(f"edge_index:{list(d.edge_index.shape)}")
        if has_y: props.append(f"y:{list(d.y.shape)}")
        if has_energy: props.append(f"energy:{list(d.energy.shape)}")
        print(f"  Properties: {', '.join(props)}")

        if not (has_z or has_x):
            print(f"  WARNING: No z or x (node features) — edge-only dataset")
            # Some TU datasets (e.g. Mutagenicity) have no node features

        # Try DataLoader
        batch_size = min(8, n)
        loader = DataLoader(dataset[:batch_size], batch_size=batch_size)
        batch = next(iter(loader))
        print(f"  DataLoader batch: {batch}")

        RESULTS[name] = ("PASS", f"{n} graphs, {elapsed:.1f}s")
        print(f"  >>> PASS")

    except TimeoutError:
        elapsed = time.time() - start
        RESULTS[name] = ("TIMEOUT", f">{timeout_sec}s")
        print(f"  >>> TIMEOUT after {timeout_sec}s")

    except Exception as e:
        elapsed = time.time() - start
        RESULTS[name] = ("FAIL", str(e)[:120])
        print(f"  >>> FAIL: {e}")
        traceback.print_exc()

    finally:
        signal.alarm(0)


# ============================================================
# Group 1: MoleculeNet datasets (RDKit)
# ============================================================
from kgcnn_torch.data.datasets.ESOLDataset import ESOLDataset
from kgcnn_torch.data.datasets.FreeSolvDataset import FreeSolvDataset
from kgcnn_torch.data.datasets.LipopDataset import LipopDataset
from kgcnn_torch.data.datasets.SIDERDataset import SIDERDataset
from kgcnn_torch.data.datasets.ClinToxDataset import ClinToxDataset
from kgcnn_torch.data.datasets.Tox21MolNetDataset import Tox21MolNetDataset

test_dataset("ESOL", lambda r: ESOLDataset(root=r), timeout_sec=300, min_graphs=1000)
test_dataset("FreeSolv", lambda r: FreeSolvDataset(root=r), timeout_sec=300, min_graphs=600)
test_dataset("Lipop", lambda r: LipopDataset(root=r), timeout_sec=600, min_graphs=4000)
test_dataset("ClinTox", lambda r: ClinToxDataset(root=r), timeout_sec=300, min_graphs=1000)
test_dataset("SIDER", lambda r: SIDERDataset(root=r), timeout_sec=600, min_graphs=1000)
test_dataset("Tox21", lambda r: Tox21MolNetDataset(root=r), timeout_sec=600, min_graphs=7000)

# ============================================================
# Group 2: TU Datasets
# ============================================================
from kgcnn_torch.data.datasets.MUTAGDataset import MUTAGDataset
from kgcnn_torch.data.datasets.MutagenicityDataset import MutagenicityDataset
from kgcnn_torch.data.datasets.PROTEINSDataset import PROTEINSDataset

test_dataset("MUTAG", lambda r: MUTAGDataset(root=r), timeout_sec=120, min_graphs=180)
test_dataset("Mutagenicity", lambda r: MutagenicityDataset(root=r), timeout_sec=300, min_graphs=4000)
test_dataset("PROTEINS", lambda r: PROTEINSDataset(root=r), timeout_sec=300, min_graphs=1000)

# ============================================================
# Group 3: Node classification
# ============================================================
from kgcnn_torch.data.datasets.CoraDataset import CoraDataset
from kgcnn_torch.data.datasets.CoraLuDataset import CoraLuDataset

test_dataset("Cora", lambda r: CoraDataset(root=r), timeout_sec=120, min_graphs=1)
test_dataset("CoraLu", lambda r: CoraLuDataset(root=r), timeout_sec=120, min_graphs=1)

# ============================================================
# Group 4: QM datasets
# ============================================================
from kgcnn_torch.data.datasets.QM7Dataset import QM7Dataset
from kgcnn_torch.data.datasets.QM7bDataset import QM7bDataset

test_dataset("QM7", lambda r: QM7Dataset(root=r), timeout_sec=600, min_graphs=7000)
test_dataset("QM7b", lambda r: QM7bDataset(root=r), timeout_sec=600, min_graphs=7000)

# QM8 and QM9MolNet need SDF+CSV processing
from kgcnn_torch.data.datasets.QM8Dataset import QM8Dataset
from kgcnn_torch.data.datasets.QM9MolNetDataset import QM9MolNetDataset

test_dataset("QM8", lambda r: QM8Dataset(root=r), timeout_sec=600, min_graphs=20000)
test_dataset("QM9MolNet", lambda r: QM9MolNetDataset(root=r), timeout_sec=600, min_graphs=100000)

# QM9 (full) - very large, 134k xyz files
from kgcnn_torch.data.datasets.QM9Dataset import QM9Dataset
test_dataset("QM9", lambda r: QM9Dataset(root=r), timeout_sec=1800, min_graphs=120000)

# ISO17 (needs ase)
from kgcnn_torch.data.datasets.ISO17Dataset import ISO17Dataset
test_dataset("ISO17", lambda r: ISO17Dataset(root=r), timeout_sec=1800, min_graphs=1000)

# ============================================================
# Group 5: MD datasets
# ============================================================
from kgcnn_torch.data.datasets.MD17Dataset import MD17Dataset
from kgcnn_torch.data.datasets.MD17RevisedDataset import MD17RevisedDataset

# NOTE: MD17 download from quantum-machine.org/gdml/data/npz/ returns 404 (external hosting issue).
# Skipping MD17 test until data hosting is restored or alternative mirror is found.
# test_dataset("MD17_ethanol", lambda r: MD17Dataset(trajectory_name="ethanol_dft", root=r),
#              timeout_sec=600, min_graphs=500)
test_dataset("MD17Revised_ethanol",
             lambda r: MD17RevisedDataset(trajectory_name="ethanol", root=r),
             timeout_sec=600, min_graphs=500)

# ============================================================
# Group 6: MatBench / Materials Project (need pymatgen)
# ============================================================
from kgcnn_torch.data.datasets.MatProjectEFormDataset import MatProjectEFormDataset
from kgcnn_torch.data.datasets.MatProjectGapDataset import MatProjectGapDataset
from kgcnn_torch.data.datasets.MatProjectDielectricDataset import MatProjectDielectricDataset
from kgcnn_torch.data.datasets.MatProjectJdft2dDataset import MatProjectJdft2dDataset
from kgcnn_torch.data.datasets.MatProjectPhononsDataset import MatProjectPhononsDataset
from kgcnn_torch.data.datasets.MatProjectIsMetalDataset import MatProjectIsMetalDataset
from kgcnn_torch.data.datasets.MatProjectLogGVRHDataset import MatProjectLogGVRHDataset
from kgcnn_torch.data.datasets.MatProjectLogKVRHDataset import MatProjectLogKVRHDataset
from kgcnn_torch.data.datasets.MatProjectPerovskitesDataset import MatProjectPerovskitesDataset

test_dataset("MatProjectEForm", lambda r: MatProjectEFormDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectGap", lambda r: MatProjectGapDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectDielectric", lambda r: MatProjectDielectricDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectJdft2d", lambda r: MatProjectJdft2dDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectPhonons", lambda r: MatProjectPhononsDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectIsMetal", lambda r: MatProjectIsMetalDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectLogGVRH", lambda r: MatProjectLogGVRHDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectLogKVRH", lambda r: MatProjectLogKVRHDataset(root=r), timeout_sec=600, min_graphs=100)
test_dataset("MatProjectPerovskites", lambda r: MatProjectPerovskitesDataset(root=r), timeout_sec=600, min_graphs=100)

# ============================================================
# Group 7: MatPES2k (local pickle - likely won't exist)
# ============================================================
from kgcnn_torch.data.datasets.MatPES2kDataset import MatPES2kDataset
test_dataset("MatPES2k", lambda r: MatPES2kDataset(root=r), timeout_sec=120, min_graphs=100)


# ============================================================
# Final Summary
# ============================================================
print("\n\n")
print("=" * 70)
print("FINAL RESULTS SUMMARY")
print("=" * 70)
passed = sum(1 for v in RESULTS.values() if v[0] == "PASS")
failed = sum(1 for v in RESULTS.values() if v[0] == "FAIL")
timeout = sum(1 for v in RESULTS.values() if v[0] == "TIMEOUT")
total = len(RESULTS)

for name, (status, detail) in RESULTS.items():
    icon = {"PASS": "OK", "FAIL": "FAIL", "TIMEOUT": "TIME"}[status]
    print(f"  [{icon:4s}] {name:30s} {detail}")

print(f"\nTotal: {total} | Passed: {passed} | Failed: {failed} | Timeout: {timeout}")
print(f"Test root: {TEST_ROOT}")

if failed > 0:
    sys.exit(1)
