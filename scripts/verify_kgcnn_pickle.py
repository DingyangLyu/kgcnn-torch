#!/usr/bin/env python
"""Verify a KGCNN ``*.kgcnn.pickle`` cache can be consumed by kgcnn-torch.

This script intentionally does not modify the pickle contents. It only:
1) Loads the cached list of graph dicts into ``MemoryGraphDataset``.
2) Converts to a list of PyG ``Data`` objects via ``to_pyg_list()``.

Usage:
    python scripts/verify_kgcnn_pickle.py /path/to/DATASET.kgcnn.pickle
"""

import sys
from pathlib import Path

from kgcnn_torch.data.base import MemoryGraphDataset


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/verify_kgcnn_pickle.py /path/to/DATASET.kgcnn.pickle")
        return 2

    pickle_path = Path(sys.argv[1]).expanduser().resolve()
    if not pickle_path.exists():
        print(f"ERROR: file not found: {pickle_path}")
        return 2

    ds = MemoryGraphDataset(dataset_name="VerifyPickle")
    ds.load(str(pickle_path))
    print(f"Loaded graphs: {len(ds)}")

    pyg_list = ds.to_pyg_list()
    print(f"Converted to PyG Data: {len(pyg_list)}")
    if pyg_list:
        print(f"First graph keys: {sorted(list(pyg_list[0].keys()))}")
        print(f"First graph num_nodes: {getattr(pyg_list[0], 'num_nodes', None)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

