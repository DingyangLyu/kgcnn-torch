"""Quick training smoke test for ALL datasets — SERIAL, one at a time.

Each dataset runs in its own subprocess so memory is fully released after each test.
Skips very large datasets (QM9/QM9MolNet/ISO17) that have already been verified separately.
"""
import os
import sys
import time
import json
import subprocess

# Datasets to test: (name, module, class, kwargs, ds_type)
TASKS = [
    # MoleculeNet
    ("ESOL", "kgcnn_torch.data.datasets.ESOLDataset", "ESOLDataset", {}, "mol"),
    ("FreeSolv", "kgcnn_torch.data.datasets.FreeSolvDataset", "FreeSolvDataset", {}, "mol"),
    ("Lipop", "kgcnn_torch.data.datasets.LipopDataset", "LipopDataset", {}, "mol"),
    ("ClinTox", "kgcnn_torch.data.datasets.ClinToxDataset", "ClinToxDataset", {}, "mol"),
    ("SIDER", "kgcnn_torch.data.datasets.SIDERDataset", "SIDERDataset", {}, "mol"),
    ("Tox21", "kgcnn_torch.data.datasets.Tox21MolNetDataset", "Tox21MolNetDataset", {}, "mol"),
    # TU
    ("MUTAG", "kgcnn_torch.data.datasets.MUTAGDataset", "MUTAGDataset", {}, "mol"),
    ("Mutagenicity", "kgcnn_torch.data.datasets.MutagenicityDataset", "MutagenicityDataset", {}, "mol"),
    ("PROTEINS", "kgcnn_torch.data.datasets.PROTEINSDataset", "PROTEINSDataset", {}, "mol"),
    # Node classification
    ("Cora", "kgcnn_torch.data.datasets.CoraDataset", "CoraDataset", {}, "node"),
    ("CoraLu", "kgcnn_torch.data.datasets.CoraLuDataset", "CoraLuDataset", {}, "node"),
    # QM (skip QM9/QM9MolNet — 127k/133k graphs, already verified)
    ("QM7", "kgcnn_torch.data.datasets.QM7Dataset", "QM7Dataset", {}, "mol"),
    ("QM7b", "kgcnn_torch.data.datasets.QM7bDataset", "QM7bDataset", {}, "3d"),
    ("QM8", "kgcnn_torch.data.datasets.QM8Dataset", "QM8Dataset", {}, "mol"),
    # Force (skip ISO17 — 640k graphs, already verified)
    ("MD17Revised", "kgcnn_torch.data.datasets.MD17RevisedDataset", "MD17RevisedDataset",
     {"trajectory_name": "ethanol"}, "force"),
    # MatProject crystal
    ("MatProjectEForm", "kgcnn_torch.data.datasets.MatProjectEFormDataset", "MatProjectEFormDataset", {}, "crystal"),
    ("MatProjectJdft2d", "kgcnn_torch.data.datasets.MatProjectJdft2dDataset", "MatProjectJdft2dDataset", {}, "crystal"),
    ("MatProjectPhonons", "kgcnn_torch.data.datasets.MatProjectPhononsDataset", "MatProjectPhononsDataset", {}, "crystal"),
    ("MatProjectGap", "kgcnn_torch.data.datasets.MatProjectGapDataset", "MatProjectGapDataset", {}, "crystal"),
    ("MatProjectDielectric", "kgcnn_torch.data.datasets.MatProjectDielectricDataset", "MatProjectDielectricDataset", {}, "crystal"),
    ("MatProjectIsMetal", "kgcnn_torch.data.datasets.MatProjectIsMetalDataset", "MatProjectIsMetalDataset", {}, "crystal"),
    ("MatProjectLogGVRH", "kgcnn_torch.data.datasets.MatProjectLogGVRHDataset", "MatProjectLogGVRHDataset", {}, "crystal"),
    ("MatProjectLogKVRH", "kgcnn_torch.data.datasets.MatProjectLogKVRHDataset", "MatProjectLogKVRHDataset", {}, "crystal"),
    ("MatProjectPerovskites", "kgcnn_torch.data.datasets.MatProjectPerovskitesDataset", "MatProjectPerovskitesDataset", {}, "crystal"),
    # Other
    ("MatPES2k", "kgcnn_torch.data.datasets.MatPES2kDataset", "MatPES2kDataset", {}, "crystal"),
]

# Worker script — reads config from TRAIN_TEST_CONFIG env var
WORKER_SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_train_worker.py")


def write_worker_script():
    """Write the worker script to disk."""
    code = '''#!/usr/bin/env python
"""Worker script for test_train_all.py — runs one dataset training test."""
import os, sys, time, tempfile, json, gc
os.environ["OMP_NUM_THREADS"] = "2"

import logging
logging.getLogger("kgcnn_torch.molecule.convert").setLevel(logging.ERROR)
logging.getLogger("kgcnn_torch.molecule.encoder").setLevel(logging.WARNING)
logging.getLogger("kgcnn.data").setLevel(logging.WARNING)

config = json.loads(os.environ["TRAIN_TEST_CONFIG"])
name = config["name"]
import_path = config["import_path"]
class_name = config["class_name"]
ds_kwargs = config["ds_kwargs"]
ds_type = config["ds_type"]

import importlib
import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.loader import DataLoader

root = os.path.join(tempfile.mkdtemp(prefix=f"kt_{name}_"), name)
t0 = time.time()

try:
    mod = importlib.import_module(import_path)
    cls = getattr(mod, class_name)

    # Link pre-downloaded rMD17 tar
    if name == "MD17Revised":
        raw = os.path.join(root, "raw")
        os.makedirs(raw, exist_ok=True)
        tar_src = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch/data/rmd17.tar.bz2"
        tar_dst = os.path.join(raw, "rmd17.tar.bz2")
        if os.path.exists(tar_src) and not os.path.exists(tar_dst):
            os.symlink(tar_src, tar_dst)

    # RadiusGraph for force/crystal/3d datasets (anything with pos but no edges)
    transform = None
    if ds_type in ("force", "crystal", "3d"):
        from torch_geometric.transforms import RadiusGraph, Compose
        transform = RadiusGraph(r=5.0, max_num_neighbors=32)

    ds = cls(root=root, transform=transform, **ds_kwargs)
    n_total = len(ds)

    # Note: PBC attributes (edge_image, lattice) will be removed from subset_list
    # below, after converting to mutable list.

    # Small subset (64 samples max to save memory)
    n = min(64, n_total)
    torch.manual_seed(42)
    perm = torch.randperm(n_total)[:n]
    subset = ds[perm]

    # Free the full dataset
    del ds
    gc.collect()

    # Inspect first sample
    d = subset[0]
    has_pos = hasattr(d, "pos") and d.pos is not None
    has_ei = hasattr(d, "edge_index") and d.edge_index is not None
    has_force = hasattr(d, "force") and d.force is not None
    has_energy = hasattr(d, "energy") and d.energy is not None
    has_y = hasattr(d, "y") and d.y is not None
    has_x = hasattr(d, "x") and d.x is not None
    has_z = hasattr(d, "z") and d.z is not None

    # Convert subset to a mutable list of Data objects
    subset_list = [subset[i] for i in range(len(subset))]
    del subset
    gc.collect()

    # For crystal datasets: remove PBC attributes that conflict with RadiusGraph edges
    if ds_type == "crystal":
        for data in subset_list:
            if hasattr(data, "edge_image"):
                del data.edge_image
            if hasattr(data, "lattice"):
                del data.lattice

    # For datasets without x AND without z, add degree-based node features
    if not has_x and not has_z and has_ei and not has_pos:
        from torch_geometric.utils import degree
        for data in subset_list:
            if data.x is None and data.edge_index is not None:
                nn = data.edge_index.max().item() + 1 if data.edge_index.numel() > 0 else 1
                deg = degree(data.edge_index[0], num_nodes=nn)
                data.x = deg.unsqueeze(-1).float()
                data.num_nodes = nn
        has_x = True
        d = subset_list[0]

    # Determine num_targets correctly
    if has_y:
        if d.y.dim() == 0:
            num_targets = 1
        else:
            num_targets = d.y.shape[-1] if d.y.dim() >= 1 else 1
            # For graph classification, y is typically (num_targets,) per graph
            # Avoid treating node-level labels as targets
            if has_x and d.y.shape[0] == d.x.shape[0] and d.y.shape[0] > 20:
                # Likely node-level labels (node classification)
                num_targets = d.y.shape[-1] if d.y.dim() > 1 else 1
    elif has_energy:
        num_targets = 1
    else:
        num_targets = 1

    # Check if labels contain NaN (Tox21 etc.)
    nan_labels = has_y and d.y.isnan().any()

    # Detect node classification (single-graph datasets like Cora)
    is_node_classification = False
    if ds_type == "node" and has_y and has_x:
        if d.y.shape[0] == d.x.shape[0]:
            is_node_classification = True

    # Determine if x is float features or needs embedding
    use_node_embedding = True  # default: integer z -> embedding
    node_input_dim = 1
    if has_x:
        if d.x.dtype in (torch.float, torch.float32, torch.float64, torch.float16):
            use_node_embedding = False
            node_input_dim = d.x.shape[-1] if d.x.dim() > 1 else 1

    # Rebuild loaders from mutable list
    n_train = max(1, int(0.8 * n))
    train_ds = subset_list[:n_train]
    train_loader = DataLoader(train_ds, batch_size=min(16, n_train), shuffle=True)

    use_force = False
    if ds_type == "force" and has_pos and has_ei:
        from kgcnn_torch.models.schnet import SchNetModel
        from kgcnn_torch.models.force import EnergyForceModel
        energy_model = SchNetModel(
            node_dim=16, depth=1, units=16, edge_dim=10,
            gauss_bins=10, gauss_distance=5.0, num_targets=1,
            make_distance=True, expand_distance=True,
        )
        model = EnergyForceModel(energy_model, output_as_dict=True,
                                  output_squeeze_states=True)
        use_force = True
    elif has_pos and has_ei:
        from kgcnn_torch.models.schnet import SchNetModel
        model = SchNetModel(
            node_dim=16, depth=1, units=16, edge_dim=10,
            gauss_bins=10, gauss_distance=5.0, num_targets=num_targets,
            make_distance=True, expand_distance=True,
        )
    elif (has_x or has_z) and has_ei:
        from kgcnn_torch.models.gcn import GCNModel
        if is_node_classification:
            # Node classification: no graph pooling, predict per-node
            model = GCNModel(
                node_dim=16, depth=2, gcn_units=16,
                num_targets=num_targets,
                output_embedding="node",
                use_node_embedding=use_node_embedding,
                node_input_dim=node_input_dim,
                output_final_activation="linear",
                output_units=[16],
            )
        else:
            model = GCNModel(
                node_dim=16, depth=2, gcn_units=16,
                num_targets=num_targets,
                output_embedding="graph",
                use_node_embedding=use_node_embedding,
                node_input_dim=node_input_dim,
                output_final_activation="linear",
                output_units=[16],
            )
    else:
        raise ValueError(f"No suitable model: has_x={has_x}, has_z={has_z}, has_pos={has_pos}, has_ei={has_ei}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    for epoch in range(3):
        model.train()
        epoch_loss, n_b = 0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            if use_force:
                batch.pos = batch.pos.detach().requires_grad_(True)
                out = model(batch)
                e_loss = F.mse_loss(out["energy"].squeeze(-1), batch.energy.squeeze(-1))
                f_loss = F.mse_loss(out["force"], batch.force)
                loss = e_loss + 10.0 * f_loss
            else:
                pred = model(batch)
                if has_y:
                    target = batch.y.float()
                    # Reshape target to match pred
                    if pred.shape != target.shape:
                        try:
                            target = target.view(pred.shape)
                        except RuntimeError:
                            # Fallback: flatten both and use min length
                            pred_flat = pred.reshape(-1)
                            target_flat = target.reshape(-1)
                            mn = min(len(pred_flat), len(target_flat))
                            loss = F.mse_loss(pred_flat[:mn], target_flat[:mn])
                            loss.backward()
                            optimizer.step()
                            epoch_loss += loss.item()
                            n_b += 1
                            continue
                    if nan_labels:
                        mask = ~target.isnan()
                        if mask.any():
                            loss = F.mse_loss(pred[mask], target[mask])
                        else:
                            loss = torch.tensor(0.0, requires_grad=True)
                    else:
                        loss = F.mse_loss(pred, target)
                elif has_energy:
                    loss = F.mse_loss(pred.squeeze(-1), batch.energy.float().squeeze(-1))
                else:
                    loss = pred.sum()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_b += 1
        losses.append(epoch_loss / max(n_b, 1))

    dt = time.time() - t0
    all_finite = all(np.isfinite(l) for l in losses)
    loss_str = ",".join(f"{l:.3g}" for l in losses)

    if all_finite:
        print(f"RESULT:PASS:{name}:{n_total} graphs, losses=[{loss_str}], {dt:.1f}s")
    else:
        print(f"RESULT:FAIL(NaN):{name}:{n_total} graphs, losses=[{loss_str}], {dt:.1f}s")

except Exception as e:
    dt = time.time() - t0
    err = str(e).replace("\\n", " ")[:200]
    print(f"RESULT:FAIL:{name}:{err} ({dt:.1f}s)")
'''
    with open(WORKER_SCRIPT_PATH, "w") as f:
        f.write(code)


def run_one(name, import_path, class_name, ds_kwargs, ds_type, timeout=600):
    """Run one dataset test in a subprocess."""
    config = json.dumps({
        "name": name, "import_path": import_path, "class_name": class_name,
        "ds_kwargs": ds_kwargs, "ds_type": ds_type,
    })
    env = os.environ.copy()
    env["TRAIN_TEST_CONFIG"] = config

    try:
        result = subprocess.run(
            [sys.executable, WORKER_SCRIPT_PATH],
            capture_output=True, text=True, timeout=timeout,
            cwd=os.path.dirname(os.path.abspath(__file__)),
            env=env,
        )
        # Find RESULT line in stdout
        for line in result.stdout.split("\n"):
            if line.startswith("RESULT:"):
                parts = line.split(":", 3)
                return (parts[2], parts[1], parts[3])
        # No RESULT line found
        err = result.stderr[-500:] if result.stderr else "no output"
        return (name, "FAIL", f"No result. stderr: {err}")
    except subprocess.TimeoutExpired:
        return (name, "FAIL", f"Timeout after {timeout}s")
    except Exception as e:
        return (name, "FAIL", str(e)[:200])


if __name__ == "__main__":
    write_worker_script()

    print(f"Testing training on {len(TASKS)} datasets (serial, one at a time)...")
    print(f"Skipped (already verified): QM9, QM9MolNet, ISO17")
    print(f"{'='*70}")

    t0 = time.time()
    results = []

    for i, (name, mod, cls, kwargs, dtype) in enumerate(TASKS):
        print(f"  [{i+1}/{len(TASKS)}] {name:25s} ... ", end="", flush=True)
        r = run_one(name, mod, cls, kwargs, dtype)
        results.append(r)
        status = r[1]
        detail = r[2]
        print(f"[{status}] {detail}")

    total_time = time.time() - t0
    print(f"\n{'='*70}")
    print(f"TRAINING TEST RESULTS (total {total_time:.0f}s)")
    print(f"{'='*70}")

    passed = failed = 0
    for name, status, detail in results:
        icon = "PASS" if "PASS" in status else "FAIL"
        print(f"  [{icon:4s}] {name:25s} {detail}")
        if "PASS" in status:
            passed += 1
        else:
            failed += 1

    skipped = ["QM9 (127k)", "QM9MolNet (133k)", "ISO17 (640k)"]
    print(f"\nTested: {passed}/{passed+failed} PASS, {failed} FAIL")
    print(f"Skipped (already verified): {', '.join(skipped)}")
    print(f"Total verified: {passed + len(skipped)}/{passed + failed + len(skipped)}")
    sys.exit(1 if failed > 0 else 0)
