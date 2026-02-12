"""Test all hyper configs: load dataset, create model, run 2 training steps.

Each (dataset, model) combination runs in a separate subprocess to prevent
memory accumulation and OOM kills.

Usage:
    python scripts/test_all_hyper.py
"""
import json
import glob
import os
import sys
import subprocess
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
HYPER_DIR = os.path.join(PROJECT_DIR, "training_scripts", "hyper")
PYTHON = sys.executable

# Available datasets (have raw pickle data locally)
AVAILABLE_DATASETS = {
    "ClinTox", "Cora", "CoraLu", "MUTAG", "Mutagenicity", "PROTEINS",
    "Tox21MolNet", "ESOL",
}

# Worker script as a separate file
WORKER_SCRIPT = os.path.join(SCRIPT_DIR, "_test_worker.py")


def ensure_worker_script():
    """Write the worker script to disk."""
    code = '''
import json, sys, os, warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

project_dir = os.environ["TEST_PROJECT_DIR"]
hyper_file = os.environ["TEST_HYPER_FILE"]
model_name = os.environ["TEST_MODEL_NAME"]

sys.path.insert(0, os.path.join(project_dir, "training_scripts"))

import torch
from torch_geometric.loader import DataLoader
from train_graph import (
    get_model_class, translate_model_config, adapt_model_config_from_data,
    make_optimizer, make_loss, MATBatchWrapper, load_dataset_pyg
)

with open(hyper_file) as f:
    hyper = json.load(f)

entry = hyper[model_name]
data_config = entry.get("data", {})

pyg_list = load_dataset_pyg(data_config)

model_config = dict(entry["model"]["config"])
model_config.pop("model_name", None)
model_config = translate_model_config(model_name, model_config)
model_config = adapt_model_config_from_data(model_name, model_config, pyg_list)

ModelClass = get_model_class(model_name)
model = ModelClass(**model_config)
if model_name == "MAT":
    model = MATBatchWrapper(model)
model.train()

training_config = entry.get("training", {})
compile_config = training_config.get("compile", {})
optimizer = make_optimizer(model, compile_config)
loss_fn = make_loss(compile_config)

output_embedding = entry["model"]["config"].get("output_embedding", "graph")
label_key = "node_labels" if output_embedding == "node" else "y"

subset = pyg_list[:min(8, len(pyg_list))]
loader = DataLoader(subset, batch_size=min(4, len(subset)))
batch = next(iter(loader))

optimizer.zero_grad()
pred = model(batch)
target = getattr(batch, label_key, batch.y)
if target.dim() == 1:
    target = target.unsqueeze(-1)
if pred.shape != target.shape:
    target = target[:pred.shape[0]]
    if target.dim() > 1 and pred.dim() > 1 and target.shape[1] != pred.shape[1]:
        target = target[:, :pred.shape[1]]
target = target.float()
loss = loss_fn(pred, target)
loss.backward()
optimizer.step()

optimizer.zero_grad()
pred2 = model(batch)
target2 = getattr(batch, label_key, batch.y)
if target2.dim() == 1:
    target2 = target2.unsqueeze(-1)
if pred2.shape != target2.shape:
    target2 = target2[:pred2.shape[0]]
    if target2.dim() > 1 and pred2.dim() > 1 and target2.shape[1] != pred2.shape[1]:
        target2 = target2[:, :pred2.shape[1]]
target2 = target2.float()
loss2 = loss_fn(pred2, target2)

from torch.nn.parameter import UninitializedParameter
n_params = sum(p.numel() for p in model.parameters()
               if not isinstance(p, UninitializedParameter))
print("OK loss=%.4f->%.4f shape=%s params=%d" % (
    loss.item(), loss2.item(), tuple(pred.shape), n_params))
'''
    with open(WORKER_SCRIPT, "w") as f:
        f.write(code)


def run_test(hyper_file, model_name):
    """Run a single test in a subprocess."""
    env = os.environ.copy()
    env["TEST_PROJECT_DIR"] = PROJECT_DIR
    env["TEST_HYPER_FILE"] = hyper_file
    env["TEST_MODEL_NAME"] = model_name
    env["CUDA_VISIBLE_DEVICES"] = ""

    try:
        result = subprocess.run(
            [PYTHON, WORKER_SCRIPT],
            capture_output=True, text=True, timeout=120,
            cwd=PROJECT_DIR, env=env,
        )
        stdout = result.stdout.strip()
        stderr = result.stderr.strip()

        if result.returncode == 0 and stdout.startswith("OK"):
            return "OK", stdout[3:]
        else:
            err_lines = [l for l in stderr.split("\n")
                         if l.strip() and not l.startswith("Traceback")
                         and "Warning" not in l and "numpy" not in l.lower()
                         and "ARRAY_API" not in l and "transformer" not in l
                         and "pybind" not in l and "compiled" not in l
                         and "module may" not in l.lower()
                         and "downgrade" not in l and "upgrade" not in l]
            err_msg = err_lines[-1].strip() if err_lines else stderr[-200:]
            return "FAIL", err_msg
    except subprocess.TimeoutExpired:
        return "FAIL", "TIMEOUT (>120s)"


def main():
    ensure_worker_script()

    results = {"OK": [], "SKIP": [], "FAIL": []}
    hyper_files = sorted(glob.glob(os.path.join(HYPER_DIR, "hyper_*.json")))

    # Count testable combos
    total = 0
    for fpath in hyper_files:
        fname = os.path.basename(fpath)
        if "test" in fname:
            continue
        with open(fpath) as f:
            hyper = json.load(f)
        for model_name, entry in hyper.items():
            ds_class = entry.get("data", {}).get("dataset", {}).get("class_name", "")
            if ds_class in AVAILABLE_DATASETS:
                total += 1

    print(f"Testing {total} (dataset x model) combinations...\n")

    for fpath in hyper_files:
        fname = os.path.basename(fpath)
        if "test" in fname:
            continue

        with open(fpath) as f:
            hyper = json.load(f)

        for model_name, entry in hyper.items():
            tag = f"{fname}:{model_name}"
            ds_class = entry.get("data", {}).get("dataset", {}).get("class_name", "")

            if ds_class not in AVAILABLE_DATASETS:
                results["SKIP"].append((tag, f"dataset {ds_class} not available"))
                continue

            t0 = time.time()
            status, msg = run_test(fpath, model_name)
            elapsed = time.time() - t0

            results[status].append((tag, msg))
            symbol = {"OK": "+", "SKIP": "-", "FAIL": "X"}[status]
            done = len(results["OK"]) + len(results["FAIL"])
            print(f"  [{symbol}] ({done}/{total}) {tag:50s} ({elapsed:.1f}s) {msg[:80]}")
            sys.stdout.flush()

    print(f"\n{'='*70}")
    print(f"RESULTS: {len(results['OK'])} OK, {len(results['SKIP'])} SKIP, {len(results['FAIL'])} FAIL")
    print(f"{'='*70}")

    if results["FAIL"]:
        print(f"\nFAILURES ({len(results['FAIL'])}):")
        for tag, msg in results["FAIL"]:
            print(f"  {tag}")
            print(f"    {msg}")

    print(f"\nSUCCESS ({len(results['OK'])}):")
    for tag, _ in results["OK"]:
        print(f"  {tag}")

    # Cleanup
    if os.path.exists(WORKER_SCRIPT):
        os.unlink(WORKER_SCRIPT)


if __name__ == "__main__":
    main()
