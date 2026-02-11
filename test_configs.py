"""Quick smoke test for hyper configs: create model, load a few fake graphs, run 2 epochs.

Tests that the full pipeline (HyperParameter -> model creation -> loss -> scheduler -> training)
works end-to-end for representative configs WITHOUT downloading real datasets.
"""
import sys
import json
import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "training_scripts"))

from kgcnn_torch.training.hyper import HyperParameter
from kgcnn_torch.training.scheduler import get_scheduler
from training_scripts.train_graph import (
    get_model_class, translate_model_config, make_optimizer, make_loss,
    MATBatchWrapper, adapt_model_config_from_data,
)


def make_fake_graph(num_nodes=12, num_edges=30, node_dim=None, edge_dim=8,
                    include_pos=True, num_targets=1, classification=False,
                    model_name=""):
    """Create a fake PyG Data object for testing."""
    z = torch.randint(1, 30, (num_nodes,))
    x = torch.randn(num_nodes, node_dim) if node_dim else None
    # Build valid edge_index (no self-loops for some models)
    src = torch.randint(0, num_nodes, (num_edges,))
    dst = torch.randint(0, num_nodes, (num_edges,))
    edge_index = torch.stack([src, dst])
    edge_attr = torch.randn(num_edges, edge_dim) if edge_dim else None
    pos = torch.randn(num_nodes, 3) if include_pos else None

    # CGCNN with expand_distance needs scalar distances as edge_attr
    if model_name == "CGCNN" and edge_dim == 0 and include_pos:
        diff = pos[dst] - pos[src]
        edge_attr = diff.norm(dim=-1, keepdim=True)  # (M, 1)

    if classification:
        y = torch.tensor([float(np.random.randint(0, 2)) for _ in range(num_targets)])
    else:
        y = torch.randn(num_targets)
    data = Data(z=z, x=x, edge_index=edge_index, edge_attr=edge_attr,
                pos=pos, y=y, num_nodes=num_nodes)

    # DMPNN needs edge_pair_index (reverse edge mapping)
    if model_name == "DMPNN":
        # For each edge (i->j), find index of reverse edge (j->i)
        edge_pair_index = torch.full((num_edges,), -1, dtype=torch.long)
        for e in range(num_edges):
            s, t = src[e].item(), dst[e].item()
            for r in range(num_edges):
                if src[r].item() == t and dst[r].item() == s and r != e:
                    edge_pair_index[e] = r
                    break
            if edge_pair_index[e] == -1:
                edge_pair_index[e] = e  # fallback: self
        data.edge_pair_index = edge_pair_index

    # DimeNetPP needs angle_index (triplet indices)
    if model_name == "DimeNetPP":
        # Build angle pairs: for edges sharing a target node
        angle_src = []
        angle_dst = []
        for e1 in range(min(num_edges, 60)):
            t1 = dst[e1].item()
            for e2 in range(min(num_edges, 60)):
                if e1 != e2 and dst[e2].item() == t1:
                    angle_src.append(e1)
                    angle_dst.append(e2)
                    if len(angle_src) > 200:
                        break
            if len(angle_src) > 200:
                break
        if not angle_src:
            angle_src = [0]
            angle_dst = [0]
        data.angle_index = torch.tensor([angle_src, angle_dst], dtype=torch.long)

    return data


def test_config(hyper_path, category, task_type="regression", num_epochs=2):
    """Test a single hyper config end-to-end."""
    print(f"\n{'='*70}")
    print(f"Testing: {os.path.basename(hyper_path)} -> {category} ({task_type})")
    print(f"{'='*70}")

    hyper = HyperParameter(hyper_path, model_name=category)
    model_config = hyper.model_config
    compile_config = hyper.compile_config
    scheduler_config = hyper.scheduler_config
    fit_config = hyper.fit_config

    model_name = model_config.pop("model_name", category)
    model_config = translate_model_config(model_name, model_config)
    num_targets = model_config.get("num_targets", 1)

    # Create fake data
    classification = task_type == "classification"
    include_pos = model_name in ("SchNet", "PAiNN", "DimeNetPP", "EGNN", "CGCNN", "Megnet")
    # Match edge_dim from config for models that expect specific dimensions
    edge_dim_from_config = model_config.get("edge_dim", 0)
    if model_name in ("SchNet", "PAiNN", "DimeNetPP", "CGCNN"):
        edge_dim = 0  # These compute edge features from positions
    elif edge_dim_from_config > 0:
        edge_dim = edge_dim_from_config
    else:
        edge_dim = 8
    fake_data = [make_fake_graph(
        num_nodes=15, num_edges=40, edge_dim=edge_dim,
        include_pos=include_pos, num_targets=num_targets,
        classification=classification, model_name=model_name
    ) for _ in range(32)]

    # Adapt config from data
    model_config = adapt_model_config_from_data(model_name, model_config, fake_data)

    # Create model
    try:
        ModelClass = get_model_class(model_name)
    except ValueError as e:
        print(f"  SKIP: {e}")
        return True

    model = ModelClass(**model_config)
    if model_name == "MAT":
        model = MATBatchWrapper(model)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {model_name}, params: {n_params:,}")

    # Loss
    loss_fn = make_loss(compile_config)
    print(f"  Loss: {loss_fn.__class__.__name__}")

    # Optimizer
    optimizer = make_optimizer(model, compile_config)

    # Models with custom edge-level indices (angle_index, edge_pair_index) need
    # batch_size=1 in this test because PyG collate doesn't know how to offset
    # these edge-referencing indices across graphs.
    bs = 1 if model_name in ("DimeNetPP", "DMPNN", "CMPNN", "DGIN") else 8
    loader = DataLoader(fake_data, batch_size=bs, shuffle=True)
    scheduler = None
    if scheduler_config:
        sched_config = dict(scheduler_config)
        sched_name = sched_config.pop("class_name", None)
        if sched_name:
            sched_config.setdefault("steps_per_epoch", len(loader))
            scheduler = get_scheduler(sched_name, optimizer, **sched_config)
            print(f"  Scheduler: {sched_name}")

    # Train a few epochs
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        n_batch = 0
        for batch in loader:
            optimizer.zero_grad()
            pred = model(batch)
            target = batch.y.unsqueeze(0) if batch.y.dim() == 1 else batch.y
            # Reshape pred/target to match
            if pred.shape != target.shape:
                target = target.view(pred.shape)
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batch += 1
        avg_loss = epoch_loss / max(n_batch, 1)

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_loss)
            else:
                scheduler.step()

        lr = optimizer.param_groups[0]['lr']
        print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, lr={lr:.2e}")

    print(f"  PASSED")
    return True


def main():
    hyper_dir = os.path.join(os.path.dirname(__file__), "training_scripts", "hyper")

    tests = [
        # (config file, category, task_type)
        # Regression - molecular
        ("hyper_esol.json", "SchNet", "regression"),
        ("hyper_esol.json", "GIN", "regression"),
        ("hyper_esol.json", "GAT", "regression"),
        ("hyper_esol.json", "DMPNN", "regression"),
        # Regression - QM
        ("hyper_qm9.json", "SchNet", "regression"),
        ("hyper_qm9.json", "DimeNetPP", "regression"),
        ("hyper_qm9.json", "PAiNN", "regression"),
        # Regression - more molecular
        ("hyper_freesolv.json", "SchNet", "regression"),
        ("hyper_lipop.json", "GIN", "regression"),
        # Classification - molecular (the bce_with_logits fix)
        ("hyper_clintox.json", "SchNet", "classification"),
        ("hyper_clintox.json", "GIN", "classification"),
        ("hyper_clintox.json", "GAT", "classification"),
        ("hyper_clintox.json", "DMPNN", "classification"),
        # Classification - graph (use models that exist in mutag config)
        ("hyper_mutag.json", "GIN", "classification"),
        ("hyper_mutag.json", "GAT", "classification"),
        # Materials - regression
        ("hyper_mp_e_form.json", "SchNet", "regression"),
        ("hyper_mp_e_form.json", "PAiNN", "regression"),
        ("hyper_mp_e_form.json", "DimeNetPP", "regression"),
        ("hyper_mp_e_form.json", "CGCNN", "regression"),
        # Materials - classification
        ("hyper_mp_is_metal.json", "SchNet", "classification"),
        ("hyper_mp_is_metal.json", "CGCNN", "classification"),
    ]

    passed = 0
    failed = 0
    errors = []

    for config_file, category, task_type in tests:
        hyper_path = os.path.join(hyper_dir, config_file)
        if not os.path.exists(hyper_path):
            print(f"SKIP: {config_file} not found")
            continue
        try:
            ok = test_config(hyper_path, category, task_type, num_epochs=2)
            if ok:
                passed += 1
        except Exception as e:
            failed += 1
            errors.append((config_file, category, str(e)))
            import traceback
            print(f"  FAILED: {e}")
            traceback.print_exc()

    print(f"\n{'='*70}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
    if errors:
        print(f"\nFailed tests:")
        for cfg, cat, err in errors:
            print(f"  {cfg} / {cat}: {err}")
    print(f"{'='*70}")
    return failed == 0


if __name__ == "__main__":
    torch.manual_seed(42)
    np.random.seed(42)
    ok = main()
    sys.exit(0 if ok else 1)
