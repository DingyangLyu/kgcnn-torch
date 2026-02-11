#!/usr/bin/env python3
"""Best-effort forward smoke test for all *Model classes in kgcnn_torch.models."""
from __future__ import annotations

import importlib
import inspect
import os
import sys
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, Dict, List, Tuple

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


torch.manual_seed(7)


@dataclass
class Result:
    name: str
    status: str
    detail: str = ""


MODEL_SPECS: List[Tuple[str, str]] = [
    ("kgcnn_torch.models.attentivefp", "AttentiveFPModel"),
    ("kgcnn_torch.models.cgcnn", "CGCNNModel"),
    ("kgcnn_torch.models.cmpnn", "CMPNNModel"),
    ("kgcnn_torch.models.dgin", "DGINModel"),
    ("kgcnn_torch.models.dimenetpp", "DimeNetPPModel"),
    ("kgcnn_torch.models.dimenetpp", "DimeNetPPCrystalModel"),
    ("kgcnn_torch.models.dmpnn", "DMPNNModel"),
    ("kgcnn_torch.models.egnn", "EGNNModel"),
    ("kgcnn_torch.models.force", "EnergyForceModel"),
    ("kgcnn_torch.models.gat", "GATModel"),
    ("kgcnn_torch.models.gatv2", "GATv2Model"),
    ("kgcnn_torch.models.gcn", "GCNModel"),
    ("kgcnn_torch.models.gcn", "GCNWeightedModel"),
    ("kgcnn_torch.models.gin", "GINModel"),
    ("kgcnn_torch.models.gnnfilm", "GNNFilmModel"),
    ("kgcnn_torch.models.graphsage", "GraphSAGEModel"),
    ("kgcnn_torch.models.hamnet", "HamNetModel"),
    ("kgcnn_torch.models.hdnnp2nd", "HDNNP2ndModel"),
    ("kgcnn_torch.models.hdnnp2nd", "HDNNP2ndBehlerModel"),
    ("kgcnn_torch.models.hdnnp2nd", "HDNNP2ndAtomWiseModel"),
    ("kgcnn_torch.models.inorp", "INorpModel"),
    ("kgcnn_torch.models.mat", "MATModel"),
    ("kgcnn_torch.models.megan", "MEGANModel"),
    ("kgcnn_torch.models.megnet", "MEGNetModel"),
    ("kgcnn_torch.models.megnet", "MEGNetCrystalModel"),
    ("kgcnn_torch.models.mogat", "MoGATModel"),
    ("kgcnn_torch.models.mxmnet", "MXMNetModel"),
    ("kgcnn_torch.models.nmpn", "NMPNModel"),
    ("kgcnn_torch.models.nmpn", "NMPNCrystalModel"),
    ("kgcnn_torch.models.painn", "PAiNNModel"),
    ("kgcnn_torch.models.painn", "PAiNNCrystalModel"),
    ("kgcnn_torch.models.rgcn", "RGCNModel"),
    ("kgcnn_torch.models.rgin", "rGINModel"),
    ("kgcnn_torch.models.schnet", "SchNetModel"),
    ("kgcnn_torch.models.schnet", "SchNetCrystalModel"),
]


def _base_data(n: int = 20, m: int = 40, k: int = 60) -> SimpleNamespace:
    batch = torch.cat([torch.zeros(n // 2, dtype=torch.long),
                       torch.ones(n - n // 2, dtype=torch.long)])
    edge_index = torch.randint(0, n, (2, m), dtype=torch.long)
    edge_pair_index = torch.randint(0, m, (m,), dtype=torch.long)
    range_index = torch.randint(0, n, (2, m), dtype=torch.long)
    data = SimpleNamespace(
        z=torch.randint(0, 10, (n,), dtype=torch.long),
        x=torch.randint(0, 10, (n,), dtype=torch.long),
        pos=torch.randn(n, 3),
        edge_index=edge_index,
        edge_attr=torch.randn(m, 8),
        edge_weight=torch.randn(m, 1),
        node_weight=torch.randn(n, 1),
        edge_type=torch.randint(0, 4, (m,), dtype=torch.long),
        edge_pair_index=edge_pair_index,
        range_index=range_index,
        angle_index=torch.randint(0, m, (2, k), dtype=torch.long),
        angle_index_1=torch.randint(0, m, (2, k), dtype=torch.long),
        angle_index_2=torch.randint(0, m, (2, k), dtype=torch.long),
        batch=batch,
        batch_edge=batch[edge_index[0]],
        edge_image=torch.randint(-1, 2, (m, 3), dtype=torch.float32),
        lattice=torch.randn(2, 3, 3),
        graph_state=torch.randn(2, 1),
    )
    return data


def _data_for_hdnnp() -> SimpleNamespace:
    data = _base_data(n=20, m=40, k=30)
    data.z = torch.tensor([1, 6, 7, 8] * 5, dtype=torch.long)
    data.angle_index = torch.stack([
        torch.randint(0, 20, (30,), dtype=torch.long),
        torch.randint(0, 20, (30,), dtype=torch.long),
        torch.randint(0, 20, (30,), dtype=torch.long),
    ])
    return data


def _data_for_hdnnp_atomwise() -> SimpleNamespace:
    data = _base_data(n=20, m=40, k=30)
    data.z = torch.tensor([1, 6, 7, 8] * 5, dtype=torch.long)
    data.x = torch.randn(20, 40)
    return data


def _data_for_mogan() -> SimpleNamespace:
    # MoGAT expects edge_attr as distances-like scalar features in its default stack.
    data = _base_data()
    data.edge_attr = torch.randn(data.edge_index.shape[1], 1).abs()
    return data


def _data_for_cgcnn() -> SimpleNamespace:
    data = _base_data()
    data.edge_attr = torch.randn(data.edge_index.shape[1], 1).abs()
    return data


def _data_for_rgcn() -> SimpleNamespace:
    data = _base_data()
    data.edge_attr = None
    data.edge_type = torch.randint(0, 4, (data.edge_index.shape[1],), dtype=torch.long)
    return data


def _data_for_dmpnn_like() -> SimpleNamespace:
    n, m = 20, 40
    src = torch.randint(0, n, (m // 2,), dtype=torch.long)
    dst = torch.randint(0, n, (m // 2,), dtype=torch.long)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    half = edge_index.size(1) // 2
    edge_pair_index = torch.cat([torch.arange(half, edge_index.size(1)),
                                 torch.arange(0, half)]).long()
    batch = torch.cat([torch.zeros(n // 2, dtype=torch.long),
                       torch.ones(n - n // 2, dtype=torch.long)])
    return SimpleNamespace(
        z=torch.randint(0, 10, (n,), dtype=torch.long),
        x=torch.randint(0, 10, (n,), dtype=torch.long),
        pos=torch.randn(n, 3),
        edge_index=edge_index,
        edge_attr=torch.randn(edge_index.size(1), 14),
        edge_pair_index=edge_pair_index,
        batch=batch,
        graph_state=torch.randn(2, 1),
    )


def _mat_inputs() -> Tuple[torch.Tensor, ...]:
    b, n = 2, 10
    node_input = torch.randn(b, n, 16)
    xyz_input = torch.randn(b, n, 3)
    adjacency = torch.ones(b, n, n)
    node_mask = torch.ones(b, n)
    node_mask[1, 7:] = 0
    adj_mask = node_mask.unsqueeze(1) * node_mask.unsqueeze(2)
    return node_input, xyz_input, adjacency, node_mask, adj_mask


def _filtered_kwargs(cls_name: str, cls) -> Dict:
    sig = inspect.signature(cls.__init__)
    common = {
        "node_dim": 16,
        "edge_dim": 8,
        "depth": 2,
        "units": 16,
        "num_targets": 1,
        "num_radial": 8,
        "num_spherical": 3,
        "cutoff": 5.0,
        "attention_units": 8,
        "attention_heads_num": 2,
        "use_node_embedding": True,
    }
    special = {
        "DimeNetPPModel": dict(
            emb_size=16, out_emb_size=16, int_emb_size=8, basis_emb_size=4,
            num_blocks=1, num_spherical=3, num_radial=4, cutoff=5.0,
            num_targets=1, num_before_skip=1, num_after_skip=1, num_dense_output=1
        ),
        "DimeNetPPCrystalModel": dict(
            emb_size=16, out_emb_size=16, int_emb_size=8, basis_emb_size=4,
            num_blocks=1, num_spherical=3, num_radial=4, cutoff=5.0,
            num_targets=1, num_before_skip=1, num_after_skip=1, num_dense_output=1
        ),
        "HDNNP2ndModel": dict(element_types=[1, 6, 7, 8], n_rad_features=8, n_ang_features=4, cutoff=5.0,
                              relational_units=[10, 10, 1], num_targets=1),
        "HDNNP2ndBehlerModel": dict(element_types=[1, 6, 7, 8], num_targets=1),
        "HDNNP2ndAtomWiseModel": dict(element_types=[1, 6, 7, 8], num_targets=1),
        "MATModel": dict(embedding_units=16, depth=2, num_heads=2, units_ff=16,
                         num_targets=1, input_node_dim=16, use_node_embedding=False),
        "MEGNetModel": dict(node_dim=16, edge_dim=16, state_dim=8, edge_input_dim=8, depth=2, num_targets=1),
        "MEGNetCrystalModel": dict(node_dim=16, edge_dim=16, state_dim=8, state_input_dim=1, depth=2, num_targets=1),
        "MEGNetModel": dict(node_dim=16, edge_dim=16, state_dim=8, edge_input_dim=8, state_input_dim=1, depth=2, num_targets=1),
        "NMPNModel": dict(node_dim=16, depth=2, units=16, edge_dim=8, num_targets=1, use_set2set=False),
        "NMPNCrystalModel": dict(node_dim=16, depth=2, units=16, num_targets=1, use_set2set=False),
        "PAiNNModel": dict(node_dim=32, depth=2, units=32, num_radial=8, cutoff=5.0, num_targets=1),
        "PAiNNCrystalModel": dict(node_dim=32, depth=2, units=32, num_radial=8, cutoff=5.0, num_targets=1),
        "SchNetModel": dict(node_dim=32, depth=2, units=32, edge_dim=10, gauss_bins=10, num_targets=1),
        "SchNetCrystalModel": dict(node_dim=32, depth=2, units=32, edge_dim=10, gauss_bins=10, num_targets=1),
        "GCNModel": dict(node_dim=16, depth=2, gcn_units=16, num_targets=1),
        "GCNWeightedModel": dict(node_dim=16, depth=2, gcn_units=16, num_targets=1),
        "GATModel": dict(node_dim=16, depth=2, attention_units=8, attention_heads_num=2, num_targets=1),
        "GATv2Model": dict(node_dim=16, depth=2, attention_units=8, attention_heads_num=2, num_targets=1),
        "HamNetModel": dict(node_dim=16, edge_dim=8, depth=2, units=16, fingerprint_dim=16,
                            fingerprint_depth=2, num_targets=1),
        "EGNNModel": dict(node_dim=16, depth=2, units=16, edge_attr_dim=8, num_targets=1),
        "GATv2Model": dict(node_dim=16, depth=2, attention_units=8, attention_heads_num=2, edge_dim=8, num_targets=1),
        "MoGATModel": dict(node_dim=16, depthato=2, depthmol=2, units=16, edge_dim=1, num_targets=1),
        "DMPNNModel": dict(node_dim=16, edge_dim=14, depth=2, units=16, num_targets=1),
        "CMPNNModel": dict(node_dim=16, edge_dim=14, depth=2, units=16, num_targets=1),
        "DGINModel": dict(node_dim=16, edge_dim=14, depth_dmpnn=2, depth_gin=2, units=16, num_targets=1),
        "MEGANModel": dict(node_dim=16, units=[16, 16], num_heads=2, depth=2, importance_channels=2, num_targets=1),
    }
    kwargs = special.get(cls_name, common)
    has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
    if has_var_kwargs:
        return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}


def _build_model(module_name: str, cls_name: str):
    mod = importlib.import_module(module_name)
    cls = getattr(mod, cls_name)
    if cls_name == "EnergyForceModel":
        from kgcnn_torch.models.schnet import SchNetModel
        energy_model = SchNetModel(node_dim=16, depth=1, units=16, edge_dim=8, gauss_bins=8, num_targets=1)
        return cls(energy_model, coordinate_input="pos", output_as_dict=False)
    kwargs = _filtered_kwargs(cls_name, cls)
    return cls(**kwargs)


def _run_model(name: str, model) -> torch.Tensor:
    if name == "MATModel":
        return model(*_mat_inputs())

    if name in {"HDNNP2ndModel", "HDNNP2ndBehlerModel", "HDNNP2ndAtomWiseModel"}:
        if name == "HDNNP2ndAtomWiseModel":
            return model(_data_for_hdnnp_atomwise())
        return model(_data_for_hdnnp())

    if name in {"DMPNNModel", "CMPNNModel", "DGINModel"}:
        return model(_data_for_dmpnn_like())

    if name == "CGCNNModel":
        return model(_data_for_cgcnn())

    if name == "MoGATModel":
        return model(_data_for_mogan())

    if name == "RGCNModel":
        return model(_data_for_rgcn())

    data = _base_data()
    if name == "EnergyForceModel":
        data.pos = data.pos.clone().requires_grad_(True)
        out = model(data)
        # (energy, force) tuple
        return out[0] if isinstance(out, tuple) else out
    return model(data)


def main():
    results: List[Result] = []
    for module_name, cls_name in MODEL_SPECS:
        label = f"{module_name}.{cls_name}"
        try:
            model = _build_model(module_name, cls_name)
            model.train()
            out = _run_model(cls_name, model)
            if not torch.is_tensor(out):
                raise RuntimeError(f"output is not Tensor: {type(out)}")
            if out.numel() == 0:
                raise RuntimeError("empty output tensor")
            loss = out.float().sum()
            loss.backward()
            results.append(Result(label, "PASS", f"shape={tuple(out.shape)}"))
        except Exception as exc:
            results.append(Result(label, "FAIL", str(exc)))

    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]

    print(f"Total: {len(results)} | PASS: {len(passed)} | FAIL: {len(failed)}")
    for r in results:
        print(f"[{r.status}] {r.name} :: {r.detail}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
