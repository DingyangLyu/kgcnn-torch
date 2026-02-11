#!/usr/bin/env python3
"""Training divergence test with REAL dataset data.

Both models start from identical weights (transferred once), then train
independently with the same SGD optimizer and real graph data.
Plots loss curves + divergence over time.

Uses:
  - MUTAG (188 molecular graphs) for models that don't need edge features
  - ClinTox (1451 molecules with 11D edge features + 3D coordinates) for
    models that require edge_attr or pos

Usage:
    KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES="" python scripts/plot_training_divergence_realdata.py
"""
import os
import sys
from types import SimpleNamespace

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import numpy as np

ROOT = "/home/yuanbai/Downloads/MLIPs"
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from plot_training_divergence import run_long_training, plot_divergence
from train_alignment_utils import collect_keras_params

PICKLE_DIR = os.path.join(ROOT, "kgcnn-torch", "datasets", "raw")
N_STEPS = 50
LR = 0.01
BATCH_SIZE = 8


# ---- Data loading ----

def load_pickle_as_pyg(pickle_name, edge_attr_keys=None):
    """Load a .kgcnn.pickle file and return a list of PyG Data objects."""
    from kgcnn_torch.data.base import MemoryGraphList
    gl = MemoryGraphList()
    gl.load(os.path.join(PICKLE_DIR, pickle_name))
    return gl.to_pyg_list(edge_attr_keys=edge_attr_keys)


def make_batch(pyg_list, batch_size, seed=42):
    """Select batch_size graphs and create a PyG batch."""
    from torch_geometric.loader import DataLoader
    torch.manual_seed(seed)
    indices = torch.randperm(len(pyg_list))[:batch_size]
    subset = [pyg_list[int(i)] for i in indices]
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)
    return next(iter(loader))


def pyg_batch_to_alignment_data(batch, batch_size):
    """Convert a PyG Batch into (torch_data, keras_data) pair.

    Handles:
      - Edge index convention: PyG [src,dst] -> KGCNN [dst,src]
      - Computes batch_id_edge, count_nodes, count_edges for Keras
    """
    num_edges = batch.edge_index.size(1)
    batch_id_edge = batch.batch[batch.edge_index[0]]
    count_nodes = torch.zeros(batch_size, dtype=torch.long)
    count_edges = torch.zeros(batch_size, dtype=torch.long)
    for i in range(batch_size):
        count_nodes[i] = (batch.batch == i).sum()
        count_edges[i] = (batch_id_edge == i).sum()

    edge_weight = torch.ones(num_edges, 1)

    torch_data = SimpleNamespace(
        z=batch.z,
        edge_index=batch.edge_index,
        edge_attr=getattr(batch, 'edge_attr', None),
        edge_weight=edge_weight,
        batch=batch.batch,
        pos=getattr(batch, 'pos', None),
    )

    keras_data = {
        "z": batch.z,
        "edge_index": batch.edge_index[[1, 0]],  # swap to [dst, src]
        "edge_attr": getattr(batch, 'edge_attr', None),
        "edge_weight": edge_weight,
        "batch_id_node": batch.batch,
        "batch_id_edge": batch_id_edge,
        "count_nodes": count_nodes,
        "count_edges": count_edges,
        "pos": getattr(batch, 'pos', None),
    }

    return torch_data, keras_data


# ---- Dataset loaders ----

_mutag_cache = None
_clintox_cache = None


def get_mutag_batch():
    global _mutag_cache
    if _mutag_cache is None:
        pyg_list = load_pickle_as_pyg("MUTAG.kgcnn.pickle",
                                       edge_attr_keys=["edge_attributes"])
        _mutag_cache = pyg_list
    batch = make_batch(_mutag_cache, BATCH_SIZE)
    return pyg_batch_to_alignment_data(batch, BATCH_SIZE)


def get_clintox_batch():
    global _clintox_cache
    if _clintox_cache is None:
        pyg_list = load_pickle_as_pyg("ClinTox.kgcnn.pickle",
                                       edge_attr_keys=["edge_attributes"])
        # Filter graphs with valid z (some ClinTox graphs may be incomplete)
        valid = [d for d in pyg_list if 'z' in d.keys() and 'edge_attr' in d.keys()]
        _clintox_cache = valid
    batch = make_batch(_clintox_cache, BATCH_SIZE)
    return pyg_batch_to_alignment_data(batch, BATCH_SIZE)


# ---- Model tests ----

def test_gcn_mutag():
    """GCN on MUTAG real data."""
    from align_gcn_model import Config, KerasGCNFullStack, GCNModel, transfer_all_weights
    torch_data, keras_data = get_mutag_batch()

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1

    torch_model = GCNModel(
        node_dim=cfg.node_dim, depth=cfg.depth, gcn_units=cfg.gcn_units,
        gcn_activation="leaky_relu2", gcn_pooling="sum", node_pooling="sum",
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        output_final_activation=cfg.output_final_activation,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasGCNFullStack(cfg)
    z_k, ew_k, ei_k = keras_data["z"], keras_data["edge_weight"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ew_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GCN (MUTAG)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ew_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_gin_mutag():
    """GIN on MUTAG real data."""
    from align_gin_model import Config, KerasGINFullStack, GINModel, transfer_all_weights
    torch_data, keras_data = get_mutag_batch()

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1

    torch_model = GINModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        gin_mlp_units=cfg.gin_mlp_units, gin_mlp_activation="relu",
        gin_mlp_use_normalization=False, gin_pooling="sum",
        epsilon_learnable=False, use_edge_features=False,
        node_pooling="sum", last_mlp_units=cfg.last_mlp_units,
        last_mlp_activation="relu", dropout_rate=0.0,
        output_units=cfg.output_units, output_activation="relu",
        output_final_activation="linear",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasGINFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GIN (MUTAG)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_megan_mutag():
    """MEGAN on MUTAG real data."""
    from align_megan_model import Config, KerasMEGANFullStack, MEGANModel, transfer_all_weights
    torch_data, keras_data = get_mutag_batch()

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1
    cfg.final_units = [cfg.num_targets]

    torch_model = MEGANModel(
        node_dim=cfg.node_dim, units=cfg.units, num_heads=cfg.num_heads,
        depth=cfg.depth, attention_activation="leaky_relu2",
        use_edge_features=False, concat_heads=cfg.concat_heads,
        importance_channels=cfg.importance_channels,
        importance_units=cfg.importance_units,
        importance_activation="relu",
        final_units=cfg.final_units, final_activation=cfg.final_activation,
        use_bias=True, final_pooling=cfg.final_pooling,
        dropout_rate=0.0, final_dropout_rate=0.0,
        regression_reference=None,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasMEGANFullStack(cfg)
    z_k = keras_data["z"]
    ei_k = keras_data["edge_index"]
    bid_k = keras_data["batch_id_node"]
    cn_k = keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("MEGAN (MUTAG)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_gat_clintox():
    """GAT on ClinTox real data."""
    from align_gat_model import Config, KerasGATFullStack, GATModel, transfer_all_weights
    torch_data, keras_data = get_clintox_batch()

    edge_dim = int(keras_data["edge_attr"].size(-1))  # 11 for ClinTox

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1
    cfg.edge_dim = edge_dim

    torch_model = GATModel(
        node_dim=cfg.node_dim, depth=cfg.depth,
        attention_units=cfg.attention_units,
        attention_heads_num=cfg.heads, attention_heads_concat=cfg.concat,
        attention_activation="leaky_relu2", use_edge_features=True,
        edge_dim=cfg.edge_dim, node_pooling="mean",
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        output_final_activation=cfg.output_final_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasGATFullStack(cfg)
    z_k, ea_k = keras_data["z"], keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GAT (ClinTox)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_gatv2_clintox():
    """GATv2 on ClinTox real data."""
    from align_gatv2_model import Config, KerasGATv2FullStack, GATv2Model, transfer_all_weights
    torch_data, keras_data = get_clintox_batch()

    edge_dim = int(keras_data["edge_attr"].size(-1))

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1
    cfg.edge_dim = edge_dim

    torch_model = GATv2Model(
        node_dim=cfg.node_dim, depth=cfg.depth,
        attention_units=cfg.attention_units,
        attention_heads_num=cfg.heads, attention_heads_concat=cfg.concat,
        attention_activation="leaky_relu2", use_edge_features=True,
        edge_dim=cfg.edge_dim, node_pooling="mean",
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasGATv2FullStack(cfg)
    z_k, ea_k = keras_data["z"], keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GATv2 (ClinTox)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_attentivefp_clintox():
    """AttentiveFP on ClinTox real data."""
    from align_attentivefp_model import (Config, KerasAttentiveFPFullStack,
                                          AttentiveFPModel, transfer_all_weights)
    torch_data, keras_data = get_clintox_batch()

    edge_dim = int(keras_data["edge_attr"].size(-1))

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1
    cfg.edge_dim = edge_dim

    torch_model = AttentiveFPModel(
        node_dim=cfg.node_dim, depth_ato=cfg.depth_ato, depth_mol=cfg.depth_mol,
        units=cfg.units, use_edge_features=True, edge_dim=cfg.edge_dim,
        attention_activation="leaky_relu2", attention_activation_context="elu",
        pooling_activation="leaky_relu2", pooling_activation_context="elu",
        node_pooling="sum", output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, dropout=0.0,
        output_embedding="graph", use_node_embedding=True,
        num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasAttentiveFPFullStack(cfg)
    z_k, ea_k = keras_data["z"], keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("AttentiveFP (ClinTox)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_graphsage_clintox():
    """GraphSAGE on ClinTox real data."""
    from align_graphsage_model import (Config, KerasGraphSAGEFullStack,
                                        GraphSAGEModel, transfer_all_weights)
    torch_data, keras_data = get_clintox_batch()

    edge_dim = int(keras_data["edge_attr"].size(-1))

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1
    cfg.edge_dim = edge_dim

    torch_model = GraphSAGEModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        node_mlp_units=cfg.node_mlp_units, edge_mlp_units=cfg.edge_mlp_units,
        edge_dim=cfg.edge_dim, use_edge_features=True, pooling_method="mean",
        node_pooling="mean", output_units=cfg.output_units,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        activation=cfg.output_activation, output_final_activation="sigmoid",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasGraphSAGEFullStack(cfg)
    z_k, ea_k = keras_data["z"], keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GraphSAGE (ClinTox)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


def test_schnet_clintox():
    """SchNet on ClinTox real data (uses 3D coordinates)."""
    from align_schnet_model import Config, KerasSchNetFullStack, SchNetModel, transfer_all_weights
    torch_data, keras_data = get_clintox_batch()

    cfg = Config()
    cfg.batch_size = BATCH_SIZE
    cfg.num_targets = 1

    torch_model = SchNetModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        gauss_bins=cfg.gauss_bins, gauss_distance=cfg.gauss_distance,
        gauss_sigma=cfg.gauss_sigma, gauss_offset=cfg.gauss_offset,
        interaction_activation="shifted_softplus", interaction_pooling="sum",
        node_pooling="sum", last_mlp_units=cfg.last_mlp_units,
        last_mlp_activation="shifted_softplus",
        output_units=cfg.output_units, output_activation="shifted_softplus",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings,
        make_distance=True, expand_distance=True, use_output_mlp=True)
    torch_model.train()

    keras_stack = KerasSchNetFullStack(cfg)
    z_k, pos_k, ei_k = keras_data["z"], keras_data["pos"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(BATCH_SIZE, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("SchNet (ClinTox)",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target, n_steps=N_STEPS, lr=LR)


# ---- Plotting override ----

def plot_divergence_realdata(all_results, save_path):
    """Generate plot with real-data title."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(n, 3, figsize=(16, 3.5 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for idx, (name, res) in enumerate(all_results.items()):
        tl = res["torch_losses"]
        kl = res["keras_losses"]
        ld = res["loss_diffs"]
        od = res["output_diffs"]
        steps = list(range(len(tl)))

        ax = axes[idx, 0]
        ax.plot(steps, tl, "-", color="#2196F3", label="Torch", linewidth=1.5)
        ax.plot(steps, kl, "--", color="#FF5722", label="Keras", linewidth=1.5)
        ax.set_title(f"{name}: Loss Curves", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        ax = axes[idx, 1]
        ax.semilogy(steps, [max(d, 1e-15) for d in ld], "-", color="#4CAF50", linewidth=1.5)
        ax.set_title(f"{name}: |Loss_torch - Loss_keras|", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Absolute Diff")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1e-6, color="gray", linestyle=":", alpha=0.5, label="1e-6")
        ax.legend(fontsize=9)

        ax = axes[idx, 2]
        ax.semilogy(steps, [max(d, 1e-15) for d in od], "-", color="#9C27B0", linewidth=1.5)
        ax.set_title(f"{name}: Output MAE", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean |out_torch - out_keras|")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Divergence Over 50 Steps — Real Data (MUTAG + ClinTox)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)


# ---- Main ----

def main():
    print(f"Real-Data Training Divergence Test")
    print(f"Datasets: MUTAG ({PICKLE_DIR}/MUTAG.kgcnn.pickle)")
    print(f"          ClinTox ({PICKLE_DIR}/ClinTox.kgcnn.pickle)")
    print(f"Batch size: {BATCH_SIZE}, Steps: {N_STEPS}, LR: {LR}")
    print("=" * 70)

    tests = [
        ("GCN (MUTAG)", test_gcn_mutag),
        ("GIN (MUTAG)", test_gin_mutag),
        ("MEGAN (MUTAG)", test_megan_mutag),
        ("GAT (ClinTox)", test_gat_clintox),
        ("GATv2 (ClinTox)", test_gatv2_clintox),
        ("AttentiveFP (ClinTox)", test_attentivefp_clintox),
        ("GraphSAGE (ClinTox)", test_graphsage_clintox),
        ("SchNet (ClinTox)", test_schnet_clintox),
    ]

    all_results = {}
    for name, test_fn in tests:
        try:
            res = test_fn()
            all_results[name] = res
        except Exception as e:
            import traceback
            print(f"\n  {name} ERROR: {e}")
            traceback.print_exc()

    if all_results:
        save_path = os.path.join(SCRIPT_DIR, "training_divergence_realdata_50steps.png")
        plot_divergence_realdata(all_results, save_path)

    # Summary table
    print(f"\n{'='*90}")
    print(f"{'Model':<25} {'Steps':>5} {'Final Loss Diff':>18} {'Final Out MAE':>18} {'Rel Loss Diff':>18}")
    print(f"{'='*90}")
    for name, res in all_results.items():
        n = len(res["loss_diffs"])
        ld = res["loss_diffs"][-1]
        od = res["output_diffs"][-1]
        tl = res["torch_losses"][-1]
        rel = ld / tl if tl > 0 else 0
        print(f"{name:<25} {n:>5} {ld:>18.6e} {od:>18.6e} {rel:>18.6e}")

    n_pass = len(all_results)
    n_total = len(tests)
    print(f"\nCompleted: {n_pass}/{n_total}")


if __name__ == "__main__":
    main()
