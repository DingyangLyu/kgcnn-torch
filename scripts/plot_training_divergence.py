#!/usr/bin/env python3
"""Measure training divergence: Torch vs Keras over many steps.

Both models start from identical weights (transferred once), then train
independently with the same SGD optimizer and data. No intermediate
weight re-alignment. Plots loss curves + divergence over time.

Usage:
    KERAS_BACKEND=torch CUDA_VISIBLE_DEVICES="" python scripts/plot_training_divergence.py
"""
import os
import sys
import math
import numpy as np

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch
import torch.nn.functional as F

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from model_alignment_utils import (
    make_disjoint_graph, make_disjoint_graph_directed,
    make_disjoint_graph_relational, keras_to_torch,
)
from train_alignment_utils import collect_keras_params

N_STEPS = 50
LR = 0.01


def run_long_training(name, torch_fwd, keras_fwd, torch_params, keras_params, target,
                      n_steps=N_STEPS, lr=LR, grad_clip=None):
    """Run n_steps of independent training on both models."""
    torch_opt = torch.optim.SGD(torch_params, lr=lr)
    keras_opt = torch.optim.SGD(keras_params, lr=lr)

    torch_losses = []
    keras_losses = []
    loss_diffs = []
    output_diffs = []  # MAE between torch and keras outputs

    print(f"\n=== {name}: {n_steps} steps, lr={lr}"
          f"{f', grad_clip={grad_clip}' if grad_clip else ''} ===")
    print(f"  params: torch={len(torch_params)}, keras={len(keras_params)}")

    for step in range(n_steps):
        torch_out = torch_fwd()
        keras_out = keras_fwd()

        torch_loss = F.mse_loss(torch_out, target)
        keras_loss = F.mse_loss(keras_out, target)

        tl = float(torch_loss.item())
        kl = float(keras_loss.item())
        torch_losses.append(tl)
        keras_losses.append(kl)
        loss_diffs.append(abs(tl - kl))

        # Measure output divergence
        with torch.no_grad():
            out_mae = float((torch_out.detach() - keras_out.detach()).abs().mean().item())
        output_diffs.append(out_mae)

        if step % 10 == 0 or step == n_steps - 1:
            print(f"  Step {step:3d}: loss_t={tl:.6e} loss_k={kl:.6e} "
                  f"loss_diff={loss_diffs[-1]:.3e} out_mae={out_mae:.3e}")

        if math.isnan(tl) or math.isnan(kl) or math.isinf(tl) or math.isinf(kl):
            print(f"  NaN/Inf at step {step}, stopping.")
            break

        torch_opt.zero_grad()
        torch_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(torch_params, grad_clip)
        torch_opt.step()

        keras_opt.zero_grad()
        keras_loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(keras_params, grad_clip)
        keras_opt.step()

    return {
        "torch_losses": torch_losses,
        "keras_losses": keras_losses,
        "loss_diffs": loss_diffs,
        "output_diffs": output_diffs,
    }


# ---- Models ----

def test_gcn():
    from align_gcn_model import Config, KerasGCNFullStack, GCNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed, include_edge_attr=False)

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
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GCN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ew_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_schnet():
    from align_schnet_model import Config, KerasSchNetFullStack, SchNetModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed, include_pos=True, include_edge_attr=False)

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
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("SchNet",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_gat():
    from align_gat_model import Config, KerasGATFullStack, GATModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)

    torch_model = GATModel(
        node_dim=cfg.node_dim, depth=cfg.depth, attention_units=cfg.attention_units,
        attention_heads_num=cfg.heads, attention_heads_concat=cfg.concat,
        attention_activation="leaky_relu2", use_edge_features=True, edge_dim=cfg.edge_dim,
        node_pooling="mean", output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        output_final_activation=cfg.output_final_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasGATFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GAT",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_megnet():
    from align_megnet_model import Config, KerasMEGNetFullStack, MEGNetModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_input_dim, seed=cfg.seed, include_edge_attr=True)

    torch_model = MEGNetModel(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, state_dim=cfg.state_dim,
        edge_input_dim=cfg.edge_input_dim, state_input_dim=0, depth=cfg.depth,
        block_units_edge=cfg.block_units_edge, block_units_node=cfg.block_units_node,
        block_units_state=cfg.block_units_state, node_ff_units=cfg.node_ff_units,
        edge_ff_units=cfg.edge_ff_units, state_ff_units=cfg.state_ff_units,
        activation="softplus2", has_ff=True, dropout=None, use_set2set=False,
        node_pooling="mean", output_units=cfg.output_units, output_activation="softplus2",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings, use_graph_embedding=False)
    torch_model.train()

    keras_stack = KerasMEGNetFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("MEGNet",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_graphsage():
    from align_graphsage_model import Config, KerasGraphSAGEFullStack, GraphSAGEModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)

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
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("GraphSAGE",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_dmpnn():
    from align_dmpnn_model import Config, KerasDMPNNFullStack, DMPNNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph_directed(
        n_nodes=cfg.n_nodes, n_edges_per_dir=cfg.n_edges_per_dir,
        batch_size=cfg.batch_size, node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)

    torch_model = DMPNNModel(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, depth=cfg.depth, units=cfg.units,
        message_activation="relu", init_activation="relu", node_activation="relu",
        message_pooling="sum", node_pooling="sum",
        output_units=cfg.output_units, output_activation="relu",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings,
        use_edge_embedding=False, dropout_rate=0.0, use_graph_state=False)
    torch_model.train()

    keras_stack = KerasDMPNNFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    ep_k = keras_data["edge_pair_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("DMPNN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_egnn():
    from align_egnn_model import Config, KerasEGNNFullStack, EGNNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_edge_attr=True, include_pos=True)

    torch_model = EGNNModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        edge_mlp_units=cfg.edge_mlp_units, edge_mlp_activation=cfg.edge_mlp_activation,
        coord_mlp_units=cfg.coord_mlp_units, coord_mlp_activation=cfg.coord_mlp_activation,
        node_mlp_units=cfg.node_mlp_units, node_mlp_activation=cfg.node_mlp_activation,
        use_edge_attr=True, edge_attr_dim=cfg.edge_dim,
        use_attention=False, use_normalize=False, use_skip=True,
        use_node_attributes=False, use_node_normalization=False,
        layer_pooling="sum", coord_pooling="mean", node_pooling="sum",
        output_units=cfg.output_units, output_activation=cfg.edge_mlp_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasEGNNFullStack(cfg)
    z_k, pos_k, ea_k = keras_data["z"], keras_data["pos"], keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, pos_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("EGNN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_painn():
    from align_painn_model import Config, KerasPAiNNFullStack, PAiNNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)

    # Remove self-loops
    src_t = torch_data.edge_index[0]; dst_t = torch_data.edge_index[1]
    mask = src_t != dst_t
    torch_data.edge_index = torch_data.edge_index[:, mask]
    torch_data.edge_weight = torch_data.edge_weight[mask]
    src_k = keras_data["edge_index"][1]; dst_k = keras_data["edge_index"][0]
    mask_k = src_k != dst_k
    keras_data["edge_index"] = keras_data["edge_index"][:, mask_k]
    keras_data["edge_weight"] = keras_data["edge_weight"][mask_k]
    keras_data["batch_id_edge"] = keras_data["batch_id_edge"][mask_k]
    for i in range(cfg.batch_size):
        keras_data["count_edges"][i] = (keras_data["batch_id_edge"] == i).sum()

    torch_model = PAiNNModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        num_radial=cfg.num_radial, cutoff=cfg.cutoff, conv_cutoff=cfg.conv_cutoff,
        envelope_exponent=cfg.envelope_exponent,
        conv_activation="swish", conv_pooling="sum",
        update_activation="swish", update_add_eps=True,
        equiv_normalization=False, node_normalization=False, node_pooling="sum",
        output_units=cfg.output_units, output_activation="swish",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()

    keras_stack = KerasPAiNNFullStack(cfg)
    from kgcnn.literature.PAiNN._layers import PAiNNUpdate as KerasPAiNNUpdate
    keras_stack.updates = [
        KerasPAiNNUpdate(units=cfg.units, activation="swish", add_eps=True)
        for _ in range(cfg.depth)]
    z_k, pos_k, ei_k = keras_data["z"], keras_data["pos"], keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)

    return run_long_training("PAiNN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target, lr=1e-4)


def test_gin():
    from align_gin_model import Config, KerasGINFullStack, GINModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed, include_edge_attr=False)
    torch_model = GINModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        gin_mlp_units=cfg.gin_mlp_units, gin_mlp_activation="relu",
        gin_mlp_use_normalization=False, gin_pooling="sum",
        epsilon_learnable=False, use_edge_features=False, node_pooling="sum",
        last_mlp_units=cfg.last_mlp_units, last_mlp_activation="relu", dropout_rate=0.0,
        output_units=cfg.output_units, output_activation="relu",
        output_final_activation="linear",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasGINFullStack(cfg)
    z_k, ei_k = keras_data["z"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("GIN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_attentivefp():
    from align_attentivefp_model import Config, KerasAttentiveFPFullStack, AttentiveFPModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = AttentiveFPModel(
        node_dim=cfg.node_dim, depth_ato=cfg.depth_ato, depth_mol=cfg.depth_mol,
        units=cfg.units, use_edge_features=True, edge_dim=cfg.edge_dim,
        attention_activation="leaky_relu2", attention_activation_context="elu",
        pooling_activation="leaky_relu2", pooling_activation_context="elu",
        node_pooling="sum", output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, dropout=0.0, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasAttentiveFPFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("AttentiveFP",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_gatv2():
    from align_gatv2_model import Config, KerasGATv2FullStack, GATv2Model, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = GATv2Model(
        node_dim=cfg.node_dim, depth=cfg.depth, attention_units=cfg.attention_units,
        attention_heads_num=cfg.heads, attention_heads_concat=cfg.concat,
        attention_activation="leaky_relu2", use_edge_features=True, edge_dim=cfg.edge_dim,
        node_pooling="mean", output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasGATv2FullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("GATv2",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_megan():
    from align_megan_model import Config, KerasMEGANFullStack, MEGANModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed, include_edge_attr=False)
    torch_model = MEGANModel(
        node_dim=cfg.node_dim, units=cfg.units, num_heads=cfg.num_heads,
        depth=cfg.depth, attention_activation="leaky_relu2",
        use_edge_features=False, concat_heads=cfg.concat_heads,
        importance_channels=cfg.importance_channels,
        importance_units=cfg.importance_units, importance_activation="relu",
        final_units=cfg.final_units, final_activation=cfg.final_activation,
        use_bias=True, final_pooling=cfg.final_pooling,
        dropout_rate=0.0, final_dropout_rate=0.0, regression_reference=None,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasMEGANFullStack(cfg)
    z_k, ei_k = keras_data["z"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("MEGAN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_mogat():
    from align_mogat_model import Config, KerasMoGATFullStack, MoGATModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = MoGATModel(
        node_dim=cfg.node_dim, depthato=cfg.depthato, depthmol=cfg.depthmol,
        units=cfg.units, edge_dim=cfg.edge_dim, use_edge_features=True,
        activation="leaky_relu2", dropout=0.0,
        output_units=cfg.output_units, output_activation="relu",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasMoGATFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("MoGAT",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_inorp():
    from align_inorp_model import Config, KerasINorpFullStack, INorpModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = INorpModel(
        node_dim=cfg.node_dim, depth=cfg.depth, edge_dim=cfg.edge_dim,
        edge_mlp_units=cfg.edge_mlp_units, edge_mlp_activation="relu",
        node_mlp_units=cfg.node_mlp_units, node_mlp_activation="relu",
        message_pooling="mean", use_set2set=False, use_graph_state=False,
        node_pooling="mean", output_units=cfg.output_units,
        output_activation=cfg.output_activation, output_final_activation="sigmoid",
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings, use_edge_embedding=False)
    torch_model.train()
    keras_stack = KerasINorpFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("INorp",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_nmpn():
    from align_nmpn_model import Config, KerasNMPNFullStack, NMPNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = NMPNModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units, edge_dim=cfg.edge_dim,
        edge_mlp_units=cfg.edge_mlp_units, edge_mlp_activation=cfg.edge_mlp_activation,
        message_pooling="sum", use_set2set=False, node_pooling="sum",
        output_units=cfg.output_units, output_activation="selu",
        output_final_activation="sigmoid",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasNMPNFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("NMPN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_hamnet():
    from align_hamnet_model import Config, KerasHamNetFullStack, HamNetModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, seed=cfg.seed,
        include_pos=True, include_edge_attr=True)
    torch_model = HamNetModel(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, depth=cfg.depth, units=cfg.units,
        fingerprint_dim=cfg.fingerprint_dim, fingerprint_depth=cfg.fingerprint_depth,
        activation=cfg.activation, activation_last="elu",
        fingerprint_activation=cfg.activation, fingerprint_activation_context=cfg.activation,
        use_gru_update=cfg.use_gru_update, use_gru_update_edge=False,
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasHamNetFullStack(cfg)
    z_k, pos_k, ea_k = keras_data["z"], keras_data["pos"], keras_data["edge_attr"]
    ei_k = keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, pos_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("HamNet",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ea_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_rgcn():
    from align_rgcn_model import Config, KerasRGCNFullStack, RGCNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph_relational(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, num_relations=cfg.num_relations,
        seed=cfg.seed, include_edge_weight=True)
    torch_model = RGCNModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        num_relations=cfg.num_relations, rgcn_activation="swish",
        rgcn_pooling="sum", use_residual=False, node_pooling="sum",
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasRGCNFullStack(cfg)
    z_k, ei_k = keras_data["z"], keras_data["edge_index"]
    et_k, ea_k = keras_data["edge_type"], keras_data["edge_attr"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, et_k, ea_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("RGCN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, et_k, ea_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_gnnfilm():
    from align_gnnfilm_model import Config, KerasGNNFilmFullStack, GNNFilmModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph_relational(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, num_relations=cfg.num_relations,
        seed=cfg.seed, include_edge_weight=False)
    torch_model = GNNFilmModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        num_relations=cfg.num_relations, activation="swish",
        modulation_activation="sigmoid", film_pooling="sum", node_pooling="sum",
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasGNNFilmFullStack(cfg)
    z_k, ei_k, et_k = keras_data["z"], keras_data["edge_index"], keras_data["edge_type"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, et_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("GNNFilm",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, et_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_cmpnn():
    from align_cmpnn_model import Config, KerasCMPNNFullStack, CMPNNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph_directed(
        n_nodes=cfg.n_nodes, n_edges_per_dir=cfg.n_edges_per_dir,
        batch_size=cfg.batch_size, node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = CMPNNModel(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim, depth=cfg.depth, units=cfg.units,
        dropout=0.0, activation="relu", node_dense_activation="linear",
        use_final_gru=False, node_pooling="sum",
        output_units=cfg.output_units, output_activation="relu",
        output_use_bias=[True] * len(cfg.output_units) + [False],
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasCMPNNFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    ep_k = keras_data["edge_pair_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("CMPNN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target, lr=1e-4, grad_clip=1.0)


def test_dgin():
    from align_dgin_model import Config, KerasDGINFullStack, DGINModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph_directed(
        n_nodes=cfg.n_nodes, n_edges_per_dir=cfg.n_edges_per_dir,
        batch_size=cfg.batch_size, node_dim=cfg.node_dim,
        edge_dim=cfg.edge_dim, seed=cfg.seed, include_edge_attr=True)
    torch_model = DGINModel(
        node_dim=cfg.node_dim, edge_dim=cfg.edge_dim,
        depth_dmpnn=cfg.depth_dmpnn, depth_gin=cfg.depth_gin, units=cfg.units,
        dropout_dmpnn=0.0, dropout_gin=0.0, activation="relu",
        gin_mlp_units=cfg.gin_mlp_units, gin_mlp_activation=["relu", "linear"],
        gin_mlp_use_normalization=False, last_mlp_units=cfg.last_mlp_units,
        node_pooling="mean", output_units=cfg.output_units, output_activation="relu",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasDGINFullStack(cfg)
    z_k, ea_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    ep_k = keras_data["edge_pair_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("DGIN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ea_k, ei_k, ep_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target)


def test_dimenetpp():
    from align_dimenetpp_model import (
        Config, KerasDimeNetPPFullStack, DimeNetPPModel,
        transfer_all_weights, generate_angle_index)
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_index = generate_angle_index(torch_data.edge_index)
    torch_data.angle_index = angle_index
    torch_model = DimeNetPPModel(
        emb_size=cfg.emb_size, out_emb_size=cfg.out_emb_size,
        int_emb_size=cfg.int_emb_size, basis_emb_size=cfg.basis_emb_size,
        num_blocks=cfg.num_blocks, num_spherical=cfg.num_spherical,
        num_radial=cfg.num_radial, cutoff=cfg.cutoff,
        envelope_exponent=cfg.envelope_exponent,
        num_before_skip=cfg.num_before_skip, num_after_skip=cfg.num_after_skip,
        num_dense_output=cfg.num_dense_output,
        num_targets=cfg.num_targets, activation=cfg.activation,
        extensive=cfg.extensive, output_init=cfg.output_init, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings,
        use_output_mlp=cfg.use_output_mlp, output_mlp_units=cfg.output_mlp_units,
        output_mlp_activation=cfg.output_mlp_activation)
    torch_model.train()
    keras_stack = KerasDimeNetPPFullStack(cfg)
    z_k, pos_k, ei_k = keras_data["z"], keras_data["pos"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("DimeNetPP",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k),
        torch_params, keras_params, target, lr=1e-7, grad_clip=0.1)


def test_mxmnet():
    from align_mxmnet_model import (
        Config, KerasMXMNetFullStack, MXMNetModel,
        transfer_all_weights, generate_angle_index_mxmnet)
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_idx = generate_angle_index_mxmnet(torch_data.edge_index)
    torch_data.angle_index_1 = angle_idx
    torch_data.angle_index_2 = angle_idx
    torch_model = MXMNetModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        num_radial=cfg.num_radial, num_spherical=cfg.num_spherical,
        num_radial_spherical=cfg.num_radial_spherical,
        cutoff=cfg.cutoff, envelope_exponent=cfg.envelope_exponent,
        activation=cfg.activation, mp_pooling=cfg.mp_pooling,
        global_mp_pooling=cfg.global_mp_pooling,
        use_local_mp=True, node_pooling=cfg.node_pooling,
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings, use_output_mlp=True)
    torch_model.train()
    keras_stack = KerasMXMNetFullStack(cfg)
    z_k, pos_k, ei_k = keras_data["z"], keras_data["pos"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, angle_idx, angle_idx, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("MXMNet",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, pos_k, ei_k, angle_idx, angle_idx, bid_k, cn_k),
        torch_params, keras_params, target, lr=1e-7, grad_clip=0.1)


def test_cgcnn():
    from align_cgcnn_model import Config, KerasCGCNNFullStack, CGCNNModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed,
        include_edge_attr=True, include_pos=False)
    torch.manual_seed(cfg.seed + 1)
    distances = torch.rand(torch_data.edge_index.shape[1], 1) * cfg.gauss_distance
    torch_data.edge_attr = distances
    keras_data["edge_attr"] = distances
    torch_model = CGCNNModel(
        node_dim=cfg.node_dim, depth=cfg.depth, conv_units=cfg.conv_units,
        gauss_bins=cfg.gauss_bins, gauss_distance=cfg.gauss_distance,
        gauss_offset=cfg.gauss_offset, gauss_sigma=cfg.gauss_sigma,
        batch_normalization=False, node_pooling="mean",
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasCGCNNFullStack(cfg)
    z_k, dist_k, ei_k = keras_data["z"], keras_data["edge_attr"], keras_data["edge_index"]
    bid_k, bie_k = keras_data["batch_id_node"], keras_data["batch_id_edge"]
    cn_k, ce_k = keras_data["count_nodes"], keras_data["count_edges"]
    _ = keras_stack.forward(z_k, dist_k, ei_k, bid_k, bie_k, cn_k, ce_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("CGCNN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, dist_k, ei_k, bid_k, bie_k, cn_k, ce_k),
        torch_params, keras_params, target, lr=1e-3)


def test_rgin():
    from align_rgin_model import Config, KerasrGINFullStack, rGINModel, transfer_all_weights
    cfg = Config()
    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.node_dim, edge_dim=1, seed=cfg.seed, include_edge_attr=False)
    torch_model = rGINModel(
        node_dim=cfg.node_dim, depth=cfg.depth, units=cfg.units,
        gin_mlp_units=cfg.gin_mlp_units, gin_mlp_activation=["relu", "linear"],
        gin_mlp_use_normalization=False, gin_pooling="sum",
        epsilon_learnable=False, random_range=cfg.random_range, dropout=0.0,
        node_pooling="sum", last_mlp_units=cfg.last_mlp_units, last_mlp_activation="relu",
        output_units=cfg.output_units, output_activation="relu",
        output_final_activation="linear",
        num_targets=cfg.num_targets, output_embedding="graph",
        use_node_embedding=True, num_embeddings=cfg.num_embeddings)
    torch_model.train()
    keras_stack = KerasrGINFullStack(cfg)
    z_k, ei_k = keras_data["z"], keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, ei_k, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)
    # Patch fixed randoms for deterministic comparison
    np.random.seed(cfg.seed + 200)
    n_total = torch_data.z.shape[0]
    fixed_randoms = [
        torch.tensor(np.random.rand(n_total, 1).astype(np.float32))
        for _ in range(cfg.depth)]
    for i in range(cfg.depth):
        conv = torch_model.convs[i]
        def _ptf(x, edge_index, _conv=conv, _i=i):
            num_nodes = x.size(0)
            x_aug = torch.cat([x, fixed_randoms[_i].to(x.device)], dim=-1)
            from kgcnn_torch.layers.gather import gather_nodes_outgoing
            x_j = gather_nodes_outgoing(x_aug, edge_index)
            agg = _conv.aggr(x_j, edge_index, num_nodes)
            return x_aug + agg
        conv.forward = _ptf
    for i in range(cfg.depth):
        k_conv = keras_stack.convs[i]
        def _pkc(inputs, _k_conv=k_conv, _i=i, **kwargs):
            node, edge_index = inputs
            node = _k_conv.lay_concat([node, fixed_randoms[_i]])
            ed = _k_conv.lay_gather([node, edge_index], **kwargs)
            nu = _k_conv.lay_pool([node, ed, edge_index], **kwargs)
            out = _k_conv.lay_add([node, nu], **kwargs)
            return out
        k_conv.call = _pkc
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("rGIN",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z_k, ei_k, bid_k, cn_k),
        torch_params, keras_params, target)


def test_mat():
    from align_mat_model import (
        Config, KerasMATFullStack, MATModel, transfer_all_weights, make_padded_data)
    cfg = Config()
    node_input, xyz_input, adjacency, node_mask, adj_mask = make_padded_data(cfg)
    torch_model = MATModel(
        embedding_units=cfg.embedding_units, depth=cfg.depth,
        num_heads=cfg.num_heads, attention_units=cfg.attention_units,
        merge_heads=cfg.merge_heads, lambda_attention=cfg.lambda_attention,
        lambda_distance=cfg.lambda_distance, add_identity=cfg.add_identity,
        attention_dropout=None, distance_trafo=cfg.distance_trafo,
        units_ff=cfg.units_ff, ff_activations=cfg.ff_activations,
        output_units=cfg.output_units, output_activations=cfg.output_activations,
        num_targets=cfg.num_targets, use_node_embedding=True,
        num_embeddings=cfg.num_embeddings, input_node_dim=cfg.input_node_dim,
        output_embedding="graph")
    torch_model.train()
    keras_stack = KerasMATFullStack(cfg)
    _ = keras_stack.forward(node_input, xyz_input, adjacency, node_mask, adj_mask)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("MAT",
        lambda: torch_model(node_input, xyz_input, adjacency, node_mask.float(), adj_mask.float()),
        lambda: keras_stack.forward(node_input, xyz_input, adjacency, node_mask, adj_mask),
        torch_params, keras_params, target, lr=1e-3, grad_clip=1.0)


def test_hdnnp2nd():
    from align_hdnnp2nd_model import Config, KerasHDNNP2ndFullStack, HDNNP2ndModel, transfer_all_weights
    from types import SimpleNamespace
    cfg = Config()
    torch.manual_seed(cfg.seed)
    element_types = list(range(cfg.n_types))
    z = torch.randint(0, cfg.n_types, (cfg.n_nodes,), dtype=torch.long)
    pos = torch.randn(cfg.n_nodes, 3)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    center = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr2 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    angle_index = torch.stack([center, nbr1, nbr2], dim=0)
    nodes_per_graph = cfg.n_nodes // cfg.batch_size
    batch_node = torch.cat([torch.full((nodes_per_graph,), i, dtype=torch.long) for i in range(cfg.batch_size)])
    remainder = cfg.n_nodes - nodes_per_graph * cfg.batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node, torch.full((remainder,), cfg.batch_size - 1, dtype=torch.long)])
    count_nodes = torch.zeros(cfg.batch_size, dtype=torch.long)
    for i in range(cfg.batch_size):
        count_nodes[i] = (batch_node == i).sum()
    torch_model = HDNNP2ndModel(
        element_types=element_types, n_rad_features=cfg.n_rad_features,
        n_ang_features=cfg.n_ang_features, cutoff=cfg.cutoff,
        num_relations=cfg.num_relations, relational_units=cfg.relational_units,
        relational_activation=cfg.relational_activation, use_batch_norm=False,
        node_pooling=cfg.node_pooling, use_output_mlp=True,
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph")
    torch_model.train()
    eta_mu = np.stack([torch_model.acsf_rad.eta.detach().cpu().numpy(),
                       torch_model.acsf_rad.mu.detach().cpu().numpy()], axis=-1)
    eta_mu_lz = np.stack([torch_model.acsf_ang.eta.detach().cpu().numpy(),
                          torch_model.acsf_ang.mu.detach().cpu().numpy(),
                          torch_model.acsf_ang.lam.detach().cpu().numpy(),
                          torch_model.acsf_ang.zeta.detach().cpu().numpy()], axis=-1)
    keras_stack = KerasHDNNP2ndFullStack(cfg, eta_mu, eta_mu_lz)
    _ = keras_stack.forward(z, pos, edge_index_keras, angle_index, batch_node, count_nodes)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_data = SimpleNamespace(z=z, pos=pos, edge_index=edge_index_torch,
                                 angle_index=angle_index, batch=batch_node)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("HDNNP2nd",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z, pos, edge_index_keras, angle_index, batch_node, count_nodes),
        torch_params, keras_params, target, lr=1e-3)


def test_hdnnp2nd_behler():
    from align_hdnnp2nd_behler_model import (
        Config, KerasHDNNP2ndBehlerFullStack, HDNNP2ndBehlerModel, transfer_all_weights)
    from types import SimpleNamespace
    cfg = Config()
    torch.manual_seed(cfg.seed)
    z_choices = torch.tensor(sorted(cfg.element_types), dtype=torch.long)
    z = z_choices[torch.randint(0, len(cfg.element_types), (cfg.n_nodes,))]
    pos = torch.randn(cfg.n_nodes, 3)
    src = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    dst = torch.randint(0, cfg.n_nodes, (cfg.n_edges,), dtype=torch.long)
    edge_index_torch = torch.stack([src, dst], dim=0)
    edge_index_keras = torch.stack([dst, src], dim=0)
    center = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr2 = torch.randint(0, cfg.n_nodes, (cfg.n_angles,), dtype=torch.long)
    nbr1 = torch.where(nbr1 == center, (nbr1 + 1) % cfg.n_nodes, nbr1)
    nbr2 = torch.where(nbr2 == center, (nbr2 + 2) % cfg.n_nodes, nbr2)
    nbr2 = torch.where(nbr2 == nbr1, (nbr2 + 1) % cfg.n_nodes, nbr2)
    angle_index = torch.stack([center, nbr1, nbr2], dim=0)
    nodes_per_graph = cfg.n_nodes // cfg.batch_size
    batch_node = torch.cat([torch.full((nodes_per_graph,), i, dtype=torch.long) for i in range(cfg.batch_size)])
    remainder = cfg.n_nodes - nodes_per_graph * cfg.batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node, torch.full((remainder,), cfg.batch_size - 1, dtype=torch.long)])
    count_nodes = torch.zeros(cfg.batch_size, dtype=torch.long)
    for i in range(cfg.batch_size):
        count_nodes[i] = (batch_node == i).sum()
    torch_model = HDNNP2ndBehlerModel(
        element_types=cfg.element_types, g2_eta=cfg.g2_eta, g2_rs=cfg.g2_rs, g2_rc=cfg.g2_rc,
        g4_eta=cfg.g4_eta, g4_zeta=cfg.g4_zeta, g4_lamda=cfg.g4_lamda,
        g4_rc=cfg.g4_rc, g4_multiplicity=cfg.g4_multiplicity,
        relational_units=cfg.relational_units, relational_activation=cfg.relational_activation,
        use_batch_norm=False, node_pooling=cfg.node_pooling,
        output_units=cfg.output_units, output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph")
    torch_model.train()
    keras_stack = KerasHDNNP2ndBehlerFullStack(cfg)
    _ = keras_stack.forward(z, pos, edge_index_keras, angle_index, batch_node, count_nodes)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_data = SimpleNamespace(z=z, pos=pos, edge_index=edge_index_torch,
                                 angle_index=angle_index, batch=batch_node)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("HDNNP2ndBehler",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z, pos, edge_index_keras, angle_index, batch_node, count_nodes),
        torch_params, keras_params, target, lr=1e-3)


def test_hdnnp2nd_atomwise():
    from align_hdnnp2nd_atomwise_model import (
        Config, KerasHDNNP2ndAtomWiseFullStack, HDNNP2ndAtomWiseModel, transfer_all_weights)
    from types import SimpleNamespace
    cfg = Config()
    torch.manual_seed(cfg.seed)
    nodes_per_graph = cfg.n_nodes // cfg.batch_size
    batch_node = torch.cat([torch.full((nodes_per_graph,), i, dtype=torch.long) for i in range(cfg.batch_size)])
    remainder = cfg.n_nodes - nodes_per_graph * cfg.batch_size
    if remainder > 0:
        batch_node = torch.cat([batch_node, torch.full((remainder,), cfg.batch_size - 1, dtype=torch.long)])
    total_nodes = batch_node.shape[0]
    count_nodes = torch.zeros(cfg.batch_size, dtype=torch.long)
    for i in range(cfg.batch_size):
        count_nodes[i] = (batch_node == i).sum()
    z_choices = torch.tensor(cfg.element_types, dtype=torch.long)
    z = z_choices[torch.randint(0, len(cfg.element_types), (total_nodes,))]
    x = torch.randn(total_nodes, cfg.input_dim)
    torch_data = SimpleNamespace(z=z, x=x, batch=batch_node)
    torch_model = HDNNP2ndAtomWiseModel(
        element_types=cfg.element_types, input_dim=cfg.input_dim,
        relational_units=cfg.relational_units, relational_activation=cfg.relational_activation,
        node_pooling=cfg.node_pooling, output_units=cfg.output_units,
        output_activation=cfg.output_activation,
        num_targets=cfg.num_targets, output_embedding="graph")
    torch_model.train()
    keras_stack = KerasHDNNP2ndAtomWiseFullStack(cfg)
    _ = keras_stack.forward(z, x, batch_node, count_nodes)
    transfer_all_weights(torch_model, keras_stack, cfg)
    torch_params = list(torch_model.parameters())
    keras_params = collect_keras_params(keras_stack)
    target = torch.randn(cfg.batch_size, cfg.num_targets); target.requires_grad_(False)
    return run_long_training("HDNNP2ndAtomWise",
        lambda: torch_model(torch_data),
        lambda: keras_stack.forward(z, x, batch_node, count_nodes),
        torch_params, keras_params, target)


# ---- Plotting ----

def plot_divergence(all_results, save_path):
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

        # Col 1: Loss curves
        ax = axes[idx, 0]
        ax.plot(steps, tl, "-", color="#2196F3", label="Torch", linewidth=1.5)
        ax.plot(steps, kl, "--", color="#FF5722", label="Keras", linewidth=1.5)
        ax.set_title(f"{name}: Loss Curves", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("MSE Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        # Col 2: Loss difference (absolute)
        ax = axes[idx, 1]
        ax.semilogy(steps, [max(d, 1e-15) for d in ld], "-", color="#4CAF50", linewidth=1.5)
        ax.set_title(f"{name}: |Loss_torch - Loss_keras|", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Absolute Diff")
        ax.grid(True, alpha=0.3)
        ax.axhline(y=1e-6, color="gray", linestyle=":", alpha=0.5, label="1e-6")
        ax.legend(fontsize=9)

        # Col 3: Output MAE divergence
        ax = axes[idx, 2]
        ax.semilogy(steps, [max(d, 1e-15) for d in od], "-", color="#9C27B0", linewidth=1.5)
        ax.set_title(f"{name}: Output MAE (Torch vs Keras)", fontsize=11, fontweight="bold")
        ax.set_xlabel("Step")
        ax.set_ylabel("Mean |out_torch - out_keras|")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Training Divergence Over 50 Steps (No Intermediate Weight Sync)",
                 fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved to: {save_path}")
    plt.close(fig)


def main():
    tests = [
        ("GCN", test_gcn),
        ("SchNet", test_schnet),
        ("GAT", test_gat),
        ("GIN", test_gin),
        ("PAiNN", test_painn),
        ("AttentiveFP", test_attentivefp),
        ("GATv2", test_gatv2),
        ("MEGAN", test_megan),
        ("MoGAT", test_mogat),
        ("GraphSAGE", test_graphsage),
        ("INorp", test_inorp),
        ("NMPN", test_nmpn),
        ("MEGNet", test_megnet),
        ("EGNN", test_egnn),
        ("HamNet", test_hamnet),
        ("RGCN", test_rgcn),
        ("GNNFilm", test_gnnfilm),
        ("DMPNN", test_dmpnn),
        ("CMPNN", test_cmpnn),
        ("DGIN", test_dgin),
        ("DimeNetPP", test_dimenetpp),
        ("MXMNet", test_mxmnet),
        ("CGCNN", test_cgcnn),
        ("rGIN", test_rgin),
        ("MAT", test_mat),
        ("HDNNP2nd", test_hdnnp2nd),
        ("HDNNP2ndBehler", test_hdnnp2nd_behler),
        ("HDNNP2ndAtomWise", test_hdnnp2nd_atomwise),
    ]

    all_results = {}
    for name, test_fn in tests:
        try:
            res = test_fn()
            all_results[name] = res
        except Exception as e:
            print(f"  {name} ERROR: {e}")

    if all_results:
        save_path = os.path.join(os.path.dirname(__file__), "training_divergence_50steps.png")
        plot_divergence(all_results, save_path)

    # Print summary table
    print(f"\n{'='*85}")
    print(f"{'Model':<15} {'Steps':>5} {'Final Loss Diff':>18} {'Final Out MAE':>18} {'Rel Loss Diff':>18}")
    print(f"{'='*85}")
    for name, res in all_results.items():
        n = len(res["loss_diffs"])
        ld = res["loss_diffs"][-1]
        od = res["output_diffs"][-1]
        tl = res["torch_losses"][-1]
        rel = ld / tl if tl > 0 else 0
        print(f"{name:<15} {n:>5} {ld:>18.6e} {od:>18.6e} {rel:>18.6e}")


if __name__ == "__main__":
    main()
