#!/usr/bin/env python3
"""Check intermediate values in DimeNetPP: are interaction block outputs different?

The zero-init dense_final masks any differences — forward output is always 0.
This script compares intermediate values to find where divergence begins.
"""
import os
import sys

os.environ.setdefault("KERAS_BACKEND", "torch")

import torch

ROOT = "/home/yuanbai/Downloads/MLIPs"
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch", "scripts"))
sys.path.insert(0, os.path.join(ROOT, "kgcnn-torch"))
sys.path.insert(0, os.path.join(ROOT, "gcnn_keras-master"))

from model_alignment_utils import make_disjoint_graph, compare_outputs
from align_dimenetpp_model import (
    Config, KerasDimeNetPPFullStack, DimeNetPPModel,
    transfer_all_weights, generate_angle_index)

# Also import Keras layers for geometry
from kgcnn.layers.geom import NodePosition, NodeDistanceEuclidean, EdgeAngle
from keras.layers import Subtract


def compare(name, t, k, max_mae=1e-4, max_abs=1e-3):
    """Compare two tensors and print results."""
    if t.shape != k.shape:
        print(f"  {name}: SHAPE MISMATCH torch={list(t.shape)} keras={list(k.shape)}")
        return
    diff = (t - k).abs()
    mae = diff.mean().item()
    maxabs = diff.max().item()
    status = "OK" if mae < max_mae and maxabs < max_abs else "DIFFER"
    print(f"  {name:<35} shape={str(list(t.shape)):<18} MAE={mae:.3e} MAX={maxabs:.3e} [{status}]")


def main():
    cfg = Config()
    torch.manual_seed(cfg.seed)

    torch_data, keras_data = make_disjoint_graph(
        n_nodes=cfg.n_nodes, n_edges=cfg.n_edges, batch_size=cfg.batch_size,
        node_dim=cfg.emb_size, edge_dim=1, seed=cfg.seed,
        include_pos=True, include_edge_attr=False)
    angle_index = generate_angle_index(torch_data.edge_index)
    torch_data.angle_index = angle_index

    # Build models
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
    torch_model.eval()

    keras_stack = KerasDimeNetPPFullStack(cfg)
    z_k, pos_k = keras_data["z"], keras_data["pos"]
    ei_k = keras_data["edge_index"]
    bid_k, cn_k = keras_data["batch_id_node"], keras_data["count_nodes"]
    _ = keras_stack.forward(z_k, pos_k, ei_k, angle_index, bid_k, cn_k)
    transfer_all_weights(torch_model, keras_stack, cfg)

    # ===== TORCH forward with intermediates =====
    print("=" * 80)
    print("Comparing intermediate values: Torch vs Keras DimeNetPP")
    print("=" * 80)

    with torch.no_grad():
        # --- Torch intermediates ---
        z_t = torch_data.z
        pos_t = torch_data.pos
        ei_t = torch_data.edge_index
        batch_t = torch_data.batch

        n_t = torch_model.node_embedding(z_t)

        # Geometry
        from kgcnn_torch.layers.geom import compute_edge_distances
        pos_i_t = pos_t[ei_t[1]]
        pos_j_t = pos_t[ei_t[0]]
        dist_t = compute_edge_distances(pos_t, ei_t)
        rbf_t = torch_model.bessel_basis(dist_t)

        v12_t = pos_i_t - pos_j_t
        vec_a_t = v12_t[angle_index[0]]
        vec_b_t = v12_t[angle_index[1]]
        dot_t = (vec_a_t * vec_b_t).sum(dim=-1)
        cross_t = torch.cross(vec_a_t, vec_b_t, dim=-1)
        cross_norm_t = torch.norm(cross_t, dim=-1)
        angles_t = torch.atan2(cross_norm_t, dot_t).unsqueeze(-1)
        sbf_t = torch_model.spherical_basis(dist_t, angles_t, angle_index)

        rbf_emb_t = torch_model.rbf_emb(rbf_t)
        n_target_t = n_t[ei_t[1]]
        n_source_t = n_t[ei_t[0]]
        n_pairs_t = torch.cat([n_target_t, n_source_t], dim=-1)
        x_t = torch.cat([n_pairs_t, rbf_emb_t], dim=-1)
        x_t = torch_model.edge_emb(x_t)

        # --- Keras intermediates ---
        n_k = keras_stack.node_embedding(z_k)
        pos1_k, pos2_k = NodePosition()([pos_k, ei_k])
        dist_k = NodeDistanceEuclidean()([pos1_k, pos2_k])
        rbf_k = keras_stack.bessel_basis(dist_k)
        v12_k = Subtract()([pos1_k, pos2_k])
        angles_k = EdgeAngle()([v12_k, angle_index])
        sbf_k = keras_stack.spherical_basis([dist_k, angles_k, angle_index])

        # Convert keras tensors to torch for comparison
        def to_t(x):
            if hasattr(x, 'numpy'):
                return torch.from_numpy(x.numpy())
            return x

        print("\n--- Geometry ---")
        compare("node_embedding", n_t, to_t(n_k))
        compare("distances", dist_t, to_t(dist_k))
        compare("rbf", rbf_t, to_t(rbf_k))
        compare("angles", angles_t, to_t(angles_k))
        compare("sbf", sbf_t, to_t(sbf_k))

        print("\n--- Edge embedding ---")
        rbf_emb_k = keras_stack.rbf_emb_dense(rbf_k)
        compare("rbf_emb", rbf_emb_t, to_t(rbf_emb_k))
        compare("edge_embedding (x0)", x_t, to_t(keras_stack.edge_emb_dense(
            torch.cat([
                keras_stack.node_embedding(z_k)[ei_k[0]],  # target in keras convention
                keras_stack.node_embedding(z_k)[ei_k[1]],  # source
                rbf_emb_k
            ], dim=-1))))

        # Interaction blocks
        print("\n--- Interaction blocks ---")
        x_torch = x_t.clone()
        # Run Keras forward manually to get intermediates
        from kgcnn.layers.gather import GatherNodes
        from keras.layers import Concatenate
        n_k2 = keras_stack.node_embedding(z_k)
        pos1_k2, pos2_k2 = NodePosition()([pos_k, ei_k])
        dist_k2 = NodeDistanceEuclidean()([pos1_k2, pos2_k2])
        rbf_k2 = keras_stack.bessel_basis(dist_k2)
        v12_k2 = Subtract()([pos1_k2, pos2_k2])
        angles_k2 = EdgeAngle()([v12_k2, angle_index])
        sbf_k2 = keras_stack.spherical_basis([dist_k2, angles_k2, angle_index])
        rbf_emb_k2 = keras_stack.rbf_emb_dense(rbf_k2)
        n_pairs_k2 = GatherNodes()([n_k2, ei_k])
        x_keras = keras_stack.edge_emb_dense(
            Concatenate(axis=-1)([n_pairs_k2, rbf_emb_k2]))

        compare("x0 (initial edge emb)", x_torch, to_t(x_keras))

        for i in range(cfg.num_blocks):
            x_torch = torch_model.interaction_blocks[i](
                x_torch, rbf_t, sbf_t, angle_index)
            x_keras = keras_stack.interaction_blocks[i](
                [x_keras, rbf_k2, sbf_k2, angle_index])
            compare(f"x{i+1} (after interaction {i})", x_torch, to_t(x_keras))

        # Output blocks internal values
        print("\n--- Output block internal values ---")
        for block_idx in range(cfg.num_blocks + 1):
            if block_idx == 0:
                t_block = torch_model.output_block_0
                k_block = keras_stack.output_block_0
                x_input_t = x_t  # initial edge embedding
                x_input_k = keras_stack.edge_emb_dense(
                    Concatenate(axis=-1)([
                        GatherNodes()([keras_stack.node_embedding(z_k), ei_k]),
                        keras_stack.rbf_emb_dense(
                            keras_stack.bessel_basis(
                                NodeDistanceEuclidean()(
                                    NodePosition()([pos_k, ei_k]))))
                    ]))
            else:
                t_block = torch_model.output_blocks[block_idx - 1]
                k_block = keras_stack.output_blocks[block_idx - 1]
                # x_input_t and x_input_k from the interaction block output above
                # (already computed in the loop above)
                # We need to recompute...
                pass  # Skip for now, focus on the interaction block outputs

            # Compare internal values of output block
            g_t = t_block.dense_rbf(rbf_t)
            g_k = k_block.dense_rbf(rbf_k2)
            compare(f"output_block_{block_idx}.dense_rbf", g_t, to_t(g_k))

        # The key test: does the output block's dense_final have different inputs?
        print("\n--- Output block dense_final inputs (the gradient source) ---")
        # Rerun the full forward with hooks to capture intermediate values
        # For simplicity, manually trace output_block outputs

        # Rebuild interaction block outputs
        x_torch_list = [x_t.clone()]
        x_keras_list = [to_t(keras_stack.edge_emb_dense(
            Concatenate(axis=-1)([
                GatherNodes()([keras_stack.node_embedding(z_k), ei_k]),
                keras_stack.rbf_emb_dense(
                    keras_stack.bessel_basis(
                        NodeDistanceEuclidean()(
                            NodePosition()([pos_k, ei_k]))))
            ])))]

        x_t_cur = x_torch_list[0]
        # Need fresh keras forward
        n_k3 = keras_stack.node_embedding(z_k)
        pos1_k3, pos2_k3 = NodePosition()([pos_k, ei_k])
        dist_k3 = NodeDistanceEuclidean()([pos1_k3, pos2_k3])
        rbf_k3 = keras_stack.bessel_basis(dist_k3)
        v12_k3 = Subtract()([pos1_k3, pos2_k3])
        angles_k3 = EdgeAngle()([v12_k3, angle_index])
        sbf_k3 = keras_stack.spherical_basis([dist_k3, angles_k3, angle_index])
        rbf_emb_k3 = keras_stack.rbf_emb_dense(rbf_k3)
        x_k_cur = keras_stack.edge_emb_dense(
            Concatenate(axis=-1)([
                GatherNodes()([n_k3, ei_k]), rbf_emb_k3]))

        for i in range(cfg.num_blocks):
            x_t_cur = torch_model.interaction_blocks[i](x_t_cur, rbf_t, sbf_t, angle_index)
            x_k_cur = keras_stack.interaction_blocks[i]([x_k_cur, rbf_k3, sbf_k3, angle_index])
            x_torch_list.append(x_t_cur)
            x_keras_list.append(to_t(x_k_cur))

        # Now trace each output block's internal values
        for block_idx in range(cfg.num_blocks + 1):
            if block_idx == 0:
                t_block = torch_model.output_block_0
                k_block = keras_stack.output_block_0
            else:
                t_block = torch_model.output_blocks[block_idx - 1]
                k_block = keras_stack.output_blocks[block_idx - 1]

            x_in_t = x_torch_list[block_idx]
            x_in_k = x_keras_list[block_idx]

            print(f"\n  Output block {block_idx}:")
            compare(f"  edge_input", x_in_t, x_in_k)

            # Internal: g = dense_rbf(rbf), weighted = g * x
            g_t = t_block.dense_rbf(rbf_t)
            g_k = to_t(k_block.dense_rbf(rbf_k3))
            compare(f"  dense_rbf_out", g_t, g_k)

            weighted_t = g_t * x_in_t
            weighted_k = g_k * x_in_k
            compare(f"  g*x", weighted_t, weighted_k)

            # Aggregate
            num_nodes_t = n_t.size(0)
            agg_t = t_block.aggr(weighted_t, ei_t, num_nodes_t)
            agg_k = to_t(k_block.lay_pool([n_k3, weighted_k if isinstance(weighted_k, torch.Tensor) else to_t(weighted_k), ei_k]))
            compare(f"  aggregated", agg_t, agg_k)

            # up_projection
            up_t = t_block.up_projection(agg_t)
            up_k = to_t(k_block.up_projection(agg_k))
            compare(f"  up_projection", up_t, up_k)

            # dense_mlp
            mlp_t = t_block.dense_mlp(up_t)
            mlp_k = to_t(k_block.dense_mlp(up_k))
            compare(f"  dense_mlp_out", mlp_t, mlp_k)

            # dense_final
            final_t = t_block.dense_final(mlp_t)
            final_k = to_t(k_block.dense_final(mlp_k))
            compare(f"  dense_final_out", final_t, final_k)

            print(f"  dense_mlp_out norm: torch={mlp_t.norm():.4e}, keras={mlp_k.norm():.4e}")


if __name__ == "__main__":
    main()
