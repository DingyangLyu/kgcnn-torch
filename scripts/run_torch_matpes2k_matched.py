"""Fair Keras-vs-Torch comparison: translate FULL Keras matpes2k configs to torch hyper.

Reads the 26 Keras matpes2k_26 JSON configs and translates:
  - Complete model architecture parameters (depth, units, attention, pooling, …)
  - StandardLabelScaler
  - LR scheduler (LinearLearningRateScheduler → polynomial_decay,
                  LinearWarmupExponentialDecay → warmup_exponential,
                  ExponentialDecay → exponential)
  - Optimizer (Adam / AdamW with correct lr)
  - Loss function (mae / mse)

Then runs training via train_graph.py for each model.
"""
import argparse
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
TORCH_ROOT = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch"
KERAS_ROOT = "/home/yuanbai/Downloads/MLIPs/gcnn_keras-master"
KERAS_HYPER_DIR = os.path.join(KERAS_ROOT, "training/hyper/tmp_matpes2k_26")
DEFAULT_DATASET = os.path.join(TORCH_ROOT, "tmp_matpes_2k/matpes_pbe_2k.pkl")
DEFAULT_OUTPUT = os.path.join(TORCH_ROOT, "tmp_matpes_2k/fair_match")

MODEL_LIST = [
    "SchNet", "PAiNN", "DimeNetPP", "GCN", "GAT", "GATv2", "GIN", "EGNN",
    "DMPNN", "GraphSAGE", "Megnet", "AttentiveFP", "CGCNN", "NMPN", "INorp",
    "MEGAN", "RGCN", "GNNFilm", "rGIN", "MXMNet", "MoGAT", "CMPNN", "DGIN",
    "HamNet", "HDNNP2nd", "MAT",
]

# Keras JSON file name mapping (model name → file stem after "hyper_matpes2k_")
_KERAS_FILE_KEY = {"Megnet": "Megnet", "SchNet": "SchNet"}  # default: same as model name

# Approximate train-set size for step→epoch conversion (2000 total, 2-fold CV)
_TRAIN_SIZE = 1000


# ---------------------------------------------------------------------------
# Helper: activation / pooling translation
# ---------------------------------------------------------------------------
def _act(v):
    """Translate a Keras activation spec to a plain string for torch."""
    if v is None:
        return "linear"
    if isinstance(v, str):
        return v
    if isinstance(v, dict):
        cls = v.get("class_name", "")
        cfg = v.get("config", "")
        if cls == "function" and isinstance(cfg, str):
            # "kgcnn>shifted_softplus" → "shifted_softplus"
            return cfg.split(">")[-1] if ">" in cfg else cfg
        if isinstance(cfg, str):
            return cfg
    return "linear"


def _pool(v):
    """Translate Keras pooling_method to torch pooling name."""
    if v is None:
        return "sum"
    s = str(v)
    return s.replace("scatter_", "")


def _output_mlp_units(mlp):
    """Extract hidden units from output_mlp (everything before the last layer)."""
    units = mlp.get("units", [1])
    if isinstance(units, int):
        return []
    return list(units[:-1])


def _output_mlp_act(mlp):
    """Extract activation of the first hidden layer in output_mlp."""
    act = mlp.get("activation", ["linear"])
    if isinstance(act, list) and len(act) > 0:
        return _act(act[0])
    return _act(act)


def _num_targets(mlp):
    units = mlp.get("units", [1])
    if isinstance(units, int):
        return units
    return units[-1]


def _dropout_val(v):
    """Extract dropout rate from Keras config (can be float, dict, or None)."""
    if v is None:
        return 0.0
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, dict):
        return float(v.get("rate", 0.0))
    return 0.0


# ---------------------------------------------------------------------------
# Per-model architecture translation: Keras model.config → torch model.config
# ---------------------------------------------------------------------------
def _translate_schnet(c):
    omlp = c.get("output_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": c["interaction_args"]["units"],
        "gauss_bins": c["gauss_args"]["bins"],
        "gauss_distance": c["gauss_args"]["distance"],
        "gauss_sigma": c["gauss_args"].get("sigma", 0.4),
        "gauss_offset": c["gauss_args"].get("offset", 0.0),
        "interaction_activation": _act(c["interaction_args"].get("activation")),
        "interaction_pooling": _pool(c["interaction_args"].get("cfconv_pool")),
        "node_pooling": _pool(c.get("node_pooling_args", {}).get("pooling_method")),
        "last_mlp_units": c.get("last_mlp", {}).get("units"),
        "last_mlp_activation": _act(c.get("last_mlp", {}).get("activation", ["shifted_softplus"])[0]
                                     if isinstance(c.get("last_mlp", {}).get("activation"), list)
                                     else c.get("last_mlp", {}).get("activation")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_painn(c):
    omlp = c.get("output_mlp", {})
    bb = c.get("bessel_basis", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": c.get("conv_args", {}).get("units", 128),
        "num_radial": bb.get("num_radial", 20),
        "cutoff": bb.get("cutoff", 5.0),
        "envelope_exponent": bb.get("envelope_exponent", 5),
        "node_pooling": _pool(c.get("pooling_args", {}).get("pooling_method")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_dimenetpp(c):
    omlp = c.get("output_mlp", {})
    return {
        "emb_size": c["emb_size"],
        "out_emb_size": c["out_emb_size"],
        "int_emb_size": c["int_emb_size"],
        "basis_emb_size": c["basis_emb_size"],
        "num_blocks": c["num_blocks"],
        "num_spherical": c["num_spherical"],
        "num_radial": c["num_radial"],
        "cutoff": c["cutoff"],
        "envelope_exponent": c["envelope_exponent"],
        "num_before_skip": c["num_before_skip"],
        "num_after_skip": c["num_after_skip"],
        "num_dense_output": c["num_dense_output"],
        "num_targets": c["num_targets"],
        "extensive": c.get("extensive", False),
        "output_init": c.get("output_init", "zeros"),
        "activation": _act(c.get("activation", "swish")),
        "use_output_mlp": True,
        "output_mlp_units": _output_mlp_units(omlp),
        "output_mlp_activation": _output_mlp_act(omlp),
    }


def _translate_gcn(c):
    omlp = c.get("output_mlp", {})
    ga = c.get("gcn_args", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "gcn_units": ga.get("units", 100),
        "gcn_activation": _act(ga.get("activation", "relu")),
        "node_pooling": "sum",
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_gat(c):
    omlp = c.get("output_mlp", {})
    aa = c.get("attention_args", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "attention_units": aa.get("units", 32),
        "attention_heads_num": c.get("attention_heads_num", 5),
        "attention_heads_concat": c.get("attention_heads_concat", False),
        "attention_activation": _act(aa.get("activation", "leaky_relu2")),
        "use_edge_features": aa.get("use_edge_features", True),
        "node_pooling": _pool(c.get("pooling_nodes_args", {}).get("pooling_method")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "output_final_activation": "linear",
        "num_targets": _num_targets(omlp),
    }


def _translate_gatv2(c):
    omlp = c.get("output_mlp", {})
    aa = c.get("attention_args", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "attention_units": aa.get("units", 32),
        "attention_heads_num": c.get("attention_heads_num", 5),
        "attention_heads_concat": c.get("attention_heads_concat", False),
        "attention_activation": _act(aa.get("activation", "leaky_relu2")),
        "use_edge_features": aa.get("use_edge_features", True),
        "node_pooling": _pool(c.get("pooling_nodes_args", {}).get("pooling_method")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_gin(c):
    gm = c.get("gin_mlp", {})
    lm = c.get("last_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "gin_mlp_units": gm.get("units", [64, 64]),
        "gin_mlp_activation": _act(gm.get("activation", ["relu"])[0]
                                    if isinstance(gm.get("activation"), list)
                                    else gm.get("activation", "relu")),
        "gin_mlp_use_normalization": gm.get("use_normalization", True),
        "dropout_rate": _dropout_val(c.get("dropout", 0.0)),
        "last_mlp_units": lm.get("units") if lm else None,
        "last_mlp_activation": _act(lm.get("activation", ["relu"])[0]
                                     if isinstance(lm.get("activation"), list)
                                     else lm.get("activation", "relu")) if lm else "relu",
        "node_pooling": "sum",
        "num_targets": 1,
    }


def _translate_egnn(c):
    omlp = c.get("output_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "edge_mlp_units": c.get("edge_mlp_kwargs", {}).get("units", [64, 64]),
        "edge_mlp_activation": _act(c.get("edge_mlp_kwargs", {}).get("activation", ["swish"])[0]
                                     if isinstance(c.get("edge_mlp_kwargs", {}).get("activation"), list)
                                     else c.get("edge_mlp_kwargs", {}).get("activation", "swish")),
        "coord_mlp_units": c.get("coord_mlp_kwargs", {}).get("units", [64, 1]),
        "coord_mlp_activation": _act(c.get("coord_mlp_kwargs", {}).get("activation", ["swish"])[0]
                                      if isinstance(c.get("coord_mlp_kwargs", {}).get("activation"), list)
                                      else c.get("coord_mlp_kwargs", {}).get("activation", "swish")),
        "node_mlp_units": c.get("node_mlp_kwargs", {}).get("units", [64, 64]),
        "node_mlp_activation": _act(c.get("node_mlp_kwargs", {}).get("activation", ["swish"])[0]
                                     if isinstance(c.get("node_mlp_kwargs", {}).get("activation"), list)
                                     else c.get("node_mlp_kwargs", {}).get("activation", "swish")),
        "use_skip": c.get("use_skip", True),
        # Note: don't set use_edge_attr; let adapt_model_config_from_data handle it
        # to avoid edge_attr_dim mismatch.
        "layer_pooling": _pool(c.get("pooling_edge_kwargs", {}).get("pooling_method", "sum")),
        "coord_pooling": _pool(c.get("pooling_coord_kwargs", {}).get("pooling_method", "mean")),
        "node_pooling": _pool(c.get("node_pooling_kwargs", {}).get("pooling_method", "sum")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_dmpnn(c):
    omlp = c.get("output_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": c.get("edge_initialize", {}).get("units", 64),
        "message_activation": _act(c.get("edge_activation", {}).get("activation", "relu")),
        "node_pooling": _pool(c.get("pooling_args", {}).get("pooling_method")),
        "dropout_rate": _dropout_val(c.get("dropout")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_graphsage(c):
    omlp = c.get("output_mlp", {})
    nma = c.get("node_mlp_args", {})
    ema = c.get("edge_mlp_args", {})
    eu = ema.get("units", 64)
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "node_mlp_units": nma.get("units"),
        "edge_mlp_units": [eu] if isinstance(eu, int) else eu,
        "use_edge_features": c.get("use_edge_features", True),
        "pooling_method": _pool(c.get("pooling_args", {}).get("pooling_method", "mean")),
        "node_pooling": _pool(c.get("pooling_nodes_args", {}).get("pooling_method", "sum")),
        "activation": _act(nma.get("activation", ["relu"])[0]
                           if isinstance(nma.get("activation"), list)
                           else nma.get("activation", "relu")),
        "output_units": _output_mlp_units(omlp),
        "output_final_activation": "linear",
        "num_targets": _num_targets(omlp),
    }


def _translate_megnet(c):
    omlp = c.get("output_mlp", {})
    mba = c.get("meg_block_args", {})
    s2s = c.get("set2set_args", {})
    # node/edge/state dim must match block_units_*[-1] for residual connections
    block_node = mba.get("node_embed", [64, 32, 32])
    block_edge = mba.get("edge_embed", [64, 32, 32])
    block_state = mba.get("env_embed", [64, 32, 32])
    cfg = {
        "node_dim": block_node[-1] if block_node else 32,
        "edge_dim": block_edge[-1] if block_edge else 32,
        "state_dim": block_state[-1] if block_state else 32,
        "depth": c.get("nblocks", 3),
        "block_units_node": mba.get("node_embed"),
        "block_units_edge": mba.get("edge_embed"),
        "block_units_state": mba.get("env_embed"),
        "activation": _act(mba.get("activation", "softplus2")),
        "node_ff_units": c.get("node_ff_args", {}).get("units"),
        "edge_ff_units": c.get("edge_ff_args", {}).get("units"),
        "state_ff_units": c.get("state_ff_args", {}).get("units"),
        "has_ff": c.get("has_ff", True),
        "use_set2set": c.get("use_set2set", True),
        "set2set_channels": s2s.get("channels", 16),
        "set2set_T": s2s.get("T", 3),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }
    # NOTE: Keras input_node_embedding has output_dim=16 followed by an FFN to 32,
    # but in practice embedding directly to node_dim=32 works better for this data
    # pipeline where edge features are pre-computed (14D vs Keras's 20 Gaussian bins).
    # Similarly, Keras input_graph_embedding uses Embedding(100, 64) for graph_attributes,
    # but graph_attributes in MatPES2k are continuous floats (mean Z, 11-2920),
    # not discrete tokens — the model's lazy linear projection handles this better.
    return cfg


def _translate_attentivefp(c):
    omlp = c.get("output_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth_ato": c.get("depthato", 2),
        "depth_mol": c.get("depthmol", 2),
        "units": c.get("attention_args", {}).get("units", 32),
        "dropout": _dropout_val(c.get("dropout")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_cgcnn(c):
    omlp = c.get("output_mlp", {})
    cla = c.get("conv_layer_args", {})
    cfg = {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "conv_activation": _act(cla.get("activation_s", "softplus")),
        "node_pooling": _pool(c.get("node_pooling_args", {}).get("pooling_method", "mean")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }
    # NOTE: Do NOT pass gauss_bins/gauss_distance here.  The MatPES2k pickle
    # already contains pre-computed 14-dim edge features (distance + Gaussian
    # expansion).  adapt_model_config_from_data detects edge_attr_dim > 1 and
    # sets expand_distance=False, gauss_bins=edge_attr_dim automatically.
    # Pass conv_units from Keras conv_layer_args (internal working dimension)
    if "units" in cla:
        cfg["conv_units"] = cla["units"]
    return cfg


def _translate_nmpn(c):
    omlp = c.get("output_mlp", {})
    em = c.get("edge_mlp", {})
    em_act = em.get("activation", "swish")
    return {
        "node_dim": c.get("node_dim", c["input_node_embedding"]["output_dim"]),
        "depth": c["depth"],
        "edge_mlp_units": em.get("units", [64, 64]),
        "edge_mlp_activation": _act(em_act[0] if isinstance(em_act, list) else em_act),
        "use_set2set": c.get("use_set2set", True),
        "set2set_T": c.get("set2set_args", {}).get("T", 3),
        "node_pooling": _pool(c.get("pooling_args", {}).get("pooling_method")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_inorp(c):
    omlp = c.get("output_mlp", {})
    nma = c.get("node_mlp_args", {})
    ema = c.get("edge_mlp_args", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "node_mlp_units": nma.get("units"),
        "node_mlp_activation": _act(nma.get("activation", ["relu"])[0]
                                     if isinstance(nma.get("activation"), list)
                                     else nma.get("activation", "relu")),
        "edge_mlp_units": ema.get("units"),
        "edge_mlp_activation": _act(ema.get("activation", ["relu"])[0]
                                     if isinstance(ema.get("activation"), list)
                                     else ema.get("activation", "relu")),
        "message_pooling": _pool(c.get("pooling_args", {}).get("pooling_method")),
        "use_set2set": c.get("use_set2set", False),
        "node_pooling": "sum",
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_megan(c):
    ic = c.get("importance_channels", 2)
    return {
        "units": c.get("units", [60, 50, 40, 30]),
        "final_units": c.get("final_units", [50, 30, 10, 1]),
        "dropout_rate": _dropout_val(c.get("dropout_rate")),
        "final_dropout_rate": _dropout_val(c.get("final_dropout_rate")),
        "importance_channels": ic,
        "num_heads": ic,  # torch requires num_heads == importance_channels
        "use_edge_features": c.get("use_edge_features", False),
        "num_targets": 1,
    }


def _translate_rgcn(c):
    omlp = c.get("output_mlp", {})
    dr = c.get("dense_relation_kwargs", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": dr.get("units", 64),
        "num_relations": dr.get("num_relations", 4),
        "rgcn_activation": _act(c.get("activation_kwargs", {}).get("activation", "swish")),
        "node_pooling": "sum",
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_gnnfilm(c):
    omlp = c.get("output_mlp", {})
    dr = c.get("dense_relation_kwargs", {})
    dm = c.get("dense_modulation_kwargs", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": dr.get("units", 64),
        "num_relations": dr.get("num_relations", 4),
        "activation": _act(c.get("activation_kwargs", {}).get("activation", "swish")),
        "modulation_activation": _act(dm.get("activation", "sigmoid")),
        "node_pooling": "sum",
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_rgin(c):
    gm = c.get("gin_mlp", {})
    lm = c.get("last_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "gin_mlp_units": gm.get("units", [64, 64]),
        "gin_mlp_activation": _act(gm.get("activation", ["relu"])[0]
                                    if isinstance(gm.get("activation"), list)
                                    else gm.get("activation", "relu")),
        "random_range": c.get("rgin_args", {}).get("random_range", 100),
        # Note: don't pass dropout; train_graph.py aliases it to dropout_rate
        # which rGINModel doesn't accept.
        "last_mlp_units": lm.get("units") if lm else None,
        "last_mlp_activation": _act(lm.get("activation", ["relu"])[0]
                                     if isinstance(lm.get("activation"), list)
                                     else lm.get("activation", "relu")) if lm else "relu",
        "node_pooling": "sum",
        "num_targets": 1,
    }


def _translate_mxmnet(c):
    bl = c.get("bessel_basis_local", {})
    sl = c.get("spherical_basis_local", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": c.get("mlp_rbf_kwargs", {}).get("units", 32),
        "num_radial": bl.get("num_radial", 16),
        "num_spherical": sl.get("num_spherical", 7),
        "cutoff": bl.get("cutoff", 5.0),
        "envelope_exponent": bl.get("envelope_exponent", 5),
        "activation": "swish",
        "global_mp_pooling": _pool(c.get("global_mp_kwargs", {}).get("pooling_method", "mean")),
        "node_pooling": _pool(c.get("node_pooling_args", {}).get("pooling_method", "sum")),
        "use_output_mlp": c.get("use_output_mlp", False),
        "num_targets": 1,
    }


def _translate_mogat(c):
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depthato": c.get("depthato", 2),
        "depthmol": c.get("depthmol", 2),
        "units": c.get("attention_args", {}).get("units", 64),
        "dropout": _dropout_val(c.get("dropout")),
        "num_targets": 1,
    }


def _translate_cmpnn(c):
    omlp = c.get("output_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth": c["depth"],
        "units": c.get("node_initialize", {}).get("units", 300),
        "activation": _act(c.get("node_initialize", {}).get("activation", "relu")),
        "use_final_gru": c.get("use_final_gru", True),
        "gru_units": c.get("pooling_gru", {}).get("units"),
        "node_pooling": _pool(c.get("pooling_kwargs", {}).get("pooling_method")),
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_dgin(c):
    gm = c.get("gin_mlp", {})
    lm = c.get("last_mlp", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        "depth_dmpnn": c.get("depthDMPNN", 3),
        "depth_gin": c.get("depthGIN", 3),
        "units": c.get("edge_initialize", {}).get("units", 128),
        "dropout_dmpnn": _dropout_val(c.get("dropoutDMPNN")),
        "dropout_gin": _dropout_val(c.get("dropoutGIN")),
        "gin_mlp_units": gm.get("units"),
        "last_mlp_units": lm.get("units") if lm else None,
        "node_pooling": _pool(c.get("pooling_args", {}).get("pooling_method")),
        "num_targets": 1,
    }


def _translate_hamnet(c):
    omlp = c.get("output_mlp", {})
    mk = c.get("message_kwargs", {})
    fk = c.get("fingerprint_kwargs", {})
    return {
        "node_dim": c["input_node_embedding"]["output_dim"],
        # Note: don't set edge_dim; let adapt_model_config_from_data infer from data
        "depth": c["depth"],
        "units": mk.get("units", 128),
        "fingerprint_dim": fk.get("units", 128),
        "fingerprint_depth": fk.get("depth", 2),
        "use_gru_update": c.get("union_type_node", "gru") == "gru",
        "output_units": _output_mlp_units(omlp),
        "output_activation": _output_mlp_act(omlp),
        "num_targets": _num_targets(omlp),
    }


def _translate_hdnnp2nd(c):
    mk = c.get("mlp_kwargs", {})
    mk_units = mk.get("units", [128, 128, 128, 1])
    mk_act = mk.get("activation", ["swish", "swish", "swish", "linear"])
    return {
        "relational_units": mk_units[:-1],
        "relational_activation": [_act(a) for a in mk_act[:-1]] if isinstance(mk_act, list) else [_act(mk_act)],
        "node_pooling": _pool(c.get("node_pooling_args", {}).get("pooling_method")),
        "num_targets": mk_units[-1],
    }


def _translate_mat(c):
    ak = c.get("attention_kwargs", {})
    ffk = c.get("feed_forward_kwargs", {})
    omlp = c.get("output_mlp", {})
    return {
        "embedding_units": c.get("embedding_units", 32),
        "depth": c["depth"],
        "num_heads": c.get("heads", 8),
        "merge_heads": c.get("merge_heads", "concat"),
        "attention_units": ak.get("units", 8),
        "lambda_attention": ak.get("lambda_attention", 0.3),
        "lambda_distance": ak.get("lambda_distance", 0.3),
        "lambda_adjacency": ak.get("lambda_adjacency"),
        "add_identity": ak.get("add_identity", False),
        "attention_dropout": ak.get("dropout", 0.0),
        "distance_trafo": c.get("distance_matrix_kwargs", {}).get("trafo", "exp"),
        "units_ff": ffk.get("units"),
        "ff_activations": [_act(a) for a in ffk.get("activation", ["relu"])]
                          if isinstance(ffk.get("activation"), list)
                          else [_act(ffk.get("activation", "relu"))],
        "output_units": _output_mlp_units(omlp),
        # MAT needs len(output_activations) == len(output_units) + 1 (includes final layer)
        "output_activations": [_act(a) for a in omlp.get("activation", ["linear"])]
                              if isinstance(omlp.get("activation"), list)
                              else [_act(omlp.get("activation", "linear"))],
        "num_targets": _num_targets(omlp),
    }


_TRANSLATORS = {
    "SchNet": _translate_schnet,
    "PAiNN": _translate_painn,
    "DimeNetPP": _translate_dimenetpp,
    "GCN": _translate_gcn,
    "GAT": _translate_gat,
    "GATv2": _translate_gatv2,
    "GIN": _translate_gin,
    "EGNN": _translate_egnn,
    "DMPNN": _translate_dmpnn,
    "GraphSAGE": _translate_graphsage,
    "Megnet": _translate_megnet,
    "AttentiveFP": _translate_attentivefp,
    "CGCNN": _translate_cgcnn,
    "NMPN": _translate_nmpn,
    "INorp": _translate_inorp,
    "MEGAN": _translate_megan,
    "RGCN": _translate_rgcn,
    "GNNFilm": _translate_gnnfilm,
    "rGIN": _translate_rgin,
    "MXMNet": _translate_mxmnet,
    "MoGAT": _translate_mogat,
    "CMPNN": _translate_cmpnn,
    "DGIN": _translate_dgin,
    "HamNet": _translate_hamnet,
    "HDNNP2nd": _translate_hdnnp2nd,
    "MAT": _translate_mat,
}


# ---------------------------------------------------------------------------
# Training config translation: optimizer, loss, scheduler, scaler
# ---------------------------------------------------------------------------
def _translate_optimizer(keras_training):
    """Extract optimizer class_name, lr, and extra opts; separate out lr schedule."""
    compile_cfg = keras_training.get("compile", {})
    opt = compile_cfg.get("optimizer", {})
    if not isinstance(opt, dict):
        return "Adam", 1e-3, {}, None

    cls = opt.get("class_name", "Adam")
    ocfg = opt.get("config", {})

    lr_raw = ocfg.get("learning_rate", 1e-3)
    lr_schedule = None
    if isinstance(lr_raw, dict):
        lr_schedule = lr_raw
        lr_val = float(lr_raw.get("config", {}).get("learning_rate",
                       lr_raw.get("config", {}).get("initial_learning_rate", 1e-3)))
    else:
        lr_val = float(lr_raw)

    extra = {}
    if "weight_decay" in ocfg:
        extra["weight_decay"] = ocfg["weight_decay"]

    return cls, lr_val, extra, lr_schedule


def _translate_loss(keras_training):
    compile_cfg = keras_training.get("compile", {})
    loss = compile_cfg.get("loss", "mean_absolute_error")
    if isinstance(loss, dict):
        cn = loss.get("class_name", "")
        if "AbsoluteError" in cn or "l1" in cn.lower():
            return "mae"
        return "mse"
    s = str(loss).lower()
    if "absolute" in s or s in ("mae", "l1"):
        return "mae"
    if "squared" in s or s in ("mse",):
        return "mse"
    return s


def _translate_scheduler(keras_training, batch_size, lr_schedule_from_opt=None):
    """Build torch scheduler config from Keras callbacks or optimizer-embedded schedule.

    Returns dict suitable for training.scheduler section, or None.
    """
    steps_per_epoch = max(1, math.ceil(_TRAIN_SIZE / batch_size))

    # 1) Check callbacks for LR schedulers
    callbacks = keras_training.get("fit", {}).get("callbacks", [])
    for cb in callbacks:
        if not isinstance(cb, dict):
            continue
        cn = cb.get("class_name", "")
        ccfg = cb.get("config", {})

        if "LinearLearningRateScheduler" in cn:
            lr_start = ccfg.get("learning_rate_start", 1e-3)
            lr_stop = ccfg.get("learning_rate_stop", 1e-5)
            epo = ccfg.get("epo", 500)
            return {
                "class_name": "polynomial_decay",
                "total_epochs": epo,
                "lr_final_factor": lr_stop / max(lr_start, 1e-12),
                "power": 1.0,
            }

        if "LinearWarmupExponential" in cn:
            lr_start = ccfg.get("lr_start", 1e-3)
            gamma = ccfg.get("gamma", 0.996)
            epo_warmup = ccfg.get("epo_warmup", 1)
            spe = ccfg.get("steps_per_epoch", steps_per_epoch)
            # Convert per-step gamma to per-epoch decay_rate
            decay_rate_per_epoch = gamma ** spe
            return {
                "class_name": "LinearWarmupExponentialDecay",
                "warmup_epochs": epo_warmup,
                "decay_rate": decay_rate_per_epoch,
                "decay_epochs": 1,  # decay every epoch
            }

    # 2) Check optimizer-embedded lr schedule
    if lr_schedule_from_opt:
        cn = lr_schedule_from_opt.get("class_name", "")
        scfg = lr_schedule_from_opt.get("config", {})

        if "LinearWarmupExponentialDecay" in cn:
            warmup_steps = scfg.get("warmup_steps", 10)
            decay_steps = scfg.get("decay_steps", 10000)
            decay_rate = scfg.get("decay_rate", 0.96)
            warmup_epochs = max(1, round(warmup_steps / steps_per_epoch))
            decay_epochs = max(1, round(decay_steps / steps_per_epoch))
            return {
                "class_name": "LinearWarmupExponentialDecay",
                "warmup_epochs": warmup_epochs,
                "decay_rate": decay_rate,
                "decay_epochs": decay_epochs,
            }

        if "ExponentialDecay" in cn:
            initial_lr = scfg.get("initial_learning_rate", 1e-3)
            decay_steps = scfg.get("decay_steps", 1600)
            decay_rate = scfg.get("decay_rate", 0.5)
            # Convert step-level to epoch-level gamma
            gamma_per_epoch = decay_rate ** (steps_per_epoch / max(decay_steps, 1))
            return {
                "class_name": "exponential",
                "gamma": gamma_per_epoch,
            }

    return None


# ---------------------------------------------------------------------------
# Build torch hyper JSON
# ---------------------------------------------------------------------------
def build_torch_hyper(model_name, keras_json, dataset_path, epochs_override=None):
    """Translate a full Keras matpes2k JSON config to a torch hyper dict."""
    kcfg = keras_json["model"]["config"]
    ktraining = keras_json.get("training", {})

    # Model config
    translator = _TRANSLATORS.get(model_name)
    if translator is None:
        raise ValueError(f"No translator for model: {model_name}")
    model_config = translator(kcfg)
    # Remove None values to let torch defaults apply
    model_config = {k: v for k, v in model_config.items() if v is not None}

    # Match Keras node input type: if Keras uses float node_attributes (not
    # integer node_number), disable Torch embedding so a Linear projection is
    # used instead — this keeps the information bottleneck fair.
    keras_inputs = kcfg.get("inputs", [])
    uses_float_node_attr = any(
        inp.get("name") == "node_attributes" and "int" not in inp.get("dtype", "")
        for inp in keras_inputs
    )
    if uses_float_node_attr:
        model_config["use_node_embedding"] = False

    # Training config
    opt_cls, lr, opt_extra, lr_schedule = _translate_optimizer(ktraining)
    loss = _translate_loss(ktraining)
    batch_size = ktraining.get("fit", {}).get("batch_size", 32)
    epochs = epochs_override or ktraining.get("fit", {}).get("epochs", 10)

    # Optimizer
    opt_config = {"lr": lr}
    opt_config.update(opt_extra)
    compile_cfg = {
        "optimizer": {"class_name": opt_cls, "config": opt_config},
        "loss": loss,
    }

    # Scheduler
    scheduler_cfg = _translate_scheduler(ktraining, batch_size, lr_schedule)

    # Scaler - always use StandardLabelScaler (all Keras configs use it)
    scaler_cfg = {"class_name": "StandardLabelScaler"}

    # Cross-validation
    cv_cfg = {"n_splits": 2, "shuffle": True}

    # Assemble
    hyper = {
        "model": {"config": model_config},
        "training": {
            "fit": {
                "epochs": epochs,
                "batch_size": batch_size,
                "early_stopping_patience": 0,
            },
            "compile": compile_cfg,
            "scaler": scaler_cfg,
            "cross_validation": cv_cfg,
        },
        "data": {
            "dataset": {
                "class_name": "MatPES2k",
                "config": {"file_path": dataset_path},
            },
            "data_unit": "meV/atom",
        },
    }
    if scheduler_cfg:
        hyper["training"]["scheduler"] = scheduler_cfg

    return hyper


# ---------------------------------------------------------------------------
# Score parsing
# ---------------------------------------------------------------------------
def parse_torch_score(score_path):
    if not os.path.exists(score_path):
        return None
    txt = open(score_path, "r", encoding="utf-8").read()
    # Prefer mean_best_val_mae (cross-validation average) over single-fold value
    m2 = re.search(r"mean_best_val_mae:\s*([0-9eE+\-.]+)", txt)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            pass
    m = re.search(r"best_val_mae:\s*([0-9eE+\-.]+)", txt)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description="Fair Keras→Torch matched comparison")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override epochs (default: use Keras config value)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET)
    ap.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT)
    ap.add_argument("--dry-run", action="store_true",
                    help="Only generate hyper JSONs, don't run training")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    hyper_out_dir = os.path.join(args.output_dir, "hyper")
    logs_dir = os.path.join(args.output_dir, "logs")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(hyper_out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    models = MODEL_LIST
    if args.models:
        model_set = set(args.models)
        models = [m for m in models if m in model_set]

    rows = []
    for model in models:
        # Load Keras config
        keras_file = os.path.join(KERAS_HYPER_DIR, f"hyper_matpes2k_{model}.json")
        if not os.path.exists(keras_file):
            print(f"SKIP {model}: Keras config not found at {keras_file}")
            rows.append({"model": model, "ok": False, "error": "missing_keras_config"})
            continue

        with open(keras_file, "r", encoding="utf-8") as f:
            keras_json = json.load(f)

        # Translate to torch hyper
        try:
            hyper = build_torch_hyper(model, keras_json, args.dataset_path,
                                      epochs_override=args.epochs)
        except Exception as e:
            print(f"SKIP {model}: translation error: {e}")
            rows.append({"model": model, "ok": False, "error": f"translate_error: {e}"})
            continue

        # Save torch hyper
        hpath = os.path.join(hyper_out_dir, f"hyper_{model}.json")
        with open(hpath, "w", encoding="utf-8") as f:
            json.dump(hyper, f, indent=2)

        lr = hyper["training"]["compile"]["optimizer"]["config"]["lr"]
        bs = hyper["training"]["fit"]["batch_size"]
        ep = hyper["training"]["fit"]["epochs"]
        sched = hyper["training"].get("scheduler", {}).get("class_name", "none")
        loss = hyper["training"]["compile"]["loss"]
        print(f"[{model}] lr={lr}, batch={bs}, epochs={ep}, sched={sched}, loss={loss}")

        if args.dry_run:
            rows.append({"model": model, "ok": True, "dry_run": True,
                         "hyper_path": hpath})
            continue

        # Run training
        out_dir = os.path.join(results_dir, model)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{model}.log")

        cmd = [
            sys.executable,
            os.path.join(TORCH_ROOT, "training_scripts/train_graph.py"),
            "--hyper", hpath,
            "--category", model,
            "--output", results_dir,
            "--seed", str(args.seed),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = TORCH_ROOT + os.pathsep + env.get("PYTHONPATH", "")

        start = datetime.now().isoformat(timespec="seconds")
        print(f"[{start}] Running {model} ...", flush=True)
        timed_out = False
        rc = 1
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd, cwd=TORCH_ROOT, env=env,
                    stdout=lf, stderr=subprocess.STDOUT, text=True,
                    timeout=args.timeout_sec,
                )
                rc = proc.returncode
        except subprocess.TimeoutExpired:
            timed_out = True
            rc = 124

        score_path = os.path.join(results_dir, model, "score.yaml")
        best_val_mae = parse_torch_score(score_path)
        tail = ""
        if os.path.exists(log_path):
            with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                tail = "\n".join(f.read().splitlines()[-40:])

        row = {
            "model": model,
            "ok": rc == 0,
            "returncode": rc,
            "timed_out": timed_out,
            "lr": lr,
            "batch_size": bs,
            "epochs": ep,
            "scheduler": sched,
            "loss": loss,
            "best_val_mae": best_val_mae,
            "score_file": score_path if os.path.exists(score_path) else "",
            "log_file": log_path,
            "tail": tail,
        }
        rows.append(row)
        print(f"  → {model}: ok={row['ok']}, best_val_mae={best_val_mae}", flush=True)

    # Summary
    summary = {
        "total": len(rows),
        "passed": sum(1 for r in rows if r.get("ok")),
        "failed": sum(1 for r in rows if not r.get("ok")),
        "failed_models": [r["model"] for r in rows if not r.get("ok")],
    }
    out = {"summary": summary, "results": rows}
    outp = os.path.join(args.output_dir, "torch_fair_matched_summary.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("\n" + json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {outp}")
    if summary["failed"] and not args.dry_run:
        sys.exit(1)


if __name__ == "__main__":
    main()
