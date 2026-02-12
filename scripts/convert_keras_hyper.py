#!/usr/bin/env python3
"""Convert Keras hyper configs to Torch JSON format.

Reads Keras .py hyper files, converts model and training configs to Torch JSON format,
and merges into existing Torch JSON files (preserving existing entries).
"""

import copy
import json
import os
import sys

KERAS_DIR = "/home/yuanbai/Downloads/MLIPs/gcnn_keras-master/training/hyper"
TORCH_DIR = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch/training_scripts/hyper"

# Mapping: Keras base name -> Torch base name (both without extension)
FILE_MAP = {
    "hyper_esol": "hyper_esol",
    "hyper_freesolv": "hyper_freesolv",
    "hyper_clintox": "hyper_clintox",
    "hyper_lipop": "hyper_lipop",
    "hyper_mutag": "hyper_mutag",
    "hyper_mutagenicity": "hyper_mutagenicity",
    "hyper_proteins": "hyper_proteins",
    "hyper_sider": "hyper_sider",
    "hyper_tox21mol": "hyper_tox21mol",
    "hyper_cora": "hyper_cora",
    "hyper_cora_lu": "hyper_cora_lu",
    "hyper_qm7": "hyper_qm7",
    "hyper_qm9_energies": "hyper_qm9_energies",
    "hyper_qm9_orbitals": "hyper_qm9_orbitals",
    "hyper_md17": "hyper_md17",
    "hyper_md17_revised": "hyper_md17_revised",
    "hyper_iso17": "hyper_iso17",
    "hyper_mp_e_form": "hyper_mp_e_form",
    "hyper_mp_gap": "hyper_mp_gap",
    "hyper_mp_dielectric": "hyper_mp_dielectric",
    "hyper_mp_is_metal": "hyper_mp_is_metal",
    "hyper_mp_jdft2d": "hyper_mp_jdft2d",
    "hyper_mp_perovskites": "hyper_mp_perovskites",
    "hyper_mp_phonons": "hyper_mp_phonons",
    "hyper_mp_log_gvrh": "hyper_mp_log_gvrh",
    "hyper_mp_log_kvrh": "hyper_mp_log_kvrh",
}

# Keras model name -> Torch model name (simple names)
_BASE_NAME_MAP = {
    "Schnet": "SchNet",
    "SchNet": "SchNet",
    "GCN": "GCN",
    "GAT": "GAT",
    "GATv2": "GATv2",
    "GIN": "GIN",
    "DMPNN": "DMPNN",
    "GraphSAGE": "GraphSAGE",
    "PAiNN": "PAiNN",
    "AttentiveFP": "AttentiveFP",
    "NMPN": "NMPN",
    "DGIN": "DGIN",
    "DimeNetPP": "DimeNetPP",
    "EGNN": "EGNN",
    "GNNFilm": "GNNFilm",
    "HamNet": "HamNet",
    "Megnet": "Megnet",
    "RGCN": "RGCN",
    "HDNNP2nd": "HDNNP2nd",
    "MoGAT": "MoGAT",
    "INorp": "INorp",
    "MEGAN": "MEGAN",
    "rGIN": "rGIN",
    "MXMNet": "MXMNet",
    "MAT": "MAT",
    "CMPNN": "CMPNN",
    "CGCNN": "CGCNN",
}

# Keras compound suffixes for crystal/force model variants
_COMPOUND_SUFFIXES = (
    ".make_crystal_model", ".EnergyForceModel",
    ".make_model", ".make_force_model",
)


def resolve_model_name(keras_name):
    """Map Keras model name (possibly compound) to Torch model name."""
    # Direct lookup
    if keras_name in _BASE_NAME_MAP:
        return _BASE_NAME_MAP[keras_name]
    # Strip compound suffixes (e.g. "Schnet.make_crystal_model" -> "Schnet")
    for suffix in _COMPOUND_SUFFIXES:
        if keras_name.endswith(suffix):
            base = keras_name[: -len(suffix)]
            if base in _BASE_NAME_MAP:
                return _BASE_NAME_MAP[base]
    return keras_name  # Unknown; caller will skip

# Dataset categories
CLASSIFICATION_DATASETS = {
    "hyper_clintox", "hyper_mutag", "hyper_mutagenicity",
    "hyper_proteins", "hyper_sider", "hyper_tox21mol", "hyper_mp_is_metal",
}
NODE_CLASSIFICATION_DATASETS = {"hyper_cora", "hyper_cora_lu"}
FORCE_DATASETS = {"hyper_md17", "hyper_md17_revised", "hyper_iso17"}
QM_DATASETS = {"hyper_qm7", "hyper_qm9_energies", "hyper_qm9_orbitals"}

# Models supported by the Torch _MODEL_REGISTRY
SUPPORTED_MODELS = {
    "SchNet", "PAiNN", "DimeNetPP", "GCN", "GAT", "GATv2", "GIN", "EGNN",
    "DMPNN", "GraphSAGE", "Megnet", "AttentiveFP", "CGCNN", "NMPN", "INorp",
    "MEGAN", "RGCN", "GNNFilm", "rGIN", "MXMNet", "MoGAT", "CMPNN", "DGIN",
    "HamNet", "HDNNP2nd", "MAT",
}


# --------------------------------------------------------------------------- #
#                          File I/O helpers                                     #
# --------------------------------------------------------------------------- #

def load_keras_hyper(filepath):
    """Load Keras hyper dict from .py file using exec()."""
    namespace = {}
    with open(filepath, "r") as f:
        exec(f.read(), namespace)
    return namespace.get("hyper", {})


def load_torch_json(filepath):
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            return json.load(f)
    return {}


def save_torch_json(filepath, data):
    with open(filepath, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


# --------------------------------------------------------------------------- #
#                       Output MLP helper                                      #
# --------------------------------------------------------------------------- #

def get_output_units(keras_config):
    """Return (hidden_units_list, final_unit) from output_mlp."""
    output_mlp = keras_config.get("output_mlp", {})
    units = output_mlp.get("units", [1])
    if isinstance(units, int):
        return [], units
    if len(units) == 0:
        return [], 1
    return list(units[:-1]), units[-1]


# --------------------------------------------------------------------------- #
#                     Model config converters                                  #
# --------------------------------------------------------------------------- #

def _strip_scatter(pooling_str):
    """'scatter_sum' -> 'sum'."""
    if isinstance(pooling_str, str):
        return pooling_str.replace("scatter_", "")
    return "sum"


def convert_schnet(cfg):
    _, output_dim = get_output_units(cfg)
    pooling = cfg.get("node_pooling_args", {}).get("pooling_method", "scatter_sum")
    return {
        "model_name": "SchNet",
        "num_features": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "num_filters": cfg.get("interaction_args", {}).get("units", 128),
        "num_interactions": cfg.get("depth", 4),
        "cutoff": cfg.get("gauss_args", {}).get("distance", 4.0),
        "num_gaussians": cfg.get("gauss_args", {}).get("bins", 20),
        "readout": _strip_scatter(pooling),
        "output_dim": output_dim,
    }


def convert_painn(cfg):
    _, output_dim = get_output_units(cfg)
    return {
        "model_name": "PAiNN",
        "num_features": cfg.get("input_node_embedding", {}).get("output_dim", 128),
        "num_interactions": cfg.get("depth", 3),
        "cutoff": cfg.get("bessel_basis", {}).get("cutoff", 5.0),
        "num_gaussians": cfg.get("bessel_basis", {}).get("num_radial", 20),
        "output_dim": output_dim,
    }


def convert_dimenetpp(cfg):
    _, output_dim = get_output_units(cfg)
    return {
        "model_name": "DimeNetPP",
        "emb_size": cfg.get("emb_size", 128),
        "out_emb_size": cfg.get("out_emb_size", 256),
        "int_emb_size": cfg.get("int_emb_size", 64),
        "basis_emb_size": cfg.get("basis_emb_size", 8),
        "num_blocks": cfg.get("num_blocks", 4),
        "num_spherical": cfg.get("num_spherical", 7),
        "num_radial": cfg.get("num_radial", 6),
        "cutoff": cfg.get("cutoff", 5.0),
        "envelope_exponent": cfg.get("envelope_exponent", 5),
        "num_before_skip": cfg.get("num_before_skip", 1),
        "num_after_skip": cfg.get("num_after_skip", 2),
        "num_dense_output": cfg.get("num_dense_output", 3),
        "output_dim": output_dim,
    }


def convert_gcn(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "GCN",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "units": cfg.get("gcn_args", {}).get("units", 140),
        "depth": cfg.get("depth", 5),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def _convert_gat_family(cfg, model_name):
    output_units, num_targets = get_output_units(cfg)
    pooling = cfg.get("pooling_nodes_args", {}).get("pooling_method", "scatter_sum")
    return {
        "model_name": model_name,
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth": cfg.get("depth", 3),
        "attention_units": cfg.get("attention_args", {}).get("units", 64),
        "attention_heads_num": cfg.get("attention_heads_num", 10),
        "attention_heads_concat": cfg.get("attention_heads_concat", False),
        "node_pooling": _strip_scatter(pooling),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_gat(cfg):
    return _convert_gat_family(cfg, "GAT")


def convert_gatv2(cfg):
    return _convert_gat_family(cfg, "GATv2")


def _safe_dropout(val, default=0.0):
    if val is None:
        return default
    if isinstance(val, dict):
        return val.get("rate", default)
    return float(val)


def convert_gin(cfg):
    gin_mlp = cfg.get("gin_mlp", {})
    gin_units = gin_mlp.get("units", [64, 64])
    last_mlp = cfg.get("last_mlp", {})
    last_units = last_mlp.get("units", None)
    if last_units and isinstance(last_units, list) and len(last_units) >= 2:
        output_units = last_units[:-1]
        num_targets = last_units[-1]
    else:
        output_units, num_targets = get_output_units(cfg)
        if not output_units:
            u = gin_units[0] if gin_units else 64
            output_units = [u, u // 2]
    return {
        "model_name": "GIN",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth": cfg.get("depth", 5),
        "units": gin_units[0] if gin_units else 64,
        "dropout": _safe_dropout(cfg.get("dropout", 0.05), 0.05),
        "use_edge_features": False,
        "node_pooling": "sum",
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_rgin(cfg):
    result = convert_gin(cfg)
    result["model_name"] = "rGIN"
    rgin_args = cfg.get("rgin_args", {})
    result["random_range"] = rgin_args.get("random_range", 100)
    return result


def convert_graphsage(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "GraphSAGE",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth": cfg.get("depth", 3),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_dmpnn(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "DMPNN",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth": cfg.get("depth", 5),
        "units": cfg.get("edge_dense", {}).get("units", 128),
        "dropout": _safe_dropout(cfg.get("dropout"), 0.0),
        "node_pooling": "sum",
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_cmpnn(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "CMPNN",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth": cfg.get("depth", 5),
        "units": cfg.get("edge_dense", {}).get("units", 300),
        "use_final_gru": cfg.get("use_final_gru", True),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_attentivefp(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "AttentiveFP",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "units": cfg.get("attention_args", {}).get("units", 200),
        "depthato": cfg.get("depthato", 2),
        "depthmol": cfg.get("depthmol", 3),
        "dropout": _safe_dropout(cfg.get("dropout", 0.2), 0.2),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_nmpn(cfg):
    output_units, num_targets = get_output_units(cfg)
    node_dim = cfg.get("node_dim", cfg.get("input_node_embedding", {}).get("output_dim", 128))
    return {
        "model_name": "NMPN",
        "node_dim": node_dim,
        "depth": cfg.get("depth", 3),
        "use_set2set": cfg.get("use_set2set", True),
        "set2set_channels": cfg.get("set2set_args", {}).get("channels", 64),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_dgin(cfg):
    last_mlp = cfg.get("last_mlp", {})
    last_units = last_mlp.get("units", [64, 32])
    output_mlp = cfg.get("output_mlp", {})
    out_u = output_mlp.get("units", 1)
    num_targets = out_u[-1] if isinstance(out_u, list) else out_u
    return {
        "model_name": "DGIN",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth_dmpnn": cfg.get("depthDMPNN", 5),
        "depth_gin": cfg.get("depthGIN", 5),
        "units": cfg.get("edge_initialize", {}).get("units", 100),
        "dropout_dmpnn": _safe_dropout(cfg.get("dropoutDMPNN"), 0.05),
        "dropout_gin": _safe_dropout(cfg.get("dropoutGIN"), 0.05),
        "num_targets": num_targets,
    }


def convert_egnn(cfg):
    _, output_dim = get_output_units(cfg)
    return {
        "model_name": "EGNN",
        "num_features": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "num_interactions": cfg.get("depth", 4),
        "output_dim": output_dim,
    }


def convert_gnnfilm(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "GNNFilm",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "units": cfg.get("dense_relation_kwargs", {}).get("units", 64),
        "depth": cfg.get("depth", 5),
        "num_relations": cfg.get("dense_relation_kwargs", {}).get("num_relations", 20),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_rgcn(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "RGCN",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "units": cfg.get("dense_relation_kwargs", {}).get("units", 64),
        "depth": cfg.get("depth", 5),
        "num_relations": cfg.get("dense_relation_kwargs", {}).get("num_relations", 20),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_hamnet(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "HamNet",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "units": cfg.get("message_kwargs", {}).get("units", 200),
        "depth": cfg.get("depth", 3),
        "fingerprint_dim": cfg.get("fingerprint_kwargs", {}).get("units", 200),
        "fingerprint_depth": cfg.get("fingerprint_kwargs", {}).get("depth", 3),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_megnet(cfg):
    _, output_dim = get_output_units(cfg)
    return {
        "model_name": "Megnet",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 16),
        "depth": cfg.get("nblocks", 3),
        "set2set_channels": cfg.get("set2set_args", {}).get("channels", 16),
        "output_dim": output_dim,
    }


def convert_hdnnp2nd(cfg):
    mlp_kwargs = cfg.get("mlp_kwargs", {})
    mlp_units = mlp_kwargs.get("units", [128, 128, 128, 1])
    return {
        "model_name": "HDNNP2nd",
        "relational_units": mlp_units,
        "num_relations": mlp_kwargs.get("num_relations", 96),
        "num_targets": mlp_units[-1],
    }


def convert_mogat(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "MoGAT",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "units": cfg.get("attention_args", {}).get("units", 100),
        "depthato": cfg.get("depthato", 2),
        "depthmol": cfg.get("depthmol", 2),
        "dropout": _safe_dropout(cfg.get("dropout", 0.2), 0.2),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_inorp(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "INorp",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 32),
        "depth": cfg.get("depth", 3),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_megan(cfg):
    final_units = cfg.get("final_units", [50, 30, 10, 1])
    return {
        "model_name": "MEGAN",
        "units": cfg.get("units", [60, 50, 40, 30]),
        "importance_units": cfg.get("importance_units", []),
        "final_units": final_units,
        "dropout_rate": cfg.get("dropout_rate", 0.3),
        "importance_channels": cfg.get("importance_channels", 3),
        "num_targets": final_units[-1],
    }


def convert_mxmnet(cfg):
    _, num_targets = get_output_units(cfg)
    return {
        "model_name": "MXMNet",
        "node_dim": cfg.get("input_node_embedding", {}).get("output_dim", 32),
        "depth": cfg.get("depth", 4),
        "num_radial": cfg.get("bessel_basis_local", {}).get("num_radial", 16),
        "num_spherical": cfg.get("spherical_basis_local", {}).get("num_spherical", 7),
        "cutoff": cfg.get("bessel_basis_local", {}).get("cutoff", 5.0),
        "num_targets": num_targets,
    }


def convert_mat(cfg):
    output_units, num_targets = get_output_units(cfg)
    return {
        "model_name": "MAT",
        "embedding_units": cfg.get("embedding_units", 32),
        "depth": cfg.get("depth", 5),
        "num_heads": cfg.get("heads", 8),
        "attention_units": cfg.get("attention_kwargs", {}).get("units", 8),
        "merge_heads": cfg.get("merge_heads", "concat"),
        "attention_dropout": cfg.get("attention_kwargs", {}).get("dropout", 0.1),
        "output_units": output_units,
        "num_targets": num_targets,
    }


def convert_cgcnn(cfg):
    _, output_dim = get_output_units(cfg)
    gauss_args = cfg.get("gauss_args", {})
    pooling = cfg.get("node_pooling_args", {}).get("pooling_method", "scatter_mean")
    return {
        "model_name": "CGCNN",
        "node_features": cfg.get("input_node_embedding", {}).get("output_dim", 64),
        "depth": cfg.get("depth", 4),
        "num_gaussians": gauss_args.get("bins", 60),
        "cutoff": gauss_args.get("distance", 6.0),
        "readout": _strip_scatter(pooling),
        "output_dim": output_dim,
    }


MODEL_CONVERTERS = {
    "SchNet": convert_schnet,
    "PAiNN": convert_painn,
    "DimeNetPP": convert_dimenetpp,
    "GCN": convert_gcn,
    "GAT": convert_gat,
    "GATv2": convert_gatv2,
    "GIN": convert_gin,
    "rGIN": convert_rgin,
    "GraphSAGE": convert_graphsage,
    "DMPNN": convert_dmpnn,
    "CMPNN": convert_cmpnn,
    "AttentiveFP": convert_attentivefp,
    "NMPN": convert_nmpn,
    "DGIN": convert_dgin,
    "EGNN": convert_egnn,
    "GNNFilm": convert_gnnfilm,
    "RGCN": convert_rgcn,
    "HamNet": convert_hamnet,
    "Megnet": convert_megnet,
    "HDNNP2nd": convert_hdnnp2nd,
    "MoGAT": convert_mogat,
    "INorp": convert_inorp,
    "MEGAN": convert_megan,
    "MXMNet": convert_mxmnet,
    "MAT": convert_mat,
    "CGCNN": convert_cgcnn,
}


# --------------------------------------------------------------------------- #
#                     Training config conversion                               #
# --------------------------------------------------------------------------- #

def convert_loss_name(loss_raw):
    """Convert Keras loss to Torch loss string."""
    if isinstance(loss_raw, dict):
        cn = loss_raw.get("class_name", "")
        if "MeanAbsoluteError" in cn:
            return "mae"
        if "MeanSquaredError" in cn:
            return "mse"
        if "BinaryCrossentropy" in cn:
            return "bce_with_logits"
        if "CategoricalCrossentropy" in cn:
            return "categorical_crossentropy"
        return "mae"
    s = str(loss_raw).lower()
    if "mean_absolute_error" in s:
        return "mae"
    if "mean_squared_error" in s:
        return "mse"
    if "binary_crossentropy" in s:
        return "bce_with_logits"
    if "categorical_crossentropy" in s:
        return "categorical_crossentropy"
    return s


def extract_scheduler_and_lr(keras_training):
    """Return (scheduler_dict, lr_float) from Keras training config."""
    fit = keras_training.get("fit", {})
    compile_cfg = keras_training.get("compile", {})
    optimizer = compile_cfg.get("optimizer", {})
    opt_config = optimizer.get("config", {})
    lr_raw = opt_config.get("learning_rate", 1e-3)
    epochs = fit.get("epochs", 300)

    # 1) Check callbacks
    for cb in fit.get("callbacks", []):
        if not isinstance(cb, dict):
            continue
        cb_class = cb.get("class_name", "")
        cb_cfg = cb.get("config", {})

        if "LinearLearningRateScheduler" in cb_class:
            lr_start = cb_cfg.get("learning_rate_start", 1e-3)
            return {
                "class_name": "LinearLearningRateScheduler",
                "learning_rate_start": lr_start,
                "learning_rate_stop": cb_cfg.get("learning_rate_stop", 1e-5),
                "epo_min": cb_cfg.get("epo_min", 0),
                "epo": cb_cfg.get("epo", epochs),
            }, lr_start

        if "LinearWarmupExponentialLRScheduler" in cb_class:
            lr_start = cb_cfg.get("lr_start", 1e-3)
            return {
                "class_name": "LinearWarmupExponentialDecay",
                "lr_start": lr_start,
                "gamma": cb_cfg.get("gamma", 0.9961697),
                "epo_warmup": cb_cfg.get("epo_warmup", 1),
            }, lr_start

    # 2) Check learning_rate dict (LinearWarmupExponentialDecay / ExponentialDecay)
    #    Check LinearWarmup FIRST since "ExponentialDecay" is a substring of it.
    if isinstance(lr_raw, dict):
        lr_class = lr_raw.get("class_name", "")
        lr_cfg = lr_raw.get("config", {})

        if "LinearWarmupExponentialDecay" in lr_class:
            lr_val = lr_cfg.get("learning_rate", 1e-3)
            return {
                "class_name": "LinearWarmupExponentialDecay",
                "warmup_steps": int(lr_cfg.get("warmup_steps", 30)),
                "decay_steps": int(lr_cfg.get("decay_steps", 40000)),
                "decay_rate": lr_cfg.get("decay_rate", 0.01),
            }, lr_val

        if "ExponentialDecay" in lr_class:
            initial_lr = lr_cfg.get("initial_learning_rate", 1e-3)
            return {
                "class_name": "LinearWarmupExponentialDecay",
                "warmup_steps": 200,
                "decay_steps": int(lr_cfg.get("decay_steps", 40000)),
                "decay_rate": lr_cfg.get("decay_rate", 0.5),
            }, initial_lr

    # 3) Fallback: ReduceLROnPlateau
    lr_val = lr_raw if isinstance(lr_raw, (int, float)) else 1e-3
    return {
        "class_name": "ReduceLROnPlateau",
        "patience": 25 if epochs <= 500 else 50,
        "factor": 0.5,
        "min_lr": 1e-6,
    }, lr_val


def extract_n_splits(keras_entry):
    """Extract n_splits from cross-validation or dataset methods."""
    training = keras_entry.get("training", {})
    cv = training.get("cross_validation", {})
    if cv:
        cv_config = cv.get("config", {})
        if "n_splits" in cv_config:
            return cv_config["n_splits"]
    # Check dataset methods
    for key in ("dataset", "data"):
        ds = keras_entry.get(key, {})
        if isinstance(ds, dict):
            if "dataset" in ds:
                ds = ds["dataset"]
            for method in ds.get("methods", []):
                if isinstance(method, dict):
                    kf = method.get("set_train_test_indices_k_fold")
                    if kf and "n_splits" in kf:
                        return kf["n_splits"]
    return 5


def extract_scaler_class(keras_entry):
    """Extract scaler class name (or None) from Keras config."""
    training = keras_entry.get("training", {})
    scaler = training.get("scaler", {})
    if scaler:
        return scaler.get("class_name", "StandardLabelScaler")
    model_cfg = keras_entry.get("model", {}).get("config", {})
    output_scaling = model_cfg.get("output_scaling", {})
    if output_scaling:
        return output_scaling.get("name", "StandardLabelScaler")
    return None


def convert_training_config(keras_entry, *, is_classification, is_node_classification,
                            is_force, scaler_override, force_template):
    """Convert Keras training to Torch JSON training section."""
    training = keras_entry.get("training", {})
    fit = training.get("fit", {})
    compile_cfg = training.get("compile", {})

    epochs = fit.get("epochs", 300)
    batch_size = fit.get("batch_size", 32)
    early_stopping = 50 if epochs <= 500 else 100

    optimizer = compile_cfg.get("optimizer", {})
    opt_class = optimizer.get("class_name", "Adam").split(".")[-1]
    opt_config = optimizer.get("config", {})

    scheduler, lr_value = extract_scheduler_and_lr(training)
    loss = convert_loss_name(compile_cfg.get("loss", "mean_absolute_error"))

    if is_classification and loss not in ("bce_with_logits",):
        loss = "bce_with_logits"
    if is_node_classification:
        loss = "categorical_crossentropy"

    opt_result = {"class_name": opt_class, "config": {"lr": lr_value}}
    if "AdamW" in opt_class:
        opt_result["config"]["weight_decay"] = opt_config.get("weight_decay", 1e-5)

    result = {
        "fit": {
            "epochs": epochs,
            "batch_size": batch_size,
            "early_stopping_patience": early_stopping,
        },
        "compile": {
            "optimizer": opt_result,
            "loss": loss,
        },
        "scheduler": scheduler,
    }

    # Scaler
    scaler_class = scaler_override or extract_scaler_class(keras_entry)
    if scaler_class and not is_classification and not is_node_classification:
        result["scaler"] = {"class_name": scaler_class}

    # Cross-validation
    n_splits = extract_n_splits(keras_entry)
    result["cross_validation"] = {"n_splits": n_splits, "shuffle": True}

    # Force model
    if is_force:
        if force_template:
            # Reuse energy/force keys and weights from existing entry
            tmpl_fit = force_template.get("fit", {})
            tmpl_compile = force_template.get("compile", {})
            result["fit"]["energy_key"] = tmpl_fit.get("energy_key", "y")
            result["fit"]["force_key"] = tmpl_fit.get("force_key", "force")
            result["compile"]["loss"] = "energy_force"
            result["compile"]["energy_weight"] = tmpl_compile.get("energy_weight", 1.0)
            result["compile"]["force_weight"] = tmpl_compile.get("force_weight", 100.0)
            # Reuse scaler from template if present
            if "scaler" in force_template and "scaler" not in result:
                result["scaler"] = copy.deepcopy(force_template["scaler"])
            elif "scaler" in force_template:
                result["scaler"] = copy.deepcopy(force_template["scaler"])
        else:
            result["fit"]["energy_key"] = "y"
            result["fit"]["force_key"] = "force"
            result["compile"]["loss"] = "energy_force"
            result["compile"]["energy_weight"] = 1.0
            result["compile"]["force_weight"] = 100.0

    return result


# --------------------------------------------------------------------------- #
#                     Data section helpers                                     #
# --------------------------------------------------------------------------- #

def extract_data_template(torch_json):
    """Get data section from first existing entry as template."""
    for entry in torch_json.values():
        return copy.deepcopy(entry.get("data", {}))
    return {}


def extract_force_template(torch_json):
    """Get force-model training settings from first existing entry."""
    for entry in torch_json.values():
        t = entry.get("training", {})
        if t.get("compile", {}).get("loss") == "energy_force":
            return copy.deepcopy(t)
    return None


# --------------------------------------------------------------------------- #
#                     Main conversion loop                                     #
# --------------------------------------------------------------------------- #

def main():
    total_added = 0

    for keras_base, torch_base in sorted(FILE_MAP.items()):
        keras_path = os.path.join(KERAS_DIR, keras_base + ".py")
        torch_path = os.path.join(TORCH_DIR, torch_base + ".json")

        if not os.path.exists(keras_path):
            print(f"  SKIP: {keras_path} not found")
            continue

        keras_hyper = load_keras_hyper(keras_path)
        torch_json = load_torch_json(torch_path)
        data_template = extract_data_template(torch_json)

        is_classification = keras_base in CLASSIFICATION_DATASETS
        is_node_classification = keras_base in NODE_CLASSIFICATION_DATASETS
        is_force = keras_base in FORCE_DATASETS
        scaler_override = "QMGraphLabelScaler" if keras_base in QM_DATASETS else None
        force_template = extract_force_template(torch_json) if is_force else None

        added = []

        for keras_model_name, keras_entry in keras_hyper.items():
            torch_model_name = resolve_model_name(keras_model_name)

            # Skip already existing
            if torch_model_name in torch_json:
                continue

            # Skip unsupported models
            if torch_model_name not in SUPPORTED_MODELS:
                print(f"  WARN: '{torch_model_name}' not in Torch registry, skipping ({keras_base})")
                continue

            if torch_model_name not in MODEL_CONVERTERS:
                print(f"  WARN: No converter for '{torch_model_name}', skipping ({keras_base})")
                continue

            # Convert model config
            keras_model_config = keras_entry.get("model", {}).get("config", {})
            model_config = MODEL_CONVERTERS[torch_model_name](keras_model_config)

            # Classification flags
            if is_classification:
                model_config["output_activation"] = "sigmoid"
            if is_node_classification:
                model_config["output_embedding"] = "node"

            # Convert training config
            training_config = convert_training_config(
                keras_entry,
                is_classification=is_classification,
                is_node_classification=is_node_classification,
                is_force=is_force,
                scaler_override=scaler_override,
                force_template=force_template,
            )

            torch_entry = {
                "model": {"config": model_config},
                "training": training_config,
                "data": copy.deepcopy(data_template),
            }

            torch_json[torch_model_name] = torch_entry
            added.append(torch_model_name)

        if added:
            save_torch_json(torch_path, torch_json)
            print(f"  {torch_base}.json: +{len(added)} models: {', '.join(added)}")
            total_added += len(added)
        else:
            print(f"  {torch_base}.json: no new models")

    print(f"\nTotal: {total_added} model entries added")


# --------------------------------------------------------------------------- #
#                     Manual additions for edge cases                          #
# --------------------------------------------------------------------------- #

def add_manual_entries():
    """Add entries that Keras doesn't have but the plan requires."""

    # EGNN for MD17 (force model) - Keras MD17 may not have EGNN
    md17_path = os.path.join(TORCH_DIR, "hyper_md17.json")
    md17 = load_torch_json(md17_path)
    if "EGNN" not in md17:
        data_template = extract_data_template(md17)
        force_template = extract_force_template(md17)
        ft = force_template or {}
        ft_fit = ft.get("fit", {})
        ft_compile = ft.get("compile", {})

        md17["EGNN"] = {
            "model": {
                "config": {
                    "model_name": "EGNN",
                    "num_features": 64,
                    "num_interactions": 4,
                    "output_dim": 1,
                }
            },
            "training": {
                "fit": {
                    "epochs": 1000,
                    "batch_size": 64,
                    "early_stopping_patience": 100,
                    "energy_key": ft_fit.get("energy_key", "y"),
                    "force_key": ft_fit.get("force_key", "force"),
                },
                "compile": {
                    "optimizer": {"class_name": "Adam", "config": {"lr": 5e-4}},
                    "loss": "energy_force",
                    "energy_weight": ft_compile.get("energy_weight", 1.0),
                    "force_weight": ft_compile.get("force_weight", 100.0),
                },
                "scheduler": {
                    "class_name": "ReduceLROnPlateau",
                    "patience": 50,
                    "factor": 0.5,
                    "min_lr": 1e-7,
                },
                "cross_validation": {"n_splits": 3, "shuffle": True},
            },
            "data": copy.deepcopy(data_template),
        }
        save_torch_json(md17_path, md17)
        print("  hyper_md17.json: +1 manual (EGNN)")
        return 1
    return 0


if __name__ == "__main__":
    print("=== Converting Keras hyper configs to Torch JSON ===\n")
    main()
    print("\n=== Adding manual entries ===\n")
    count = add_manual_entries()
    print(f"\nManual additions: {count}")
    print("\nDone!")
