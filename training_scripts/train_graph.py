"""Training script for graph-level property prediction.

Supports cross-validation, data scaling, model checkpointing, and result logging.

Usage:
    python train_graph.py --hyper hyper/hyper_esol.json --category SchNet --output results/
    python train_graph.py --hyper hyper/hyper_esol.json --category GIN --fold 0 1 2
"""
import argparse
import copy
import json
import os
import time
import logging
import inspect
import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import UninitializedParameter
from datetime import timedelta
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_adj, to_dense_batch
from sklearn.model_selection import KFold

from kgcnn_torch.training.hyper import HyperParameter
from kgcnn_torch.training.trainer import fit
from kgcnn_torch.training.scheduler import get_scheduler
from kgcnn_torch.metrics.metrics import mae, rmse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model registry
_MODEL_REGISTRY = {
    "SchNet": "kgcnn_torch.models.schnet.SchNetModel",
    "PAiNN": "kgcnn_torch.models.painn.PAiNNModel",
    "DimeNetPP": "kgcnn_torch.models.dimenetpp.DimeNetPPModel",
    "GCN": "kgcnn_torch.models.gcn.GCNModel",
    "GAT": "kgcnn_torch.models.gat.GATModel",
    "GATv2": "kgcnn_torch.models.gatv2.GATv2Model",
    "GIN": "kgcnn_torch.models.gin.GINModel",
    "EGNN": "kgcnn_torch.models.egnn.EGNNModel",
    "DMPNN": "kgcnn_torch.models.dmpnn.DMPNNModel",
    "GraphSAGE": "kgcnn_torch.models.graphsage.GraphSAGEModel",
    "Megnet": "kgcnn_torch.models.megnet.MEGNetModel",
    "AttentiveFP": "kgcnn_torch.models.attentivefp.AttentiveFPModel",
    "CGCNN": "kgcnn_torch.models.cgcnn.CGCNNModel",
    "NMPN": "kgcnn_torch.models.nmpn.NMPNModel",
    "INorp": "kgcnn_torch.models.inorp.INorpModel",
    "MEGAN": "kgcnn_torch.models.megan.MEGANModel",
    "RGCN": "kgcnn_torch.models.rgcn.RGCNModel",
    "GNNFilm": "kgcnn_torch.models.gnnfilm.GNNFilmModel",
    "rGIN": "kgcnn_torch.models.rgin.rGINModel",
    "MXMNet": "kgcnn_torch.models.mxmnet.MXMNetModel",
    "MoGAT": "kgcnn_torch.models.mogat.MoGATModel",
    "CMPNN": "kgcnn_torch.models.cmpnn.CMPNNModel",
    "DGIN": "kgcnn_torch.models.dgin.DGINModel",
    "HamNet": "kgcnn_torch.models.hamnet.HamNetModel",
    "HDNNP2nd": "kgcnn_torch.models.hdnnp2nd.HDNNP2ndModel",
    "MAT": "kgcnn_torch.models.mat.MATModel",
}


def get_model_class(name: str):
    """Import and return model class by name."""
    if name in _MODEL_REGISTRY:
        module_path, class_name = _MODEL_REGISTRY[name].rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")


# Mapping from common hyper-config parameter names to model constructor parameter names.
# This handles the naming differences between Keras-style configs and kgcnn-torch models.
_PARAM_ALIASES = {
    "SchNet": {
        "num_features": "node_dim",
        "num_filters": "units",
        "num_interactions": "depth",
        "cutoff": "gauss_distance",
        "num_gaussians": "gauss_bins",
        "readout": "node_pooling",
        "output_dim": "num_targets",
    },
    "PAiNN": {
        "num_features": "node_dim",
        "num_interactions": "depth",
        "num_gaussians": "num_radial",
        "output_dim": "num_targets",
    },
    "DimeNetPP": {
        "output_dim": "num_targets",
    },
    "EGNN": {
        "num_features": "node_dim",
        "num_interactions": "depth",
        "output_dim": "num_targets",
    },
    "GCN": {
        "units": "gcn_units",
    },
    "GIN": {
        "dropout": "dropout_rate",
    },
    "rGIN": {},
    "DMPNN": {
        "dropout": "dropout_rate",
    },
    "CMPNN": {},
    "CGCNN": {
        "node_features": "node_dim",
        "num_gaussians": "gauss_bins",
        "cutoff": "gauss_distance",
        "readout": "node_pooling",
        "output_dim": "num_targets",
    },
    "Megnet": {
        "output_dim": "num_targets",
    },
    "AttentiveFP": {
        "depthato": "depth_ato",
        "depthmol": "depth_mol",
    },
    "GraphSAGE": {
        "output_activation": "output_final_activation",
    },
}

# Config keys to silently ignore (e.g., data preprocessing params not used by model).
_IGNORED_PARAMS = {
    "GCN": {"dropout", "use_edge_features"},
    "CGCNN": {"edge_features", "units"},
    "DimeNetPP": {"output_activation"},
    "rGIN": {"use_edge_features"},
}


def translate_model_config(model_name, config):
    """Translate hyper config parameter names to model constructor parameter names.

    Args:
        model_name: Name of the model (e.g., 'SchNet', 'GCN').
        config: Dict of model config parameters from hyper file.

    Returns:
        Translated config dict with model-compatible parameter names.
    """
    aliases = _PARAM_ALIASES.get(model_name, {})
    ignored = _IGNORED_PARAMS.get(model_name, set())
    translated = {}
    for k, v in config.items():
        if k in ignored:
            continue
        translated[aliases.get(k, k)] = v
    return translated


def make_optimizer(model, compile_config):
    """Create optimizer from config."""
    opt_name = compile_config.get("optimizer", {}).get("class_name", "Adam")
    opt_config = compile_config.get("optimizer", {}).get("config", {"lr": 1e-3})
    opt_name_lower = opt_name.lower().replace("torch.optim.", "")
    if opt_name_lower in ("adam",):
        return torch.optim.Adam(model.parameters(), **opt_config)
    elif opt_name_lower in ("adamw",):
        return torch.optim.AdamW(model.parameters(), **opt_config)
    elif opt_name_lower in ("sgd",):
        return torch.optim.SGD(model.parameters(), **opt_config)
    else:
        return torch.optim.Adam(model.parameters(), lr=opt_config.get("lr", 1e-3))


def make_loss(compile_config):
    """Create loss function from config."""
    loss_name = compile_config.get("loss", "mse")
    if loss_name in ("mse", "mean_squared_error"):
        return nn.MSELoss()
    elif loss_name in ("mae", "mean_absolute_error", "l1"):
        return nn.L1Loss()
    elif loss_name == "huber":
        return nn.HuberLoss()
    elif loss_name in ("bce", "bce_with_logits", "binary_crossentropy"):
        return nn.BCEWithLogitsLoss()
    elif loss_name in ("categorical_crossentropy", "cross_entropy"):
        return nn.CrossEntropyLoss()
    else:
        logger.warning(f"Unknown loss '{loss_name}', falling back to MSELoss.")
        return nn.MSELoss()


class MATBatchWrapper(nn.Module):
    """Wrap MATModel to accept a PyG disjoint batch.

    Converts disjoint batch tensors into dense padded tensors expected by MAT:
    ``(node_input, xyz_input, adjacency, node_mask, adj_mask)``.
    """

    def __init__(self, mat_model: nn.Module):
        super().__init__()
        self.mat_model = mat_model

    def forward(self, batch):
        if hasattr(batch, "z"):
            node_input, node_mask = to_dense_batch(batch.z, batch.batch)
        elif hasattr(batch, "x"):
            node_input, node_mask = to_dense_batch(batch.x, batch.batch)
        else:
            raise ValueError("MAT requires node features in batch.z or batch.x")

        if hasattr(batch, "pos") and batch.pos is not None:
            xyz_input, _ = to_dense_batch(batch.pos, batch.batch)
        else:
            xyz_input = torch.zeros(
                node_input.size(0), node_input.size(1), 3,
                dtype=torch.float, device=node_input.device
            )

        adjacency = to_dense_adj(batch.edge_index, batch=batch.batch, max_num_nodes=node_input.size(1))
        adj_mask = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        return self.mat_model(node_input, xyz_input, adjacency, node_mask, adj_mask)


def count_model_parameters(model):
    """Count initialized trainable and non-trainable parameters.

    Lazy modules may contain ``UninitializedParameter`` objects before the first
    forward pass; those are skipped to avoid runtime errors in logging.
    """
    n_params = 0
    n_uninitialized = 0
    for p in model.parameters():
        if isinstance(p, UninitializedParameter):
            n_uninitialized += 1
            continue
        n_params += p.numel()
    return n_params, n_uninitialized


def adapt_model_config_from_data(model_name: str, model_config: dict, pyg_data_list: list) -> dict:
    """Fill model config defaults that depend on dataset tensor shapes."""
    if not pyg_data_list:
        return model_config

    cfg = dict(model_config)
    sample = pyg_data_list[0]

    edge_attr_dim = None
    if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
        edge_attr_dim = int(sample.edge_attr.size(-1)) if sample.edge_attr.dim() > 1 else 1

    if edge_attr_dim is not None:
        if model_name in {"DMPNN", "CMPNN", "DGIN", "NMPN", "INorp", "HamNet"}:
            cfg.setdefault("edge_dim", edge_attr_dim)
        if model_name in {"AttentiveFP", "GraphSAGE", "GAT", "GATv2"}:
            cfg.setdefault("edge_dim", edge_attr_dim)
        if model_name == "Megnet":
            cfg.setdefault("edge_input_dim", edge_attr_dim)

        # CGCNN expects scalar distances when expand_distance=True. If edge_attr is already a vector,
        # use it directly as fixed edge features.
        if model_name == "CGCNN" and edge_attr_dim > 1:
            cfg.setdefault("expand_distance", False)
            cfg.setdefault("gauss_bins", edge_attr_dim)

    # Handle node input: embedding vs projection.
    supports_node_embedding = False
    try:
        sig = inspect.signature(get_model_class(model_name).__init__)
        supports_node_embedding = "use_node_embedding" in sig.parameters
    except Exception:
        supports_node_embedding = False

    if supports_node_embedding:
        if "use_node_embedding" not in cfg:
            # Auto-detect: use embedding if integer atomic numbers are available
            # and no dense float node features exist.
            if hasattr(sample, "z") and sample.z is not None and sample.z.dim() == 1:
                has_dense_x = (
                    hasattr(sample, "x") and sample.x is not None and
                    sample.x.dim() > 1 and torch.is_floating_point(sample.x)
                )
                if not has_dense_x:
                    cfg["use_node_embedding"] = True

        # When use_node_embedding is explicitly False, set node_input_dim
        # from data.x so the model can create a Linear projection layer.
        if cfg.get("use_node_embedding") is False:
            if hasattr(sample, "x") and sample.x is not None and sample.x.dim() > 1:
                cfg.setdefault("node_input_dim", int(sample.x.size(-1)))
            elif hasattr(sample, "x") and sample.x is not None and sample.x.dim() == 1:
                cfg.setdefault("node_input_dim", 1)
            elif hasattr(sample, "z") and sample.z is not None:
                cfg.setdefault("node_input_dim", 1)

    # HDNNP2nd requires explicit element type map that covers all atomic numbers.
    if model_name == "HDNNP2nd" and "element_types" not in cfg:
        z_all = torch.cat([d.z.view(-1).cpu() for d in pyg_data_list if hasattr(d, "z") and d.z is not None], dim=0)
        cfg["element_types"] = sorted(torch.unique(z_all).tolist())

    return cfg


def load_dataset_pyg(data_config):
    """Load dataset and return list of PyG Data objects.

    Supports:
    1. Pickle file with MemoryGraphList
    2. PyG InMemoryDataset path
    3. Custom dataset class from kgcnn_torch.data.datasets

    Args:
        data_config: Dict with 'dataset' section containing 'class_name', 'config', etc.

    Returns:
        List of PyG Data objects.
    """
    dataset_config = data_config.get("dataset", {})
    class_name = dataset_config.get("class_name", "")
    config = dataset_config.get("config", {})

    # Try loading from pickle (kgcnn format)
    pickle_path = config.get("file_path", None)
    if pickle_path and os.path.exists(pickle_path):
        from kgcnn_torch.data.base import MemoryGraphList
        gl = MemoryGraphList()
        gl.load(pickle_path)
        # Ensure common edge features are preserved from kgcnn GraphDict pickles.
        return gl.to_pyg_list(edge_attr_keys=["edge_attributes"])

    # Try loading from kgcnn_torch.data.datasets module
    if class_name:
        try:
            import importlib
            module = importlib.import_module("kgcnn_torch.data.datasets")
            # Try exact name first, then with "Dataset" suffix
            try:
                DatasetClass = getattr(module, class_name)
            except AttributeError:
                DatasetClass = getattr(module, class_name + "Dataset")
            dataset = DatasetClass(**config)
            if hasattr(dataset, 'to_pyg_list'):
                return dataset.to_pyg_list()
            return list(dataset)
        except (ImportError, AttributeError):
            pass

    # Try loading PyG built-in datasets
    if class_name:
        try:
            import importlib
            module = importlib.import_module("torch_geometric.datasets")
            DatasetClass = getattr(module, class_name)
            root = config.get("root", "data/")
            name = config.get("name", None)
            if name:
                dataset = DatasetClass(root=root, name=name)
            else:
                dataset = DatasetClass(root=root)
            return list(dataset)
        except (ImportError, AttributeError):
            pass

    raise ValueError(
        f"Could not load dataset. Provide a valid 'file_path' to a pickle file, "
        f"or a valid 'class_name' in data config. Got: {dataset_config}"
    )


def save_results(filepath, history_list, time_list, hyper, label_units=None, seed=42):
    """Save training results summary to YAML-like text file."""
    lines = []
    lines.append(f"model_name: {hyper._model_name or 'unknown'}")
    lines.append(f"seed: {seed}")
    lines.append(f"device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
    lines.append(f"torch_version: {torch.__version__}")
    lines.append(f"n_folds: {len(history_list)}")
    lines.append("")

    for i, (hist, elapsed) in enumerate(zip(history_list, time_list)):
        lines.append(f"fold_{i}:")
        lines.append(f"  time: {elapsed}")
        lines.append(f"  epochs: {len(hist.get('train_loss', []))}")
        if 'val_loss' in hist and hist['val_loss']:
            best_val = min(hist['val_loss'])
            best_epoch = hist['val_loss'].index(best_val) + 1
            lines.append(f"  best_val_loss: {best_val:.6f}")
            lines.append(f"  best_epoch: {best_epoch}")
        if 'val_mae' in hist and hist['val_mae']:
            lines.append(f"  best_val_mae: {min(hist['val_mae']):.6f}")
            lines.append(f"  final_val_mae: {hist['val_mae'][-1]:.6f}")
        if 'val_rmse' in hist and hist['val_rmse']:
            lines.append(f"  best_val_rmse: {min(hist['val_rmse']):.6f}")
        lines.append("")

    # Summary across folds
    if len(history_list) > 1:
        val_losses = [min(h['val_loss']) for h in history_list if 'val_loss' in h and h['val_loss']]
        if val_losses:
            lines.append("summary:")
            lines.append(f"  mean_best_val_loss: {np.mean(val_losses):.6f} +/- {np.std(val_losses):.6f}")
        val_maes = [min(h['val_mae']) for h in history_list if 'val_mae' in h and h['val_mae']]
        if val_maes:
            lines.append(f"  mean_best_val_mae: {np.mean(val_maes):.6f} +/- {np.std(val_maes):.6f}")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train graph-level GNN model")
    parser.add_argument("--hyper", type=str, required=True, help="Path to hyperparameter file (.json or .py)")
    parser.add_argument("--category", type=str, default=None, help="Model category in hyper file")
    parser.add_argument("--model", type=str, default=None, help="Override model name")
    parser.add_argument("--output", type=str, default="results/", help="Output directory")
    parser.add_argument("--device", type=str, default="auto", help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--fold", type=int, nargs="+", default=None, help="Specific fold indices to run")
    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # Load hyperparameters
    hyper = HyperParameter(args.hyper, model_name=args.category)
    model_config = hyper.model_config
    fit_config = hyper.fit_config
    compile_config = hyper.compile_config
    scheduler_config = hyper.scheduler_config
    scaler_config = hyper.scaler_config
    cv_config = hyper.cross_validation_config

    # Model name
    model_name = args.model or model_config.pop("model_name", args.category or "SchNet")
    model_config = translate_model_config(model_name, model_config)
    logger.info(f"Model: {model_name}")
    logger.info(f"Model config: {json.dumps(model_config, indent=2, default=str)}")

    # Load dataset
    data_config = hyper.data_config
    logger.info("Loading dataset...")
    pyg_data_list = load_dataset_pyg(data_config)
    data_length = len(pyg_data_list)
    logger.info(f"Dataset loaded with {data_length} graphs")
    model_config = adapt_model_config_from_data(model_name, model_config, pyg_data_list)
    logger.info(f"Adapted model config: {json.dumps(model_config, indent=2, default=str)}")

    # Training parameters
    epochs = fit_config.get("epochs", 100)
    batch_size = fit_config.get("batch_size", 32)
    early_stopping = fit_config.get("early_stopping_patience", 0)

    # Cross-validation splits
    if cv_config:
        n_splits = cv_config.get("n_splits", cv_config.get("config", {}).get("n_splits", 5))
        shuffle = cv_config.get("shuffle", cv_config.get("config", {}).get("shuffle", True))
        kfold = KFold(n_splits=n_splits, shuffle=shuffle, random_state=args.seed)
        train_test_indices = list(kfold.split(np.zeros(data_length)))
    else:
        # Default 80/20 split
        indices = np.arange(data_length)
        np.random.shuffle(indices)
        split_point = int(0.8 * data_length)
        train_test_indices = [(indices[:split_point], indices[split_point:])]

    # Output directory
    output_dir = os.path.join(args.output, model_name)
    os.makedirs(output_dir, exist_ok=True)
    hyper.save(os.path.join(output_dir, "hyper_used.json"))

    # Determine which folds to run
    execute_folds = args.fold

    # Metrics
    metrics = {"mae": mae, "rmse": rmse}

    # Run cross-validation
    history_list = []
    time_list = []

    for fold_idx, (train_index, test_index) in enumerate(train_test_indices):
        if execute_folds is not None and fold_idx not in execute_folds:
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx+1}/{len(train_test_indices)}")
        logger.info(f"Train: {len(train_index)}, Test: {len(test_index)}")
        logger.info(f"{'='*60}")

        # Split dataset - deep copy to avoid mutating original data across folds
        # (scaler transforms labels in-place, so without copy fold 1 would
        #  fit on already-Z-scored labels from fold 0)
        train_data = [copy.deepcopy(pyg_data_list[i]) for i in train_index]
        test_data = [copy.deepcopy(pyg_data_list[i]) for i in test_index]

        # Apply scaler if configured
        scaler = None
        if scaler_config:
            scaler_class = scaler_config.get("class_name", "StandardLabelScaler")
            if scaler_class in ("StandardLabelScaler", "StandardScaler"):
                from kgcnn_torch.data.transform import StandardLabelScaler
                scaler = StandardLabelScaler()
                train_labels = torch.stack([d.y for d in train_data if hasattr(d, 'y') and d.y is not None])
                scaler.fit(train_labels.numpy())
                # Transform labels
                for d in train_data:
                    if hasattr(d, 'y') and d.y is not None:
                        d.y = torch.tensor(scaler.transform(d.y.numpy()), dtype=torch.float)
                for d in test_data:
                    if hasattr(d, 'y') and d.y is not None:
                        d.y = torch.tensor(scaler.transform(d.y.numpy()), dtype=torch.float)
                # Save scaler
                scaler.save(os.path.join(output_dir, f"scaler_fold_{fold_idx}.json"))

        # DataLoaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        # Create model
        ModelClass = get_model_class(model_name)
        model = ModelClass(**model_config)
        if model_name == "MAT":
            model = MATBatchWrapper(model)

        # Match Keras Glorot/Xavier initialization for node_projection layers.
        # PyTorch default (Kaiming) gives weights in [-1, 1] for fan_in=1,
        # while Keras Dense uses Glorot uniform [-0.3, 0.3].  With atomic
        # numbers (1-90) as input, the 3x larger initial outputs cause
        # gradient explosion in attention-based models (GAT, AttentiveFP).
        for name, module in model.named_modules():
            if 'node_projection' in name and isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

        model = model.to(device)

        n_params, n_uninitialized = count_model_parameters(model)
        if n_uninitialized > 0:
            logger.info(
                f"Model parameters (initialized): {n_params:,} "
                f"(uninitialized: {n_uninitialized})"
            )
        else:
            logger.info(f"Model parameters: {n_params:,}")

        # Optimizer
        optimizer = make_optimizer(model, compile_config)

        # Loss
        loss_fn = make_loss(compile_config)

        # Scheduler
        scheduler = None
        if scheduler_config:
            sched_config = dict(scheduler_config)
            sched_name = sched_config.pop("class_name", None)
            if sched_name:
                # Pass steps_per_epoch so step-based Keras configs are converted correctly
                sched_config.setdefault("steps_per_epoch", len(train_loader))
                scheduler = get_scheduler(sched_name, optimizer, **sched_config)

        # Checkpoint path
        checkpoint_path = os.path.join(output_dir, f"best_model_fold_{fold_idx}.pt")

        # Train
        start_time = time.time()
        history = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_fn=loss_fn,
            scheduler=scheduler,
            epochs=epochs,
            device=device,
            metrics=metrics,
            verbose=1,
            early_stopping_patience=early_stopping,
            checkpoint_path=checkpoint_path,
            scaler=scaler
        )
        elapsed = str(timedelta(seconds=time.time() - start_time))
        logger.info(f"Fold {fold_idx+1} training time: {elapsed}")

        # Save history
        history_list.append(history)
        time_list.append(elapsed)

        with open(os.path.join(output_dir, f"history_fold_{fold_idx}.json"), 'w') as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

        # Save final model
        torch.save(model.state_dict(), os.path.join(output_dir, f"model_fold_{fold_idx}.pt"))

        # Log best results
        if 'val_loss' in history and history['val_loss']:
            best_val = min(history['val_loss'])
            logger.info(f"Best val_loss: {best_val:.6f}")
        if 'val_mae' in history and history['val_mae']:
            best_mae = min(history['val_mae'])
            logger.info(f"Best val_mae: {best_mae:.6f}")

    # Save overall results
    if history_list:
        save_results(
            os.path.join(output_dir, "score.yaml"),
            history_list, time_list, hyper, seed=args.seed
        )

    # Save indices
    train_indices_all = [ti.tolist() if hasattr(ti, 'tolist') else list(ti)
                         for ti, _ in train_test_indices]
    test_indices_all = [ti.tolist() if hasattr(ti, 'tolist') else list(ti)
                        for _, ti in train_test_indices]
    np.savez(os.path.join(output_dir, "train_indices.npz"), *train_indices_all)
    np.savez(os.path.join(output_dir, "test_indices.npz"), *test_indices_all)

    logger.info(f"\nTraining complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
