"""Training script for node-level classification/regression.

Supports transductive learning on single-graph datasets (e.g., citation networks
like Cora, CiteSeer, PubMed) where the task is to predict per-node labels.

Key differences from train_graph.py:
    - No graph-level pooling: predictions are per-node.
    - Uses train/val/test masks (transductive setting) instead of splitting graphs.
    - Cross-entropy loss for classification, MSE for regression.
    - Metrics: accuracy, F1-score for classification.

Usage:
    python train_node.py --hyper hyper/hyper_cora.json --category GCN --output results/
    python train_node.py --dataset Cora --model GCN --epochs 200 --lr 0.01
"""
import argparse
import json
import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import timedelta
from sklearn.model_selection import KFold

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model registry (node-level models)
_MODEL_REGISTRY = {
    "GCN": "kgcnn_torch.models.gcn.GCNModel",
    "GAT": "kgcnn_torch.models.gat.GATModel",
    "GATv2": "kgcnn_torch.models.gatv2.GATv2Model",
    "GIN": "kgcnn_torch.models.gin.GINModel",
    "GraphSAGE": "kgcnn_torch.models.graphsage.GraphSAGEModel",
    "RGCN": "kgcnn_torch.models.rgcn.RGCNModel",
    "GNNFilm": "kgcnn_torch.models.gnnfilm.GNNFilmModel",
}


def get_model_class(name: str):
    """Import and return model class by name."""
    if name in _MODEL_REGISTRY:
        module_path, class_name = _MODEL_REGISTRY[name].rsplit('.', 1)
        import importlib
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    raise ValueError(f"Unknown model '{name}'. Available: {list(_MODEL_REGISTRY.keys())}")


class NodeGCN(nn.Module):
    """Simple GCN model for node-level classification/regression.

    This wraps GCN convolution layers without graph-level pooling, outputting
    per-node predictions suitable for node classification or regression tasks.

    Args:
        input_dim: Dimension of input node features.
        hidden_dim: Hidden dimension for GCN layers.
        output_dim: Number of output classes or regression targets.
        depth: Number of GCN layers.
        dropout: Dropout rate applied between layers.
        use_node_embedding: Whether to use an embedding layer for integer node features.
        num_embeddings: Vocabulary size for node embedding.
        activation: Activation function for hidden layers.
    """

    def __init__(self,
                 input_dim: int = 1433,
                 hidden_dim: int = 64,
                 output_dim: int = 7,
                 depth: int = 2,
                 dropout: float = 0.5,
                 use_node_embedding: bool = False,
                 num_embeddings: int = 95,
                 activation: str = "relu"):
        super().__init__()
        from kgcnn_torch.layers.conv import GCNConv
        from kgcnn_torch.ops.activ import get_activation

        self.use_node_embedding = use_node_embedding
        self.dropout_rate = dropout

        if use_node_embedding:
            self.node_embedding = nn.Embedding(num_embeddings, input_dim)

        self.convs = nn.ModuleList()
        # First layer
        self.convs.append(GCNConv(
            in_features=input_dim, out_features=hidden_dim,
            pooling_method="sum", activation=activation
        ))
        # Hidden layers
        for _ in range(depth - 2):
            self.convs.append(GCNConv(
                in_features=hidden_dim, out_features=hidden_dim,
                pooling_method="sum", activation=activation
            ))
        # Output layer (no activation, applied later via loss)
        if depth > 1:
            self.convs.append(GCNConv(
                in_features=hidden_dim, out_features=output_dim,
                pooling_method="sum", activation="linear"
            ))
        else:
            # Single-layer: overwrite first conv to go directly to output_dim
            self.convs = nn.ModuleList([GCNConv(
                in_features=input_dim, out_features=output_dim,
                pooling_method="sum", activation="linear"
            )])

    def forward(self, data) -> torch.Tensor:
        """Forward pass producing per-node predictions.

        Args:
            data: PyG Data object with x, edge_index, edge_weight (or edge_attr).

        Returns:
            Node-level logits/predictions of shape (N, output_dim).
        """
        x = data.x
        edge_index = data.edge_index
        edge_weight = data.edge_weight if hasattr(data, 'edge_weight') and data.edge_weight is not None else None
        if edge_weight is None and hasattr(data, 'edge_attr') and data.edge_attr is not None:
            edge_weight = data.edge_attr

        # If edge_weight is missing, use ones
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), 1, device=edge_index.device)

        if self.use_node_embedding:
            x = self.node_embedding(x.long())

        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index, edge_weight)
            x = F.dropout(x, p=self.dropout_rate, training=self.training)

        x = self.convs[-1](x, edge_index, edge_weight)
        return x


def compute_accuracy(pred: torch.Tensor, target: torch.Tensor,
                     mask: torch.Tensor = None) -> float:
    """Compute classification accuracy.

    Args:
        pred: Predicted logits of shape (N, C).
        target: Target labels of shape (N,) (class indices).
        mask: Boolean mask of shape (N,). Only masked nodes are evaluated.

    Returns:
        Accuracy as a float.
    """
    pred_class = pred.argmax(dim=-1)
    if mask is not None:
        correct = (pred_class[mask] == target[mask]).sum().item()
        total = mask.sum().item()
    else:
        correct = (pred_class == target).sum().item()
        total = target.size(0)
    return correct / max(total, 1)


def compute_f1(pred: torch.Tensor, target: torch.Tensor,
               mask: torch.Tensor = None, average: str = "macro") -> float:
    """Compute F1-score.

    Args:
        pred: Predicted logits of shape (N, C).
        target: Target labels of shape (N,).
        mask: Boolean mask of shape (N,).
        average: Averaging mode ('macro', 'micro', 'weighted').

    Returns:
        F1-score as a float.
    """
    from sklearn.metrics import f1_score

    pred_class = pred.argmax(dim=-1).cpu().numpy()
    target_np = target.cpu().numpy()

    if mask is not None:
        mask_np = mask.cpu().numpy().astype(bool)
        pred_class = pred_class[mask_np]
        target_np = target_np[mask_np]

    if len(target_np) == 0:
        return 0.0

    return f1_score(target_np, pred_class, average=average, zero_division=0)


def load_node_dataset(dataset_name: str, root: str = "data/"):
    """Load a node classification dataset.

    Supports standard citation network datasets from PyG (Cora, CiteSeer,
    PubMed) and other Planetoid datasets.

    Args:
        dataset_name: Name of the dataset (e.g., 'Cora', 'CiteSeer', 'PubMed').
        root: Root directory for dataset storage.

    Returns:
        PyG Data object (single graph for transductive setting).
    """
    try:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=root, name=dataset_name)
        return dataset[0]
    except (ImportError, Exception):
        pass

    # Fallback: try loading from torch_geometric.datasets generically.
    try:
        import importlib
        module = importlib.import_module("torch_geometric.datasets")
        DatasetClass = getattr(module, dataset_name)
        dataset = DatasetClass(root=root)
        return dataset[0]
    except (ImportError, AttributeError, Exception):
        pass

    raise ValueError(
        f"Could not load node dataset '{dataset_name}'. "
        f"Install torch_geometric and ensure the dataset name is valid."
    )


def train_node_epoch(model, data, optimizer, loss_fn, train_mask, device):
    """Run one training epoch for node-level prediction.

    Args:
        model: The GNN model.
        data: Single PyG Data object (the full graph).
        optimizer: PyTorch optimizer.
        loss_fn: Loss function.
        train_mask: Boolean tensor indicating training nodes.
        device: Device.

    Returns:
        Training loss (float).
    """
    model.train()
    data = data.to(device)
    optimizer.zero_grad()

    out = model(data)  # (N, C) for classification

    # Handle target format
    target = data.y
    if target.dim() > 1 and target.shape[-1] > 1:
        # One-hot encoded -> class indices
        target = target.argmax(dim=-1)

    loss = loss_fn(out[train_mask], target[train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()


@torch.no_grad()
def eval_node_epoch(model, data, loss_fn, mask, device):
    """Evaluate model on a subset of nodes.

    Args:
        model: The GNN model.
        data: Single PyG Data object.
        loss_fn: Loss function.
        mask: Boolean tensor indicating evaluation nodes.
        device: Device.

    Returns:
        Dict with loss, accuracy, and f1.
    """
    model.eval()
    data = data.to(device)
    out = model(data)

    target = data.y
    if target.dim() > 1 and target.shape[-1] > 1:
        target = target.argmax(dim=-1)

    loss = loss_fn(out[mask], target[mask]).item()
    acc = compute_accuracy(out, target, mask)
    f1 = compute_f1(out, target, mask)

    return {"loss": loss, "accuracy": acc, "f1": f1}


def save_results(filepath, history_list, time_list, model_name, dataset_name, seed):
    """Save training results summary."""
    lines = []
    lines.append(f"model_name: {model_name}")
    lines.append(f"dataset: {dataset_name}")
    lines.append(f"seed: {seed}")
    lines.append(f"device: {torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'}")
    lines.append(f"torch_version: {torch.__version__}")
    lines.append(f"n_folds: {len(history_list)}")
    lines.append("")

    for i, (hist, elapsed) in enumerate(zip(history_list, time_list)):
        lines.append(f"fold_{i}:")
        lines.append(f"  time: {elapsed}")
        lines.append(f"  epochs: {len(hist.get('train_loss', []))}")
        if 'val_accuracy' in hist and hist['val_accuracy']:
            best_acc = max(hist['val_accuracy'])
            best_epoch = hist['val_accuracy'].index(best_acc) + 1
            lines.append(f"  best_val_accuracy: {best_acc:.4f}")
            lines.append(f"  best_epoch: {best_epoch}")
        if 'val_f1' in hist and hist['val_f1']:
            lines.append(f"  best_val_f1: {max(hist['val_f1']):.4f}")
        if 'val_loss' in hist and hist['val_loss']:
            lines.append(f"  best_val_loss: {min(hist['val_loss']):.6f}")
        lines.append("")

    # Summary across folds
    if len(history_list) > 1:
        val_accs = [max(h['val_accuracy']) for h in history_list
                    if 'val_accuracy' in h and h['val_accuracy']]
        if val_accs:
            lines.append("summary:")
            lines.append(f"  mean_best_val_accuracy: {np.mean(val_accs):.4f} +/- {np.std(val_accs):.4f}")
        val_f1s = [max(h['val_f1']) for h in history_list
                   if 'val_f1' in h and h['val_f1']]
        if val_f1s:
            lines.append(f"  mean_best_val_f1: {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")

    with open(filepath, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Results saved to {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Train node-level GNN model")
    parser.add_argument("--hyper", type=str, default=None,
                        help="Path to hyperparameter file (.json)")
    parser.add_argument("--category", type=str, default="GCN",
                        help="Model category in hyper file")
    parser.add_argument("--model", type=str, default="GCN",
                        help="Model name (GCN, GAT, etc.)")
    parser.add_argument("--dataset", type=str, default="Cora",
                        help="Dataset name (Cora, CiteSeer, PubMed)")
    parser.add_argument("--output", type=str, default="results/",
                        help="Output directory")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device (cpu/cuda/auto)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay")
    parser.add_argument("--hidden_dim", type=int, default=64, help="Hidden dimension")
    parser.add_argument("--depth", type=int, default=2, help="Number of GNN layers")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--n_splits", type=int, default=None,
                        help="Number of KFold splits. If None, use dataset train/val/test masks.")
    parser.add_argument("--task", type=str, default="classification",
                        choices=["classification", "regression"],
                        help="Node-level task type")
    parser.add_argument("--early_stopping", type=int, default=0,
                        help="Early stopping patience. 0 = disabled.")
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

    # Load hyperparameters if provided
    hyper_config = {}
    if args.hyper and os.path.exists(args.hyper):
        from kgcnn_torch.training.hyper import HyperParameter
        hyper = HyperParameter(args.hyper, model_name=args.category)
        hyper_config = hyper.model_config
        fit_config = hyper.fit_config
        args.epochs = fit_config.get("epochs", args.epochs)
        compile_config = hyper.compile_config
        args.lr = compile_config.get("optimizer", {}).get("config", {}).get("lr", args.lr)
        logger.info(f"Loaded hyperparameters from {args.hyper}")

    model_name = args.model
    dataset_name = args.dataset
    logger.info(f"Model: {model_name}, Dataset: {dataset_name}")

    # Load dataset
    logger.info("Loading dataset...")
    data = load_node_dataset(dataset_name, root=os.path.join("data", dataset_name))
    logger.info(f"Dataset loaded: {data.num_nodes} nodes, {data.num_edges} edges")
    logger.info(f"Node features shape: {data.x.shape}")
    logger.info(f"Number of classes: {data.y.max().item() + 1 if args.task == 'classification' else 'N/A'}")

    input_dim = data.x.shape[1] if data.x.dim() > 1 else 1
    if args.task == "classification":
        if data.y.dim() > 1 and data.y.shape[-1] > 1:
            output_dim = data.y.shape[-1]
        else:
            output_dim = int(data.y.max().item()) + 1
    else:
        output_dim = data.y.shape[-1] if data.y.dim() > 1 else 1

    # Output directory
    output_dir = os.path.join(args.output, f"{model_name}_{dataset_name}")
    os.makedirs(output_dir, exist_ok=True)

    # Determine cross-validation splits or use built-in masks
    history_list = []
    time_list = []

    if args.n_splits is not None and args.n_splits > 1:
        # KFold cross-validation over nodes
        kf = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)
        node_indices = np.arange(data.num_nodes)
        splits = list(kf.split(node_indices))
    elif hasattr(data, 'train_mask') and data.train_mask is not None:
        # Use built-in train/val/test masks (standard for citation networks)
        train_mask = data.train_mask
        val_mask = data.val_mask if hasattr(data, 'val_mask') and data.val_mask is not None else None
        test_mask = data.test_mask if hasattr(data, 'test_mask') and data.test_mask is not None else None

        # Handle multiple splits stored in mask columns
        if train_mask.dim() > 1:
            n_mask_splits = train_mask.shape[1]
            splits = []
            for col in range(n_mask_splits):
                train_idx = train_mask[:, col].nonzero(as_tuple=True)[0].numpy()
                val_idx = val_mask[:, col].nonzero(as_tuple=True)[0].numpy() if val_mask is not None else np.array([])
                splits.append((train_idx, val_idx))
        else:
            train_idx = train_mask.nonzero(as_tuple=True)[0].numpy()
            val_idx = val_mask.nonzero(as_tuple=True)[0].numpy() if val_mask is not None else np.array([])
            splits = [(train_idx, val_idx)]
    else:
        # Default 60/20/20 split
        n = data.num_nodes
        indices = np.random.permutation(n)
        train_end = int(0.6 * n)
        val_end = int(0.8 * n)
        splits = [(indices[:train_end], indices[train_end:val_end])]

    logger.info(f"Number of splits: {len(splits)}")

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        logger.info(f"\n{'='*60}")
        logger.info(f"Fold {fold_idx+1}/{len(splits)}")
        logger.info(f"Train nodes: {len(train_idx)}, Val nodes: {len(val_idx)}")
        logger.info(f"{'='*60}")

        # Create masks
        train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        train_mask[train_idx] = True
        val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        if len(val_idx) > 0:
            val_mask[val_idx] = True
        else:
            # If no val set, use 20% of train as validation
            n_val = max(1, int(0.2 * len(train_idx)))
            val_from_train = np.random.choice(train_idx, n_val, replace=False)
            val_mask[val_from_train] = True
            train_mask[val_from_train] = False

        train_mask = train_mask.to(device)
        val_mask = val_mask.to(device)

        # Create model
        if model_name in _MODEL_REGISTRY:
            ModelClass = get_model_class(model_name)
            # Build config from CLI args and hyper file, adapting to model interface.
            _model_kwargs = dict(hyper_config)
            _model_kwargs.setdefault("output_embedding", "node")
            _model_kwargs.setdefault("output_units", [output_dim])
            model = ModelClass(**_model_kwargs)
        else:
            model = NodeGCN(
                input_dim=input_dim,
                hidden_dim=args.hidden_dim,
                output_dim=output_dim,
                depth=args.depth,
                dropout=args.dropout,
            )
        model = model.to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Model parameters: {n_params:,}")

        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        # Loss function
        if args.task == "classification":
            loss_fn = nn.CrossEntropyLoss()
        else:
            loss_fn = nn.MSELoss()

        # Training loop
        history = {
            "train_loss": [], "val_loss": [],
            "val_accuracy": [], "val_f1": [], "lr": []
        }
        best_val_metric = 0.0 if args.task == "classification" else float('inf')
        patience_counter = 0

        start_time = time.time()

        for epoch in range(args.epochs):
            # Train
            train_loss = train_node_epoch(model, data, optimizer, loss_fn,
                                          train_mask, device)
            history["train_loss"].append(train_loss)
            history["lr"].append(optimizer.param_groups[0]['lr'])

            # Validate
            val_results = eval_node_epoch(model, data, loss_fn, val_mask, device)
            history["val_loss"].append(val_results["loss"])
            history["val_accuracy"].append(val_results["accuracy"])
            history["val_f1"].append(val_results["f1"])

            # Early stopping on accuracy (for classification) or loss (for regression)
            if args.early_stopping > 0:
                if args.task == "classification":
                    current_metric = val_results["accuracy"]
                    if current_metric > best_val_metric:
                        best_val_metric = current_metric
                        patience_counter = 0
                        torch.save(model.state_dict(),
                                   os.path.join(output_dir, f"best_model_fold_{fold_idx}.pt"))
                    else:
                        patience_counter += 1
                else:
                    current_metric = val_results["loss"]
                    if current_metric < best_val_metric:
                        best_val_metric = current_metric
                        patience_counter = 0
                        torch.save(model.state_dict(),
                                   os.path.join(output_dir, f"best_model_fold_{fold_idx}.pt"))
                    else:
                        patience_counter += 1

                if patience_counter >= args.early_stopping:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            # Logging
            if (epoch + 1) % max(1, args.epochs // 20) == 0:
                logger.info(
                    f"Epoch {epoch+1}/{args.epochs} - "
                    f"loss: {train_loss:.4f} - "
                    f"val_loss: {val_results['loss']:.4f} - "
                    f"val_acc: {val_results['accuracy']:.4f} - "
                    f"val_f1: {val_results['f1']:.4f}"
                )

        elapsed = str(timedelta(seconds=time.time() - start_time))
        logger.info(f"Fold {fold_idx+1} training time: {elapsed}")

        # Load best model checkpoint if early stopping was used
        best_ckpt = os.path.join(output_dir, f"best_model_fold_{fold_idx}.pt")
        if os.path.exists(best_ckpt) and args.early_stopping > 0:
            model.load_state_dict(torch.load(best_ckpt, weights_only=True))
            logger.info("Loaded best model checkpoint")

        # Save history and model
        history_list.append(history)
        time_list.append(elapsed)

        with open(os.path.join(output_dir, f"history_fold_{fold_idx}.json"), 'w') as f:
            json.dump({k: [float(v) for v in vs] for k, vs in history.items()}, f, indent=2)

        torch.save(model.state_dict(),
                   os.path.join(output_dir, f"model_fold_{fold_idx}.pt"))

        # Log best results
        if history['val_accuracy']:
            best_acc = max(history['val_accuracy'])
            logger.info(f"Best val_accuracy: {best_acc:.4f}")
        if history['val_f1']:
            best_f1 = max(history['val_f1'])
            logger.info(f"Best val_f1: {best_f1:.4f}")

    # Save overall results
    if history_list:
        save_results(
            os.path.join(output_dir, "score.yaml"),
            history_list, time_list,
            model_name=model_name, dataset_name=dataset_name, seed=args.seed
        )

    logger.info(f"\nTraining complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
