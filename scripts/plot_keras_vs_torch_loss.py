"""Plot per-epoch val_mae comparison: Keras vs Torch for all 26 models."""
import json
import os
import re
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paths
KERAS_LOG_DIR = "/home/yuanbai/Downloads/MLIPs/gcnn_keras-master/results/matpes2k_26/logs"
TORCH_RESULT_DIR = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch/tmp_matpes_2k/fair_match_final/results"
OUTPUT_DIR = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch/tmp_matpes_2k/fair_match_final/plots"

MODEL_LIST = [
    "SchNet", "PAiNN", "DimeNetPP", "GCN", "GAT", "GATv2", "GIN", "EGNN",
    "DMPNN", "GraphSAGE", "Megnet", "AttentiveFP", "CGCNN", "NMPN", "INorp",
    "MEGAN", "RGCN", "GNNFilm", "rGIN", "MXMNet", "MoGAT", "CMPNN", "DGIN",
    "HamNet", "HDNNP2nd", "MAT",
]


def parse_keras_log(log_path):
    """Parse Keras training log to extract per-fold, per-epoch metrics.

    Returns list of dicts (one per fold), each with keys:
      train_loss, val_loss, val_mae (= val_scaled_mean_absolute_error if available)
    """
    if not os.path.exists(log_path):
        return []

    with open(log_path, "r") as f:
        text = f.read()

    # Split by fold markers
    # Look for "Fold X/Y" or "KFold ... split X"
    fold_sections = re.split(r"(?:Info: Running fold|KFold.*?split)\s*\d+", text)
    if len(fold_sections) <= 1:
        # No fold markers, treat whole file as one fold
        fold_sections = [text]
    else:
        # First section is pre-fold header, skip it
        fold_sections = fold_sections[1:]

    all_folds = []
    for section in fold_sections:
        train_loss = []
        val_loss = []
        val_mae = []

        # Parse epoch lines
        for line in section.split("\n"):
            if "loss:" not in line or "val_loss:" not in line:
                continue

            # Extract metrics
            tl = re.search(r"(?<!\w)loss:\s*([0-9.e+-]+)", line)
            vl = re.search(r"val_loss:\s*([0-9.e+-]+)", line)

            # Try val_scaled_mean_absolute_error first (transform_dataset path)
            vm = re.search(r"val_scaled_mean_absolute_error:\s*([0-9.e+-]+)", line)
            if not vm:
                # For set_scale path, val_loss IS the MAE in original scale
                # (since loss="mean_absolute_error" and model has set_scale)
                vm = vl

            if tl:
                train_loss.append(float(tl.group(1)))
            if vl:
                val_loss.append(float(vl.group(1)))
            if vm:
                val_mae.append(float(vm.group(1)))

        if train_loss:
            all_folds.append({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_mae": val_mae,
            })

    return all_folds


def load_torch_history(model_name):
    """Load Torch per-fold history JSON files.

    Returns list of dicts with keys: train_loss, val_loss, val_mae, val_rmse
    """
    model_dir = os.path.join(TORCH_RESULT_DIR, model_name)
    folds = []
    for i in range(10):  # up to 10 folds
        hist_path = os.path.join(model_dir, f"history_fold_{i}.json")
        if not os.path.exists(hist_path):
            break
        with open(hist_path) as f:
            folds.append(json.load(f))
    return folds


def plot_model(model_name, keras_folds, torch_folds, output_path):
    """Plot val_mae comparison for one model."""
    n_folds = max(len(keras_folds), len(torch_folds))
    if n_folds == 0:
        return

    fig, axes = plt.subplots(1, n_folds, figsize=(6 * n_folds, 4.5), squeeze=False)
    fig.suptitle(f"{model_name} — Validation MAE (meV/atom)", fontsize=14, fontweight="bold")

    for fold_idx in range(n_folds):
        ax = axes[0, fold_idx]
        ax.set_title(f"Fold {fold_idx}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Val MAE (meV/atom)")

        # Keras fold
        if fold_idx < len(keras_folds):
            kf = keras_folds[fold_idx]
            val_mae_k = kf.get("val_mae", [])
            if val_mae_k:
                epochs_k = list(range(1, len(val_mae_k) + 1))
                ax.plot(epochs_k, val_mae_k, "o-", color="#2196F3", label="Keras",
                        linewidth=2, markersize=4)

        # Torch fold
        if fold_idx < len(torch_folds):
            tf = torch_folds[fold_idx]
            val_mae_t = tf.get("val_mae", [])
            if val_mae_t:
                epochs_t = list(range(1, len(val_mae_t) + 1))
                ax.plot(epochs_t, val_mae_t, "s--", color="#FF5722", label="Torch",
                        linewidth=2, markersize=4)

        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)

        # Set reasonable y-axis limits (clip extreme values)
        all_vals = []
        if fold_idx < len(keras_folds):
            all_vals.extend(keras_folds[fold_idx].get("val_mae", []))
        if fold_idx < len(torch_folds):
            all_vals.extend(torch_folds[fold_idx].get("val_mae", []))
        if all_vals:
            # Filter out NaN/Inf
            finite_vals = [v for v in all_vals if math.isfinite(v)]
            if finite_vals:
                ymin = min(finite_vals) * 0.8
                ymax = max(finite_vals) * 1.2
                # Clip extremely large values for readability
                if ymax > 50 and min(finite_vals) < 10:
                    clipped = [v for v in finite_vals if v < 50]
                    if clipped:
                        ymax = max(clipped) * 1.3
                if math.isfinite(ymin) and math.isfinite(ymax) and ymax > ymin:
                    ax.set_ylim(bottom=max(0, ymin), top=ymax)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_all_models_grid(all_data, output_path):
    """Create a single large grid figure with all models."""
    n_models = len(all_data)
    n_cols = 4
    n_rows = math.ceil(n_models / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    fig.suptitle("Keras vs Torch — Validation MAE per Epoch (meV/atom)\nMatPES2k, 2-fold CV, 10 epochs",
                 fontsize=16, fontweight="bold", y=1.01)

    for idx, (model, keras_folds, torch_folds) in enumerate(all_data):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]

        has_data = False

        # Keras: use last fold
        if keras_folds:
            kf = keras_folds[-1]
            val_mae_k = kf.get("val_mae", [])
            if val_mae_k:
                finite_k = [v if math.isfinite(v) else None for v in val_mae_k]
                epochs_k = list(range(1, len(finite_k) + 1))
                ax.plot(epochs_k, finite_k, "o-", color="#2196F3", label="Keras",
                        linewidth=1.8, markersize=3.5)
                has_data = True

        # Torch: use last fold
        if torch_folds:
            tf = torch_folds[-1]
            val_mae_t = tf.get("val_mae", [])
            if val_mae_t:
                finite_t = [v if math.isfinite(v) else None for v in val_mae_t]
                epochs_t = list(range(1, len(finite_t) + 1))
                ax.plot(epochs_t, finite_t, "s--", color="#FF5722", label="Torch",
                        linewidth=1.8, markersize=3.5)
                has_data = True

        ax.set_title(model, fontsize=11, fontweight="bold")
        ax.set_xlabel("Epoch", fontsize=9)
        ax.set_ylabel("Val MAE", fontsize=9)
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

        # Set y limits
        all_vals = []
        if keras_folds:
            all_vals.extend(v for v in keras_folds[-1].get("val_mae", []) if math.isfinite(v))
        if torch_folds:
            all_vals.extend(v for v in torch_folds[-1].get("val_mae", []) if math.isfinite(v))
        if all_vals:
            ymin = min(all_vals) * 0.7
            ymax = max(all_vals) * 1.3
            # Clip extreme outliers
            if ymax > 100 and min(all_vals) < 20:
                clipped = [v for v in all_vals if v < 100]
                if clipped:
                    ymax = max(clipped) * 1.3
            if math.isfinite(ymin) and math.isfinite(ymax) and ymax > ymin:
                ax.set_ylim(bottom=max(0, ymin), top=ymax)

    # Hide unused subplots
    for idx in range(n_models, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col] if n_rows > 1 else axes[col]
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Grid plot saved: {output_path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_data = []
    success = 0
    skip = 0
    for model in MODEL_LIST:
        keras_log = os.path.join(KERAS_LOG_DIR, f"{model}.log")
        keras_folds = parse_keras_log(keras_log)
        torch_folds = load_torch_history(model)

        if not keras_folds and not torch_folds:
            print(f"SKIP {model}: no data")
            skip += 1
            continue

        all_data.append((model, keras_folds, torch_folds))

        # Individual plot
        out_path = os.path.join(OUTPUT_DIR, f"{model}_val_mae.png")
        plot_model(model, keras_folds, torch_folds, out_path)

        # Print summary
        k_final = keras_folds[-1]["val_mae"][-1] if keras_folds and keras_folds[-1].get("val_mae") else None
        t_final = torch_folds[-1]["val_mae"][-1] if torch_folds and torch_folds[-1].get("val_mae") else None
        k_str = f"{k_final:.3f}" if k_final else "N/A"
        t_str = f"{t_final:.3f}" if t_final else "N/A"
        print(f"{model}: keras_final={k_str}, torch_final={t_str} → {out_path}")
        success += 1

    # Combined grid plot
    if all_data:
        grid_path = os.path.join(OUTPUT_DIR, "ALL_models_val_mae_grid.png")
        plot_all_models_grid(all_data, grid_path)

    print(f"\nDone: {success} individual plots + 1 grid plot saved, {skip} skipped")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
