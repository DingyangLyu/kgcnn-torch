#!/usr/bin/env python3
"""Run all 28 model-level alignment scripts and generate comparison plots.

Executes each align_*_model.py script, parses output metrics (MAE, RMSE, MAX),
and creates a comprehensive alignment summary plot.
"""
import os
import sys
import re
import subprocess
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

SCRIPT_DIR = Path(__file__).parent
PYTHON = sys.executable
ENV = {
    **os.environ,
    "KERAS_BACKEND": "torch",
    "CUDA_VISIBLE_DEVICES": "",  # Force CPU for reproducible alignment
}

# All model alignment scripts
ALIGN_SCRIPTS = sorted(SCRIPT_DIR.glob("align_*_model.py"))


@dataclass
class AlignResult:
    model_name: str
    script: str
    status: str  # "PASS" or "FAIL"
    mae: float = 0.0
    rmse: float = 0.0
    max_abs: float = 0.0
    error: str = ""


def parse_metrics(output: str) -> dict:
    """Parse MAE, RMSE, MAX from alignment script output."""
    # Pattern: | MAE=3.725290e-09 | RMSE=1.053671e-08 | MAX=2.980232e-08
    m = re.search(r"MAE=([\d.e+-]+)\s*\|\s*RMSE=([\d.e+-]+)\s*\|\s*MAX=([\d.e+-]+)", output)
    if m:
        return {
            "mae": float(m.group(1)),
            "rmse": float(m.group(2)),
            "max_abs": float(m.group(3)),
        }
    return {}


def extract_model_name(script_path: Path) -> str:
    """Extract human-readable model name from script filename."""
    name = script_path.stem  # align_gcn_model -> align_gcn_model
    name = name.replace("align_", "").replace("_model", "")
    # Capitalize well-known abbreviations
    known = {
        "gcn": "GCN", "gat": "GAT", "gatv2": "GATv2", "gin": "GIN",
        "rgin": "rGIN", "rgcn": "RGCN", "graphsage": "GraphSAGE",
        "gnnfilm": "GNNFilm", "schnet": "SchNet", "painn": "PAiNN",
        "dimenetpp": "DimeNet++", "egnn": "EGNN", "mxmnet": "MXMNet",
        "megnet": "MEGNet", "cgcnn": "CGCNN", "dmpnn": "DMPNN",
        "cmpnn": "CMPNN", "dgin": "DGIN", "nmpn": "NMPN",
        "attentivefp": "AttentiveFP", "mogat": "MoGAT", "inorp": "INorp",
        "hamnet": "HamNet", "mat": "MAT", "megan": "MEGAN",
        "hdnnp2nd": "HDNNP2nd", "hdnnp2nd_behler": "HDNNP2nd-Behler",
        "hdnnp2nd_atomwise": "HDNNP2nd-AtomWise",
    }
    return known.get(name, name.upper())


def run_alignment(script_path: Path) -> AlignResult:
    """Run a single alignment script and return results."""
    model_name = extract_model_name(script_path)
    try:
        result = subprocess.run(
            [PYTHON, str(script_path)],
            cwd=str(SCRIPT_DIR),
            env=ENV,
            capture_output=True,
            text=True,
            timeout=120,
        )
        output = result.stdout + result.stderr
        metrics = parse_metrics(output)

        if result.returncode == 0 and metrics:
            return AlignResult(
                model_name=model_name,
                script=script_path.name,
                status="PASS",
                mae=metrics["mae"],
                rmse=metrics["rmse"],
                max_abs=metrics["max_abs"],
            )
        elif result.returncode != 0:
            # Extract error message
            err_lines = result.stderr.strip().split("\n")
            err_msg = err_lines[-1] if err_lines else "Unknown error"
            return AlignResult(
                model_name=model_name,
                script=script_path.name,
                status="FAIL",
                mae=metrics.get("mae", 0),
                rmse=metrics.get("rmse", 0),
                max_abs=metrics.get("max_abs", 0),
                error=err_msg[:200],
            )
        else:
            return AlignResult(
                model_name=model_name,
                script=script_path.name,
                status="PASS",
                error="No metrics parsed but exit 0",
            )
    except subprocess.TimeoutExpired:
        return AlignResult(
            model_name=model_name,
            script=script_path.name,
            status="FAIL",
            error="Timeout (120s)",
        )
    except Exception as e:
        return AlignResult(
            model_name=model_name,
            script=script_path.name,
            status="FAIL",
            error=str(e)[:200],
        )


def plot_results(results: List[AlignResult], output_path: str):
    """Generate alignment summary plot."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    # Sort by MAE (passed first, then failed)
    passed = [r for r in results if r.status == "PASS"]
    failed = [r for r in results if r.status == "FAIL"]
    passed.sort(key=lambda r: r.mae)

    all_sorted = passed + failed
    names = [r.model_name for r in all_sorted]
    maes = [r.mae for r in all_sorted]
    rmses = [r.rmse for r in all_sorted]
    max_abs = [r.max_abs for r in all_sorted]
    colors = ["#2ecc71" if r.status == "PASS" else "#e74c3c" for r in all_sorted]

    n = len(all_sorted)

    fig, axes = plt.subplots(2, 1, figsize=(max(14, n * 0.55), 12), gridspec_kw={"height_ratios": [2, 1]})

    # --- Top panel: MAE and MAX_ABS bar chart (log scale) ---
    ax1 = axes[0]
    x = np.arange(n)
    bar_width = 0.35

    bars_mae = ax1.bar(x - bar_width / 2, [max(v, 1e-15) for v in maes],
                       bar_width, label="MAE", color=colors, alpha=0.85, edgecolor="white")
    bars_max = ax1.bar(x + bar_width / 2, [max(v, 1e-15) for v in max_abs],
                       bar_width, label="MAX ABS", color=colors, alpha=0.5, edgecolor="white")

    ax1.set_yscale("log")
    ax1.set_ylabel("Error (log scale)", fontsize=12)
    ax1.set_title("Cross-Framework Alignment: Torch vs Keras (Weight Transfer + Forward Pass)",
                  fontsize=14, fontweight="bold", pad=15)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=45, ha="right", fontsize=9)

    # Threshold lines
    ax1.axhline(y=1e-5, color="orange", linestyle="--", linewidth=1, alpha=0.7, label="Layer threshold (1e-5)")
    ax1.axhline(y=1e-4, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Model threshold (1e-4)")

    # Legend
    legend_elements = [
        Patch(facecolor="#2ecc71", alpha=0.85, label="MAE (PASS)"),
        Patch(facecolor="#2ecc71", alpha=0.5, label="MAX ABS (PASS)"),
        Patch(facecolor="#e74c3c", alpha=0.85, label="MAE (FAIL)"),
        plt.Line2D([0], [0], color="orange", linestyle="--", label="Layer threshold (1e-5)"),
        plt.Line2D([0], [0], color="red", linestyle="--", label="Model threshold (1e-4)"),
    ]
    ax1.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax1.grid(axis="y", alpha=0.3)

    # --- Bottom panel: Summary table ---
    ax2 = axes[1]
    ax2.axis("off")

    n_pass = len(passed)
    n_fail = len(failed)
    summary_text = f"Summary: {n_pass}/{n} models PASS alignment"
    if n_fail > 0:
        summary_text += f" | {n_fail} FAIL"

    ax2.text(0.5, 0.95, summary_text, transform=ax2.transAxes,
             fontsize=14, fontweight="bold", ha="center", va="top",
             color="#2ecc71" if n_fail == 0 else "#e74c3c")

    # Table with details
    if passed:
        mae_values = [r.mae for r in passed]
        table_text = (
            f"Passed Models:\n"
            f"  MAE range: [{min(mae_values):.2e}, {max(mae_values):.2e}]\n"
            f"  Median MAE: {np.median(mae_values):.2e}\n"
            f"  All below model threshold (1e-4): {'Yes' if all(v < 1e-4 for v in mae_values) else 'No'}"
        )
        ax2.text(0.05, 0.75, table_text, transform=ax2.transAxes,
                 fontsize=10, fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#d5f5e3", alpha=0.8))

    if failed:
        fail_text = "Failed Models:\n"
        for r in failed:
            fail_text += f"  {r.model_name}: {r.error[:80]}\n"
        ax2.text(0.55, 0.75, fail_text.strip(), transform=ax2.transAxes,
                 fontsize=9, fontfamily="monospace", va="top",
                 bbox=dict(boxstyle="round,pad=0.5", facecolor="#fadbd8", alpha=0.8))

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {output_path}")
    plt.close()


def plot_training_alignment(results: List[AlignResult], output_path: str):
    """Generate a per-model alignment detail plot with horizontal bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    passed = [r for r in results if r.status == "PASS"]
    if not passed:
        print("No passed models to plot training alignment.")
        return

    # Sort by MAE descending for visual clarity
    passed.sort(key=lambda r: r.mae, reverse=True)

    n = len(passed)
    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.4)))

    y_pos = np.arange(n)
    bar_height = 0.25

    # Replace zeros with a small value for log scale display
    def safe_log(v):
        return max(v, 1e-15)

    maes = [safe_log(r.mae) for r in passed]
    rmses = [safe_log(r.rmse) for r in passed]
    maxes = [safe_log(r.max_abs) for r in passed]
    names = [r.model_name for r in passed]

    ax.barh(y_pos - bar_height, maxes, bar_height, label="MAX ABS",
            color="#e67e22", alpha=0.8, edgecolor="white")
    ax.barh(y_pos, rmses, bar_height, label="RMSE",
            color="#9b59b6", alpha=0.8, edgecolor="white")
    ax.barh(y_pos + bar_height, maes, bar_height, label="MAE",
            color="#3498db", alpha=0.8, edgecolor="white")

    ax.set_xscale("log")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()

    # Threshold lines
    ax.axvline(x=1e-5, color="orange", linestyle="--", linewidth=1.2, alpha=0.7, label="Layer threshold (1e-5)")
    ax.axvline(x=1e-4, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Model threshold (1e-4)")

    ax.set_xlabel("Error (log scale)", fontsize=11)
    ax.set_title("Per-Model Alignment Error Detail\n(Torch → Keras weight transfer, lower is better)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # Annotate exact values for top-3 MAE models
    for i in range(min(3, n)):
        r = passed[i]
        if r.mae > 0:
            ax.text(safe_log(r.mae) * 3, y_pos[i] + bar_height,
                    f"{r.mae:.1e}", va="center", fontsize=7, color="#2c3e50")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Per-model plot saved to: {output_path}")
    plt.close()


def main():
    print(f"Found {len(ALIGN_SCRIPTS)} alignment scripts")
    print(f"Python: {PYTHON}")
    print(f"Using CPU for alignment (CUDA_VISIBLE_DEVICES='')")
    print("=" * 70)

    results = []
    for i, script in enumerate(ALIGN_SCRIPTS, 1):
        model_name = extract_model_name(script)
        print(f"[{i:2d}/{len(ALIGN_SCRIPTS)}] Running {model_name:20s} ... ", end="", flush=True)
        result = run_alignment(script)
        results.append(result)
        if result.status == "PASS":
            print(f"PASS  (MAE={result.mae:.2e}, MAX={result.max_abs:.2e})")
        else:
            print(f"FAIL  ({result.error[:60]})")

    print("=" * 70)
    n_pass = sum(1 for r in results if r.status == "PASS")
    n_fail = sum(1 for r in results if r.status == "FAIL")
    print(f"Total: {len(results)} | PASS: {n_pass} | FAIL: {n_fail}")

    # Save raw results as JSON
    json_path = str(SCRIPT_DIR / "alignment_results.json")
    with open(json_path, "w") as f:
        json.dump([{
            "model": r.model_name, "script": r.script, "status": r.status,
            "mae": r.mae, "rmse": r.rmse, "max_abs": r.max_abs, "error": r.error,
        } for r in results], f, indent=2)
    print(f"Results saved to: {json_path}")

    # Generate plots
    plot_path = str(SCRIPT_DIR / "alignment_summary.png")
    plot_results(results, plot_path)

    detail_path = str(SCRIPT_DIR / "alignment_per_model.png")
    plot_training_alignment(results, detail_path)


if __name__ == "__main__":
    main()
