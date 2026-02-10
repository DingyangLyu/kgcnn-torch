import argparse
import copy
import importlib.util
import inspect
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime


def load_py_hyper(path):
    spec = importlib.util.spec_from_file_location("hyper_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hyper


def load_train_graph_helpers(torch_root):
    if torch_root not in sys.path:
        sys.path.insert(0, torch_root)
    path = os.path.join(torch_root, "training_scripts", "train_graph.py")
    spec = importlib.util.spec_from_file_location("train_graph_module", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.get_model_class, module.translate_model_config


def parse_activation(value):
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        cfg = value.get("config")
        if isinstance(cfg, str):
            if "leaky_relu2" in cfg:
                return "leaky_relu2"
            if "shifted_softplus" in cfg:
                return "shifted_softplus"
            if "softplus2" in cfg:
                return "softplus2"
            return cfg.split(">")[-1]
    return None


def sanitize_activation_value(key, value):
    if "activation" not in key.lower():
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return parse_activation(value)
    if isinstance(value, list):
        out = []
        for v in value:
            if isinstance(v, str):
                out.append(v)
            elif isinstance(v, dict):
                p = parse_activation(v)
                if p is not None:
                    out.append(p)
        return out if out else None
    return None


def keras_lr_from_optimizer_config(opt_cfg):
    lr = opt_cfg.get("learning_rate", 1e-3)
    if isinstance(lr, (int, float)):
        return float(lr), None
    if isinstance(lr, dict):
        schedule = {
            "class_name": lr.get("class_name", ""),
            "config": lr.get("config", {}),
        }
        init_lr = schedule["config"].get("initial_learning_rate", 1e-3)
        return float(init_lr), schedule
    return 1e-3, None


def extract_cv_from_keras_cfg(kcfg):
    methods = kcfg.get("dataset", {}).get("methods", [])
    for m in methods:
        if "set_train_test_indices_k_fold" in m:
            c = m["set_train_test_indices_k_fold"]
            return {
                "n_splits": int(c.get("n_splits", 5)),
                "shuffle": bool(c.get("shuffle", True)),
                "random_state": int(c.get("random_state", 42)),
            }
    return {"n_splits": 5, "shuffle": True, "random_state": 42}


def extract_scheduler_from_keras(kcfg, epochs, batch_size, dataset_size, n_splits):
    tr = kcfg.get("training", {})
    fit_cfg = tr.get("fit", {})
    callbacks = fit_cfg.get("callbacks", [])
    for cb in callbacks:
        name = cb.get("class_name", "")
        cfg = cb.get("config", {})
        if "LinearLearningRateScheduler" in name:
            lr_start = float(cfg.get("learning_rate_start", 1e-3))
            lr_stop = float(cfg.get("learning_rate_stop", lr_start * 0.01))
            total_epochs = int(cfg.get("epo", epochs))
            if total_epochs <= 0:
                total_epochs = epochs
            lr_final_factor = max(min(lr_stop / max(lr_start, 1e-12), 1.0), 1e-8)
            sched = {
                "class_name": "polynomial_decay",
                "total_epochs": total_epochs,
                "lr_final_factor": lr_final_factor,
                "power": 1.0,
            }
            return sched, lr_start

    opt_cfg = tr.get("compile", {}).get("optimizer", {}).get("config", {})
    _, lr_schedule = keras_lr_from_optimizer_config(opt_cfg)
    if lr_schedule and lr_schedule.get("class_name") == "ExponentialDecay":
        scfg = lr_schedule.get("config", {})
        decay_rate = float(scfg.get("decay_rate", 0.5))
        decay_steps = float(scfg.get("decay_steps", 1600))
        train_size = int(dataset_size * (n_splits - 1) / n_splits)
        steps_per_epoch = max(1, math.ceil(train_size / max(batch_size, 1)))
        gamma = decay_rate ** (steps_per_epoch / max(decay_steps, 1.0))
        sched = {"class_name": "exponential", "gamma": gamma}
        return sched, None
    return None, None


def extract_output_mlp_args(keras_model_cfg):
    out = {}
    mlp = keras_model_cfg.get("output_mlp")
    if not isinstance(mlp, dict):
        return out
    units = mlp.get("units")
    activation = mlp.get("activation")
    if isinstance(units, int):
        if units > 1:
            out["output_units"] = [units]
        else:
            out["output_units"] = []
        out["num_targets"] = 1
    elif isinstance(units, list) and units:
        if len(units) >= 2:
            out["output_units"] = list(units[:-1])
            out["num_targets"] = int(units[-1])
        else:
            out["output_units"] = []
            out["num_targets"] = int(units[0])
    if isinstance(activation, str):
        out["output_final_activation"] = activation
    elif isinstance(activation, list) and activation:
        first_act = activation[0]
        last_act = activation[-1]
        if isinstance(first_act, dict):
            first_act = parse_activation(first_act)
        if isinstance(last_act, dict):
            last_act = parse_activation(last_act)
        if first_act is not None:
            out["output_activation"] = first_act
        if last_act is not None:
            out["output_final_activation"] = last_act
    elif isinstance(activation, dict):
        parsed = parse_activation(activation)
        if parsed is not None:
            out["output_final_activation"] = parsed
    return out


def build_model_config(model_name, keras_model_cfg, get_model_class, translate_model_config):
    ModelClass = get_model_class(model_name)
    sig = inspect.signature(ModelClass.__init__)
    accepted = set(sig.parameters.keys()) - {"self", "args", "kwargs"}

    src = copy.deepcopy(keras_model_cfg)
    mapped = translate_model_config(model_name, src)
    out = {"model_name": model_name}

    for k, v in mapped.items():
        if k in accepted:
            v2 = sanitize_activation_value(k, v)
            if v2 is not None:
                out[k] = v2

    if "depth" in src and "depth" in accepted:
        out["depth"] = src["depth"]
    if "depthDMPNN" in src and "depth_dmpnn" in accepted:
        out["depth_dmpnn"] = src["depthDMPNN"]
    if "depthGIN" in src and "depth_gin" in accepted:
        out["depth_gin"] = src["depthGIN"]

    # Common nested configs from keras hyper.
    if "gcn_args" in src:
        gcn = src["gcn_args"]
        if "units" in gcn and "gcn_units" in accepted:
            out["gcn_units"] = gcn["units"]
        if "activation" in gcn and "gcn_activation" in accepted:
            out["gcn_activation"] = gcn["activation"]

    if "edge_initialize" in src and isinstance(src["edge_initialize"], dict):
        ei = src["edge_initialize"]
        if "units" in ei and "units" in accepted:
            out["units"] = ei["units"]
        if "activation" in ei and "message_activation" in accepted:
            out["message_activation"] = ei["activation"]

    if "node_dense" in src and isinstance(src["node_dense"], dict):
        nd = src["node_dense"]
        if "activation" in nd and "node_activation" in accepted:
            out["node_activation"] = nd["activation"]

    if "attention_args" in src:
        att = src["attention_args"]
        if "units" in att and "attention_units" in accepted:
            out["attention_units"] = att["units"]
        if "use_edge_features" in att and "use_edge_features" in accepted:
            out["use_edge_features"] = bool(att["use_edge_features"])
        act = parse_activation(att.get("activation"))
        if act and "attention_activation" in accepted:
            out["attention_activation"] = act

    for pool_key in ("pooling_nodes_args", "pooling_args"):
        if pool_key in src and isinstance(src[pool_key], dict):
            pm = src[pool_key].get("pooling_method")
            if pm:
                if "node_pooling" in accepted:
                    out["node_pooling"] = pm
                elif "readout" in accepted:
                    out["readout"] = pm

    if "gin_mlp" in src and isinstance(src["gin_mlp"], dict):
        g = src["gin_mlp"]
        if "units" in g and "gin_mlp_units" in accepted:
            out["gin_mlp_units"] = g["units"]
        if "activation" in g and "gin_mlp_activation" in accepted:
            act = g["activation"][0] if isinstance(g["activation"], list) and g["activation"] else g["activation"]
            out["gin_mlp_activation"] = act
        if "use_normalization" in g and "gin_mlp_use_normalization" in accepted:
            out["gin_mlp_use_normalization"] = bool(g["use_normalization"])
        if "normalization_technique" in g and "gin_mlp_normalization_technique" in accepted:
            out["gin_mlp_normalization_technique"] = g["normalization_technique"]

    if "last_mlp" in src and isinstance(src["last_mlp"], dict):
        lm = src["last_mlp"]
        if "units" in lm and "last_mlp_units" in accepted:
            units = lm["units"]
            out["last_mlp_units"] = units[:-1] if isinstance(units, list) and len(units) > 1 else units
        if "activation" in lm and "last_mlp_activation" in accepted:
            act = lm["activation"][0] if isinstance(lm["activation"], list) and lm["activation"] else lm["activation"]
            if isinstance(act, dict):
                act = parse_activation(act)
            if act is not None:
                out["last_mlp_activation"] = act

    if "dropout" in src:
        d = src["dropout"]
        rate = d.get("rate", 0.0) if isinstance(d, dict) else d
        if "dropout_rate" in accepted:
            out["dropout_rate"] = float(rate)

    if "input_node_embedding" in src and isinstance(src["input_node_embedding"], dict):
        emb = src["input_node_embedding"]
        if "output_dim" in emb and "node_dim" in accepted:
            out["node_dim"] = int(emb["output_dim"])
        if "input_dim" in emb and "num_embeddings" in accepted:
            out["num_embeddings"] = int(emb["input_dim"])
        if "use_node_embedding" in accepted:
            out["use_node_embedding"] = True

    # Do not force edge embedding from Keras "input_edge_embedding":
    # torch models in this repo generally expect continuous edge_attr tensors
    # and infer edge_dim from data; forcing embedding on float edge features can
    # create rank mismatches (e.g., DMPNN).

    out.update(extract_output_mlp_args(src))

    if model_name == "MEGAN":
        k = src.get("importance_channels")
        if k is not None and "num_heads" in accepted and "num_heads" not in out:
            out["num_heads"] = int(k)

    if "num_targets" not in out:
        if "num_targets" in accepted:
            out["num_targets"] = 1
        elif "output_dim" in accepted:
            out["output_dim"] = 1

    # Keep only constructor-supported args.
    out = {k: v for k, v in out.items() if k in accepted or k == "model_name"}
    return out


def parse_torch_score(score_path):
    if not os.path.exists(score_path):
        return None
    txt = open(score_path, "r", encoding="utf-8").read()
    m = re.search(r"best_val_mae:\s*([0-9eE+\-.]+)", txt)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    m2 = re.search(r"mean_best_val_mae:\s*([0-9eE+\-.]+)", txt)
    if m2:
        try:
            return float(m2.group(1))
        except ValueError:
            return None
    return None


def dataset_size_from_pickle(torch_root, pkl_path):
    try:
        sys.path.insert(0, torch_root)
        from kgcnn_torch.data.base import MemoryGraphList
        gl = MemoryGraphList()
        gl.load(pkl_path)
        return len(gl)
    except Exception:
        return 2000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--max-batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--timeout-sec", type=int, default=1800)
    ap.add_argument("--models", nargs="+", default=None)
    ap.add_argument("--dataset-path", type=str,
                    default="/home/yuanbai/Downloads/MLIPs/kgcnn-torch/tmp_matpes_2k/matpes_pbe_2k.pkl")
    ap.add_argument("--output-dir", type=str,
                    default="/home/yuanbai/Downloads/MLIPs/kgcnn-torch/tmp_matpes_2k/apple2apple")
    args = ap.parse_args()

    torch_root = "/home/yuanbai/Downloads/MLIPs/kgcnn-torch"
    keras_root = "/home/yuanbai/Downloads/MLIPs/gcnn_keras-master"
    get_model_class, translate_model_config = load_train_graph_helpers(torch_root)

    os.makedirs(args.output_dir, exist_ok=True)
    hyper_out_dir = os.path.join(args.output_dir, "hyper")
    logs_dir = os.path.join(args.output_dir, "logs")
    results_dir = os.path.join(args.output_dir, "results")
    os.makedirs(hyper_out_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    keras_esol = load_py_hyper(os.path.join(keras_root, "training/hyper/hyper_esol.py"))
    keras_mp = load_py_hyper(os.path.join(keras_root, "training/hyper/hyper_mp_e_form.py"))
    dataset_size = dataset_size_from_pickle(torch_root, args.dataset_path)

    model_list = [
        "SchNet", "PAiNN", "DimeNetPP", "GCN", "GAT", "GATv2", "GIN", "EGNN", "DMPNN", "GraphSAGE",
        "Megnet", "AttentiveFP", "CGCNN", "NMPN", "INorp", "MEGAN", "RGCN", "GNNFilm", "rGIN", "MXMNet",
        "MoGAT", "CMPNN", "DGIN", "HamNet", "HDNNP2nd", "MAT"
    ]
    if args.models:
        model_set = set(args.models)
        model_list = [m for m in model_list if m in model_set]

    keras_key = {"SchNet": "Schnet", "CGCNN": "CGCNN.make_crystal_model"}
    rows = []
    for model in model_list:
        kkey = keras_key.get(model, model)

        if kkey in keras_esol:
            kcfg = keras_esol[kkey]
        elif kkey in keras_mp:
            kcfg = keras_mp[kkey]
        else:
            rows.append({"model": model, "ok": False, "error": f"missing_keras_hyper:{kkey}"})
            continue

        k_model_cfg = kcfg.get("model", {}).get("config", {})
        model_cfg = build_model_config(model, k_model_cfg, get_model_class, translate_model_config)

        fit_cfg = copy.deepcopy(kcfg.get("training", {}).get("fit", {}))
        batch_size = int(fit_cfg.get("batch_size", 32))
        batch_size = min(batch_size, args.max_batch_size)
        epochs = args.epochs

        cv_cfg = extract_cv_from_keras_cfg(kcfg)
        n_splits = int(cv_cfg.get("n_splits", 5))

        opt = kcfg.get("training", {}).get("compile", {}).get("optimizer", {})
        opt_name = opt.get("class_name", "Adam")
        opt_cfg = copy.deepcopy(opt.get("config", {}))
        lr, _ = keras_lr_from_optimizer_config(opt_cfg)

        sched_cfg, override_lr = extract_scheduler_from_keras(
            kcfg=kcfg,
            epochs=epochs,
            batch_size=batch_size,
            dataset_size=dataset_size,
            n_splits=n_splits
        )
        if override_lr is not None:
            lr = override_lr

        loss_cfg = kcfg.get("training", {}).get("compile", {}).get("loss", "mean_absolute_error")
        if isinstance(loss_cfg, dict):
            loss_name = str(loss_cfg.get("class_name", "mean_absolute_error")).lower()
        else:
            loss_name = str(loss_cfg).lower()
        if "mse" in loss_name or "mean_squared_error" in loss_name:
            torch_loss = "mse"
        else:
            torch_loss = "mae"

        compile_cfg = {
            "optimizer": {"class_name": opt_name, "config": {"lr": float(lr)}},
            "loss": torch_loss,
        }
        if "weight_decay" in opt_cfg:
            compile_cfg["optimizer"]["config"]["weight_decay"] = opt_cfg["weight_decay"]

        scaler_cfg = copy.deepcopy(kcfg.get("training", {}).get("scaler", {}))
        if not scaler_cfg:
            scaler_cfg = {"class_name": "StandardLabelScaler"}

        cfg = {
            "model": {"config": model_cfg},
            "training": {
                "fit": {
                    "epochs": epochs,
                    "batch_size": batch_size,
                    "early_stopping_patience": 0
                },
                "compile": compile_cfg,
                "cross_validation": {
                    "n_splits": n_splits,
                    "shuffle": bool(cv_cfg.get("shuffle", True))
                },
                "scaler": scaler_cfg,
            },
            "data": {
                "dataset": {
                    "class_name": "MatPES2k",
                    "config": {"file_path": args.dataset_path}
                },
                "data_unit": "meV/atom"
            }
        }
        if sched_cfg:
            cfg["training"]["scheduler"] = sched_cfg

        hpath = os.path.join(hyper_out_dir, f"hyper_{model}.json")
        with open(hpath, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        out_dir = os.path.join(results_dir, model)
        os.makedirs(out_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, f"{model}.log")

        cmd = [
            sys.executable,
            os.path.join(torch_root, "training_scripts/train_graph.py"),
            "--hyper", hpath,
            "--category", model,
            "--output", results_dir,
            "--seed", str(args.seed),
        ]
        env = os.environ.copy()
        env["PYTHONPATH"] = torch_root + os.pathsep + env.get("PYTHONPATH", "")

        start = datetime.now().isoformat(timespec="seconds")
        print(f"[{start}] Running {model} (lr={lr}, batch={batch_size})", flush=True)
        timed_out = False
        rc = 1
        try:
            with open(log_path, "w", encoding="utf-8") as lf:
                proc = subprocess.run(
                    cmd,
                    cwd=torch_root,
                    env=env,
                    stdout=lf,
                    stderr=subprocess.STDOUT,
                    text=True,
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
            "batch_size": batch_size,
            "best_val_mae": best_val_mae,
            "score_file": score_path if os.path.exists(score_path) else "",
            "log_file": log_path,
            "tail": tail,
        }
        rows.append(row)
        print(f"Finished {model}: ok={row['ok']}, best_val_mae={best_val_mae}", flush=True)

    summary = {
        "total": len(rows),
        "passed": sum(1 for r in rows if r.get("ok")),
        "failed": sum(1 for r in rows if not r.get("ok")),
        "failed_models": [r["model"] for r in rows if not r.get("ok")],
    }
    out = {"summary": summary, "results": rows}
    outp = os.path.join(args.output_dir, "torch_apple2apple_26_summary.json")
    with open(outp, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {outp}")
    if summary["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
