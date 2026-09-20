"""Bounded, offline CUDA tutorial; not a reproduction of full-training results."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parents[1]
CONFIG_KEYS = {
    "seed", "hidden_size", "num_hidden_layers", "batch_size", "num_workers",
    "precision", "warmup_steps", "epochs", "mlm_max_len", "decision_max_len",
    "mlm_max_steps", "decision_max_steps", "tokenizer_n_synth", "mlm_n_synth",
    "per_gen_train", "per_gen_val", "per_gen_calib", "per_gen_test_known",
    "per_gen_test_ood", "lbfgs_steps",
}


def resolve_path(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, value):
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path):
    config = read_json(path)
    if not isinstance(config, dict) or set(config) != CONFIG_KEYS:
        raise ValueError(f"Quickstart config must contain exactly {sorted(CONFIG_KEYS)}.")
    for key, value in config.items():
        if key == "precision":
            if value != "bf16":
                raise ValueError("Quickstart supports precision='bf16' only.")
            continue
        minimum = 0 if key in {"seed", "num_workers", "warmup_steps", "per_gen_test_ood"} else 1
        if type(value) is not int or value < minimum:
            raise ValueError(f"Config {key} must be an integer >= {minimum}.")
    if config["num_workers"] != 0 or config["per_gen_test_ood"] != 0:
        raise ValueError("Quickstart requires num_workers=0 and per_gen_test_ood=0.")
    return config


def check_destination(out):
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise FileExistsError(f"Refusing nonempty destination {out}; choose a new --out directory.")


def environment_info():
    # Imported here so --help and command construction do not initialize CUDA.
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required. Run this script with the CUDA-enabled minimind "
                           "Python environment; no CPU fallback or package downloads are used.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("This bf16 quickstart requires a CUDA device with bf16 support.")
    device = torch.cuda.current_device()
    versions = {name: importlib.metadata.version(name) for name in
                ("torch", "transformers", "tokenizers", "numpy", "datasets")}
    return {
        "python": sys.version, "executable": sys.executable, "platform": platform.platform(),
        "versions": versions, "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device), "device": f"cuda:{device}",
        "vram_gb": torch.cuda.get_device_properties(device).total_memory / 1024 ** 3,
        "bf16_supported": True,
        "env": {key: os.environ.get(key) for key in
                ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "PYTHONHASHSEED",
                 "CUDA_VISIBLE_DEVICES", "CONDA_DEFAULT_ENV")},
    }


def git_info():
    def git(*args):
        return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True,
                              text=True, encoding="utf-8").stdout.strip()
    status = git("status", "--porcelain", "--untracked-files=normal")
    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(status), "status": status}


def build_stages(config, out):
    c = config
    tok, data = out / "tokenizer", out / "data"
    ckpt = out / "decision" / "decision.pth"
    temperature = out / "calibration" / "T.json"

    def command(script, *args):
        return [sys.executable, str(ROOT / script), *map(str, args)]

    common = ["--tokenizer", tok, "--hidden_size", c["hidden_size"],
              "--num_hidden_layers", c["num_hidden_layers"], "--batch_size", c["batch_size"],
              "--num_workers", c["num_workers"], "--epochs", c["epochs"],
              "--warmup_steps", c["warmup_steps"], "--seed", c["seed"],
              "--log_interval", 1, "--save_optimizer", "--no_swanlab"]
    build_counts = []
    for split in ("train", "val", "calib", "test_known", "test_ood"):
        key = f"per_gen_{split}"
        build_counts.extend([f"--{key}", c[key]])
    stages = [
        ("tokenizer", command("trainer/train_tokenizer.py", "--synthetic_only", "--n_docs", 0,
                              "--n_synth", c["tokenizer_n_synth"], "--seed", c["seed"],
                              "--skip_eval", "--out_dir", tok)),
        ("data", command("scripts/build_dataset.py", "--out", data, "--tokenizer", tok,
                         "--seed", c["seed"], "--max_len", c["decision_max_len"], *build_counts)),
        ("mlm", command("trainer/train_mlm.py", *common, "--synthetic_only", "--n_docs", 0,
                        "--n_synth", c["mlm_n_synth"], "--max_len", c["mlm_max_len"],
                        "--max_steps", c["mlm_max_steps"], "--out", out / "mlm",
                        "--log_dir", out / "mlm" / "logs")),
        ("decision", command("trainer/train_decision.py", *common, "--data", data,
                             "--encoder", out / "mlm" / "mlm.pth",
                             "--max_len", c["decision_max_len"],
                             "--max_steps", c["decision_max_steps"], "--out", out / "decision",
                             "--log_dir", out / "decision" / "logs")),
        ("calibration", command("trainer/calibrate_temperature.py", "--ckpt", ckpt,
                                "--tokenizer", tok, "--data", data, "--out", out / "calibration",
                                "--batch_size", c["batch_size"], "--lbfgs_steps", c["lbfgs_steps"],
                                "--max_len", c["decision_max_len"])),
        ("eval", command("eval/eval_harness.py", "--ckpt", ckpt, "--tokenizer", tok,
                         "--data", data, "--temperature", temperature, "--sets", "test_known",
                         "--out", out / "eval", "--batch_size", c["batch_size"],
                         "--max_len", c["decision_max_len"], "--no_per_sample")),
    ]
    for name in ("noul", "choice", "score"):
        stages.append((f"inference_{name}", command(
            "inference.py", "--ckpt", ckpt, "--tokenizer", tok,
            "--input", ROOT / "examples" / "inference" / f"{name}.json",
            "--temperature", temperature)))
    return stages


def check_data_lengths(out, max_len):
    """Catch the existing minimum-state/fixed-candidate overflow before training."""
    counts = {}
    for split in ("train", "val", "calib", "test_known"):
        count = 0
        with open(out / "data" / f"{split}.jsonl", encoding="utf-8") as f:
            for line in f:
                record = json.loads(line)
                actual = record["meta"]["approx_tokens"]
                if actual > max_len:
                    raise ValueError(
                        f"{split} record {record['id']} needs {actual} tokens, above "
                        f"decision_max_len={max_len}. Fixed question/candidates plus minimum "
                        "state do not fit; review the config rather than silently dropping candidates.")
                count += 1
        if not count:
            raise ValueError(f"Generated {split} is empty; cannot complete quickstart.")
        counts[split] = count
    return counts


def collect_stage_result(name, out, config):
    if name in ("mlm", "decision"):
        path = out / name / "summary.json"
        result = read_json(path)
        for field in ("step", "elapsed_s", "peak_vram_gb"):
            if field not in result:
                raise ValueError(f"{path} is missing trainer summary field {field}.")
        if result["step"] != config[f"{name}_max_steps"]:
            raise ValueError(f"{name} stopped at {result['step']}, expected "
                             f"{config[f'{name}_max_steps']} optimizer updates.")
        return {"summary_path": str(path), "training": result}
    if name == "eval":
        path = out / "eval" / "decision" / "data_test_known.json"
        result = read_json(path)
        return {"metrics_path": str(path), "n": result["n"],
                "metrics": result["metrics"], "calibrated": result["calibrated"]}
    return {}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quickstart.json")
    parser.add_argument("--out", default="out/quickstart")
    args = parser.parse_args()
    config_path, out = resolve_path(args.config), resolve_path(args.out)
    config = load_config(config_path)
    check_destination(out)
    # Set offline flags for this process AND every checked child, regardless of caller env.
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      PYTHONHASHSEED=str(config["seed"]), PYTHONUTF8="1")
    environment = environment_info()
    git = git_info()
    stages = build_stages(config, out)
    for _, cmd in stages:
        if not Path(cmd[1]).is_file():
            raise FileNotFoundError(f"Required quickstart entry point is missing: {cmd[1]}")
    # Check again after preflight; never offer an implicit --force overwrite.
    check_destination(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "logs").mkdir()
    summary = {
        "status": "running", "started_at": datetime.now(timezone.utc).isoformat(),
        "note": "Offline synthetic-template tutorial; not full pretraining or business-quality evidence.",
        "config_path": str(config_path), "config_sha256": sha256(config_path),
        "config": config, "seed": config["seed"], "environment": environment,
        "git": git, "out": str(out), "stages": [],
    }
    write_json(out / "config.json", config)
    write_json(out / "summary.json", summary)
    started = time.perf_counter()
    try:
        for name, cmd in stages:
            log = out / "logs" / f"{name}.log"
            stage = {"name": name, "command": cmd, "log_path": str(log), "status": "running"}
            summary["stages"].append(stage)
            write_json(out / "summary.json", summary)
            print(f"[{name}] {subprocess.list2cmdline(cmd)}\nLog: {log}", flush=True)
            stage_started = time.perf_counter()
            try:
                with open(log, "w", encoding="utf-8") as stderr:
                    if name.startswith("inference_"):
                        prediction = out / f"{name}.json"
                        with open(prediction, "w", encoding="utf-8") as stdout:
                            subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True,
                                           stdout=stdout, stderr=stderr)
                        stage["output_path"] = str(prediction)
                        stage["output"] = read_json(prediction)
                    else:
                        subprocess.run(cmd, cwd=ROOT, env=os.environ.copy(), check=True,
                                       stdout=stderr, stderr=subprocess.STDOUT)
                if name == "data":
                    stage["counts"] = check_data_lengths(out, config["decision_max_len"])
                stage.update(collect_stage_result(name, out, config))
                stage["status"] = "completed"
            except BaseException as exc:
                stage["status"] = "failed"
                stage["error"] = str(exc)
                raise
            finally:
                stage["elapsed_s"] = time.perf_counter() - stage_started
                write_json(out / "summary.json", summary)
        artifacts = [out / "tokenizer" / "tokenizer.json",
                     out / "tokenizer" / "tokenizer_config.json",
                     out / "data" / "manifest.json", out / "mlm" / "mlm.pth",
                     out / "decision" / "decision.pth", out / "calibration" / "T.json"]
        artifacts.extend(out / "data" / f"{s}.jsonl" for s in
                         ("train", "val", "calib", "test_known", "test_ood"))
        summary["artifact_sha256"] = {str(path): sha256(path) for path in artifacts}
        summary["status"] = "completed"
    except BaseException as exc:
        summary["status"] = "failed"
        summary["error"] = str(exc)
        raise
    finally:
        summary["elapsed_s"] = time.perf_counter() - started
        summary["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(out / "summary.json", summary)
    print(f"Quickstart complete: {out / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
