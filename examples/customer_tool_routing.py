import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.eval_metrics import compute_metrics, risk_coverage_curve, top1

SPLITS = ("train", "val", "calib", "test_known")
HUMAN = "human"
SCHEMA = {"primitive": "choice", "name": "customer_tool_routing",
          "desc": "选择一个客服工具；信息不足或超出支持范围时转人工。"}
TOOLS = {
    "order": "查询订单物流状态",
    "refund": "登记订单退款申请（不直接退款）",
    "account": "查询账户登录问题",
    HUMAN: "转人工客服，进一步确认需求",
}
QUESTION = "这个请求应交给哪个工具或人工客服？"
TEMPLATES = {
    "order": ("请查询订单 {entity} 的物流状态。", "想知道 {entity} 这笔订单送到哪里了。",
              "帮我查一下订单 {entity} 的运输进度。", "订单 {entity} 现在配送到哪一站？"),
    "refund": ("请为订单 {entity} 登记退款申请。", "我想申请退回订单 {entity} 的款项。",
               "帮我提交 {entity} 这单的退款申请。", "订单 {entity} 不要了，需要申请退款。"),
    "account": ("账户 {entity} 无法登录，请排查。", "账号 {entity} 登录失败，需要帮助。",
                "请检查账户 {entity} 为什么登不进去。", "我用 {entity} 这个账号一直无法登录。"),
    HUMAN: ("我的编号是 {entity}，有个问题但还没说清楚，请人工联系。",
            "关于 {entity} 的事情说不清楚，想找人工客服聊聊。",
            "编号 {entity}，暂时不知道具体需要处理什么，请转人工确认。",
            "我是 {entity}，需求还不明确，需要人工先沟通。"),
}


def request_for(state):
    return {"schema": dict(SCHEMA), "state": state, "question": QUESTION,
            "candidates": [{"label": label, "text": text, "meta": {"level": None}}
                           for label, text in TOOLS.items()]}


def write_json(path, value):
    path = Path(path)
    text = json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        stream.write(text)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def generate(seed=0):
    pools = {split: [f"C{1000 + i * 100 + j}" for j in range(32 if i == 0 else 8)]
             for i, split in enumerate(SPLITS)}
    assignments = {split: [(label, f"{label}-{i}", texts[i])
                           for label, texts in TEMPLATES.items()]
                   for i, split in enumerate(SPLITS)}
    result = {}
    for split in SPLITS:
        records = []
        for label, template_id, template in assignments[split]:
            for entity in pools[split]:
                record = request_for(template.format(entity=entity))
                rng = random.Random(f"{seed}|{split}|{template_id}|{entity}")
                rng.shuffle(record["candidates"])
                record.update({
                    "id": f"customer_tool_routing::{split}::{len(records):07d}",
                    "source": "synth:customer_tool_routing", "gen_version": "1.0.0",
                    "split": split,
                    "state_sections": [{"seg": "notes", "text": record["state"], "priority": 3}],
                    "question_paraphrases": ["应该选择哪个客服工具？", "请选一个处理渠道。", "此请求应如何分派？"],
                    "target": {"kind": "hard", "provenance": "hard", "renormalized": False,
                               "p": [int(c["label"] == label) for c in record["candidates"]],
                               "audit": {"correct_label": label}},
                    "meta": {"K_full": len(TOOLS), "approx_tokens": 256,
                             "template_id": template_id, "entity_pool": split,
                             "entity_id": entity},
                })
                records.append(record)
        random.Random(f"{seed}|{split}|rows").shuffle(records)
        result[split] = records
    assert_disjoint(result)
    return result


def assert_disjoint(splits):
    seen = {field: set() for field in ("id", "state", "template_id", "entity_id", "entity_pool")}
    for records in splits.values():
        for field, previous in seen.items():
            values = [r[field] if field in ("id", "state") else r["meta"][field] for r in records]
            if field in ("id", "state") and len(values) != len(set(values)):
                raise ValueError(f"duplicate {field} within split")
            if previous.intersection(values):
                raise ValueError(f"split leakage: {field}")
            previous.update(values)


def build(out, seed=0):
    out = Path(out)
    if out.exists() and (not out.is_dir() or any(out.iterdir())):
        raise ValueError(f"refusing nonempty output: {out}")
    records = generate(seed)
    out.mkdir(parents=True, exist_ok=True)
    for split, rows in records.items():
        with (out / f"{split}.jsonl").open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    return {"data": str(out.resolve()), "seed": seed,
            "counts": {split: len(rows) for split, rows in records.items()}}


def validate_routing_request(request):
    if not isinstance(request, dict) or not isinstance(request.get("schema"), dict):
        raise ValueError("request and schema must be JSON objects")
    if request["schema"].get("primitive") != "choice" or request["schema"].get("name") != SCHEMA["name"]:
        raise ValueError("this policy is only for the customer_tool_routing Choice schema")
    for key in ("state", "question"):
        if not isinstance(request.get(key), str) or not request[key].strip():
            raise ValueError(f"{key} must be nonempty text")
    candidates = request.get("candidates", [])
    if not isinstance(candidates, list) or any(not isinstance(c, dict) for c in candidates):
        raise ValueError("candidates must be a list of objects")
    labels = [c.get("label") for c in candidates]
    if any(not isinstance(label, str) for label in labels) or len(labels) != len(TOOLS) or set(labels) != set(TOOLS):
        raise ValueError("routing requires exactly order/refund/account/human candidates")
    if any(c.get("text") != TOOLS[c["label"]] for c in candidates):
        raise ValueError("candidate meanings must match the frozen routing tools")


def read_split(data, split):
    with (Path(data) / f"{split}.jsonl").open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if not rows:
        raise ValueError(f"empty {split} split")
    for row in rows:
        validate_routing_request(row)
        if row.get("split") != split:
            raise ValueError(f"{split} file contains another split")
        target = row.get("target", {})
        p = target.get("p", [])
        if (target.get("provenance") != "hard" or target.get("kind") != "hard"
                or len(p) != len(TOOLS) or any(x not in (0, 1) for x in p) or sum(p) != 1):
            raise ValueError("routing evaluation requires aligned one-hot hard targets")
    return rows


def choose_threshold(p, t, abstain_idx, max_risk):
    if not math.isfinite(max_risk) or not 0 <= max_risk <= 1:
        raise ValueError("max_risk must be between 0 and 1")
    coverage, risk, thresholds = risk_coverage_curve(p, t, abstain_idx=abstain_idx)
    feasible = np.flatnonzero((coverage > 0) & np.isfinite(risk) & (risk <= max_risk))
    if not len(feasible):
        return None
    best = feasible[np.argmax(coverage[feasible])]
    return float(thresholds[best])


def policy_report(p, t, abstain_idx, threshold):
    idx, confidence = top1(p)
    model_human = idx == np.asarray(abstain_idx)
    accepted = np.zeros(len(idx), dtype=bool) if threshold is None else (~model_human & (confidence >= threshold))
    errors = idx != np.asarray(t).argmax(-1)
    n = len(idx)
    n_accepted = int(accepted.sum())
    return {
        "n": n, "accepted": n_accepted, "accepted_errors": int((accepted & errors).sum()),
        "model_human": int(model_human.sum()),
        "threshold_handoff": int((~model_human & ~accepted).sum()),
        "human_handoff": int((~accepted).sum()),
        "coverage": n_accepted / n,
        "accepted_risk": float(errors[accepted].mean()) if n_accepted else None,
        "handoff_fraction": float((~accepted).mean()),
        "metrics_all": compute_metrics(p, t),
    }


def fingerprint(ckpt, tokenizer, temperature, data):
    paths = {"checkpoint": Path(ckpt).resolve(), "temperature": Path(temperature).resolve()}
    tok = Path(tokenizer).resolve()
    if not (tok / "tokenizer.json").is_file():
        raise ValueError("missing tokenizer.json")
    paths.update({f"tokenizer/{p.name}": p for p in sorted(tok.iterdir()) if p.is_file()})
    paths.update({f"data/{split}": (Path(data) / f"{split}.jsonl").resolve() for split in SPLITS})
    return {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}


def load_policy(path, ckpt=None, tokenizer=None, temperature=None, data=None):
    with Path(path).open(encoding="utf-8") as stream:
        policy = json.load(stream)
    if (policy.get("version") != 1 or policy.get("schema") != SCHEMA
            or policy.get("tools") != TOOLS or policy.get("granularity") != "global"
            or policy.get("selection_split") != "val" or policy.get("temperature_split") != "calib"):
        raise ValueError("unsupported or invalid routing policy")
    threshold = policy.get("threshold")
    if threshold is not None and (isinstance(threshold, bool) or not isinstance(threshold, (int, float))
                                  or not math.isfinite(threshold) or not 0 <= threshold <= 1):
        raise ValueError("invalid policy threshold")
    args = policy["paths"]
    paths = dict(ckpt=ckpt or args["ckpt"], tokenizer=tokenizer or args["tokenizer"],
                 temperature=temperature or args["temperature"], data=data or args["data"])
    actual = fingerprint(**paths)
    expected = policy["artifacts"]
    if set(actual) != set(expected) or any(actual[key]["sha256"] != expected[key]["sha256"] for key in actual):
        raise ValueError("policy artifact mismatch; do not reuse a threshold after changing model/tokenizer/temperature/data")
    return policy, paths


def predictor_for(paths):
    from inference import Predictor
    predictor = Predictor(paths["ckpt"], tokenizer=paths["tokenizer"],
                          temperature=paths["temperature"], device="cuda", granularity="global")
    if predictor.temperature["meta"].get("split") != "calib":
        raise ValueError("temperature must have been fitted on calib")
    return predictor


def predict_rows(predictor, rows):
    p, t, human = [], [], []
    for row in rows:
        result = predictor.predict({key: row[key] for key in ("schema", "state", "question", "candidates")})
        if [c["label"] for c in result["candidates"]] != [c["label"] for c in row["candidates"]]:
            raise ValueError("predictor changed candidate order")
        p.append([c["probability"] for c in result["candidates"]])
        t.append(row["target"]["p"])
        human.append(next(i for i, c in enumerate(row["candidates"]) if c["label"] == HUMAN))
    return np.asarray(p), np.asarray(t), np.asarray(human)


def select(args):
    if Path(args.policy).exists():
        raise ValueError("policy already exists; use a new path for a new experiment")
    paths = {key: str(Path(getattr(args, key)).resolve()) for key in ("ckpt", "tokenizer", "temperature", "data")}
    artifacts = fingerprint(**paths)
    predictor = predictor_for(paths)
    rows = read_split(args.data, "val")
    p, t, human = predict_rows(predictor, rows)
    threshold = choose_threshold(p, t, human, args.max_risk)
    policy = {"version": 1, "schema": SCHEMA, "tools": TOOLS, "granularity": "global",
              "selection_split": "val", "temperature_split": "calib", "max_risk": args.max_risk,
              "threshold": threshold, "paths": paths, "artifacts": artifacts,
              "validation": policy_report(p, t, human, threshold),
              "note": "Empirical validation constraint only; not a test/OOD risk guarantee."}
    write_json(args.policy, policy)
    return policy


def evaluate(args):
    if Path(args.out).exists():
        raise ValueError("report exists; refusing to overwrite frozen test results")
    policy, paths = load_policy(args.policy, args.ckpt, args.tokenizer, args.temperature, args.data)
    rows = read_split(paths["data"], "test_known")
    p, t, human = predict_rows(predictor_for(paths), rows)
    report = {"split": "test_known", "threshold": policy["threshold"],
              "policy_sha256": sha256(args.policy), "artifacts": policy["artifacts"],
              "note": "Frozen val policy; no threshold fitting on test. No OOD guarantee.",
              **policy_report(p, t, human, policy["threshold"])}
    write_json(args.out, report)
    return report


def dispatch_stub(label):
    if label not in TOOLS:
        raise ValueError("tool is not allowlisted")
    return {"handler": label, "dry_run": True, "side_effects": False,
            "message": f"Local stub only: {TOOLS[label]}"}


def dispatch(args):
    policy, paths = load_policy(args.policy)
    if args.input:
        with Path(args.input).open(encoding="utf-8") as stream:
            request = json.load(stream)
    else:
        request = request_for(args.state)
    validate_routing_request(request)
    result = predictor_for(paths).predict({key: request[key] for key in ("schema", "state", "question", "candidates")})
    label = result["selected_label"]
    if label not in TOOLS:
        raise ValueError("prediction is not allowlisted")
    _, confidence = top1([[c["probability"] for c in result["candidates"]]])
    policy_confidence = float(confidence[0])
    accepted = label != HUMAN and policy["threshold"] is not None and policy_confidence >= policy["threshold"]
    return {"prediction": result, "threshold": policy["threshold"], "accepted": accepted,
            "policy_confidence": policy_confidence, "not_trained_example": args.unseen,
            "scope_note": "--unseen is a user annotation, not an OOD detector. No unknown-task guarantee.",
            "dispatch": dispatch_stub(label if accepted else HUMAN)}


def main():
    parser = argparse.ArgumentParser(description="Offline customer routing tutorial; dispatch is always a local dry run.")
    commands = parser.add_subparsers(dest="command", required=True)
    builder = commands.add_parser("build")
    builder.add_argument("--out", required=True)
    builder.add_argument("--seed", type=int, default=0)
    selector = commands.add_parser("select", help="fit a confidence threshold on val only")
    evaluator = commands.add_parser("evaluate", help="apply the frozen policy to test_known only")
    for sub in (selector, evaluator):
        for flag in ("data", "ckpt", "tokenizer", "temperature"):
            sub.add_argument(f"--{flag}", required=sub is selector)
        sub.add_argument("--policy", required=True)
    selector.add_argument("--max_risk", type=float, default=0.1)
    evaluator.add_argument("--out", required=True)
    dispatcher = commands.add_parser("dispatch")
    dispatcher.add_argument("--policy", required=True)
    source = dispatcher.add_mutually_exclusive_group(required=True)
    source.add_argument("--state")
    source.add_argument("--input")
    dispatcher.add_argument("--unseen", action="store_true", help="annotate this as an untrained business example, not detection")
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(args.out, args.seed)
        else:
            result = {"select": select, "evaluate": evaluate, "dispatch": dispatch}[args.command](args)
        print(json.dumps(result, ensure_ascii=False, allow_nan=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        parser.exit(2, f"routing failed: {error}\n")


if __name__ == "__main__":
    main()
