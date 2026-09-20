"""Fit temperature scaling on held-out calibration data; transfer is not guaranteed."""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from dataset.decision_dataset import DecisionDataset
from eval.eval_inference import collect, counts_array, effective_k, load_decision
from eval.eval_metrics import (
    K_BUCKETS, apply_temperature, metrics_by_provenance, temp_key,
)
from trainer.trainer_utils import file_sha1, unbuffer_stdout


def artifact_sha1(path):
    return file_sha1(path, n=12)

T_MIN, T_MAX = 0.05, 20.0

# 屏蔽槽在 logit 空间里的填充值。见 `fit_temperature`：必须是**有限**的大负数，
# 不能用 `-inf`（会让 log_temp 的梯度出现 0·∞ 而变 NaN）。
NEG_LP = -1e4


def parse_args():
    p = argparse.ArgumentParser(description="温度校准（只在 calib split 上拟合）")
    p.add_argument("--ckpt", default="out/decision/decision.pth")
    p.add_argument("--data", default="dataset/synth")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--out", default="out/calibration")
    p.add_argument("--split", default="calib")
    p.add_argument("--max_len", type=int)
    p.add_argument("--hidden_size", type=int)
    p.add_argument("--num_hidden_layers", type=int)
    p.add_argument("--crosstalk", action="store_true", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lbfgs_steps", type=int, default=200)
    return p.parse_args()


def fit_temperature(log_p, target, mask, group, max_iter=200, device="cuda"):
    """Fit masked NLL on one calibration group."""
    lp = torch.tensor(log_p, dtype=torch.float64, device=device)
    t = torch.tensor(target, dtype=torch.float64, device=device)
    sel = torch.tensor(group, dtype=torch.bool, device=device)
    lp = lp.masked_fill(~torch.tensor(np.asarray(mask), dtype=torch.bool, device=device), NEG_LP)
    lp, t = lp[sel], t[sel]
    if lp.numel() == 0:
        return None

    def objective(log_temp):
        ls = torch.log_softmax(lp / log_temp.exp(), dim=-1)
        # 屏蔽槽上 `t=0` 且 `ls=-inf`，直接相乘得到 NaN。先把 `t=0` 处换成 0 再乘。
        return -(t * torch.where(t > 0, ls, torch.zeros_like(ls))).sum(-1).mean()

    before = float(objective(torch.zeros((), dtype=torch.float64, device=device)))
    log_temp = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(log_temp)
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(torch.exp(log_temp).clamp(T_MIN, T_MAX))
    after = float(objective(torch.tensor(np.log(temp), dtype=torch.float64, device=device)))
    return temp, before, after


def main():
    unbuffer_stdout()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本项目的所有实测数字都基于 GPU；--device cpu 不在支持范围内。")

    tok, model, meta = load_decision(
        args.ckpt, args.tokenizer, device, hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers, candidate_crosstalk=args.crosstalk,
    )
    args.max_len = args.max_len or meta.get("max_len", 1024)

    if args.split != "calib":
        raise SystemExit("温度只允许在 calib split 上拟合，不可用 val/test 替代。")
    path = os.path.join(args.data, f"{args.split}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"缺少 {path} —— 校准必须在一份**没有参与训练**的 split 上做")
    ds = DecisionDataset(path, tok, max_len=args.max_len, augment_k=False)
    if any(rec.get("split") != "calib" for rec in ds.records):
        raise SystemExit("calib 文件中包含其他 split 的样本")

    print("\n========== 收 calib 上的逐样本分布 ==========")
    t0 = time.time()
    col = collect(model, ds, device, args.batch_size, args.limit)
    if col is None:
        raise SystemExit("calib 数据为空，无法拟合温度")
    k = effective_k(col["mask"])
    n = col["n"]
    print(f"  {n} 条，K_used {k.min()}–{k.max()}，{time.time()-t0:.0f} 秒")

    # log p 而不是 p：见 fit_temperature 的注释。
    log_p = np.log(np.clip(col["p"], 1e-30, None))
    target, mask = col["target"], col["mask"]
    prim = np.asarray(col["primitive"])

    print("\n========== 拟合 ==========")
    temp = {"global": None}
    rows = []

    groups = [("global", np.ones(n, dtype=bool))]
    for pr in sorted(set(prim.tolist())):
        groups.append((f"primitive:{pr}", prim == pr))
    kk = [temp_key(p, x) for p, x in zip(prim, k)]
    kk = np.asarray(kk)
    for key in sorted(set(kk.tolist())):
        groups.append((f"primitive_k:{key}", kk == key))

    fitted = {}
    for name, sel in groups:
        res = fit_temperature(log_p, target, mask, sel, args.lbfgs_steps)
        if res is None:
            continue
        T, before, after = res
        fitted[name] = T
        rows.append({"group": name, "n": int(sel.sum()), "T": T,
                     "nll_before": before, "nll_after": after})
        print(f"  {name:26s} n={int(sel.sum()):5d}  T={T:6.3f}  "
              f"NLL {before:.4f} -> {after:.4f}  (Δ {after-before:+.4f})")

    temp["global"] = fitted["global"]
    temp["primitive"] = {nm.split(":", 1)[1]: T for nm, T in fitted.items()
                         if nm.startswith("primitive:")}
    temp["primitive_k"] = {nm.split(":", 1)[1]: T for nm, T in fitted.items()
                           if nm.startswith("primitive_k:")}

    # 拟合完立刻在校准集本身上报 ECE —— **并且明确标注这是在拟合集上**。
    # 这个数字必然偏乐观，写进 JSON 是为了让读者能看到"过拟合有多大"，
    # 而不是为了让谁引用它。
    print("\n========== 校准集上的自评（拟合集，必然偏乐观）==========")
    from eval.eval_metrics import apply_temperature
    self_report = {}
    for gran in ("global", "primitive", "primitive_k"):
        if not temp[gran if gran != "primitive_k" else "primitive_k"]:
            continue
        pg = apply_temperature(col["p"], temp, prim, k, mask, gran)
        m = metrics_by_provenance(pg, target, mask, provenance=col["provenance"],
                                  is_ord=col["is_ord"],
                                  levels=np.where(col["levels"] >= 0, col["levels"], 0),
                                  counts=counts_array(col))
        self_report[gran] = m
        print(f"  {gran:12s} ECE {m['all']['ece']:.4f}  distribution_l2 {m['all']['distribution_l2']:.4f}  "
              f"NLL {m['all']['nll']:.4f}")
    raw = metrics_by_provenance(col["p"], target, mask, provenance=col["provenance"],
                                is_ord=col["is_ord"],
                                levels=np.where(col["levels"] >= 0, col["levels"], 0),
                                counts=counts_array(col))
    print(f"  {'未校准':12s} ECE {raw['all']['ece']:.4f}  distribution_l2 {raw['all']['distribution_l2']:.4f}  "
          f"NLL {raw['all']['nll']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    out = dict(temp)
    out["meta"] = {
        "ckpt": os.path.abspath(args.ckpt),
        # **哈希是必需的，不是装饰。** 温度绑定到具体权重与具体词表：换一版
        # checkpoint 或重训 tokenizer 后，旧的 T.json 就静默失效了，而它会照常
        # 被 `eval_harness.py --temperature` 吃进去并产出一张看起来正常的可靠性图。
        # 把两个哈希写进来，读者才能判断一个 T.json 是否适用于手上的模型。
        "ckpt_sha1": artifact_sha1(args.ckpt),
        "tokenizer_sha1": artifact_sha1(os.path.join(args.tokenizer, "tokenizer.json")),
        "tokenizer_path": os.path.abspath(args.tokenizer),
        "split": args.split, "n": n,
        "k_buckets": [list(b) for b in K_BUCKETS],
        "fit_objective": "NLL on calib split (LBFGS on log T)",
        "clamp": [T_MIN, T_MAX],
        "fits": rows,
        "self_eval_on_fit_split": {
            "note": "在**拟合集**上算的，必然偏乐观；正式数字用 eval_harness.py",
            "uncalibrated": raw["all"], **{g: m["all"] for g, m in self_report.items()},
        },
    }
    out_path = os.path.join(args.out, "T.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  -> {out_path}")
    print("  下一步：python eval/eval_harness.py --ckpt "
          f"{args.ckpt} --temperature {out_path}")


if __name__ == "__main__":
    main()
