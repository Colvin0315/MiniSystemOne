"""
离线评测 harness：跑一个 checkpoint，把逐样本结果与全部指标落盘。

输出 `out/eval/<ckpt>/<set>.json`，含四块：

  - `metrics`     未校准的 per-provenance 指标（`all` / `soft_targets` / 每个 provenance）
  - `calibrated`  温度校准后的同一套指标，**三种粒度各一份**（global / primitive /
                  primitive_k）。三数并列比一句断言更有信息量 —— 如果 global 一个
                  标量就能追平 15 个桶，那"per-(primitive×K) 温度"就不是必需的复杂度。
  - `per_sample`  逐样本 p / 目标 / 元信息。**保留它是为了让改图不用重跑模型**：
                  可靠性图与误差分析都从这一份重算。（风险-覆盖率曲线**不在**
                  其中：`eval_metrics.risk_coverage_curve` 有函数，但本仓库没有
                  任何脚本产出它。）
  - `env`         checkpoint 哈希、tokenizer 哈希、gen_version、分桶方案。

**不在这里做的事**：拟合温度（那是 `trainer/calibrate_temperature.py`，它只碰
`calib` split）。本文件只读 `out/calibration/T.json` 并应用。职责分开是因为
"拟合在哪份数据上"必须一眼可见 —— 混在一个脚本里就容易在校准集上评测。

用法：
    python eval/eval_harness.py --ckpt out/decision/decision.pth
    python eval/eval_harness.py --ckpt out/decision/decision.pth \
        --temperature out/calibration/T.json --sets val test_known
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch

from dataset.decision_dataset import DecisionDataset
from eval.eval_inference import collect, counts_array, effective_k, load_decision, load_temperature
from eval.eval_metrics import apply_temperature, metrics_by_provenance, top1
from trainer.trainer_utils import peak_vram_warn, unbuffer_stdout

SETS = ("val", "calib", "test_known", "test_ood")
GRANULARITIES = ("global", "primitive", "primitive_k")
# p 落盘保留的小数位。ECE 的可见差异在 1e-3 量级，1e-6 比它低三个数量级，
# 足够事后重算，又让 JSON 小一半。
P_DECIMALS = 6


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2 离线评测")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="dataset/synth")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--out", default="out/eval")
    p.add_argument("--sets", nargs="*", default=list(SETS),
                   help="要评测的 split；不存在的自动跳过")
    p.add_argument("--temperature", default=None,
                   help="out/calibration/T.json；不给则只报未校准指标")
    p.add_argument("--apply", default="primitive_k", choices=GRANULARITIES,
                   help="per_sample.p_calibrated 用哪一档温度")
    p.add_argument("--max_len", type=int)
    p.add_argument("--hidden_size", type=int)
    p.add_argument("--num_hidden_layers", type=int)
    p.add_argument("--crosstalk", action="store_true", default=None)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="每个 set 只用前 N 条（0=全部）")
    p.add_argument("--no_per_sample", action="store_true",
                   help="不写 per_sample（K 很大时 JSON 会很大）")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--binning", default="equal_mass",
                   choices=("equal_mass", "equal_width"))
    return p.parse_args()


def sha1_of(path, chunk=1 << 20):
    """文件哈希。写进结果 JSON，让"这个数字来自哪个权重"可追溯。"""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:16]


def metrics_for(col, p, args):
    """一份 p 的全部指标。counts 只在有标注者人数时才传，见 counts_array。"""
    levels = np.where(col["levels"] >= 0, col["levels"], 0)
    return metrics_by_provenance(
        p, col["target"], col["mask"], provenance=col["provenance"],
        is_ord=col["is_ord"], levels=levels, counts=counts_array(col),
        bins=args.bins, binning=args.binning,
    )


def per_sample_block(col, p_cal, gran):
    """逐样本落盘。`p` 与 `p_calibrated` 是仅有的两个大数组，其余都是小元信息。"""
    idx, conf = top1(col["p"], col["mask"])
    _, conf_cal = top1(p_cal, col["mask"])
    # 软正确率 = t[argmax p]：t 是 one-hot 时它就是硬准确率，软目标时它是
    # "命中那一项的目标概率"。用同一个 argmax 下标去查校准前后的目标概率，
    # 于是 conf 与 correct 的差就是"置信度 - 应得的正确率"，可直接画可靠性图。
    rows = np.arange(len(idx))
    tgt = col["target"]
    block = {
        "id": col["id"],
        "provenance": col["provenance"],
        "primitive": col["primitive"],
        "source": col["source"],
        "template_id": col["template_id"],
        "gen_version": col["gen_version"],
        "K_full": col["K_full"],
        "K_used": effective_k(col["mask"]).tolist(),
        "renormalized": col["renormalized"],
        "is_ord": col["is_ord"].tolist(),
        "counts": col["counts"],
        "level": col["levels"].tolist(),
        "label": col["labels"],
        "target": np.round(tgt, P_DECIMALS).tolist(),
        "p": np.round(col["p"], P_DECIMALS).tolist(),
        "p_calibrated": np.round(p_cal, P_DECIMALS).tolist(),
        "temperature_applied": gran,
        "conf": np.round(conf, P_DECIMALS).tolist(),
        "conf_calibrated": np.round(conf_cal, P_DECIMALS).tolist(),
        # 逐样本软正确率。可靠性图要的 (置信度, 正确率) 对就是
        # (conf, soft_correct) —— 所以图能从这一份直接重画，不必再碰模型。
        "soft_correct": np.round(tgt[rows, idx], P_DECIMALS).tolist(),
    }
    return block


def main():
    unbuffer_stdout()
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本项目的所有实测数字都基于 GPU；--device cpu 不在支持范围内。")

    temp = (load_temperature(args.temperature, args.ckpt, args.tokenizer)
            if args.temperature else None)
    tok, model, meta = load_decision(
        args.ckpt, args.tokenizer, device, hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers, candidate_crosstalk=args.crosstalk,
    )
    config = model.config
    args.max_len = args.max_len or meta.get("max_len", 1024)

    stem = os.path.splitext(os.path.basename(args.ckpt))[0]
    out_dir = os.path.join(args.out, stem)
    data_stem = os.path.basename(os.path.normpath(args.data))
    os.makedirs(out_dir, exist_ok=True)

    env = {
        "ckpt": os.path.abspath(args.ckpt),
        "ckpt_sha1": sha1_of(args.ckpt),
        "data": os.path.abspath(args.data),
        "tokenizer_sha1": sha1_of(os.path.join(args.tokenizer, "tokenizer.json")),
        "n_params": sum(p.numel() for p in model.parameters()),
        "max_len": args.max_len,
        "crosstalk": config.candidate_crosstalk,
        "prefix_blocked": config.prefix_blocked,
        "bins": args.bins,
        "binning": args.binning,
        "temperature_file": args.temperature,
    }

    index = {}
    for name in args.sets:
        path = os.path.join(args.data, f"{name}.jsonl")
        if not os.path.exists(path):
            print(f"  {name:11s} 跳过（无 {path}）")
            continue
        ds = DecisionDataset(path, tok, max_len=args.max_len, augment_k=False)
        if len(ds) == 0:
            print(f"  {name:11s} 跳过（空）")
            continue

        t0 = time.time()
        col = collect(model, ds, device, args.batch_size, args.limit)
        if col is None:
            print(f"  {name:11s} 跳过（0 条前向）")
            continue
        k = effective_k(col["mask"])
        raw = metrics_for(col, col["p"], args)

        p_cal = col["p"]
        calibrated = {}
        if temp is not None:
            for gran in GRANULARITIES:
                if gran != "global" and not temp.get(gran):
                    continue
                pg = apply_temperature(col["p"], temp, col["primitive"], k,
                                       col["mask"], gran)
                calibrated[gran] = metrics_for(col, pg, args)
            if args.apply not in calibrated:
                raise ValueError(f"温度文件没有 {args.apply}，请显式选择 --apply global")
            p_cal = apply_temperature(col["p"], temp, col["primitive"], k,
                                      col["mask"], args.apply)

        gen_versions = sorted({g for g in col["gen_version"] if g})
        result = {
            "set": name, "n": col["n"], "env": env,
            "gen_version": gen_versions,
            "metrics": raw,
            "calibrated": calibrated,
            "elapsed_s": round(time.time() - t0, 1),
        }
        if not args.no_per_sample:
            # **没给 `--temperature` 时 `temperature_applied` 必须是 None。**
            # 以前这里恒写 `args.apply`（默认 "primitive_k"），于是一份根本没做温度
            # 校准的评测，落盘的 per_sample 却声称"已应用 primitive_k"，可靠性图上
            # 右图标题也印成"温度校准后" —— 而两条曲线逐点相同。读者会把它读成
            # "温度校准毫无效果"，真相是它没被运行过。这是本项目最不该出的那类错：
            # 数字没错，标签错了。
            result["per_sample"] = per_sample_block(
                col, p_cal, args.apply if temp is not None else None)

        # 文件名带上是哪个数据集。**这不是美观问题**：旗舰流程要在
        # `dataset/synth` 和 `dataset/public` 上各跑一次 `--sets test_known`，
        # 而两次的 `--out` 是同一个目录 —— 只用 `<set>.json` 命名的话，第二次会
        # 静默覆盖第一次，于是"合成集与真实集并排"变成"两次都只看得到真实集"。
        # 图照样出得来，只是那张图不再是它标题所说的东西。
        out_path = os.path.join(out_dir, f"{data_stem}_{name}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False)
        size_mb = os.path.getsize(out_path) / 1e6
        index[f"{data_stem}_{name}"] = {
            "path": os.path.relpath(out_path, args.out), "n": col["n"],
            "size_mb": round(size_mb, 1)}

        a = raw["all"]
        print(f"\n  {name}  n={col['n']}  K_used max {int(k.max())}  ({time.time()-t0:.0f}s)")
        print(f"    未校准      acc {a['accuracy']:.3f}  ECE {a['ece']:.4f}  "
              f"distribution_l2 {a['distribution_l2']:.4f}  NLL {a['nll']:.4f}")
        for gran in GRANULARITIES:
            if gran not in calibrated:
                continue
            g = calibrated[gran]["all"]
            print(f"    温度/{gran:12s} ECE {g['ece']:.4f}  distribution_l2 {g['distribution_l2']:.4f}  "
                  f"NLL {g['nll']:.4f}  (acc 不变 {g['accuracy']:.3f})")
        if "soft_targets" in raw:
            c = raw["soft_targets"]
            print(f"    soft_targets 子集 acc {c['accuracy']:.3f}  "
                  f"ECE {c['ece']:.4f}  distribution_l2 {c['distribution_l2']:.4f}")
        for prov in sorted(raw):
            if prov in ("all", "soft_targets"):
                continue
            m = raw[prov]
            print(f"      {prov:17s} n={m['n']:6d}  acc {m['accuracy']:.3f}  "
                  f"ECE {m['ece']:.4f}")
        print(f"    -> {out_path}  ({size_mb:.1f} MB)")

    with open(os.path.join(out_dir, "index.json"), "w", encoding="utf-8") as f:
        json.dump({"env": env, "sets": index}, f, ensure_ascii=False, indent=2)
    peak = peak_vram_warn()
    print(f"\n  峰值显存 {peak:.2f} GB  -> {out_dir}/index.json")
    print(f"  下一步：python eval/make_reliability_plot.py --eval {out_dir}")


if __name__ == "__main__":
    main()
