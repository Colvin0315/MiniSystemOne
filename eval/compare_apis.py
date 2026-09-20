"""
外部对比：ours vs TypeSafe `jev-latest` vs 一个通用 LLM（DeepSeek）。

**这份脚本是刻意的旁路。** 它不在 `build_dataset` / `train_*` / `eval_harness` 的
任何一条路径上，需要显式 opt-in 与 API key，而且**不把任何外部模型的逐样本输出
落盘** —— 只写聚合指标。本项目的训练数据全部来自程序合成 + 公开数据集，外部模型的
输出一个字节都没有进过训练。这与"蒸馏"是两件事，见 README 的 "Not a Jev
reproduction" 一节。

## 为什么对比要做在合成集上

判决性的指标是 **ECE —— 对着已知 P***。只有合成集有精确的条件分布（`explicit_rng`
的银行 RNG、`tie_set` 的并列集、`marginalized` 的边缘化）；公开集上的 target 是
有限个标注者投出来的，噪声底本身就压过了要测的差异。

**代价必须写明：ours 是在这个分布上训出来的，另两个是零样本。** 所以准确率这一列
对我们有利，不能当作"我们更准"的证据。这张表要读的是 ECE、Brier、延迟，以及
LLM 的 schema 错误率 —— 那几列没有这个问题。

## 指标口径

三个模型都产出"候选顺序上的分布"，然后**全部交给 `eval.eval_metrics` 的同一份
`soft_accuracy` / `ece` / `brier`**。不在这里另写一份评分代码：两份实现迟早漂移，
而漂移方向通常是"自己想展示的那个赢"。

用法：
    export TYPESAFE_API_KEY=...
    export DEEPSEEK_API_KEY=...
    python eval/compare_apis.py --n_per_source 8 --k_max 8
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import requests
from transformers import AutoTokenizer

from dataset.decision_dataset import DecisionDataset
from eval.eval_inference import collect
from eval.eval_metrics import brier, ece, soft_accuracy
from trainer.trainer_utils import unbuffer_stdout

TYPESAFE_URL = "https://api.typesafe.ai/v1/systemone"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
BINS = 15


def parse_args():
    p = argparse.ArgumentParser(description="外部 API 对比（不做训练，只测）")
    p.add_argument("--data", default="dataset/synth")
    p.add_argument("--split", default="test_known")
    p.add_argument("--ckpt", default="out/decision/decision.pth")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--n_per_source", type=int, default=8,
                   help="每个生成器取多少条；总数 = 6 × 这个值")
    p.add_argument("--k_max", type=int, default=8,
                   help="只取候选数 ≤ 这个值的样本，让三个系统的提示词都可控")
    p.add_argument("--systems", nargs="*", default=["ours", "jev", "deepseek"],
                   choices=["ours", "jev", "deepseek"])
    p.add_argument("--max_tokens", type=int, default=8192,
                   help="DeepSeek 是推理模型，reasoning 会先吃掉预算，给小了正文为空")
    p.add_argument("--timeout", type=int, default=90)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="out/compare_apis.json")
    return p.parse_args()


# ---------------------------------------------------------------------------
def pick_subset(path, n_per_source, k_max, seed):
    """按来源分层抽样，每个生成器取 n_per_source 条候选数 ≤ k_max 的。

    分层是必须的：不分层的话小 K 的生成器会被大 K 的淹没，而两者的难度完全不同。
    """
    rng = np.random.default_rng(seed)
    by_src = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if len(r["candidates"]) > k_max:
                continue
            by_src.setdefault(r["source"], []).append(r)
    picked = []
    for src in sorted(by_src):
        rows = by_src[src]
        idx = rng.permutation(len(rows))[:n_per_source]
        picked.extend(rows[i] for i in sorted(idx))
    return picked


def record_view(rec):
    """把一条 record 摊成三个系统共用的视图：state / question / 选项标签与文本。

    选项用 `label` 而不是 `text`：Noul 的 text 是 `<yes>` 这种特殊 token，对 LLM
    是噪声；`label` 才是语义标识（`yes` / `no` / `get_invoice`）。
    """
    labels = [c["label"] for c in rec["candidates"]]
    texts = [c["text"] for c in rec["candidates"]]
    assert len(set(labels)) == len(labels), f"候选标签重复：{labels}"
    return rec["state"], rec["question"], labels, texts


# ---------------------------------------------------------------------------
def run_ours(args, records):
    """我们自己的模型。走 `collect` —— 与 eval_harness / 训练循环同一条前向路径。"""
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
    from trainer.trainer_utils import init_model, verify_tokenizer

    verify_tokenizer(args.ckpt, args.tokenizer)
    cfg = DecisionConfig(hidden_size=512, num_hidden_layers=8, vocab_size=len(tok),
                         pad_token_id=tok.convert_tokens_to_ids("<pad>"),
                         sep_token_id=tok.convert_tokens_to_ids("<sep>"))
    model = init_model(MiniSystemOneForDecision, cfg, args.ckpt, "cuda")
    model.eval()

    # 只把选中的样本喂进去。临时落一份 jsonl，保证 `__getitem__` 的打包路径
    # 与真实评测完全一致（自己手工打包 = 第二个实现 = 迟早漂移）。
    tmp = os.path.join("out", "_cmp_subset.jsonl")
    os.makedirs("out", exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    ds = DecisionDataset(tmp, tok, max_len=1024, augment_k=False, seed=0)

    collect(model, ds, "cuda", batch_size=16, limit=1)      # 预热，不进计时
    torch.cuda.synchronize()
    t0 = time.time()
    col = collect(model, ds, "cuda", batch_size=16)
    torch.cuda.synchronize()
    dt = time.time() - t0

    # collect 输出是 (n, K_padded)；按每条的 K_used 截回候选长度
    k_used = np.asarray(col["K_used"] if "K_used" in col
                        else [int(m.sum()) for m in col["mask"]])
    probs = [col["p"][i, :int(k)].astype(np.float64) for i, k in enumerate(k_used)]
    return probs, dt / len(records) * 1000, {"latency_ms_per_item": dt / len(records) * 1000}


def run_jev(records, key, timeout):
    """TypeSafe。每个样本一次 Choice 调用。"""
    probs, lat, meta = [], [], {"calls": 0, "errors": 0, "in_tok": 0, "out_tok": 0}
    for rec in records:
        state, question, labels, texts = record_view(rec)
        body = {"state": state, "model": "jev-latest",
                "questions": {"q": {"type": "choice", "instructions": question,
                                    "criteria": {lb: tx for lb, tx in zip(labels, texts)}}}}
        t0 = time.time()
        try:
            r = requests.post(TYPESAFE_URL, json=body, timeout=timeout,
                              headers={"Authorization": f"Bearer {key}",
                                       "Content-Type": "application/json"})
            r.raise_for_status()
            a = r.json()
            lat.append((time.time() - t0) * 1000)
            meta["calls"] += 1
            u = a.get("usage") or {}
            meta["in_tok"] += u.get("input_tokens", 0)
            meta["out_tok"] += u.get("output_tokens", 0)
            got = a["answers"]["q"]["probabilities"]
            probs.append(np.asarray([float(got.get(lb, 0.0)) for lb in labels]))
        except Exception as e:
            meta["errors"] += 1
            print(f"    jev 失败：{type(e).__name__} {str(e)[:80]}", flush=True)
            probs.append(None)
    p50 = float(np.median(lat)) if lat else float("nan")
    return probs, p50, meta


def run_deepseek(records, key, timeout, max_tokens, model="deepseek-flash"):
    """通用 LLM。要求它输出一个分布 —— 这就是"verbalized confidence"那一档。"""
    probs, lat, meta = [], [], {"calls": 0, "errors": 0, "parse_fail": 0,
                                "in_tok": 0, "out_tok": 0}
    for rec in records:
        state, question, labels, texts = record_view(rec)
        opts = "\n".join(f"- {lb}  (={tx})" if tx != lb else f"- {lb}"
                         for lb, tx in zip(labels, texts))
        prompt = (
            "You are a decision model. Read the STATE, then answer the QUESTION by "
            "giving a probability distribution over the OPTIONS.\n"
            "Probabilities must be non-negative and sum to 1. Judge from the evidence "
            "in the STATE; do not assume missing information.\n\n"
            f"STATE:\n{state}\n\nQUESTION: {question}\n\nOPTIONS:\n{opts}\n\n"
            "Reply with JSON only, no prose, in exactly this form:\n"
            '{"probabilities": {' + ", ".join(f'"{lb}": 0.0' for lb in labels) + "}}"
        )
        t0 = time.time()
        try:
            r = requests.post(DEEPSEEK_URL, timeout=timeout,
                              headers={"Authorization": f"Bearer {key}"},
                              json={"model": model, "temperature": 0,
                                    "max_tokens": max_tokens,
                                    "response_format": {"type": "json_object"},
                                    "messages": [{"role": "user", "content": prompt}]})
            r.raise_for_status()
            d = r.json()
            lat.append((time.time() - t0) * 1000)
            meta["calls"] += 1
            u = d.get("usage") or {}
            meta["in_tok"] += u.get("prompt_tokens", 0)
            meta["out_tok"] += u.get("completion_tokens", 0)
            txt = d["choices"][0]["message"]["content"] or ""
            if not txt.strip():
                fr = d["choices"][0].get("finish_reason")
                raise ValueError(
                    f"正文为空（finish_reason={fr}，completion_tokens="
                    f"{u.get('completion_tokens')}，reasoning="
                    f"{(u.get('completion_tokens_details') or {}).get('reasoning_tokens')}）"
                    f" —— 推理预算被吃光了，加 --max_tokens")
            got = json.loads(txt).get("probabilities", {})
            probs.append(np.asarray([float(got.get(lb, 0.0)) for lb in labels]))
        except Exception as e:
            meta["errors"] += 1
            meta["parse_fail"] += 1
            print(f"    deepseek 失败：{type(e).__name__} {str(e)[:80]}", flush=True)
            probs.append(None)
    p50 = float(np.median(lat)) if lat else float("nan")
    return probs, p50, meta


# ---------------------------------------------------------------------------
def score(probs, records, tag):
    """把一个系统的逐样本分布交给 `eval_metrics` 打分。对不齐的样本剔除并报数。"""
    keep, P, T = [], [], []
    for i, (p, rec) in enumerate(zip(probs, records)):
        if p is None:
            continue
        K = len(rec["candidates"])
        p = np.asarray(p, dtype=np.float64)[:K]
        s = p.sum()
        if not np.isfinite(s) or s <= 0:
            continue
        p = np.clip(p / s, 0, None)                      # 外部模型的和未必恰好为 1
        t = np.asarray(rec["target"]["p"], dtype=np.float64)[:K]
        keep.append(i); P.append(p); T.append(t)
    if not P:
        return None
    Kmax = max(len(p) for p in P)
    Pm = np.zeros((len(P), Kmax)); Tm = np.zeros((len(P), Kmax))
    Mk = np.zeros((len(P), Kmax), bool)
    for i, (p, t) in enumerate(zip(P, T)):
        Pm[i, :len(p)] = p; Tm[i, :len(t)] = t; Mk[i, :len(p)] = True
    return {
        "n": len(P),
        "dropped": len(records) - len(P),
        # `soft_accuracy` 返回的是**逐样本**数组（见 eval_metrics:118），不是标量。
        "soft_acc": float(np.mean(soft_accuracy(Pm, Tm, Mk))),
        "ece": float(ece(Pm, Tm, Mk, BINS, "equal_mass")),
        "brier": float(brier(Pm, Tm, Mk)),
        "by_source": {
            src: {
                "n": int(sum(1 for i in keep if records[i]["source"] == src)),
                "soft_acc": float(np.mean(soft_accuracy(
                    Pm[[j for j, i in enumerate(keep) if records[i]["source"] == src]],
                    Tm[[j for j, i in enumerate(keep) if records[i]["source"] == src]],
                    Mk[[j for j, i in enumerate(keep) if records[i]["source"] == src]]))),
            } for src in sorted({records[i]["source"] for i in keep})
        },
    }


def main():
    # 长跑脚本的第一行。重定向时 stdout 是块缓冲：48 条 × 3 系统要跑七分钟，
    # 不切行缓冲的话日志文件这七分钟里一直是 0 字节 —— 看起来和卡死一样。
    unbuffer_stdout()
    args = parse_args()
    torch.manual_seed(args.seed)
    path = os.path.join(args.data, f"{args.split}.jsonl")
    records = pick_subset(path, args.n_per_source, args.k_max, args.seed)
    print(f"子集：{len(records)} 条（每个来源 ≤{args.n_per_source} 条，候选数 ≤{args.k_max}）")
    print(f"  来源分布：{ {s: sum(1 for r in records if r['source'] == s) for s in sorted({r['source'] for r in records})} }")
    print(f"  K 分布：{ {k: sum(1 for r in records if len(r['candidates']) == k) for k in sorted({len(r['candidates']) for r in records})} }")

    results = {}
    if "ours" in args.systems:
        print("\n========== ours ==========")
        t0 = time.time()
        probs, ms, meta = run_ours(args, records)
        results["ours"] = {"probs": probs, "ms_per_item": ms, "meta": meta,
                           "total_s": time.time() - t0}
        print(f"  {ms:.1f} ms/条  共 {time.time()-t0:.0f}s")

    if "jev" in args.systems:
        print("\n========== jev-latest ==========")
        key = os.environ.get("TYPESAFE_API_KEY")
        if not key:
            raise SystemExit("需要 env TYPESAFE_API_KEY")
        t0 = time.time()
        probs, ms, meta = run_jev(records, key, args.timeout)
        results["jev"] = {"probs": probs, "ms_per_item": ms, "meta": meta,
                          "total_s": time.time() - t0}
        print(f"  {ms:.1f} ms/条  {meta}  共 {time.time()-t0:.0f}s")

    if "deepseek" in args.systems:
        print("\n========== deepseek-flash ==========")
        key = os.environ.get("DEEPSEEK_API_KEY")
        if not key:
            raise SystemExit("需要 env DEEPSEEK_API_KEY")
        t0 = time.time()
        probs, ms, meta = run_deepseek(records, key, args.timeout, args.max_tokens)
        results["deepseek"] = {"probs": probs, "ms_per_item": ms, "meta": meta,
                               "total_s": time.time() - t0}
        print(f"  {ms:.1f} ms/条  {meta}  共 {time.time()-t0:.0f}s")

    print("\n========== 结果（同一份 eval_metrics 打分）==========")
    print(f"  {'系统':<10} {'n':>4} {'剔除':>5} {'软准确率':>9} {'ECE':>8} {'Brier':>8} {'ms/条':>8}")
    summary = {}
    for tag, r in results.items():
        s = score(r["probs"], records, tag)
        if not s:
            print(f"  {tag:<10} 无有效样本")
            continue
        summary[tag] = {**s, "ms_per_item": r["ms_per_item"], "meta": r["meta"]}
        print(f"  {tag:<10} {s['n']:>4} {s['dropped']:>5} {s['soft_acc']:>9.4f} "
              f"{s['ece']:>8.4f} {s['brier']:>8.4f} {r['ms_per_item']:>8.1f}")

    print("\n  逐来源软准确率：")
    for tag, s in summary.items():
        print(f"    {tag:<10} " + "  ".join(
            f"{k.split(':')[-1]}={v['soft_acc']:.3f}(n={v['n']})"
            for k, v in s["by_source"].items()))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"subset": len(records), "k_max": args.k_max,
                   "n_per_source": args.n_per_source, "summary": summary},
                  f, ensure_ascii=False, indent=2)
    print(f"\n  -> {args.out}")
    print("  注：**逐样本输出不落盘**，只有聚合指标。外部模型输出不进训练。")
    os.remove(os.path.join("out", "_cmp_subset.jsonl"))


if __name__ == "__main__":
    main()
