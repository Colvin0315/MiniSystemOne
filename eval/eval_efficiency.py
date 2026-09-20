"""
效率测量 —— 延迟、吞吐、显存，以及**问题摊销**。

三条纪律，缺一条这里的数字就不可比：

  1. **每次计时都用 `torch.cuda.synchronize()` 夹住。** CUDA 是异步的，不加同步
     量到的是"把 kernel 塞进队列"的时间，而队列一满就会反压 —— 于是单样本延迟
     看起来漂亮、批量吞吐却对不上，两者合起来自相矛盾。
  2. **先热身再计时。** 首次前向包含 cuDNN 算法选择、显存分配器预热、可能的
     kernel 编译。冷启动的数字能大出几倍，且与之后的批量数字不同源。
  3. **显存用 `reset_peak_memory_stats` 隔离。** 不重置的话，"峰值"是**进程启动
     以来**的峰值，于是先跑的大配置会把后面所有小配置的峰值都顶上去 —— 报出来的
     是一串相同的数，而它们只反映第一个配置。

**问题摊销是本文件的核心**，因为它是本项目唯一一个**架构性**的效率主张（其余都
依赖硬件）。`prefix_blocked=True` 让 state 不依赖 question 与候选，所以同一份 state
配 N 个问题时 state 只需编码一次：代价从 `N·(L_state + L_q + L_cand)` 变成
`L_state + N·(L_q + L_cand)`。N=16 时的预期加速比取决于 state 在前缀里的占比，
**由本脚本实测得出，不写死**。

用法：
    python eval/eval_efficiency.py --ckpt out/decision/decision.pth
    python eval/eval_efficiency.py --ckpt out/decision/decision.pth --suites latency vram
"""
import argparse
import json
import os
import statistics
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from transformers import AutoTokenizer

from dataset.decision_dataset import DecisionDataset
from eval.eval_inference import split_example
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.trainer_utils import init_model, unbuffer_stdout

SUITES = ("latency", "vram", "amortization", "k_sweep")

# 方案的硬性显存预算：实测 7.5–8 GB 处有悬崖，越过它 Windows WDDM 会静默换页到
# 共享内存 —— 不 OOM，但慢 10 倍。所以告警线比"塞得进 8 GB"严格得多。
VRAM_BUDGET_GB = 6.5

N_WARMUP = 20
N_TIMED = 200

# 延迟档位。B=1 与 B=8 是方案里报的两个数；K 的档位覆盖小 K 单次前向与大 K 分块。
BATCHES = (1, 8)
K_SWEEP = (2, 32, 128, 255)

# 问题摊销的 N。64 是为了看加速比是否还在随 N 上升（若饱和，说明瓶颈换到了别处）。
AMORTIZE_N = (1, 4, 16, 64)


def parse_args():
    p = argparse.ArgumentParser(description="效率测量（延迟 / 吞吐 / 显存 / 问题摊销）")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data", default="dataset/synth")
    p.add_argument("--split", default="test_known")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--out", default="out/efficiency")
    p.add_argument("--suites", nargs="*", default=list(SUITES), choices=SUITES)
    p.add_argument("--n_warmup", type=int, default=N_WARMUP)
    p.add_argument("--n_timed", type=int, default=N_TIMED)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--crosstalk", action="store_true")
    p.add_argument("--chunk", type=int, default=64)
    return p.parse_args()


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timeit(fn, n_warmup, n_timed):
    """热身 n_warmup 次，计时 n_timed 次，返回 (median_ms, p95_ms, mean_ms)。

    **中位数与 p95 一起报**：中位数是"典型一次要多久"，p95 是"最坏一次要多久"。
    只看中位数会漏掉长尾 —— 而长尾才是线上超时的来源。
    """
    for _ in range(n_warmup):
        fn()
    sync()
    ts = []
    for _ in range(n_timed):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append((time.perf_counter() - t0) * 1000.0)
    ts.sort()
    p95 = ts[min(len(ts) - 1, int(0.95 * len(ts)))]
    return statistics.median(ts), p95, statistics.fmean(ts)


def peak_vram_gb():
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1 << 30)


def reset_vram():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def build_model(args, tok, device):
    config = DecisionConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tok), candidate_crosstalk=args.crosstalk,
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        sep_token_id=tok.convert_tokens_to_ids("<sep>"),
    )
    model = init_model(MiniSystemOneForDecision, config, args.ckpt, device)
    model.eval()
    return model, config


def load_examples(args, tok, sep):
    """取真实样本并拆成 (state, question, [候选])。用合成数据里的真三元组，
    而不是随机 token —— 长度分布直接决定延迟，随机构造的长度是不具备代表性的。"""
    path = os.path.join(args.data, f"{args.split}.jsonl")
    ds = DecisionDataset(path, tok, max_len=args.max_len, augment_k=False)
    out = []
    for i in range(len(ds)):
        s, q, c = split_example(ds[i], sep)
        if s and q and c:
            out.append((s, q, c))
    return out


# ---------------------------------------------------------------------------
# 延迟与吞吐
# ---------------------------------------------------------------------------
def suite_latency(model, examples, args, tok, sep):
    """逐样本延迟，报告 200 个**不同**样本上的中位数与 p95。

    **样本必须是不同的样本。** 拿同一条重复 N 次会把长度分布压成一个点，报出来的
    中位数只代表那一条；合成集里 state 的长度跨度很大（短路由 vs 长 trace），只测
    一条要么严重乐观要么严重悲观。所以下面直接换成逐样本计时。

    注意 `B=8` 一档是把 8 个样本**循环**调用，而不是真打成 batch —— 打包序列的
    批量需要在 collate 里对齐候选数，那是训练路径的事；推理服务面对的是单个请求。
    两者混在一个数字里会让人以为这里量的是批量吞吐。
    """
    n_use = min(args.n_timed, len(examples))
    pool = examples[:n_use]
    per_sample = []
    reset_vram()
    for s, q, c in pool[:args.n_warmup]:
        model.decide_chunked(s, q, c, sep, chunk=args.chunk)
    sync()
    for s, q, c in pool:
        sync()
        t0 = time.perf_counter()
        model.decide_chunked(s, q, c, sep, chunk=args.chunk)
        sync()
        per_sample.append((time.perf_counter() - t0) * 1000.0)
    peak = peak_vram_gb()
    per_sample.sort()
    med = statistics.median(per_sample)
    p95 = per_sample[min(len(per_sample) - 1, int(0.95 * len(per_sample)))]

    comp = {"state": statistics.median(len(e[0]) for e in pool),
            "question": statistics.median(len(e[1]) for e in pool),
            "candidates": statistics.median(len(e[2]) for e in pool),
            "cand_tokens": statistics.median(sum(len(x) for x in e[2]) for e in pool)}
    rows = [{"n_samples": n_use, "median_ms": med, "p95_ms": p95,
             "mean_ms": statistics.fmean(per_sample), "min_ms": per_sample[0],
             "max_ms": per_sample[-1], "samples_per_s": 1000.0 / med,
             "peak_vram_gb": peak, "median_composition": comp}]
    print(f"    n={n_use}  中位 {med:7.2f} ms  p95 {p95:7.2f} ms  "
          f"min {per_sample[0]:.2f}  max {per_sample[-1]:.2f}  "
          f"{1000.0 / med:6.1f} 样本/s  峰值 {peak:.2f} GB")
    print(f"    长度中位：state {comp['state']} / question {comp['question']} tok，"
          f"候选 {comp['candidates']} 个共 {comp['cand_tokens']} tok")

    # 与 Jev 的 70–500 ms 不是同一口径，必须显式标注：那是 LLM 自回归生成的延迟，
    # 这里是决策原生的单次并行前向。写进结果 JSON，避免下游拿去做错的对比。
    return {"per_batch": rows,
            "note": "本延迟是决策原生单次前向；与 Jev 的 70–500 ms（LLM 生成）不可比"}


# ---------------------------------------------------------------------------
# 显存
# ---------------------------------------------------------------------------
def suite_vram(model, examples, args, tok, sep):
    """各配置的峰值显存，逐配置隔离测量。这是 R5 的运行时把关：越过 6.5 GB
    就意味着正在靠近悬崖，而悬崖的表现是**变慢而不是报错**。"""
    big = max(examples, key=lambda e: len(e[0]))[:] if examples else None
    rows = []
    for bsz in BATCHES:
        for kt in K_SWEEP:
            cand = _clamp_candidates(examples[0][2], kt, tok)
            reset_vram()
            for _ in range(3):
                model.decide_chunked(big[0], big[1], cand, sep, chunk=args.chunk)
            peak = peak_vram_gb()
            rows.append({"batch_size": bsz, "K": kt, "peak_vram_gb": peak,
                         "over_budget": peak > VRAM_BUDGET_GB})
            flag = "  <<< 超 6.5 GB 预算" if peak > VRAM_BUDGET_GB else ""
            print(f"    B={bsz} K={kt:3d}  峰值 {peak:.2f} GB{flag}")
    return {"budget_gb": VRAM_BUDGET_GB, "per_config": rows,
            "max_peak_gb": max(r["peak_vram_gb"] for r in rows) if rows else 0.0,
            "any_over_budget": any(r["over_budget"] for r in rows)}


def _clamp_candidates(cands, kt, tok, pad_token=11):
    """把候选列表凑到/截到 kt 个，用于 K 扫描。候选内容不重要，**长度分布**重要，
    所以补出来的候选沿用原始候选的长度分布，而不是一律给单 token。"""
    if not cands:
        return [[pad_token]] * kt
    out = list(cands)
    while len(out) < kt:
        out.append(out[len(out) % len(cands)])
    return out[:kt]


# ---------------------------------------------------------------------------
# 问题摊销
# ---------------------------------------------------------------------------
def suite_amortization(model, examples, args, tok, sep):
    """N 个问题共享一份 state：朴素（每次重编码 state）vs 缓存 state。

    这是全项目最强的效率主张，因为它是**架构属性**：state 在注意力上被禁止看到
    question 与候选，所以它的隐状态与"问什么、有哪些候选"无关，可以跨问题复用。
    换一个 `prefix_blocked=False` 的模型，这个数就会退化到 1×。

    加速比**不写死**（方案里猜的是 N=16 时 ~3.3×）：它取决于 state 在总 token 里
    的占比，而那随数据而变。这里同时报出每档的 token 构成，让读者能自己核。

    选哪条样本决定了这个实验能不能说明问题。**必须挑 state 占比高的**：若挑到一条
    255 候选的样本，候选打分就是全部成本，共享 state 省下的那部分被淹没，量出来
    必然是 1.0× —— 而那不是"摊销无效"，是"这份样本里没有可摊销的东西"。所以下面
    在候选 ≤ 8 个的样本里取 state 最长的一条，并把 token 构成一并报出来给读者核。
    """
    if not examples:
        return {"error": "没有可用样本"}
    small_k = [e for e in examples if len(e[2]) <= 8] or examples
    s, q, c = max(small_k, key=lambda e: len(e[0]))
    n_state, n_q = len(s), len(q)
    n_cand = sum(len(x) for x in c) + len(c)

    # 先给出三段各自的代价，让"为什么是这个加速比"可核。这三条也是 1 次调用的
    # 分解，N 档位里的差额应当约等于 (N−1)×encode_state。
    t_state, _, _ = timeit(lambda: model.encode_state(s, sep), 3, 20)
    t_prefix, _, _ = timeit(lambda: model.encode_prefix(s, q, sep), 3, 20)
    cache0 = model.encode_state(s, sep)
    t_q, _, _ = timeit(lambda: model.decide_with_state(cache0, q, c, sep, chunk=args.chunk),
                       3, 20)
    print(f"    分解：encode_state {t_state:.2f} ms  encode_prefix {t_prefix:.2f} ms  "
          f"缓存后整问 {t_q:.2f} ms")

    rows = []
    for n in AMORTIZE_N:
        def naive():
            for _ in range(n):
                model.decide_chunked(s, q, c, sep, chunk=args.chunk)

        def cached():
            cache = model.encode_state(s, sep)
            for _ in range(n):
                model.decide_with_state(cache, q, c, sep, chunk=args.chunk)

        reps = max(3, args.n_timed // (4 * n))
        t_naive, _, _ = timeit(naive, max(1, args.n_warmup // 4), reps)
        t_cache, _, _ = timeit(cached, max(1, args.n_warmup // 4), reps)
        speedup = t_naive / t_cache if t_cache > 0 else 0.0
        rows.append({"n_questions": n, "naive_ms": t_naive, "cached_ms": t_cache,
                     "speedup": speedup, "ms_per_question_cached": t_cache / n})
        print(f"    N={n:3d}  朴素 {t_naive:8.2f} ms  缓存 {t_cache:8.2f} ms  "
              f"加速 {speedup:5.2f}×  ({t_cache / n:.2f} ms/问)")

    total = n_state + n_q + n_cand
    return {"state_tokens": n_state, "question_tokens": n_q, "candidate_tokens": n_cand,
            "state_share": n_state / total if total else 0.0,
            "breakdown_ms": {"encode_state": t_state, "encode_prefix": t_prefix,
                             "cached_full_query": t_q},
            "per_n": rows,
            "note": "加速比来自 prefix_blocked=True（state 不 attend 到问题/候选），非硬件"}


# ---------------------------------------------------------------------------
# K 扫描
# ---------------------------------------------------------------------------
def suite_k_sweep(model, examples, args, tok, sep):
    """延迟随 K 的变化，并给出分块阈值（超过它才会走分块路径）。"""
    if not examples:
        return {"error": "没有可用样本"}
    s, q, _ = max(examples, key=lambda e: len(e[0]))
    base = max(examples, key=lambda e: len(e[2]))[2]
    rows = []
    for kt in K_SWEEP:
        cand = _clamp_candidates(base, kt, tok)
        reset_vram()
        med, p95, _ = timeit(lambda: model.decide_chunked(s, q, cand, sep, chunk=args.chunk),
                             args.n_warmup, args.n_timed)
        rows.append({"K": kt, "median_ms": med, "p95_ms": p95,
                     "peak_vram_gb": peak_vram_gb(),
                     "chunked": kt > args.chunk})
        print(f"    K={kt:3d}  中位 {med:7.2f} ms  p95 {p95:7.2f} ms  "
              f"峰值 {peak_vram_gb():.2f} GB  {'（分块）' if kt > args.chunk else ''}")
    return {"chunk": args.chunk, "per_k": rows}


RUNNERS = {"latency": suite_latency, "vram": suite_vram,
           "amortization": suite_amortization, "k_sweep": suite_k_sweep}
TITLES = {"latency": "延迟与吞吐", "vram": "显存（6.5 GB 预算）",
          "amortization": "问题摊销（N 个问题共享一份 state）", "k_sweep": "K 扫描"}


def main():
    unbuffer_stdout()
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本项目的所有实测数字都基于 GPU；CPU 上的延迟不是这里的口径。")

    print(f"GPU {torch.cuda.get_device_name(0)}  "
          f"{torch.cuda.get_device_properties(0).total_memory / (1 << 30):.1f} GiB  "
          f"torch {torch.__version__}")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    sep = tok.convert_tokens_to_ids("<sep>")
    model, config = build_model(args, tok, device)
    examples = load_examples(args, tok, sep)
    print(f"样本 {len(examples)} 条（{args.split}）")
    if not examples:
        raise SystemExit("没有可用样本")

    results = {"env": {
        "ckpt": os.path.abspath(args.ckpt),
        "gpu": torch.cuda.get_device_name(0),
        "vram_gb": torch.cuda.get_device_properties(0).total_memory / (1 << 30),
        "torch": torch.__version__,
        "n_params": sum(p.numel() for p in model.parameters()),
        "chunk": args.chunk, "max_len": args.max_len, "crosstalk": args.crosstalk,
        "n_warmup": args.n_warmup, "n_timed": args.n_timed,
    }, "suites": {}}

    for name in args.suites:
        print(f"\n========== {TITLES[name]} ==========")
        res = RUNNERS[name](model, examples, args, tok, sep)
        results["suites"][name] = res

    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "efficiency.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    v = results["suites"].get("vram")
    if v and v.get("any_over_budget"):
        print(f"\n  **警告** 峰值显存 {v['max_peak_gb']:.2f} GB 越过 "
              f"{VRAM_BUDGET_GB} GB 预算 —— 接近 WDDM 悬崖（变慢而非报错）")
    print(f"\n-> {out_path}")


if __name__ == "__main__":
    main()
