"""
MiniSystemOne 设计契约的不变量测试。

这里断言的不是"模型能跑"，而是三条架构契约。它们只在
    RoPE 按 position_ids 索引  +  prefix_blocked=True  +  candidate_crosstalk=False
下成立；任何一条被改坏都会静默摧毁候选顺序不变性与分块推理，而且不会报错。

  A. 候选独立性 —— 候选 k 的 logit 只依赖 (前缀, 候选 k 自己的 token)。
     做法：把候选 k 单独喂给模型，其 logit 必须与一次喂 K 个候选时相同。
     这比"置换前后一致"更强：它直接验证独立性契约本身。
  B. 置换不变性 —— 置换候选块，p 随之置换，其余不变。
  C. 分块不变性 —— decide_chunked(chunk=1/2) 与单次前向的 p 一致。
  D. 前缀不受候选影响 —— 改掉候选后，前缀隐状态逐位相同。
     这是前缀 K/V 缓存复用的前提（也是分块推理正确的必要条件）。

另附：参数分解核对、mask 健全性、以及两个可选基准 —— `--bench_seconds` 实测训练
吞吐，`--bench_mask` 实测 SDPA 的 mask 格式选型（那组数会随 torch 版本漂移，
换版本后要重跑）。

用法：
    python scripts/smoke_test.py                      # CPU 上全部跑一遍
    python scripts/smoke_test.py --device cuda
    python scripts/smoke_test.py --device cuda --bench_seconds 30
    python scripts/smoke_test.py --device cuda --bench_mask
"""
import argparse
import os
import random
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_system_one import (
    DecisionConfig, MiniSystemOneForDecision, build_position_ids,
    SEG_STATE, SEG_QUESTION, SEG_CANDIDATE, SEG_PAD, NEG_INF,
)
from model.serialize import (
    pack_example, pack_plain, build_attn_mask, build_prefix_mask, collate_packed,
)

SEP_ID = 3          # <sep>
PAD_ID = 0
VOCAB = 6400

TIERS = {
    # 名字: (hidden, layers, heads, kv_heads, inter)   参数量见 test_param_breakdown
    "26m": (512, 8, 8, 4, 1280),
    "65m": (768, 8, 8, 4, 2304),
}


def build_config(tier="26m", **kw):
    h, l, heads, kv, inter = TIERS[tier]
    return DecisionConfig(hidden_size=h, num_hidden_layers=l, num_attention_heads=heads,
                          num_key_value_heads=kv, intermediate_size=inter,
                          vocab_size=VOCAB, **kw)


# ---------------------------------------------------------------------------
# 构造随机但合法的打包样本
# ---------------------------------------------------------------------------
def make_case(rng, prefix_len=40, cand_lens=(5, 12, 3, 20, 7)):
    state_ids = [rng.randrange(20, VOCAB) for _ in range(prefix_len // 2)]
    question_ids = [rng.randrange(20, VOCAB) for _ in range(prefix_len - prefix_len // 2)]
    candidates = [[rng.randrange(20, VOCAB) for _ in range(L)] for L in cand_lens]
    ids, seg, cid, spans = pack_example(state_ids, question_ids, candidates, SEP_ID)
    return dict(ids=ids, seg=seg, cid=cid, spans=spans,
                state_ids=state_ids, question_ids=question_ids, candidates=candidates)


def forward_ids(model, ids, seg, cid, spans, dtype):
    """跑一次前向，返回 DecisionOutput 和 collate 后的 batch。"""
    dev = next(model.parameters()).device
    batch = collate_packed(
        [{"input_ids": ids, "seg_id": seg, "cand_id": cid, "cand_spans": spans}],
        pad_id=PAD_ID, sep_id=SEP_ID, device=dev,
    )
    attn = build_attn_mask(batch["seg_id"], batch["cand_id"],
                           model.config.candidate_crosstalk,
                           model.config.prefix_blocked, dtype=dtype)
    out = model(batch["input_ids"], seg_id=batch["seg_id"], cand_id=batch["cand_id"],
                cand_span=batch["cand_span"], cand_mask=batch["cand_mask"],
                prefix_mask=batch["prefix_mask"], attention_mask=attn)
    return out, batch


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


# ---------------------------------------------------------------------------
# A. 候选独立性
# ---------------------------------------------------------------------------
def test_candidate_independence(model, case, dtype, tol):
    """候选 k 单独前向的 logit == K 个候选一起前向时第 k 个 logit。"""
    out, _ = forward_ids(model, case["ids"], case["seg"], case["cid"], case["spans"], dtype)
    worst = 0.0
    for k, cand in enumerate(case["candidates"]):
        ids, seg, cid, spans = pack_example(case["state_ids"], case["question_ids"], [cand], SEP_ID)
        solo, _ = forward_ids(model, ids, seg, cid, spans, dtype)
        worst = max(worst, maxdiff(out.logits[0, k], solo.logits[0, 0]))
    ok = worst < tol
    print(f"  [A] 候选独立性      max|Δlogit| = {worst:.3e}   (tol {tol:.0e})  "
          f"{'OK' if ok else 'FAIL'}")
    return ok, worst


# ---------------------------------------------------------------------------
# B. 置换不变性
# ---------------------------------------------------------------------------
def test_permutation(model, case, dtype, tol, n_perm=3, seed=0):
    out, _ = forward_ids(model, case["ids"], case["seg"], case["cid"], case["spans"], dtype)
    p0 = torch.softmax(out.logits.float(), dim=-1)
    rng = random.Random(seed)
    k = len(case["candidates"])
    worst_logit, worst_p = 0.0, 0.0
    for _ in range(n_perm):
        perm = list(range(k))
        rng.shuffle(perm)
        cands = [case["candidates"][j] for j in perm]
        ids, seg, cid, spans = pack_example(case["state_ids"], case["question_ids"], cands, SEP_ID)
        pout, _ = forward_ids(model, ids, seg, cid, spans, dtype)
        inv = [0] * k
        for new, old in enumerate(perm):
            inv[old] = new
        idx = torch.tensor(inv, device=out.logits.device)
        worst_logit = max(worst_logit, maxdiff(out.logits[0], pout.logits[0][idx]))
        worst_p = max(worst_p, maxdiff(p0[0], torch.softmax(pout.logits.float(), dim=-1)[0][idx]))
    ok = worst_logit < tol
    print(f"  [B] 置换不变性      max|Δlogit| = {worst_logit:.3e}   max|Δp| = {worst_p:.3e}  "
          f"(tol {tol:.0e})  {'OK' if ok else 'FAIL'}")
    return ok, worst_logit


# ---------------------------------------------------------------------------
# C. 分块不变性
# ---------------------------------------------------------------------------
def test_chunking(model, case, dtype, tol):
    ids, seg, cid, spans = pack_example(case["state_ids"], case["question_ids"],
                                        case["candidates"], SEP_ID)
    out, _ = forward_ids(model, ids, seg, cid, spans, dtype)
    p_ref = torch.softmax(out.logits.float(), dim=-1)
    worst = 0.0
    for chunk in (1, 2, 3):
        p, logits = model.decide_chunked(case["state_ids"], case["question_ids"],
                                         case["candidates"], SEP_ID, chunk=chunk)
        worst = max(worst, maxdiff(p_ref[0], p[0]))
    ok = worst < tol
    print(f"  [C] 分块不变性      max|Δp| = {worst:.3e}   (tol {tol:.0e})  "
          f"{'OK' if ok else 'FAIL'}")
    return ok, worst


# ---------------------------------------------------------------------------
# D. 前缀不受候选影响
# ---------------------------------------------------------------------------
def test_prefix_isolation(model, case, dtype, tol):
    out_a, _ = forward_ids(model, case["ids"], case["seg"], case["cid"], case["spans"], dtype)
    other = [[x + 7 for x in c] for c in case["candidates"]]
    ids, seg, cid, spans = pack_example(case["state_ids"], case["question_ids"], other, SEP_ID)
    out_b, _ = forward_ids(model, ids, seg, cid, spans, dtype)
    n_pref = build_prefix_mask(torch.tensor([case["seg"]])).sum().item()
    worst = maxdiff(out_a.hidden_states[0, :n_pref], out_b.hidden_states[0, :n_pref])
    ok = worst < tol
    print(f"  [D] 前缀不受候选影响 max|Δh_prefix| = {worst:.3e}  (tol {tol:.0e})  "
          f"{'OK' if ok else 'FAIL'}")
    return ok, worst


# ---------------------------------------------------------------------------
# mask 健全性
# ---------------------------------------------------------------------------
def test_mask_sanity(device):
    """整行全被 mask 会产生 NaN，并经 LayerNorm 污染全序列。这里直接查。"""
    rng = random.Random(1)
    case = make_case(rng)
    seg = torch.tensor([case["seg"]], device=device)
    cid = torch.tensor([case["cid"]], device=device)
    m = build_attn_mask(seg, cid, False, True, dtype=torch.float32)[0, 0]

    fully = (m.min(dim=-1).values > -1e8).sum().item()   # 没有任何可选 key 的行数
    has_inf = torch.isinf(m).any().item()
    k_real = (seg != SEG_PAD)
    bad_pad = (m[:, k_real[0]].max(dim=-1).values < -1e8).sum().item()   # 真实 key 全被挡的行

    # 候选只能看到前缀 + 自身 span
    cand_pos = (seg[0] == SEG_CANDIDATE)
    cid1d = cid[0]
    for i in torch.nonzero(cand_pos).flatten().tolist():
        allowed = (m[i] > -1e8).nonzero().flatten()
        legal = ((cid1d[allowed] == cid1d[i]) | (cid1d[allowed] < 0)).all().item()
        if not legal:
            bad_pad += 1

    ok = (fully == 0) and (not has_inf) and (bad_pad == 0)
    print(f"  [E] mask 健全性     全屏蔽行={fully}  含inf={has_inf}  非法可见={bad_pad}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok, 0.0


# ---------------------------------------------------------------------------
# 位置分配
# ---------------------------------------------------------------------------
def test_position_ids(device):
    """每个候选的位置必须从 prefix_len 起算，且与前缀自然位置不冲突地分层。"""
    rng = random.Random(2)
    case = make_case(rng)
    seg = torch.tensor([case["seg"]], device=device)
    cid = torch.tensor([case["cid"]], device=device)
    pos = build_position_ids(seg, cid)[0]
    n_pref = sum(1 for s in case["seg"] if s in (SEG_STATE, SEG_QUESTION))
    bad = 0
    for k, (s, t) in enumerate(case["spans"]):
        want = torch.arange(n_pref, n_pref + (t - s), device=device)
        if not torch.equal(pos[s:t], want):
            bad += 1
    if not torch.equal(pos[:n_pref], torch.arange(n_pref, device=device)):
        bad += 1
    # 纯文本（无候选）退回自然位置
    plain = torch.tensor([pack_plain([7, 8, 9, 10])[1]], device=device)
    if not torch.equal(build_position_ids(plain)[0], torch.arange(4, device=device)):
        bad += 1
    print(f"  [F] 位置分配        prefix_len={n_pref}  K={len(case['spans'])}  "
          f"不一致={bad}  {'OK' if bad == 0 else 'FAIL'}")
    return bad == 0, 0.0


# ---------------------------------------------------------------------------
# H. 大候选集：255 个候选单次前向 vs 分块
# ---------------------------------------------------------------------------
def test_large_k(model, dtype, device, prefix_len=700, k=255, cand_len=21, tol=2e-2):
    """README 的头号能力主张：255 个候选能在单次前向里做完，且分块路径一致。"""
    rng = random.Random(3)
    state_ids = [rng.randrange(20, VOCAB) for _ in range(prefix_len // 2)]
    question_ids = [rng.randrange(20, VOCAB) for _ in range(prefix_len - prefix_len // 2)]
    candidates = [[rng.randrange(20, VOCAB) for _ in range(cand_len)] for _ in range(k)]
    ids, seg, cid, spans = pack_example(state_ids, question_ids, candidates, SEP_ID)

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    out, _ = forward_ids(model, ids, seg, cid, spans, dtype)
    if device.type == "cuda":
        torch.cuda.synchronize()
    t_single = (time.perf_counter() - t0) * 1000
    peak = torch.cuda.max_memory_allocated() / 1024 ** 2 if device.type == "cuda" else 0.0
    p_ref = torch.softmax(out.logits.float(), dim=-1)

    p_chunk, _ = model.decide_chunked(state_ids, question_ids, candidates, SEP_ID, chunk=64)
    worst = maxdiff(p_ref[0], p_chunk[0])
    ok = worst < tol and p_ref.shape[-1] == k
    print(f"  [H] 255 候选单次前向  S={len(ids)}  logits={tuple(p_ref.shape)}  "
          f"{t_single:.0f} ms  峰值 {peak:.0f} MB  max|Δp|(vs chunk)={worst:.3e}  "
          f"{'OK' if ok else 'FAIL'}")
    return ok, worst


# ---------------------------------------------------------------------------
# 参数分解
# ---------------------------------------------------------------------------
def test_param_breakdown(device):
    print("  [G] 参数分解")
    for tier in TIERS:
        model = MiniSystemOneForDecision(build_config(tier)).to(device)
        n = lambda m: sum(p.numel() for p in m.parameters())
        enc = n(model.encoder)
        emb = model.encoder.embed_tokens.weight.numel()
        head = n(model.head)
        seg_emb = model.encoder.embed_segments.weight.numel()
        total = n(model)
        print(f"      {tier}: encoder {enc/1e6:.2f}M (含 embed {emb/1e6:.2f}M + seg {seg_emb})"
              f" + head {head/1e6:.2f}M = {total/1e6:.2f}M")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return True, 0.0


# ---------------------------------------------------------------------------
# 训练吞吐实测
# ---------------------------------------------------------------------------
def synth_train_batch(bsz, seq_len, k, rng, device):
    """造一个总长≈seq_len、K 个候选的 batch。前缀占 45%，其余均分给候选。"""
    n_pref = int(seq_len * 0.45)
    per = max((seq_len - n_pref) // k - 1, 1)
    examples = []
    for _ in range(bsz):
        state_ids = [rng.randrange(20, VOCAB) for _ in range(n_pref // 2)]
        question_ids = [rng.randrange(20, VOCAB) for _ in range(n_pref - n_pref // 2)]
        cands = [[rng.randrange(20, VOCAB) for _ in range(per)] for _ in range(k)]
        ids, seg, cid, spans = pack_example(state_ids, question_ids, cands, SEP_ID)
        t = torch.rand(k)
        t = t / t.sum()
        examples.append({"input_ids": ids, "seg_id": seg, "cand_id": cid,
                         "cand_spans": spans, "target": t.tolist(),
                         "levels": [0] * k, "primitive": "choice"})
    return collate_packed(examples, pad_id=PAD_ID, sep_id=SEP_ID, device=device)


def bench(args):
    dev = torch.device(args.device)
    print(f"========== 训练吞吐实测 ({args.tier}, B={args.bs}, S≈{args.seq}, K={args.k}) ==========")
    dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    model = MiniSystemOneForDecision(build_config(args.tier)).to(dev)
    model.train()
    model.to(dtype)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(dev.type, enabled=(dtype == torch.float16))
    rng = random.Random(0)
    batch = synth_train_batch(args.bs, args.seq, args.k, rng, dev)
    attn = build_attn_mask(batch["seg_id"], batch["cand_id"],
                           model.config.candidate_crosstalk,
                           model.config.prefix_blocked, dtype=dtype)

    step = dict(input_ids=batch["input_ids"], seg_id=batch["seg_id"], cand_id=batch["cand_id"],
                cand_span=batch["cand_span"], cand_mask=batch["cand_mask"],
                prefix_mask=batch["prefix_mask"], attention_mask=attn,
                target=batch["target"], is_ord=batch["is_ord"])

    for _ in range(3):                                  # 预热
        with torch.autocast(dev.type, dtype=dtype, enabled=(dev.type == "cuda")):
            out = model(**step)
        scaler.scale(out.loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
    if dev.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    n_steps = 0
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < args.bench_seconds:
        with torch.autocast(dev.type, dtype=dtype, enabled=(dev.type == "cuda")):
            out = model(**step)
        scaler.scale(out.loss).backward()
        scaler.step(opt)
        scaler.update()
        opt.zero_grad(set_to_none=True)
        n_steps += 1
    if dev.type == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    tok = args.bs * batch["input_ids"].shape[1]
    ms = dt / max(n_steps, 1) * 1000
    print(f"  {n_steps} 步 / {dt:.2f}s -> {ms:.0f} ms/step, {tok * n_steps / dt / 1000:.1f} k tok/s")
    if dev.type == "cuda":
        peak = torch.cuda.max_memory_allocated() / 1024 ** 2
        flag = "  ⚠ 超过 6.5 GB 预算 —— Windows 会在 ~7.5GB 静默换页" if peak > 6500 else ""
        print(f"  峰值显存 {peak:.0f} MB{flag}")
    print("  注：S 已上取整到 8 的倍数，tok/s 按实际 S 计。")


def bench_mask(args):
    """SDPA 的 mask 格式基准 —— 回答"为什么 mask 必须是 float additive"。

    这不是吞吐测试，是**后端选择**测试。`scaled_dot_product_attention` 会按
    (dtype, mask 类型, head 数) 自动挑内核，而挑错了不会报错，只会慢一个量级：

      - bool mask 走不了 EFFICIENT，只能落回 MATH；
      - float additive mask 可以走 EFFICIENT；
      - `enable_gqa=True` 在 EFFICIENT/FLASH 上直接抛错（融合内核要求 q/k/v 的
        head 数相同），只有 CUDNN 接受 —— 所以本项目沿用 MiniMind 的显式 repeat_kv。

    实测数字见 `docs/DESIGN.md` 差异表第 3 条。**这些数会随 torch 版本漂移**，
    requirements.txt 钉的版本变了就重跑这一条，不要沿用旧结论。
    """
    dev = torch.device(args.device)
    if dev.type != "cuda":
        print("========== SDPA mask 基准 ==========")
        print("  跳过：CPU 上没有 EFFICIENT/FLASH 内核，四条路径的差异不存在。")
        return

    from torch.nn.attention import SDPBackend, sdpa_kernel

    b, h, s, d = 8, 8, 1024, 64
    dt = torch.bfloat16
    print(f"========== SDPA mask 基准 (B{b}/H{h}/S{s}/D{d}, {str(dt).split('.')[-1]}) ==========")
    q = torch.randn(b, h, s, d, device=dev, dtype=dt)
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    # 下三角：任何一行都至少有一个可见列，所以不会出现整行 mask（那会出 NaN）。
    keep = torch.ones(s, s, device=dev, dtype=torch.bool).tril()
    masks = {
        "无 mask": None,
        "bool mask": keep,
        "float additive mask": torch.where(keep, 0.0, float("-inf")).to(dt),
    }

    def run(attn_mask, force_math=False):
        for _ in range(20):                                     # 预热
            _one(attn_mask, force_math)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            _one(attn_mask, force_math)
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / n * 1000
        return ms, torch.cuda.max_memory_allocated() / 1024 ** 2

    def _one(attn_mask, force_math):
        ctx = sdpa_kernel(SDPBackend.MATH) if force_math else _null()
        with ctx:
            torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

    rows = []
    for name, m in masks.items():
        ms, mb = run(m)
        rows.append((name, ms, mb))
    ms, mb = run(None, force_math=True)
    rows.append(("强制 MATH 后端（无 mask）", ms, mb))

    base = rows[0][1]
    for name, ms, mb in rows:
        print(f"  {name:26s} {ms:8.3f} ms  {mb:7.1f} MB   {ms/base:5.2f}×")
    print(f"  默认启用：flash={torch.backends.cuda.flash_sdp_enabled()} "
          f"efficient={torch.backends.cuda.mem_efficient_sdp_enabled()} "
          f"math={torch.backends.cuda.math_sdp_enabled()}")
    print("  结论：float additive mask 与无 mask 同量级；bool mask 更慢更费显存；"
          "MATH 是灾难级回退。")


class _null:
    """空上下文管理器，让 `_one` 的两条分支写法一致。"""
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="MiniSystemOne 设计契约不变量测试")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--tier", default="26m", choices=list(TIERS))
    p.add_argument("--dtype", default=None, choices=["float32", "bfloat16"],
                   help="默认在 GPU 上测 bf16、CPU 上测 fp32；两种都测需要跑两遍")
    p.add_argument("--tol", type=float, default=None, help="覆盖自动容差")
    p.add_argument("--skip_big", action="store_true", help="跳过 255 候选的长序列测试")
    p.add_argument("--bench_seconds", type=float, default=0, help=">0 时附加训练吞吐实测")
    p.add_argument("--bench_mask", action="store_true",
                   help="附加 SDPA mask 格式基准（验证 float additive mask 的选型）")
    p.add_argument("--bs", type=int, default=16)
    p.add_argument("--seq", type=int, default=512)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    dev = torch.device(args.device)
    if args.dtype is not None:
        dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    else:
        dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32
    # bf16 尾数只有 8 位，两次前向的浮点结合顺序差异会被放大；容差按 dtype 定
    tol = args.tol if args.tol is not None else (1e-5 if dtype == torch.float32 else 2e-2)

    print(f"========== MiniSystemOne 不变量测试 ({args.tier}, {dev.type}, {str(dtype).split('.')[-1]}) ==========")
    torch.manual_seed(args.seed)
    model = MiniSystemOneForDecision(build_config(args.tier)).to(dev).to(dtype).eval()
    case = make_case(random.Random(args.seed))

    results = [
        test_candidate_independence(model, case, dtype, tol),
        test_permutation(model, case, dtype, tol),
        test_chunking(model, case, dtype, tol),
        test_prefix_isolation(model, case, dtype, tol),
        test_mask_sanity(dev),
        test_position_ids(dev),
    ]
    if not args.skip_big:
        results.append(test_large_k(model, dtype, dev, tol=tol))
    del model
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    results.append(test_param_breakdown(dev))

    failed = [i for i, (ok, _) in enumerate(results) if not ok]
    print("=" * 60)
    print("全部通过" if not failed else f"失败 {len(failed)} 项: {failed}")

    if args.bench_seconds > 0:
        bench(args)
    if args.bench_mask:
        bench_mask(args)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
