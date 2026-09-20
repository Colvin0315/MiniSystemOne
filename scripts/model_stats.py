"""
参数构成表。存在的理由是：**让读者能一眼证伪 README 里的每一个参数量声明。**

这个脚本刻意不 import 任何训练代码、不读 checkpoint、不碰 GPU。它只做一件事——
把三个模型类实例化（用 CPU 上的 meta/fake 参数）然后把参数量按模块拆开打印出来。
因此它可以在任何机器上秒级跑完，是审稿人/读者复核本项目数字的第一站。

要复核的几个非平凡声明：

1. **决策模型 26.89M = encoder 22.03M + embed 3.28M + head 1.58M。**
   `encoder` 里**含** `embed_tokens`，所以三项不能直接相加 —— 脚本显式扣掉重叠，
   打印 `encoder(含 embed)` 与 `encoder(不含 embed)` 两个数，避免读者把 3.28M 算两遍。

2. **Stage 1 的 MLM 模型 25.31M。** `lm_head` 与 `embed_tokens` 绑定
   （`tie_word_embeddings`），**不新增参数** —— 这一点很容易被误读成
   "多了一个 3.28M 的输出层"，脚本把 `lm_head` 单独标成 tied 且计数为 0。

3. **两个档位（26M / 65M）的构成。** `--tier` 切换，用来核 README 的两行表格。

用法：
    python scripts/model_stats.py                 # 两个档位都报
    python scripts/model_stats.py --tier 65m      # 只报 65M
    python scripts/model_stats.py --json out/model_stats.json
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from model.model_system_one import (
    DecisionConfig, MiniSystemOneForDecision,
    MiniSystemOneForMaskedLM,
)
from trainer.trainer_utils import unbuffer_stdout

# 两个档位。**形状刻意对齐 MiniMind**（h/L/heads 同 MiniMind 的两档），
# 这样 `--encoder_init ../minimind/out/pretrain_768.pth` 式的 warm-start 消融
# 不需要改任何形状参数，只是换权重。
TIERS = {
    "26m": dict(hidden_size=512, num_hidden_layers=8, num_attention_heads=8,
                num_key_value_heads=4, intermediate_size=1280),
    "65m": dict(hidden_size=768, num_hidden_layers=8, num_attention_heads=8,
                num_key_value_heads=4, intermediate_size=2304),
}


def count(module):
    """只数**可训练**参数。buffer（RoPE 的 cos/sin 表）不计入参数量 ——
    它们随 max_position_embeddings 变，但不产生梯度，混进来会让数字随
    `--max_len` 漂移，读者会以为模型变大了。"""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def breakdown(model):
    """按顶层子模块拆参数量，**每个 Parameter 只记一次**。

    必须按 `id()` 去重，不能直接对每个子模块 `sum(p.numel())` 再相加：绑定权重
    （`lm_head.weight is embed_tokens.weight`）会被算两遍，加总就超过总参数量
    —— 而 `model.parameters()` 本身是去重的。直接相加会得到 28.59M 然后被迫写一个
    -3.28M 的"其他"来配平，那个负数会被读者合理地当成脚本的 bug。

    去重后，后出现的模块（这里是 `lm_head`）计 0，并由调用方标注它是 tied 的。
    """
    out, seen = {}, set()
    for name, mod in model.named_children():
        n = 0
        for prm in mod.parameters():
            if id(prm) in seen:
                continue
            seen.add(id(prm))
            if prm.requires_grad:
                n += prm.numel()
        out[name] = n
    total = count(model)
    assert sum(out.values()) == total, \
        f"拆分 {sum(out.values())} != 总数 {total} —— 有参数不在任何顶层子模块下"
    return out, total


def report_tier(name, cfg_kwargs, vocab):
    cfg = DecisionConfig(vocab_size=vocab, **cfg_kwargs)
    lines = []
    payload = {"config": {k: v for k, v in cfg_kwargs.items()}, "vocab_size": vocab}

    lines.append(f"  档位 {name}：h={cfg.hidden_size} L={cfg.num_hidden_layers} "
                 f"heads={cfg.num_attention_heads}/{cfg.num_key_value_heads} "
                 f"inter={cfg.intermediate_size} theta={cfg.rope_theta:g}")

    # ---- 两个模型类 ----
    models = {
        "ForDecision": MiniSystemOneForDecision(cfg),
        "ForMaskedLM": MiniSystemOneForMaskedLM(cfg),
    }
    totals, parts_all = {}, {}
    for tag, m in models.items():
        parts, total = breakdown(m)
        totals[tag] = total
        parts_all[tag] = parts
        payload[tag] = {"total": total, "parts": parts}

    d = totals["ForDecision"]
    mlm = totals["ForMaskedLM"]

    # ---- 决策模型的三段拆分 ----
    dec = models["ForDecision"]
    enc_all = count(dec.encoder)                    # 含 embed_tokens
    embed = count(dec.encoder.embed_tokens)
    head = count(dec.head)
    enc_wo = enc_all - embed
    assert enc_wo + embed + head == d, "三段拆分加总不等于总参数 —— 拆分漏了模块"

    lines.append("")
    lines.append("    决策模型 MiniSystemOneForDecision")
    lines.append(f"      encoder（含 embed_tokens）  {enc_all/1e6:8.2f}M")
    lines.append(f"        ├─ 其中 embed_tokens      {embed/1e6:8.2f}M")
    lines.append(f"        └─ encoder（不含 embed）  {enc_wo/1e6:8.2f}M")
    lines.append(f"      DecisionHead              {head/1e6:8.2f}M")
    lines.append(f"      ----------------------------------")
    lines.append(f"      合计                      {d/1e6:8.2f}M")

    # ---- AttnPool 单独报：它是 z 的来源，方案里给过 2h²+3h 的解析式，
    # 这里用实测值对照，让读者能核"解析式对不对"。
    pool = count(dec.head.pool)
    lines.append(f"        （head 内 AttnPool {pool/1e6:.3f}M，"
                 f"解析式 2h²+3h = {(2*cfg.hidden_size**2 + 3*cfg.hidden_size)/1e6:.3f}M）")

    # ---- Stage 1 的 MLM ----
    # `lm_head.weight` 与 `embed_tokens.weight` 是**同一个 Parameter 对象**，
    # 所以它的"张量大小"是 3.28M 而"新增参数量"是 0。两个数都报，否则读者会以为
    # 它比算出来的多一个 3.28M 的输出层（或者反过来，以为绑定是空话）。
    mlm_head_tensor = count(models["ForMaskedLM"].lm_head)
    assert parts_all["ForMaskedLM"]["lm_head"] == 0, (
        "lm_head 未与 embed_tokens 绑定 —— tie_word_embeddings 失效了")
    assert (models["ForMaskedLM"].lm_head.weight
            is models["ForMaskedLM"].encoder.embed_tokens.weight)
    lines.append("")
    lines.append("    Stage 1 MiniSystemOneForMaskedLM")
    lines.append(f"      encoder（含 embed_tokens）  {mlm/1e6:8.2f}M")
    lines.append(f"      lm_head                   {mlm_head_tensor/1e6:8.2f}M 张量 / "
                 f"**新增 0**（与 embed_tokens 同一个 Parameter 对象）")
    lines.append(f"      合计                      {mlm/1e6:8.2f}M  "
                 f"= 决策模型 {d/1e6:.2f}M − DecisionHead {head/1e6:.2f}M")

    payload["decision"] = {"encoder_with_embed": enc_all, "embed_tokens": embed,
                           "encoder_without_embed": enc_wo, "head": head,
                           "attn_pool": pool, "total": d}
    return lines, payload


def main():
    unbuffer_stdout()
    p = argparse.ArgumentParser(description="参数构成表（CPU-only，不读 checkpoint）")
    p.add_argument("--tier", choices=sorted(TIERS) + ["all"], default="all")
    p.add_argument("--vocab_size", type=int, default=None,
                   help="缺省从 model/tokenizer.json 读实际词表大小")
    p.add_argument("--json", default=None, help="把结果另存为 JSON")
    args = p.parse_args()

    vocab = args.vocab_size
    if vocab is None:
        from transformers import AutoTokenizer
        tok_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "model")
        vocab = len(AutoTokenizer.from_pretrained(tok_dir))

    print("========== 参数构成 ==========")
    print(f"  词表 {vocab}（从 model/ 的 tokenizer 读出；换 tokenizer 会改 embed 与 lm_head）")
    print()
    print("  **读法**：encoder 里含 embed_tokens，所以「encoder + embed + head」")
    print("  会把 embed 算两遍。下面对 decision 给出扣掉重叠后的三段。")

    tiers = sorted(TIERS) if args.tier == "all" else [args.tier]
    out = {"vocab_size": vocab, "tiers": {}}
    for name in tiers:
        print()
        lines, payload = report_tier(name, TIERS[name], vocab)
        for ln in lines:
            print(ln)
        out["tiers"][name] = payload

    print()
    print("  注：以上均为**可训练参数**。RoPE 的 cos/sin 表注册为 buffer（`named_buffers()`")
    print("      里能看到 `encoder.freqs_cos`），**不进 `parameters()`**，所以这里与")
    print("      `train_decision.py` 启动时打印的 `sum(p.numel() for p in model.parameters())`")
    print("      是同一口径，两处数字应当相等。")

    if args.json:
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"\n  -> {args.json}")


if __name__ == "__main__":
    main()
