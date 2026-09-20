"""
构建合成决策数据集：跑遍全部生成器，按 split 写成 jsonl，并落一份 manifest。

这里做三件别处做不了的事：

1. **`meta.approx_tokens` 必须用最终 tokenizer 现算。** `DecisionDataset.length_key`
   读的就是它。它若是估的、或是拿旧 tokenizer 算的，分桶采样器的"组内长度差 <
   一个桶宽"就不成立，padding 浪费会静默回升 —— 而这个退化只表现为训练变慢，
   不报错，所以不能靠"看起来没问题"过关。

2. **跨生成器的 split 不相交断言**（`assert_split_disjoint`）。各生成器分别持有
   (template_id, pool) → split 的映射；只要两个生成器的 template_id 前缀撞了，
   同一个组合就会被一个划进 train、另一个划进 test_known，留出集当场漏掉，而**所有
   指标都会因此变好看**。这个断言是唯一能挡住它的地方。

3. **`gen_version` 与 tokenizer 哈希一起冻结在 manifest 里。** 两者任一变化都必须
   显式 `--force` 才能重建，否则一批基于旧测试集的结果 JSON 会被当成与新数据可比。

`test_ood` 不来自 split_map —— 它由**整族留出**的生成器产生：`--ood_generators`
点名的生成器，其全部输出都改写成 test_ood，训练完全不碰它们。
"""
import argparse
import collections
import hashlib
import json
import os
import sys

# **必须在 torch 之前。** Windows 上 pyarrow（datasets 的底层）与 torch 的 DLL 加载
# 顺序不能颠倒：先 import torch（经 `model.serialize`）、再 `from datasets import
# load_dataset`，进程会直接**段错误退出且不打印任何东西** —— 连阶段横幅都不会出现，
# 看起来像脚本根本没跑。实测：`transformers → synth → serialize(torch) → load_dataset`
# 必崩，把这一行提到最前面就正常。
#
# 这是本仓库唯一需要它的地方：只有这个脚本同时碰 datasets 与 torch。`trainer/*` 和
# `eval/*` 都不 import datasets，`dataset/adapters/*` 不 import torch。
import datasets  # noqa: F401

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer

from dataset.synth import GENERATORS, GEN_VERSION, assert_split_disjoint, build_all
from model.serialize import encode_text
from trainer.trainer_utils import unbuffer_stdout

SPLITS = ("train", "val", "calib", "test_known", "test_ood")


def approx_tokens(tok, rec, max_len):
    """这条样本打包后的真实 token 数。

    按 `DecisionDataset.__getitem__` 的算法复算一遍（固定开销先扣，剩下的预算给
    state，state 超了就按预算截）—— 分桶要的是**实际长度**，不是 state 的自然长度。
    照抄自然长度会把所有长样本堆进同一个高桶，虽然不浪费 padding，但桶的语义就
    成了"state 原本多长"，与采样器真正要对齐的东西脱节。
    """
    q = len(encode_text(tok, rec["question"]))
    fixed = q + 2 + sum(len(encode_text(tok, c["text"])) + 1 for c in rec["candidates"])
    state = len(encode_text(tok, rec["state"]))
    return min(state, max(max_len - fixed, 8)) + fixed


def write_record(f, rec):
    f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def relabel_ood(rec, name, index):
    """留出生成器的样本改写成 test_ood。id 里的 split 段也要改，否则 `test_ood.jsonl`
    里躺着一批 id 写着 `::train::` 的记录，误差分析按 id 回查会找错文件。"""
    rec = dict(rec)
    rec["split"] = "test_ood"
    rec["id"] = f"{name}::test_ood::{index:07d}"
    return rec


def parse_args():
    p = argparse.ArgumentParser(description="构建决策数据集（合成 / 公开）")
    p.add_argument("--out", default=None,
                   help="输出目录（默认为 dataset/synth，--public 时为 dataset/public）")
    p.add_argument("--tokenizer", default="model", help="tokenizer 目录（算 approx_tokens 用）")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_len", type=int, default=1024, help="训练序列长度上限，用于截断预算")
    p.add_argument("--public", action="store_true",
                   help="构建公开语料适配器而不是合成生成器")
    p.add_argument("--adapters", nargs="*", default=[],
                   help="--public 时只构建这些适配器（默认全部）")
    p.add_argument("--limit", type=int, default=0,
                   help="--public 时每个适配器最多读多少条（0 = 全量）；冒烟用")
    p.add_argument("--per_gen_train", type=int, default=30000)
    p.add_argument("--per_gen_val", type=int, default=1000)
    p.add_argument("--per_gen_calib", type=int, default=1000)
    p.add_argument("--per_gen_test_known", type=int, default=3000)
    p.add_argument("--per_gen_test_ood", type=int, default=5000)
    p.add_argument("--ood_generators", nargs="*", default=[],
                   help="整族留出的生成器（如 agent_trace_score calendar_slot）")
    p.add_argument("--smoke", action="store_true", help="极小规模，供端到端冒烟")
    p.add_argument("--force", action="store_true",
                   help="允许在 gen_version / tokenizer 变化时覆盖已有数据集")
    return p.parse_args()


def build_public(args):
    """公开语料侧：跑遍适配器，按 split 写成 jsonl，并落一份 manifest。

    **与合成侧共用同一个 `approx_tokens` 口径和同一套 manifest 字段**，但不共用
    生成流程 —— 公开集没有 (template_id, entity_pool) 网格，它的 split 要么来自
    数据集原生划分，要么来自整桶哈希（见 `dataset/adapters/__init__.py`）。所以
    这里不做 `assert_split_disjoint`：那个断言防的是两个**生成器**的 template_id
    前缀相撞，而适配器的 source 各不相同，组合天然不相交。真正需要把关的是候选
    位置泄漏，那是 `scripts/audit_leakage.py` 的活。
    """
    from dataset.adapters import PUBLIC_GEN_VERSION, all_adapters, build as build_adapter

    names = args.adapters or [a.name for a in all_adapters()]
    picked = [a for a in all_adapters() if a.name in names]
    unknown = set(names) - {a.name for a in picked}
    if unknown:
        raise SystemExit(f"--adapters 里有未注册的适配器：{sorted(unknown)}")

    out_dir = args.out or "dataset/public"
    print("========== 1. 公开适配器 ==========")
    for a in picked:
        print(f"  {a.name:16s} {a.hf_id:42s} primitive={a.primitive} "
              f"prov={a.provenance} native_splits={a.native_splits}")

    print("\n========== 2. tokenizer ==========")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_sha = hashlib.sha1(
        open(os.path.join(args.tokenizer, "tokenizer.json"), "rb").read()).hexdigest()[:12]
    print(f"  vocab {tok.vocab_size}  sha1 {tok_sha}")

    os.makedirs(out_dir, exist_ok=True)
    manifest_path = os.path.join(out_dir, "manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        old = json.load(open(manifest_path, encoding="utf-8"))
        diffs = [f"{k}: {old.get(k)} -> {v}" for k, v in
                 (("gen_version", PUBLIC_GEN_VERSION), ("tokenizer_sha1", tok_sha))
                 if old.get(k) != v]
        if diffs:
            raise SystemExit(
                "已有数据集与本次构建不一致，重建会让旧结果 JSON 失去可比性：\n  "
                + "\n  ".join(diffs) + "\n确认要重建请加 --force")

    print("\n========== 3. 生成 ==========")
    by_source = {s: collections.Counter() for s in SPLITS}
    by_prov = {s: collections.Counter() for s in SPLITS}
    by_lang = {s: collections.Counter() for s in SPLITS}
    tok_sum = collections.Counter()
    k_hist = {s: collections.Counter() for s in SPLITS}
    # (provenance, 有没有 counts) 的计数，供下面的契约门禁用。
    by_counts = collections.Counter()

    files = {s: open(os.path.join(out_dir, f"{s}.jsonl"), "w", encoding="utf-8")
             for s in SPLITS}
    try:
        for a in picked:
            got = build_adapter(a, seed=args.seed, limit=args.limit or None)
            n = 0
            for split, recs in got.items():
                for rec in recs:
                    rec["meta"]["approx_tokens"] = approx_tokens(tok, rec, args.max_len)
                    write_record(files[split], rec)
                    by_source[split][a.name] += 1
                    by_prov[split][rec["target"]["provenance"]] += 1
                    by_lang[split][rec["meta"]["language"]] += 1
                    tok_sum[split] += rec["meta"]["approx_tokens"]
                    k_hist[split][len(rec["candidates"])] += 1
                    by_counts[(rec["target"]["provenance"],
                               rec["target"].get("counts") is not None)] += 1
                    n += 1
            dist = {s: len(v) for s, v in got.items() if v}
            print(f"  {a.name:16s} {n:6d} 条  {dist}")
    finally:
        for f in files.values():
            f.close()

    # **契约门禁：`human_annotators` 必须有 `counts`，其余必须没有。**
    # 这个字段是 `binomial_noise_floor` 的唯一输入，而它一旦缺失，症状只是报告里
    # "噪声校正后 ECE"那一行静默消失 —— 不报错、不留痕，且很容易被误读成
    # "这个模型没有噪声底"。所以在这里硬拦，而不是靠散文提醒。
    bad = [(p, h) for (p, h) in by_counts if (p == "human_annotators") != h]
    if bad:
        raise SystemExit(
            "target.counts 契约被破坏（human_annotators 必须有、其余必须没有）："
            + "；".join(f"{p} counts={'有' if h else '无'}（n={by_counts[(p, h)]}）"
                        for p, h in sorted(bad)))

    print("\n========== 4. 数据集卡片 ==========")
    for s in SPLITS:
        n = sum(by_source[s].values())
        if not n:
            print(f"  {s:11s} 空")
            continue
        print(f"  {s:11s} {n:6d} 条  {tok_sum[s] / n:6.0f} tok/条  "
              f"合计 {tok_sum[s] / 1e6:.2f}M tok")
        print(f"    source   {dict(sorted(by_source[s].items()))}")
        print(f"    prov     {dict(sorted(by_prov[s].items()))}")
        print(f"    lang     {dict(sorted(by_lang[s].items()))}")
        ks = sorted(k_hist[s])
        print(f"    K        {ks[0]}..{ks[-1]}（中位 {ks[len(ks) // 2]}）")

    manifest = {
        "gen_version": PUBLIC_GEN_VERSION, "seed": args.seed,
        "tokenizer_sha1": tok_sha, "adapters": [a.name for a in picked],
        "max_len": args.max_len, "limit": args.limit,
        "counts": {s: sum(by_source[s].values()) for s in SPLITS},
        "tokens": {s: tok_sum[s] for s in SPLITS},
        "by_source": {s: dict(sorted(by_source[s].items())) for s in SPLITS},
        "by_provenance": {s: dict(sorted(by_prov[s].items())) for s in SPLITS},
        "by_language": {s: dict(sorted(by_lang[s].items())) for s in SPLITS},
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)
    print(f"\n  manifest -> {manifest_path}（gen_version {PUBLIC_GEN_VERSION}）")
    print(f"  全部输出在 {out_dir}/")


def main():
    unbuffer_stdout()
    args = parse_args()
    if args.public:
        return build_public(args)
    # `--out` 的默认值是 None，两条路径各自解析 —— 在这里兜底，否则下面的
    # `os.path.join(args.out, ...)` 与 `makedirs` 会直接 TypeError。
    args.out = args.out or "dataset/synth"
    if args.smoke:
        args.per_gen_train, args.per_gen_val = 200, 40
        args.per_gen_calib, args.per_gen_test_known = 40, 40
        args.per_gen_test_ood = 40
        args.out = os.path.join(args.out, "smoke")

    gens = build_all(seed=args.seed)
    names = [g.name for g in gens]

    print("========== 1. split 划分 ==========")
    assert_split_disjoint({g.name: g.split_map for g in gens})
    for g in gens:
        cnt = collections.Counter(g.split_map.values())
        print(f"  {g.name:16s} 组合 {len(g.split_map):3d}  {dict(sorted(cnt.items()))}")
    print(f"  跨生成器无共用组合 OK（template_id 前缀 {[g.prefix for g in gens]}）")

    ood = set(args.ood_generators)
    unknown = ood - set(names)
    if unknown:
        raise SystemExit(f"--ood_generators 里有未注册的生成器：{sorted(unknown)}")
    if len(ood) == len(names):
        raise SystemExit("全部生成器都被留出，训练集会是空的")
    train_gens = [g for g in gens if g.name not in ood]

    print("\n========== 2. tokenizer ==========")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    tok_sha = hashlib.sha1(
        open(os.path.join(args.tokenizer, "tokenizer.json"), "rb").read()).hexdigest()[:12]
    print(f"  vocab {tok.vocab_size}  sha1 {tok_sha}")

    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.json")
    if os.path.exists(manifest_path) and not args.force:
        old = json.load(open(manifest_path, encoding="utf-8"))
        diffs = [f"{k}: {old.get(k)} -> {v}" for k, v in
                 (("gen_version", GEN_VERSION), ("tokenizer_sha1", tok_sha))
                 if old.get(k) != v]
        if diffs:
            raise SystemExit(
                "已有数据集与本次构建不一致，重建会让旧结果 JSON 失去可比性：\n  "
                + "\n  ".join(diffs) + "\n确认要重建请加 --force")

    print("\n========== 3. 生成 ==========")
    counts = {"train": args.per_gen_train, "val": args.per_gen_val,
              "calib": args.per_gen_calib, "test_known": args.per_gen_test_known}
    by_source = {s: collections.Counter() for s in SPLITS}
    by_prov = {s: collections.Counter() for s in SPLITS}
    by_lang = {s: collections.Counter() for s in SPLITS}
    tok_sum = collections.Counter()
    k_hist = {s: collections.Counter() for s in SPLITS}

    files = {s: open(os.path.join(args.out, f"{s}.jsonl"), "w", encoding="utf-8")
             for s in SPLITS}
    try:
        for g in train_gens:
            for split, n in counts.items():
                for rec in g.generate(split, n):
                    rec["meta"]["approx_tokens"] = approx_tokens(tok, rec, args.max_len)
                    write_record(files[split], rec)
                    by_source[split][g.name] += 1
                    by_prov[split][rec["target"]["provenance"]] += 1
                    by_lang[split][rec["meta"]["language"]] += 1
                    tok_sum[split] += rec["meta"]["approx_tokens"]
                    k_hist[split][len(rec["candidates"])] += 1
            print(f"  {g.name:16s} 训练集 {counts['train']} 条 × {len(counts)} 个 split OK")
        for g in gens:
            if g.name not in ood:
                continue
            # 留出族只用它自己的 train 组合，规模按 --per_gen_test_ood
            for i, rec in enumerate(g.generate("train", args.per_gen_test_ood)):
                rec = relabel_ood(rec, g.name, i)
                rec["meta"]["approx_tokens"] = approx_tokens(tok, rec, args.max_len)
                write_record(files["test_ood"], rec)
                by_source["test_ood"][g.name] += 1
                by_prov["test_ood"][rec["target"]["provenance"]] += 1
                by_lang["test_ood"][rec["meta"]["language"]] += 1
                tok_sum["test_ood"] += rec["meta"]["approx_tokens"]
                k_hist["test_ood"][len(rec["candidates"])] += 1
            print(f"  {g.name:16s} 整族留出 -> test_ood {args.per_gen_test_ood} 条")
    finally:
        for f in files.values():
            f.close()

    print("\n========== 4. 数据集卡片 ==========")
    for s in SPLITS:
        n = sum(by_source[s].values())
        if not n:
            print(f"  {s:11s} 空（未配置留出生成器）")
            continue
        print(f"  {s:11s} {n:6d} 条  {tok_sum[s] / n:6.0f} tok/条  "
              f"合计 {tok_sum[s] / 1e6:.2f}M tok")
        print(f"    source   {dict(sorted(by_source[s].items()))}")
        print(f"    prov     {dict(sorted(by_prov[s].items()))}")
        print(f"    lang     {dict(sorted(by_lang[s].items()))}")
        print(f"    K        {dict(sorted(k_hist[s].items()))}")

    manifest = {
        "gen_version": GEN_VERSION, "seed": args.seed, "tokenizer_sha1": tok_sha,
        "generators": names, "ood_generators": sorted(ood), "max_len": args.max_len,
        "counts": {s: sum(by_source[s].values()) for s in SPLITS},
        "tokens": {s: tok_sum[s] for s in SPLITS},
        "by_source": {s: dict(sorted(by_source[s].items())) for s in SPLITS},
        "by_provenance": {s: dict(sorted(by_prov[s].items())) for s in SPLITS},
        "by_language": {s: dict(sorted(by_lang[s].items())) for s in SPLITS},
    }
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)
    print(f"\n  manifest -> {manifest_path}（gen_version {GEN_VERSION}）")
    print(f"  全部输出在 {args.out}/")


if __name__ == "__main__":
    main()
