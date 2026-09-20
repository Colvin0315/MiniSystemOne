"""
位置泄漏审计 —— 证明"候选位置不携带信息"，合成集与公开集都过一遍。

**为什么必须是纯位置探针。** 生成器与适配器都在逐例重排候选，但"重排了"不等于
"重排对了"：只要漏掉一条路径（某个分支没走公共的 shuffle、某个 adapter 自己拼了候选
却没置换、`target` 置换了而 `level` 没置换），位置就会重新携带信息，而**所有指标都会
因此变好**，没有任何其他检查会报出来。这个脚本量的是那件事本身。

特征**只有 `(position_onehot, K)`**，不看文本。于是它能表达的假设只有"第 i 位更容易
是正确答案"这一族，而这正是重排要消灭的东西。换成词法特征就测不出这个了 —— 词法上界
是 `audit_synthetic.py` 第 5/6 节的活，两者分工不同。

**"正确"必须按最大位置集合算，不能按下标最小那个最大位置算。** `tie_set` 来源
（security_gate / tool_router / calendar_slot）的目标是**并列**的均匀分布，比如
`[0.5, 0.5, 0]`。若按"第一个 argmax"定义金标，那位置 0 就有 2/3 的概率被记成正确 ——
量出来的是一个**恒定的位置偏置**，与重排是否生效毫无关系。实测：这三个来源上会报
0.52/0.33 的"泄漏"，而把并列项剔除后，非并列样本的位置直方图是 472/418/473，
**完全均匀**。所以判据是"该位置属于最大集合"，零假设下 P(第 i 位正确) = m/K（m 为
最大集合大小），基线也随之从 `1/K` 修正为 `E[m]/K`。

这个坑值得记下来：并列目标的**硬** argmax 准确率（`eval_metrics.accuracy`）同样带这个
偏置，所以在并列来源上它天生偏高几个点。`eval_metrics` 说明了校准要用
`soft_accuracy`（= `t[argmax p]`）而不是硬准确率，这里是同一件事的第二个面。

**为什么不用训练一个探针。** 位置探针能做到的最优，就是给每个位置一个偏置，即
"取该位置出现频率最高的那一位"。那个量可以直接从计数算出来，而且是**精确的**；
真去训一个线性模型只会复现同一个数，还多一层随机性。所以这里算的是：
每个 `(split, source, K)` 组的金标位置直方图，以及它对应的最优固定位置策略的准确率
（= Σ_K P(K) · max_i freq(i | K)）。

零假设是"每个位置等可能"，即 `counts ~ Multinomial(n, 1/K)`。逐 K 组做蒙特卡洛
（`N_PERM` 次），得到统计量 `S = Σ_K max_i count` 的零分布与 p 值。**用模拟而不是
卡方**：这里要的不是"分布是否有差异"，而是"最优固定策略能拿到多少准确率"，后者才是
会污染指标的那个量，而它是个极值统计量，卡方给不出它的分布。

**判读**：`p < 0.01` 或"最优策略准确率显著高于均匀基线"都判泄漏。小 `n` 的组噪声大，
所以门禁用模拟出来的零分布而不是硬阈值 —— 否则 K 大的组会因为极值统计量的偏倚而被
误判。
"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from trainer.trainer_utils import unbuffer_stdout  # noqa: E402

N_PERM = 400
ALPHA = 0.01


def load_split(path):
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def gold_set(rec, tol=1e-9):
    """最大位置集合。**不取"第一个最大"** —— 并列目标上那会引入恒定位置偏置，
    理由见模块 docstring。返回 (K, frozenset(位置))。"""
    p = rec["target"]["p"]
    mx = max(p)
    return len(p), frozenset(i for i, x in enumerate(p) if x >= mx - tol)


def position_stats(recs):
    """按 K 分组，统计"该位置属于最大集合"的次数。

    返回 {K: ([hit_0, hit_1, ...], [m_0, m_1, ...])}，其中 hit_i 是位置 i 被
    判对的条数，m 是每条样本的最大集合大小（用来算修正后的均匀基线）。"""
    hits = collections.defaultdict(lambda: collections.Counter())
    ms = collections.defaultdict(list)
    for r in recs:
        k, g = gold_set(r)
        ms[k].append(len(g))
        for i in g:
            hits[k][i] += 1
    return {k: ([hits[k].get(i, 0) for i in range(k)], ms[k]) for k in ms}


def best_fixed_accuracy(stats_by_k):
    """最优固定位置策略的**期望**准确率，以及均匀随机位置策略的期望。

    基线不再是 `1/K` 而是 `E[m]/K`：并列样本上随机猜也有 `m/K` 的概率落在最大集合里，
    拿 `1/K` 当基线会把并列多的来源误判成泄漏（这正是修这个脚本的原因）。
    """
    n_total = sum(len(ms) for _, ms in stats_by_k.values())
    if not n_total:
        return 0.0, 0.0
    hit = exact = 0.0
    for k, (hits, ms) in stats_by_k.items():
        hit += max(hits)
        exact += sum(m / k for m in ms)       # 均匀策略的期望命中
    return hit / n_total, exact / n_total


def perm_pvalue(stats_by_k, rng):
    """零假设下重抽，算 `S = Σ_K max_i count` 的 p 值。

    零假设是"重排把每条样本的最大集合均匀地放到任意 m 个位置上"，所以重抽就是把每条
    样本的最大集合替换成一个均匀随机 m-子集。**不能对 `max_i freq` 做正态近似：**
    K 到 255、而某些组的 n 只有几百时，`max` 的零分布明显右偏，用正态算出来的"显著"
    多半是偏倚不是泄漏。直接模拟这个极值统计量，p 值不依赖任何分布假设。
    """
    obs = sum(max(h) for h, _ in stats_by_k.values())
    ge = 0
    for _ in range(N_PERM):
        s = 0
        for k, (_, ms) in stats_by_k.items():
            bucket = [0] * k
            for m in ms:
                if m == 1:                     # 绝大多数走这条，省掉 sample() 的开销
                    bucket[rng.randrange(k)] += 1
                else:
                    for i in rng.sample(range(k), m):
                        bucket[i] += 1
            s += max(bucket)
        if s >= obs:
            ge += 1
    # +1 的 Laplace 修正：400 次重抽下 p 的倒数第二个有效位是 0.0025，
    # 报 0.000 会让"完全没检出"看起来像"p 恰好为零"。
    return (ge + 1) / (N_PERM + 1)


def audit_dir(path, seed=0):
    manifest_path = os.path.join(path, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"  跳过 {path}（没有 manifest.json）")
        return []
    manifest = json.load(open(manifest_path, encoding="utf-8"))
    print(f"\n  {path}  gen_version {manifest.get('gen_version')}  "
          f"tokenizer {manifest.get('tokenizer_sha1')}")

    fails = []
    for split in ("train", "val", "calib", "test_known", "test_ood"):
        fp = os.path.join(path, f"{split}.jsonl")
        if not os.path.exists(fp):
            continue
        recs = load_split(fp)
        if not recs:
            continue
        by_source = collections.defaultdict(list)
        for r in recs:
            by_source[r["source"]].append(r)

        print(f"    {split:11s} {len(recs):6d} 条")
        for src in sorted(by_source):
            group = by_source[src]
            stats_by_k = position_stats(group)
            obs, uniform = best_fixed_accuracy(stats_by_k)
            rng = random.Random(f"{seed}|{path}|{split}|{src}")
            p = perm_pvalue(stats_by_k, rng)
            ks = sorted(stats_by_k)
            bad = p < ALPHA
            if bad:
                fails.append((path, split, src, obs, uniform, p))
            flag = "  <<< 泄漏" if bad else ""
            print(f"      {src.split(':')[-1]:16s} n={len(group):6d} K={ks[0]}..{ks[-1]:<3d} "
                  f"最优固定位置 {obs:.3f}  均匀 {uniform:.3f}  超额 {obs - uniform:+.3f}  "
                  f"p={p:.3f}{flag}")
    return fails


def main():
    unbuffer_stdout()
    p = argparse.ArgumentParser(description="位置泄漏审计（合成集 + 公开集）")
    p.add_argument("--data", nargs="*", default=["dataset/synth", "dataset/public"],
                   help="含 manifest.json 与 <split>.jsonl 的目录")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    print("========== 位置泄漏探针 ==========")
    print("  特征只有 (position, K)，不看文本。零假设：每个位置等可能。")
    print(f"  判据：{N_PERM} 次多项重抽的 p 值 < {ALPHA} 即判泄漏（左<<< 标记）。")

    fails = []
    for d in args.data:
        if os.path.isdir(d):
            fails += audit_dir(d, args.seed)
        else:
            print(f"  跳过 {d}（目录不存在）")

    print("\n========== 结论 ==========")
    if fails:
        print(f"  **{len(fails)} 处位置泄漏** —— 重排在这几条路径上没有生效：")
        for path, split, src, obs, uniform, p in fails:
            print(f"    {path} :: {split} :: {src}  最优固定位置 {obs:.3f} "
                  f"vs 均匀 {uniform:.3f}（p={p:.3f}）")
        print("  修完之前，这个来源上的所有指标都不可采信。")
    else:
        print("  全部在随机水平。候选位置不携带信息。")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
