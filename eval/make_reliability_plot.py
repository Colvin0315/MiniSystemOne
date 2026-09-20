"""
旗舰图：可靠性图（校准前 / 后）+ 置信度分布 + per-provenance ECE 对比。

**这张图全部从 `eval_harness.py` 落盘的 `per_sample` 重算**，不再碰模型。这不是
省事，是让"改一个分桶方案 / 换一个温度 / 只看某个 provenance"变成秒级操作 ——
否则每改一次都要 1.5 小时 GPU，实践中就等于不改、直接引用第一次的数字。

而且它**直接调 `eval_metrics` 里的 `reliability_curve` / `ece`**，不自己再实现一遍
分桶。图上的曲线和标题里的 ECE 必须来自同一份代码，否则会出现"图看着挺直、ECE
却不小"这种谁也解释不了的组合 —— 那正是这类图最容易骗人的地方。

四联图：校准前/后可靠性曲线、置信度分布、per-provenance ECE。
每桶标出样本数；不把软正确率当作独立 Bernoulli 样本画通用置信区间。
默认展示 soft_targets 是展示选择，硬标签也可用 --include_hard 计算校准。
标注 MC 参考量只描述假设模型，不从 ECE 中扣除；混合总体须说明来源组成。

**风险-覆盖率曲线不在这里**：自定义任务教程单独使用 `risk_coverage_curve`，
验证冻结的置信度/人工接管策略。不要把这里的可靠性图读成该曲线。

用法：
    python eval/make_reliability_plot.py --eval out/eval/decision --sets test_known
    python eval/make_reliability_plot.py --eval out/eval/decision \\
        --sets public_test_known --source chaosnli      # 真实人类分歧那一张
"""
import argparse
import json
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from eval.eval_metrics import (
    SOFT_TARGET_PROVENANCE, ece_annotation_reference, ece, reliability_curve, top1,
)


CJK_FONTS = ("Microsoft YaHei", "Noto Sans SC", "Source Han Sans SC", "SimHei",
             "WenQuanYi Zen Hei", "PingFang SC", "Hiragino Sans GB", "Microsoft JhengHei")


def parse_args():
    p = argparse.ArgumentParser(description="可靠性图")
    p.add_argument("--eval", default="out/eval/decision",
                   help="eval_harness.py 的输出目录（含 <set>.json）")
    p.add_argument("--sets", nargs="*", default=None,
                   help="要画哪些 set；缺省画目录里所有带 per_sample 的")
    p.add_argument("--out", default="assets")
    p.add_argument("--bins", type=int, default=15)
    p.add_argument("--binning", default="equal_mass",
                   choices=("equal_mass", "equal_width"))
    p.add_argument("--include_hard", action="store_true",
                   help="默认仅展示 soft_targets；此选项也展示合法的 hard 校准指标")
    p.add_argument("--source", nargs="*", default=None,
                   help="只画指定 source；分来源便于解释，混合总体须说明组成")
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def _slug(s):
    """把 source 名塞进文件名前先净化。

    source 是 `synth:banking_balance` 这种带冒号的串，而**冒号在 Windows 上不是
    合法文件名字符** —— `fig.savefig` 不会报错，它会把 `:banking_balance.png`
    当成 NTFS 备用数据流写进 `reliability_test_known_synth` 这个文件里。结果是
    README 要引用的那张图`根本不存在`，而命令一路显示成功。
    """
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in s)


def setup_font(plt):
    """挑一个真的装了的 CJK 字体。挑不到就**大声报**，不静默出豆腐块。

    matplotlib 在字体缺字时不报错、不中止，只把每个汉字画成一个方框 —— 图照样
    生成、照样进 README，只是所有标题都是 □□□。这种失败必须显式拦住，否则
    它会被一路带到发布产物里。
    """
    from matplotlib import font_manager
    have = {f.name for f in font_manager.fontManager.ttflist}
    plt.rcParams["axes.unicode_minus"] = False      # 负号用 ASCII，省一个缺字点
    for name in CJK_FONTS:
        if name in have:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            return name
    print("  **警告**：没找到中文字体，图上的中文会渲染成方框。"
          "装一个 Noto Sans SC / Microsoft YaHei，或把标题改成英文。")
    return None


def load_set(path, include_hard, sources=None):
    """读一个 <set>.json 的 per_sample，重建 (p, t, mask, prov, sel)。

    **mask 由 `K_used` 重建，不是由 p 的非零元重建**：padding 槽在 p 里是严格 0，
    但真实候选也可能拿到接近 0 的概率，用"p > 0"当 mask 会把它们悄悄丢掉，
    且丢的恰好是模型最不确定的那批样本 —— 那正好是校准最该看的地方。
    """
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    ps = obj.get("per_sample")
    if ps is None:
        raise SystemExit(f"{path} 里没有 per_sample —— 重跑 eval_harness.py 时"
                         f"别加 --no_per_sample")
    p = np.asarray(ps["p"], dtype=np.float64)
    p_cal = np.asarray(ps["p_calibrated"], dtype=np.float64)
    t = np.asarray(ps["target"], dtype=np.float64)
    k_used = np.asarray(ps["K_used"], dtype=int)
    mask = np.arange(p.shape[1])[None, :] < k_used[:, None]
    prov = np.asarray(ps["provenance"])
    sel = (np.ones(len(prov), bool) if include_hard
           else np.isin(prov, SOFT_TARGET_PROVENANCE))
    if sources:
        # source 与 provenance 是**两回事**：provenance 说"这份目标是怎么来的"
        # （human_annotators），source 说"是哪一批数据"（chaosnli / goemotions）。
        # ChaosNLI 与 GoEmotions 的任务和标注协议不同，分来源便于解释。
        sel &= np.isin(np.asarray(ps["source"]), list(sources))
    return obj, {"raw": (p, t, mask), "cal": (p_cal, t, mask)}, prov, sel


def panel_reliability(ax, p, t, mask, bins, binning, title, annotation_reference=None,
                      annotation_n=0):
    """一张可靠性图。桶、曲线、ECE **全部来自 eval_metrics**。"""
    e = ece(p, t, mask, bins, binning)
    cb, rb, nb = reliability_curve(p, t, mask, bins, binning)

    ax.plot([0, 1], [0, 1], "--", color="0.55", lw=1.2, zorder=1, label="完美校准")
    if len(nb):
        ax.plot(cb, rb, "o-", color="#1f5fa8", ms=4, lw=1.4,
                zorder=2, label="实测（无置信区间）")
        for x, y, n in zip(cb, rb, nb):
            ax.annotate(f"{n}", (x, y), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=6.5, color="0.35")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel(f"平均置信度（{binning}，{bins} 桶）")
    ax.set_ylabel("平均正确率（软口径 t[argmax p]）")
    sub = f"ECE = {e:.4f}"
    if annotation_reference is not None:
        sub += (f"\n标注 MC 参考 = {annotation_reference:.4f}（n={annotation_n}）"
                "\n假设模型诊断，非下界，不扣除")
    ax.set_title(f"{title}\nn={int(nb.sum()) if len(nb) else 0}   {sub}", fontsize=10)
    ax.legend(loc="upper left", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.25, lw=0.5)
    return e


def panel_hist(ax, conf_raw, conf_cal, bins):
    """置信度分布。**对数纵轴**：尾部那几十个样本正是要看的，线性纵轴会把它们压没。"""
    edges = np.linspace(min(conf_raw.min(), conf_cal.min(), 0.0), 1.0, 41)
    ax.hist(conf_raw, bins=edges, color="#1f5fa8", alpha=0.55, label="校准前")
    ax.hist(conf_cal, bins=edges, histtype="step", color="#c0504d", lw=1.6,
            label="校准后")
    ax.set_yscale("log")
    ax.set_xlabel("top-1 置信度")
    ax.set_ylabel("样本数（对数）")
    ax.set_title(f"置信度分布（{bins} 个 equal-mass 桶 → 每桶约 "
                 f"{max(1, len(conf_raw)//bins)} 条）\n"
                 f"分位数边界遇到并列时实际桶数可能减少；分桶方式影响 ECE",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5)


def panel_by_provenance(ax, prov, packs_raw, packs_cal, bins, binning):
    """per-provenance ECE 前后对比；显示来源组成而非否定混合总体。"""
    groups = sorted(set(prov.tolist()))

    def per_group(packs):
        return [ece(packs[0][prov == g], packs[1][prov == g], packs[2][prov == g],
                    bins, binning) for g in groups]

    raw, cal = per_group(packs_raw), per_group(packs_cal)
    ns = [int((prov == g).sum()) for g in groups]
    x = np.arange(len(groups))
    w = 0.38
    ax.bar(x - w / 2, raw, w, color="#1f5fa8", alpha=0.85, label="校准前")
    ax.bar(x + w / 2, cal, w, color="#c0504d", alpha=0.85, label="校准后")
    for xi, (r, c, n) in enumerate(zip(raw, cal, ns)):
        ax.annotate(f"n={n}", (xi, max(r, c)), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=7, color="0.35")
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8, rotation=15, ha="right")
    ax.set_ylabel("ECE")
    ax.set_title("各 provenance 的 ECE（校准前 / 后）\n"
                 "分来源解释；混合总体须说明组成", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, lw=0.5, axis="y")


def make_set(plt, obj, packs, prov, sel, args, out_dir, name):
    if sel.sum() < 20:
        print(f"  {name}: 可用样本只有 {int(sel.sum())} 条（--include_hard 可放开），跳过")
        return None
    prov = prov[sel]
    packs_raw = (packs["raw"][0][sel], packs["raw"][1][sel], packs["raw"][2][sel])
    packs_cal = (packs["cal"][0][sel], packs["cal"][1][sel], packs["cal"][2][sel])
    conf_raw = top1(packs_raw[0], packs_raw[2])[1]
    conf_cal = top1(packs_cal[0], packs_cal[2])[1]

    # 参考量只在有正标注数的子集计算，缺失值不是零正确率。
    annotation_reference, annotation_n = None, 0
    raw_counts = obj["per_sample"].get("counts") or []
    if len(raw_counts) == len(sel):
        counts = np.asarray(raw_counts, dtype=np.float64)[sel]
        annotation_n = int((np.isfinite(counts) & (counts > 0)).sum())
        annotation_reference = ece_annotation_reference(
            packs_raw[0], counts, packs_raw[2], args.bins, args.binning)
    gran = obj["per_sample"].get("temperature_applied")

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 10.5))
    e0 = panel_reliability(axes[0, 0], *packs_raw, args.bins, args.binning,
                           "校准前", annotation_reference=annotation_reference,
                           annotation_n=annotation_n)
    # `temperature_applied` 为 None 表示**这一次评测根本没给 `--temperature`**，
    # 此时 `p_calibrated` 就是 `p`，右图与左图逐点相同。标题必须说出来，否则
    # 读者会把"两条一模一样的曲线"读成"温度校准毫无效果"，而真相是它没被运行过。
    e1 = panel_reliability(axes[0, 1], *packs_cal, args.bins, args.binning,
                           f"温度校准后（{gran}）" if gran else
                           "未应用温度（与左图同一条曲线）")
    panel_hist(axes[1, 0], conf_raw, conf_cal, args.bins)
    panel_by_provenance(axes[1, 1], prov, packs_raw, packs_cal, args.bins, args.binning)

    n_excluded = int((~sel).sum())
    note = (f"来源：{os.path.basename(obj['env']['ckpt'])}  sha1 "
            f"{obj['env']['ckpt_sha1']}   split={obj['set']}  "
            f"bins={args.bins}/{args.binning}  "
            f"gen_version={','.join(obj.get('gen_version') or ['-'])}")
    if args.source:
        note += f"   source={','.join(args.source)}"
    if n_excluded:
        note += f"   （provenance/source 筛选排除 {n_excluded} 条）"
    fig.suptitle(f"MiniSystemOne — {obj['set']} 可靠性图\n{note}", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = os.path.join(out_dir, f"reliability_{name}.png")
    fig.savefig(path, dpi=args.dpi)
    plt.close(fig)
    print(f"  {name}: n={int(sel.sum())}  ECE {e0:.4f} -> {e1:.4f}   -> {path}")
    return path


def main():
    args = parse_args()
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        raise SystemExit("需要 matplotlib：pip install matplotlib")
    print(f"  字体 {setup_font(plt)}")

    os.makedirs(args.out, exist_ok=True)
    # 文件名是 `<数据集>_<split>.json`（`eval_harness.py` 有意如此，见那里的注释：
    # 同一个 split 会在合成集和公开集上各评测一次，不能共用一个文件名）。
    # 这里允许 `--sets test_known` 这种短写法，唯一命中就解析到实际文件。
    files = sorted(f for f in os.listdir(args.eval)
                   if f.endswith(".json") and f != "index.json")
    if args.sets:
        wanted = []
        for name in args.sets:
            hit = [f for f in files if f == f"{name}.json"
                   or f[:-5].endswith(f"_{name}")]
            if not hit:
                print(f"  {name}: {args.eval} 里没有匹配的 json，跳过")
            wanted.extend(hit)
        files = sorted(dict.fromkeys(wanted))
    print(f"从 {args.eval} 读 {files}")

    src_tag = "_" + _slug("-".join(args.source)) if args.source else ""
    for fname in files:
        path = os.path.join(args.eval, fname)
        obj, packs, prov, sel = load_set(path, args.include_hard, args.source)
        make_set(plt, obj, packs, prov, sel, args, args.out, fname[:-5] + src_tag)
    print(f"\n  -> {args.out}/reliability_*.png")


if __name__ == "__main__":
    main()
