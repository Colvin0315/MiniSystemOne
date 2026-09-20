"""
纯 NumPy 指标库。硬标签和软标签都可计算 proper scores 与分桶 top-label ECE。

`distribution_l2` 是预测与给定目标分布的类别平方差之和，再跨样本平均；
`expected_brier` 是相对于该目标分布抽取类别的期望 Brier。人工频率未必是真实条件分布。
`soft_accuracy` 使用 t[argmax p]，one-hot 下自然退化为观测正确率。
按 provenance 分组解释目标来源；混合总体也合法，但须说明组成。
ECE 不是逐条分布准确性或 OOD 能力的证明。标注 Monte Carlo 参考量不是通用下界，
不从观测 ECE 中相减。
"""
import numpy as np

SOFT_TARGET_PROVENANCE = ("explicit_rng", "marginalized", "tie_set", "human_annotators")
HARD_PROVENANCE = ("hard",)

# 温度按 (primitive, K) 分档；是否优于全局温度需在留出集实测。
K_BUCKETS = ((2, 2), (3, 4), (5, 8), (9, 32), (33, 255))


def k_bucket(k):
    for lo, hi in K_BUCKETS:
        if lo <= k <= hi:
            return f"{lo}-{hi}"
    return f">{K_BUCKETS[-1][1]}"


def temp_key(primitive, k):
    """温度查找键：`<primitive>|<K 桶>`。

    校准（写 T.json）与评测（读 T.json）**必须**用同一个键构造函数，否则会出现
    "拟合了 15 个温度、评测时只命中 3 个、其余静默落回全局值"这种查不出来的一致
    性错误 —— 数字依然会出来，只是不再是拟合的那套。
    """
    return f"{primitive}|{k_bucket(k)}"


# ---------------------------------------------------------------------------
# 温度
# ---------------------------------------------------------------------------
def apply_temperature(p, temp, primitive, k, mask=None, granularity="primitive_k"):
    """把温度作用在**已经归一化的 p** 上。返回 (B, K) 的新分布。

    等价于 `softmax(logits/T)`，不需要原始 logit：
    `softmax(z/T)_k = exp(z_k/T) / Σ_j exp(z_j/T)`，而 `p_k = exp(z_k)/Σ_j exp(z_j)`，
    所以 `p_k^(1/T) ∝ exp(z_k/T)`，共同的 `Σexp(z_j)^(1/T)` 因子在归一化时消掉。

    这条恒等式是 harness 只存 `p` 就能事后重算任何温度的依据。**零质量必须保持
    零质量**：屏蔽槽 p=0，`0^(1/T)=0`，归一化后仍是 0，不会凭空长出来。
    """
    p = np.asarray(p, dtype=np.float64)
    m = None if mask is None else np.asarray(mask, dtype=bool)
    t = _temperature_vector(temp, granularity, primitive, k)
    if t is None:
        return p
    q = np.power(np.clip(p, 0.0, None), 1.0 / t[:, None])
    q = np.where(m, q, 0.0) if m is not None else q
    return q / np.maximum(q.sum(-1, keepdims=True), 1e-300)


def _temperature_vector(temp, granularity, primitive, k):
    """按粒度取逐样本温度。粒度对应的 T.json 段缺失时返回 None（= 不做校准）。"""
    if temp is None:
        return None
    if granularity == "global":
        if "global" not in temp:
            return None
        return np.full(len(primitive), float(temp["global"]))
    table = temp.get(granularity)
    if not table:
        return None
    if granularity == "primitive":
        keys = [str(x) for x in primitive]
    elif granularity == "primitive_k":
        keys = [temp_key(pr, kk) for pr, kk in zip(primitive, k)]
    else:
        raise ValueError(f"未知粒度 {granularity}；可选 global / primitive / primitive_k")
    fallback = float(temp.get("global", 1.0))
    return np.array([float(table.get(x, fallback)) for x in keys])


# ---------------------------------------------------------------------------
# 基础量
# ---------------------------------------------------------------------------
def mask_normalize(p, mask=None):
    """Zero masked slots and normalize each row; reject invalid distributions."""
    p = np.asarray(p, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] == 0:
        raise ValueError("probabilities must have shape (N, K), K > 0")
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != p.shape:
            raise ValueError("mask must match probabilities")
        p = np.where(mask, p, 0.0)
    mass = p.sum(-1, keepdims=True)
    if not np.isfinite(p).all() or np.any(p < 0) or np.any(mass <= 0):
        raise ValueError("each row needs finite nonnegative probabilities and positive mass")
    return p / mass


def top1(p, mask=None):
    """Return predicted indices and confidence after masking/normalization."""
    p = mask_normalize(p, mask)
    idx = p.argmax(-1)
    conf = p[np.arange(len(p)), idx]
    return idx, conf


def accuracy(p, t, mask=None):
    """Argmax agreement; ties in the target use NumPy's first-index rule."""
    idx, _ = top1(p, mask)
    return float((idx == mask_normalize(t, mask).argmax(-1)).mean())


def soft_accuracy(p, t, mask=None):
    """Per-row target probability of the prediction; hard labels give 0/1."""
    idx, _ = top1(p, mask)
    return mask_normalize(t, mask)[np.arange(len(idx)), idx]


def nll(p, t, mask=None):
    p = mask_normalize(p, mask)
    t = mask_normalize(t, mask)
    return float(-(t * np.log(np.clip(p, 1e-12, None))).sum(-1).mean())


def _class_count(p, mask):
    return p.shape[-1] if mask is None else np.asarray(mask, dtype=bool).sum(-1)


def distribution_l2(p, t, mask=None, normalize=False):
    """Mean over rows of sum_k (p_k-t_k)^2 (not per-class MSE).

    Optional normalize=True divides each row by its valid class count, including
    K when no mask is supplied. Public metric reports use the unnormalized sum.
    """
    p, t = mask_normalize(p, mask), mask_normalize(t, mask)
    d = ((p - t) ** 2).sum(-1)
    if normalize:
        d = d / _class_count(p, mask)
    return float(d.mean())


def expected_brier(p, t, mask=None, normalize=False):
    """E_{Y~t}[sum_k (p_k-1[Y=k])^2], averaged over rows.

    Equals distribution_l2 + mean(1-sum_k t_k^2). For one-hot targets
    this is the observed multiclass Brier score, without an additive term.
    """
    p, t = mask_normalize(p, mask), mask_normalize(t, mask)
    d = ((p - t) ** 2).sum(-1) + 1.0 - (t ** 2).sum(-1)
    if normalize:
        d = d / _class_count(p, mask)
    return float(d.mean())


# ---------------------------------------------------------------------------
# 校准
# ---------------------------------------------------------------------------
def _bin_edges_equal_mass(conf, bins):
    """按置信度**分位数**切桶，保证每桶样本数相当。"""
    qs = np.linspace(0, 1, bins + 1)
    edges = np.quantile(conf, qs)
    edges[0], edges[-1] = -np.inf, np.inf
    # 置信度大量重复（例如 K=2 且模型饱和）时会有空桶，去重后桶数会变少 ——
    # 这是可接受的，返回实际边界即可，不强行凑数。
    return np.unique(edges)


def _bin_edges_equal_width(bins):
    return np.linspace(0.0, 1.0, bins + 1)


def reliability_curve(p, t, mask=None, bins=15, binning="equal_mass"):
    """返回 (桶平均置信度, 桶平均正确率, 桶样本数)，只保留非空桶。"""
    _, conf = top1(p, mask)
    corr = soft_accuracy(p, t, mask)
    if bins < 1 or int(bins) != bins:
        raise ValueError("bins must be a positive integer")
    if binning not in ("equal_mass", "equal_width"):
        raise ValueError("binning must be equal_mass or equal_width")
    if not len(conf):
        return np.array([]), np.array([]), np.array([], dtype=int)
    edges = (_bin_edges_equal_mass(conf, bins) if binning == "equal_mass"
             else _bin_edges_equal_width(bins))
    which = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
    conf_b, corr_b, n_b = [], [], []
    for b in range(len(edges) - 1):
        sel = which == b
        n = int(sel.sum())
        if n == 0:
            continue
        conf_b.append(float(conf[sel].mean()))
        corr_b.append(float(corr[sel].mean()))
        n_b.append(n)
    return np.array(conf_b), np.array(corr_b), np.array(n_b)


def ece(p, t, mask=None, bins=15, binning="equal_mass"):
    """Binned top-label calibration error (not a proper score).

    Report alongside distribution_l2, expected_brier and NLL, with bin settings.
    """
    conf_b, corr_b, n_b = reliability_curve(p, t, mask, bins, binning)
    if len(n_b) == 0:
        return 0.0
    return float((n_b / n_b.sum() * np.abs(corr_b - conf_b)).sum())


def ece_annotation_reference(p, counts, mask=None, bins=15, binning="equal_mass",
                             n_sim=200, seed=0):
    """Monte Carlo ECE under an assumed independent annotation model.

    Hold confidence fixed and draw Binomial(count_i, conf_i)/count_i. Only
    rows with finite positive counts participate, including in bin construction.
    Returns None when no annotation counts are available. This diagnostic depends
    on counts, binning, sample size, and the assumption that confidence is the
    true top-label probability; it is neither a universal lower bound nor a
    subtractable bias estimate for observed ECE.
    """
    if counts is None:
        return None
    counts = np.asarray(counts, dtype=np.float64)
    if counts.shape != (len(p),):
        raise ValueError("counts must contain one annotation count per row")
    keep = np.isfinite(counts) & (counts > 0)
    if not keep.any():
        return None
    if np.any(counts[keep] != np.floor(counts[keep])) or n_sim < 1:
        raise ValueError("positive counts must be integers and n_sim must be positive")
    counts = counts[keep].astype(np.int64)
    _, conf = top1(np.asarray(p)[keep], None if mask is None else np.asarray(mask)[keep])
    edges = (_bin_edges_equal_mass(conf, bins) if binning == "equal_mass"
             else _bin_edges_equal_width(bins))
    which = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
    groups = [which == b for b in range(len(edges) - 1)]
    groups = [sel for sel in groups if sel.any()]
    rng = np.random.default_rng(seed)
    total = 0.0
    for _ in range(n_sim):
        corr_sim = rng.binomial(counts, np.clip(conf, 0, 1)) / counts
        total += sum(sel.mean() * abs(corr_sim[sel].mean() - conf[sel].mean())
                     for sel in groups)
    return float(total / n_sim)


def bootstrap_ci(fn, n, n_boot=1000, alpha=0.05, seed=0):
    """对逐样本指标做 bootstrap 置信区间。

    `fn(idx)` 接受一组下标返回标量。用下标而不是重采样数组，是因为
    `per_sample` 里的每一列都要跟着一起重排，传下标是唯一不会搞错对齐的方式。
    """
    rng = np.random.default_rng(seed)
    vals = np.array([fn(rng.integers(0, n, n)) for _ in range(n_boot)])
    return float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))


# ---------------------------------------------------------------------------
# Score primitive
# ---------------------------------------------------------------------------
def expected_score(p, levels, mask=None):
    """Σ level_k · p_k。Score 的头号输出，取决于整个分布的形状而非 argmax。"""
    p = mask_normalize(p, mask)
    return (p * np.asarray(levels, dtype=np.float64)).sum(-1)


def ordinal_mae(p, t, levels=None, mask=None):
    """Wasserstein-1 distance in grade units (MAE for point distributions).

    Sort by explicit levels, not presentation order. If omitted, columns are
    assumed to already be consecutive increasing levels. Historical versions
    ignored levels and multiplied by 0.5; those results are not comparable.
    """
    p = mask_normalize(p, mask)
    t = mask_normalize(t, mask)
    valid = np.ones_like(p, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    lv = np.broadcast_to(np.arange(p.shape[-1]) if levels is None else np.asarray(levels), p.shape)
    if not np.isfinite(lv[valid]).all():
        raise ValueError("Score levels must be finite")
    order = np.argsort(np.where(valid, lv, np.inf), axis=-1)
    sorted_lv = np.take_along_axis(lv, order, -1)
    sorted_mask = np.take_along_axis(valid, order, -1)
    edges = sorted_mask[:, :-1] & sorted_mask[:, 1:]
    gaps = np.diff(sorted_lv, axis=-1)
    if np.any(valid.sum(-1) < 2) or np.any(gaps[edges] <= 0):
        raise ValueError("Score requires at least two distinct levels")
    delta = np.cumsum(np.take_along_axis(p - t, order, -1), -1)[:, :-1]
    return float((np.abs(delta) * np.where(edges, gaps, 0)).sum(-1).mean())


# ---------------------------------------------------------------------------
# 弃权
# ---------------------------------------------------------------------------
def risk_coverage_curve(p, t, mask=None, abstain_idx=None):
    """Return coverage, accepted risk, threshold for conf >= threshold.

    All equal-confidence rows enter together. Predictions equal to abstain_idx
    (a scalar or one index per row; -1 means no abstain candidate) never count
    as accepted. Coverage divides by all rows; risk divides only by accepted
    rows and uses 1-t[pred]. Zero-coverage risk is NaN (serialize as JSON null).
    The initial threshold +inf represents accepting nothing.
    """
    idx, conf = top1(p, mask)
    correct = soft_accuracy(p, t, mask)
    if abstain_idx is None:
        keep = np.ones(len(idx), dtype=bool)
    else:
        abstain = np.asarray(abstain_idx)
        if abstain.ndim > 1 or (abstain.ndim == 1 and abstain.shape != idx.shape):
            raise ValueError("abstain_idx must be scalar or one index per row")
        keep = idx != abstain
    if not len(idx):
        return np.array([0.0]), np.array([np.nan]), np.array([np.inf])
    order = np.argsort(-conf, kind="stable")
    ends = np.r_[np.flatnonzero(np.diff(conf[order]) != 0), len(order) - 1]
    accepted = keep[order].cumsum()[ends]
    hit = (correct[order] * keep[order]).cumsum()[ends]
    risk = np.full(len(ends), np.nan)
    np.divide(accepted - hit, accepted, out=risk, where=accepted > 0)
    return (np.r_[0.0, accepted / len(idx)], np.r_[np.nan, risk],
            np.r_[np.inf, conf[order][ends]])


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def compute_metrics(p, t, mask=None, provenance=None, is_ord=None, levels=None,
                    counts=None, bins=15, binning="equal_mass", with_ci=False):
    """单组样本的指标；counts 可提供假设标注模型下的 MC 参考量。"""
    out = {
        "n": int(len(p)),
        "accuracy": accuracy(p, t, mask),
        # **硬准确率与软准确率必须一起报。** 并列目标（`tie_set`）上硬 argmax 会把
        # 位置 0 记成正确 —— 于是并列越多的来源数字越好看，而那不是答对了。软准确率
        # = `t[argmax p]`，在并列目标上给出 1/m 而不是 1，是唯一与校准一致的口径
        # （见模块 docstring）。**不把 `accuracy` 直接改成软口径**：那会让历史结果
        # 与新结果不可比，而且恰好把这个问题藏起来。
        "accuracy_soft": float(np.mean(soft_accuracy(p, t, mask))),
        "nll": nll(p, t, mask),
        "distribution_l2": distribution_l2(p, t, mask),
        "expected_brier": expected_brier(p, t, mask),
        "ece": ece(p, t, mask, bins, binning),
        "ece_binning": binning,
        "ece_bins": bins,
    }
    if counts is not None:
        out["ece_annotation_reference"] = ece_annotation_reference(p, counts, mask, bins, binning)
        c = np.asarray(counts, dtype=np.float64)
        out["ece_annotation_reference_n"] = int((np.isfinite(c) & (c > 0)).sum())
    if is_ord is not None and np.any(is_ord):
        sel = np.asarray(is_ord, dtype=bool)
        ord_levels = levels[sel] if levels is not None and np.ndim(levels) == 2 else levels
        out["ordinal_mae"] = ordinal_mae(p[sel], t[sel], ord_levels, None if mask is None else mask[sel])
        if levels is not None:
            es_p = expected_score(p[sel], levels[sel] if np.ndim(levels) == 2 else levels, mask[sel] if mask is not None else None)
            es_t = (mask_normalize(t[sel], None if mask is None else mask[sel])
                    * (levels[sel] if np.ndim(levels) == 2 else levels)).sum(-1)
            out["expected_score_mae"] = float(np.abs(es_p - es_t).mean())
    if with_ci:
        n = len(p)
        lo, hi = bootstrap_ci(lambda ix: accuracy(p[ix], t[ix], None if mask is None else mask[ix]), n)
        out["accuracy_ci"] = [lo, hi]
    return out


def _subset(v, sel):
    """逐样本的量跟着 `sel` 一起切；标量配置项（`bins` / `binning`）原样传下去。

    判据是 `np.ndim(v) >= 1` 而不是列举键名：以后再加逐样本量就不必回来改这里，
    而漏改一次的后果恰好是"分桶方案被当成逐样本数组切了一刀"—— 它不会报错，
    只会让分桶数变成样本数，ECE 静默地变成一个没有意义的数。
    """
    if v is None:
        return None
    a = np.asarray(v)
    return a[sel] if a.ndim >= 1 else v


def metrics_by_provenance(p, t, mask=None, provenance=None, **kw):
    """Report all rows and each provenance, including valid hard-label ECE.

    `soft_targets` aggregates the named soft-target provenances when it differs
    from `all`. Grouping describes target construction, not calibration eligibility.
    """
    res = {"all": compute_metrics(p, t, mask, **kw)}
    if provenance is None:
        return res
    provenance = np.asarray(provenance)
    for prov in sorted(set(provenance.tolist())):
        sel = provenance == prov
        if sel.sum() == 0:
            continue
        res[prov] = compute_metrics(
            p[sel], t[sel], None if mask is None else mask[sel],
            **{k: _subset(v, sel) for k, v in kw.items()},
        )
    keep = np.isin(provenance, SOFT_TARGET_PROVENANCE)
    if 0 < keep.sum() < len(provenance):
        res["soft_targets"] = compute_metrics(
            p[keep], t[keep], None if mask is None else mask[keep],
            **{k: _subset(v, keep) for k, v in kw.items()},
        )
    return res
