"""
纯函数指标库 —— 无模型、无 I/O，输入输出都是 numpy。

**所有指标按 `provenance` 分桶报告，`hard` 一律排除在校准指标之外。** 理由不是
洁癖：`explicit_rng`、`marginalized`、`tie_set`、`human_annotators` 四种来源的
**不可约噪声底不同**。把 `hard`（贝叶斯上界就是 100% 正确）和 `tie_set`（上界是
均匀分布）混进同一个 ECE，得到的数字既不是模型性质也不是数据性质，只是混合比例。
`metrics_by_provenance` 是唯一对外的主入口，`compute_metrics` 只是它的零件。

几个刻意的选择：

- **ECE 默认 equal-mass 分桶。** 等宽分桶在过度自信的尾部会退化 —— 小模型把大量
  样本堆在 0.9–1.0 这个桶里，桶数很少、每个桶很大，误差被平摊掉，ECE 显得比实际
  好。equal-mass 保证每个桶样本数相当，尾部也能被独立看到。
- **软目标的"正确率"取 `t[argmax p]`**，而不是 `1[argmax p == argmax t]`。前者是
  软标签的正确推广（t 是 one-hot 时两者一致），也是噪声地板能对上的口径。
- **`binomial_noise_floor` 是一等函数，不是脚注。** ChaosNLI 每条约 100 个标注者，
  p̂=0.5 的标准误约 0.05，所以**逐条 ECE 在原理上测不到 0.05 以下**。报告"ECE=0.03"
  而不报同标注数下的地板，等于把标注噪声当成了模型的优点。
"""
import numpy as np

CALIBRATION_PROVENANCE = ("explicit_rng", "marginalized", "tie_set", "human_annotators")
HARD_PROVENANCE = ("hard",)

# 温度校准与 ECE 分桶都按 (primitive, K) 分档：K=2 的二分类与 K=255 的多分类，
# 其 logit 尺度天然不同，一个标量温度盖不住。
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
    """把已 mask 的槽位归零并重归一化，避免被 NEG_INF 之外的脏值污染。"""
    p = np.asarray(p, dtype=np.float64)
    if mask is None:
        return p
    p = np.where(mask, p, 0.0)
    return p / np.maximum(p.sum(-1, keepdims=True), 1e-12)


def top1(p, mask=None):
    """返回 (预测下标, 置信度, 软正确率)。软正确率 = t[argmax p]，见模块注释。"""
    p = mask_normalize(p, mask)
    idx = p.argmax(-1)
    conf = p[np.arange(len(p)), idx]
    return idx, conf


def accuracy(p, t, mask=None):
    """硬准确率：argmax(p) == argmax(t)。软目标上它就是"命中最大概率那一项"。"""
    idx, _ = top1(p, mask)
    return float((idx == np.asarray(t).argmax(-1)).mean())


def soft_accuracy(p, t, mask=None):
    """软正确率：命中项的**目标概率**，在 [0,1] 上连续。校准用这个口径。"""
    idx, _ = top1(p, mask)
    return np.asarray(t, dtype=np.float64)[np.arange(len(idx)), idx]


def nll(p, t, mask=None):
    p = mask_normalize(p, mask)
    t = np.asarray(t, dtype=np.float64)
    return float(-(t * np.log(np.clip(p, 1e-12, None))).sum(-1).mean())


def brier(p, t, mask=None, normalize=False):
    """**不归一化**的 Brier（方案要求）。

    除以 K 会让大 K 处的这一项几乎消失 —— 而大 K 正是最需要它的地方
    （候选越多，"每个候选都报 1/K"这种懒策略越接近正确，越需要被惩罚）。
    """
    p = mask_normalize(p, mask)
    t = np.asarray(t, dtype=np.float64)
    d = ((p - t) ** 2).sum(-1)
    if normalize and mask is not None:
        d = d / mask.sum(-1).clip(min=1)
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
    """期望校准误差。它**不是 proper score**，可以通过换分桶改善，所以永远与
    Brier / NLL 一起报告，单独变动的 ECE 不予采信（方案 R6）。"""
    conf_b, corr_b, n_b = reliability_curve(p, t, mask, bins, binning)
    if len(n_b) == 0:
        return 0.0
    return float((n_b / n_b.sum() * np.abs(corr_b - conf_b)).sum())


def binomial_noise_floor(p, counts, mask=None, bins=15, binning="equal_mass",
                         n_sim=200, seed=0):
    """给定**每个样本的标注者人数**，理想校准模型的期望 ECE（标注噪声那一份）。

    一个真实校准的模型，它报出的 p̂ 就是对真实频率的最好估计。但用它去和**有限
    标注**得到的经验频率比，仍会有偏差 —— 因为经验频率本身在抖。这个函数把那份
    抖动量化出来：对每个样本抽 `Binomial(counts_i, conf_i) / counts_i` 作为
    "另一批标注者会给出的标签"，重算一次 ECE，重复 n_sim 次取均值。置信度**不重抽**
    —— 理想模型的自信就是它的输出，抖动的只有标签。

    **实测说明（一开始按直觉写错过，所以写在这里）**：counts=100 时这个地板只有
    约 0.001，而不是逐条标准误 0.05 那个量级。原因很直接 —— ECE 是**桶内平均**，
    桶内噪声按 1/√n_b 衰减：0.05/√1333 ≈ 0.0014。0.05 是**逐条**误差，不是 ECE
    的误差；两者常被混为一谈。所以：

      - 想让 ECE 地板抬高到可见，得让每桶样本数很少（评测集小、或按 provenance /
        per-schema 细分到几百条）。
      - 在 15 桶 × 数千样本下这份校正是可忽略的，**真正的噪声来自评测集本身的
        有限抽样**（完美校准的模型在 2 万条上 ECE 也有约 0.007，量级反而更大）。
        那一份由 `bootstrap_ci` 负责，不由本函数负责。

    没有标注数（合成集的 P* 是精确值）时地板为 0。仍按方案要求一等报告，但要知道
    它小在哪儿，否则会以为 0.007 是模型的功劳。
    """
    # `rng.binomial` 的 `n` 必须是整数，而上游 `counts_array` 给的是 float64。
    # 这一行以前不存在，且不是疏忽 —— 是这条路径**从来没有被真的跑到过**：
    # 没有任何数据集写过 `target.counts`，于是 counts 恒为 None、整个函数在第一行
    # 就返回 0。等 `target.counts` 补上之后，第一次真跑就在下面抛 TypeError。
    counts = np.asarray(counts, dtype=np.int64)
    if counts.size == 0 or counts.max() <= 0:
        return 0.0
    _, conf = top1(p, mask)
    # 理想模型的"正确率"就是它自己的置信度，所以直接拿 conf 当 corr 的期望。
    rng = np.random.default_rng(seed)
    total, wsum = 0.0, 0
    for _ in range(n_sim):
        corr_sim = rng.binomial(counts, np.clip(conf, 0, 1)) / np.maximum(counts, 1)
        edges = (_bin_edges_equal_mass(conf, bins) if binning == "equal_mass"
                 else _bin_edges_equal_width(bins))
        which = np.clip(np.digitize(conf, edges[1:-1], right=False), 0, len(edges) - 2)
        num = 0.0
        n_tot = 0
        for b in range(len(edges) - 1):
            sel = which == b
            n = int(sel.sum())
            if n == 0:
                continue
            num += n * abs(float(corr_sim[sel].mean()) - float(conf[sel].mean()))
            n_tot += n
        if n_tot:
            total += num / n_tot
            wsum += 1
    return float(total / wsum) if wsum else 0.0


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
    """0.5 · Σ|CDF_p − CDF_t| —— CDF 形式对序数距离更敏感（方案指定）。

    levels 只用于给出 MAE 的量纲（等级数）；纯 CDF 版本不需要它。
    """
    p = mask_normalize(p, mask)
    t = mask_normalize(t, mask)
    cdf_p = np.cumsum(p, -1)[..., :-1]
    cdf_t = np.cumsum(t, -1)[..., :-1]
    return float((0.5 * np.abs(cdf_p - cdf_t).sum(-1)).mean())


# ---------------------------------------------------------------------------
# 弃权
# ---------------------------------------------------------------------------
def risk_coverage_curve(p, t, mask=None, abstain_idx=None):
    """按置信度从高到低排序，返回 (覆盖率, 风险=1−准确率, 阈值)。

    `abstain_idx` 给出时，把"预测为弃权"的样本视为不覆盖（方案：弃权是一个候选，
    熵阈值规则只是事后补充的画图手段）。
    """
    idx, conf = top1(p, mask)
    correct = (idx == np.asarray(t).argmax(-1)).astype(np.float64)
    if abstain_idx is not None:
        keep = idx != abstain_idx
    else:
        keep = np.ones_like(conf, dtype=bool)
    order = np.argsort(-conf)
    ks = np.arange(1, len(order) + 1)
    hit = correct[order].cumsum()
    cov = np.where(keep[order].cumsum() > 0, ks / len(order), 0.0)
    risk = 1.0 - hit / ks
    thr = conf[order]
    return cov, risk, thr


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------
def compute_metrics(p, t, mask=None, provenance=None, is_ord=None, levels=None,
                    counts=None, bins=15, binning="equal_mass", with_ci=False):
    """单组样本的全部指标。`provenance` / `counts` 给了就算噪声地板。"""
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
        "brier": brier(p, t, mask),
        "ece": ece(p, t, mask, bins, binning),
        "ece_binning": binning,
        "ece_bins": bins,
    }
    if counts is not None:
        out["ece_noise_floor"] = binomial_noise_floor(p, counts, mask, bins, binning)
        out["ece_corrected"] = max(0.0, out["ece"] - out["ece_noise_floor"])
    if is_ord is not None and np.any(is_ord):
        sel = np.asarray(is_ord, dtype=bool)
        out["ordinal_mae"] = ordinal_mae(p[sel], t[sel], levels, None if mask is None else mask[sel])
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
    """按 provenance 分桶 + 一个 `all` 总桶。**这是对外的主入口。**

    `all` 桶同时给出两份：`all`（全部样本）和 `calibration`（剔除 `hard`）。
    后者才是能拿去和论文比的那个 —— 前者混进了贝叶斯上界为 100% 的样本，
    会把 ECE 稀释得很好看。

    注意 `hard` 与 `calibration` **不是互斥的两类**：`calibration` 是
    `explicit_rng ∪ marginalized ∪ tie_set ∪ human_annotators` 的并集，所以在
    同时含 `hard` 与软来源的集合上，`all` 的 n = `calibration` 的 n + `hard` 的 n；
    若整个集合都是软来源，则 `calibration` 不会单独出现（它与 `all` 完全重合，
    再给一行只会让人误以为少算了一部分样本）。
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
    keep = np.isin(provenance, CALIBRATION_PROVENANCE)
    if 0 < keep.sum() < len(provenance):
        res["calibration"] = compute_metrics(
            p[keep], t[keep], None if mask is None else mask[keep],
            **{k: _subset(v, keep) for k, v in kw.items()},
        )
    return res
