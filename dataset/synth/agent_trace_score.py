"""
G3 `agent_trace_score` —— **唯一的 `score` primitive 来源**，provenance = `marginalized`。

任务：给一段 agent 执行轨迹（成功/失败/重试/违规的步数统计）评一个 1–5 档。
它是全项目唯一产出**序数分布**的生成器，因此是 `is_ord` 那条 CDF-MSE 损失、
`ordinal_mae` 与 `expected_score` 三个指标的唯一数据来源 —— 没有它，这三处代码
从未被真正执行过。

## 为什么是 `marginalized` 而不是 `explicit_rng`

方案 (b) 号软目标：生成器维护一个**隐藏变量**，只在 state 里粗粒度暴露。这里的
隐藏变量是**哪位评审来打分**。state 说清了评审委员会的构成（宽松/标准/严格各几人），
但没说这次是谁评的，所以

    P(等级 s | x) = Σ_h w_h · P(s | q, h)

其中 `h ∈ {宽松, 标准, 严格}` 是严格度，`w_h` 是委员比例（**印在 state 里**），
`q` 是轨迹质量分（**由 state 里的计数算出**）。不确定在评审池里，不在 state 里。

## 两个变体：`fine` 与 `coarse`，差在 q 知不知道

  - **fine**（偶数模板）：state 印出**精确**的成功/失败/重试/违规步数 → q 精确已知，
    分布只因评审严格度混合而变宽。
  - **coarse**（奇数模板）：state 只印出成功的**分桶区间**（`成功步数 3-5`），
    于是 q 本身也是区间的：

        P(等级 s | x) = Σ_h w_h · (1/(q_hi−q_lo)) ∫_{q_lo}^{q_hi} P(s | q, h) dq

两者共用评审混合，coarse 只多叠一层 q 的区间不确定性。**刻意做成嵌套而不是 2×2**
（不做"单评审 vs 三评审 × 精确 vs 分桶"四格）：那会引入第二个需要单独归因的旋钮，
而这里要展示的对比只有一句话 —— **同一套规则下，state 说得越粗，分布越宽**。
教学上这是"模型在做推断而不是模式匹配"最好的证据。

分桶是真实存在的工程事实（遥测系统报的就是直方图分桶），不是为造难度而造。

## R1（本项目最高风险）在这里的具体防线

`q` 的真实值依赖**内部计数**。fine 变体把计数印进 state，所以 `q_true == q_used`。
coarse 变体**不能**用 `q_true` 算目标 —— 那一半信息没印出来，用它等于把答案泄漏
进去，而泄漏的表现恰好是"ECE 很漂亮"。所以：

  - audit 同时留 `q_true` 与 `q_used_lo` / `q_used_hi`（fine 时三者相等）；
  - 目标只由 `q_used_*` 算出，oracle 上界也读 `q_used_*`，于是**构造上一致**；
  - `q_true` 只供"模型到底丢了多少信息"的误差分析，不进任何目标。

渲染回读由 `RENDERED` 逐字段声明，`audit_synthetic.py` 批量核对 —— 没印出来的量
会在那里被抓住，而不是等到报告里出现一个谁也解释不了的坏 ECE。

## 分桶是**离散**的，不是连续区间

成功步数是计数，"成功步数 4-7" 指的是四个整数值，所以正确模型是**四个值上的离散
均匀**，目标是在这四个点上对 Φ 取算术平均（见 `_target`）。

这一点必须和**措辞**对齐：早先的版本把区间当连续量做积分，同时政策段里只写了门槛
而**没写**分桶内如何分布。两处都错，且错的方向是 R1 最怕的那一种 —— 目标里含一个
state 无从推断的假设，模型再努力也够不到，而报告出来的 ECE 度量的是这个缺口，
不是模型。所以现在：算的是离散平均，政策段里也明写"桶内各取值等可能"。

代价是那半个 `_Phi` 的闭式原函数不再需要（`_mean_phi`/`_phi` 已删）——
省掉的复杂度比换来的精度更值钱。
"""
import math

from dataset.synth.base import Generator, StateBuilder
from dataset.synth.lexicon import LEVELS, STATUSES, eid, fmt_count, term

# 质量分基线。**必须有**，且必须印在 state 里：加上一个常数等价于把所有阈值平移
# 同一距离，规则不变，但它让 q 与阈值都落在非负区间 —— 负号在 `scan_numbers` 里
# 有自己的解析分支（`-¥56.66` 的负号在货币符之前），渲染非负数少一个出错的地方。
BASE = 20

# q 的整数权重。**规则的一部分，不是拟合出来的** —— 这一点是 R1 的前提：只有规则
# 完全写死在 state 里，目标才谈得上"可从 state 重算"。
# 违规格扣得最重（3），其次失败与成功对称（2），重试只轻微扣分（1）。
WEIGHTS = {"ok": 2, "bad": -2, "retry": -1, "violation": -3}

# 质量分算式里的四个计数项。**规则句从这里拼出来**，见 `_formula` ——
# 术语表和权重表是同一件事的两半，手写句子会让它们各存一份而互不知道。
TERMS = (
    ("ok", "成功步数", "successful steps"),
    ("bad", "失败步数", "failed steps"),
    ("retry", "重试次数", "retries"),
    ("violation", "违规次数", "violations"),
)

# 评审严格度：类别顺序即 h，与 `_target` 里的 (-1, 0, +1) 一一对应。
STRICTNESS = (("lenient", "宽松"), ("standard", "标准"), ("strict", "严格"))
H_SHIFT = (-1, 0, 1)

# 分桶变体的桶宽，以及成功步数的取值上界（必须是桶宽的整数倍，否则最后一个桶
# 会被截断成窄桶 —— 桶宽不一致会让"粗粒度"这件事只发生在部分样本上）。
#
# 桶宽是这两个变体的**唯一自变量**，所以要按效果定而不是随手取。实测（离散平均
# 之后，3000 条，熵为 5 级分布的香农熵）：
#
#     桶宽 2 → fine 0.672 / coarse 0.712   差 +6%
#     桶宽 3 → fine 0.672 / coarse 0.773   差 +15%
#     桶宽 4 → fine 0.672 / coarse 0.846   差 +26%   ← 选它
#     桶宽 6 → fine 0.672 / coarse 1.013   差 +51%
#
# 取 4 而不是 6：6 时 coarse 的熵已到均匀分布（ln5≈1.609）的 63%，模型很容易退化成
# "看到区间就报接近均匀"，那又变回模式匹配 —— 而这条曲线的意义恰恰是"在推断"。
# 也不取 2：+6% 在图上读不出来。桶宽 4 也符合"遥测按直方图分桶上报"的实际粒度。
OK_MAX = 12
OK_BUCKET = 4

SCORE_QUESTIONS = (
    ("这条执行轨迹应该评为几级？", "What grade should this execution trace receive?"),
    ("该 agent 的表现属于哪一档？", "Which band does this agent's performance fall into?"),
    ("这次执行的质量评分是多少？", "What is the quality grade of this run?"),
    ("这段轨迹应归入哪个等级？", "Which level should this trace be assigned to?"),
    ("这次作业的评定档位是？", "What is the assessment band for this task?"),
    ("该次运行应当打几分？", "What score should this run be given?"),
    ("执行结果的等级是？", "What is the level of the execution outcome?"),
    ("这次调度应评为哪一档？", "Which rating band applies to this dispatch?"),
    ("该轨迹的质量档位是？", "What quality band does this trace have?"),
    ("评定结果落在哪个等级？", "Which level does the assessment land in?"),
    ("该 agent 本次表现的评级是？", "What is this agent's rating for this run?"),
    ("应当给出哪个档次的评价？", "Which tier of evaluation should be given?"),
)

AGENTS = (
    ("scheduler", "调度器"), ("worker", "工作节点"), ("collector", "采集器"),
    ("planner", "规划器"), ("dispatcher", "分派器"), ("crawler", "爬取器"),
    ("migrator", "迁移器"), ("monitor", "监控器"),
)

_SQRT2 = math.sqrt(2.0)


def _Phi(x):
    """标准正态分布函数 Φ。"""
    return 0.5 * (1.0 + math.erf(x / _SQRT2))


def _side(rng, lang):
    """把 "mix" 落成具体一侧。

    **整组候选必须同侧**：逐候选抽边会造出 `[1] 很差 / [2] poor / [3] 一般` 这种
    中英杂排的候选表，那是噪声不是混排 —— 混排是句间切换。
    """
    if lang == "zh":
        return "zh"
    if lang == "en":
        return "en"
    return "zh" if rng.random() < 0.5 else "en"


def _formula(lang):
    """把质量分算式拼成字符串。**数字的唯一来源是 `WEIGHTS` / `BASE`。**

    手写规则句会让权重存两份 —— 一份算目标、一份印进 state —— 而两者一旦漂移，
    state 描述的就是一条模型**算不出来**的规则：坏 ECE 度量的是那句话，不是模型，
    且没有任何检查会报出来。拼出来则让这种漂移在结构上不可能发生。

    系数恒写出（含 `× 1`）。省掉系数会让算式在两种语言、两种措辞下有不同的外观，
    而 `scan_numbers` 的读回是按出现与否判定的 —— 统一形状少一类需要单独核对的
    情况，代价只是一个多余的字符。
    """
    parts = []
    for i, (key, zh, en) in enumerate(TERMS):
        w = WEIGHTS[key]
        name = zh if lang == "zh" else en
        if i == 0:
            parts.append(f"{name} × {w}" if w >= 0 else f"−{name} × {abs(w)}")
        else:
            parts.append(f"{' + ' if w >= 0 else ' − '}{name} × {abs(w)}")
    return "".join(parts) + f" + {BASE}"


class AgentTraceScore(Generator):
    name = "agent_trace_score"
    prefix = "ats"
    n_templates = 24

    # 目标赖以计算的量必须能在它所在的段里读回来。分变体声明，因为两个变体印的
    # 字段不同 —— fine 印 `ok`，coarse 印 `ok_lo`/`ok_hi`。
    #
    # `thresholds` / `reviewers` 是列表，`sufficiency()` 会逐个成员核对。
    # **注意这是集合成员判定**：阈值 12 与某个计数 12 撞车时检查会通过得比实际宽松，
    # 所以它证明的是"这些量在文本里出现过"，不是"每个量各出现了一次"。
    _COMMON_RENDERED = (
        ("bad", "count", "history"), ("retry", "count", "history"),
        ("violation", "count", "history"),
        ("thresholds", "count", "policy"), ("step", "count", "policy"),
        ("sigma", "count", "policy"),
        ("reviewers", "count", "verification"),
    )
    RENDERED = {
        "agent_score_fine": (("ok", "count", "history"),) + _COMMON_RENDERED,
        "agent_score_coarse": (("ok_lo", "count", "history"),
                               ("ok_hi", "count", "history")) + _COMMON_RENDERED,
    }
    # 规则句必须逐次出现。它是"目标可从 state 重算"的**成文依据** —— 少了它，
    # 阈值该往哪个方向比就只是我的约定，模型无从推断，R1 的口子就开了。
    RULE_MARKERS = {
        "agent_score_fine": ("评分规则：", "Grading rule:"),
        "agent_score_coarse": ("评分规则：", "Grading rule:"),
    }

    def sufficiency(self, rec):
        """在基类的数字回读之外，补一条 coarse 变体特有的检查。

        `ok_lo`/`ok_hi` 必须**成对**出现：只印上界或只印下界，区间宽度就少了半边，
        而目标按完整区间算 —— 那会让目标比 state 支持的信息更宽，方向是把模型
        冤枉成"没学好"。
        """
        out = super().sufficiency(rec)
        if rec["schema"]["name"] != "agent_score_coarse":
            return out
        audit = rec["target"]["audit"]
        if audit["ok_hi"] - audit["ok_lo"] != OK_BUCKET - 1:
            out.append(f"成功步数分桶宽度不是 {OK_BUCKET}："
                       f"[{audit['ok_lo']}, {audit['ok_hi']}]")
        return out

    # -- 语义 ---------------------------------------------------------------
    @staticmethod
    def _draw_counts(rng):
        return {"ok": rng.randrange(0, OK_MAX), "bad": rng.randrange(0, 6),
                "retry": rng.randrange(0, 5), "violation": rng.randrange(0, 3)}

    @staticmethod
    def _quality(c, ok=None):
        """质量分。`ok` 只为分桶变体替换成功步数而存在（其余计数取自 c）。"""
        return (BASE + WEIGHTS["ok"] * (c["ok"] if ok is None else ok)
                + WEIGHTS["bad"] * c["bad"] + WEIGHTS["retry"] * c["retry"]
                + WEIGHTS["violation"] * c["violation"])

    def _thresholds(self, rng):
        """从 q 的分布取分位数当四个等级阈值。

        **手写固定阈值是行不通的**：q 的分布形状依赖上面那些计数范围，一改系数阈值
        就错位，而错位不报错 —— 它只会让绝大多数样本落进同一级，序数指标随之变成
        一个没有意义的数。分位数把"五级大致均衡"变成构造上成立的性质。

        分位数还保证阈值非负：q 的下界是 BASE + 4 项最小值 = 20−15 = 5 > 0。
        """
        pilot = sorted(self._quality(self._draw_counts(rng)) for _ in range(400))
        t = [pilot[int(len(pilot) * f)] for f in (0.2, 0.4, 0.6, 0.8)]
        # 严格递增且间隔 ≥2。相等会让某一级在构造上永远为空（那一级的目标恒为 0，
        # 模型学到的"这一级从不出现"会被读数的人当成模型缺陷）。
        for i in range(1, 4):
            if t[i] < t[i - 1] + 2:
                t[i] = t[i - 1] + 2
        return t

    def _reviewers(self, rng):
        """评审委员会构成。允许 0，但**必须印出来**（含 0）—— 见 `_verification`。"""
        counts = [rng.randrange(0, 3) for _ in range(3)]
        if sum(counts) == 0:
            counts[rng.randrange(3)] = 1
        # 单一严格度时 w_h 退化成 one-hot，混合这一层就消失了。留 25% 的退化样本是
        # 有意的 —— 它们是这个生成器内部的对照组：混合真的存在时才该让分布变宽。
        if rng.random() < 0.75 and sum(c > 0 for c in counts) < 2:
            j = counts.index(max(counts))
            counts[rng.choice([i for i in range(3) if i != j])] = rng.randrange(1, 3)
        return counts

    def _target(self, qs, t, step, sigma, counts):
        """Σ_h w_h · P(s | q ∈ qs, h)。返回 5 级分布。

        `qs` 是 q 的**等可能取值列表**：fine 变体只有一个元素（q 精确已知），
        coarse 变体是桶内那几个整数对应的 q。先对 `qs` 取算术平均，再按门槛切开 ——
        顺序不能反。先切再平均会在 `Σ_h` 之外多一层耦合，而 `P(s|h)` 里 q 与 h
        本来就是独立的两个不确定性来源。
        """
        probs = [0.0] * 5
        for h, cnt in zip(H_SHIFT, counts):
            if cnt <= 0:
                continue
            T = [x + h * step for x in t]
            prev = 0.0
            for k in range(4):                    # 第 k 级 = Φ(T_k) − Φ(T_{k−1})
                cur = sum(_Phi((T[k] - q) / sigma) for q in qs) / len(qs)
                probs[k] += cnt * (cur - prev)
                prev = cur
            probs[4] += cnt * (1.0 - prev)        # 最高一级没有上阈值
        s = sum(probs)
        return [x / s for x in probs]

    # -- 组装 ---------------------------------------------------------------
    def _item(self, rng, pool, template_id, lang):
        idx = int(template_id[len(self.prefix) + 1:])
        qidx = idx // 2
        coarse = idx % 2 == 1

        counts = self._draw_counts(rng)
        t = self._thresholds(rng)
        reviewers = self._reviewers(rng)
        step = rng.choice((2, 3))
        sigma = rng.choice((1, 2, 3))

        q_true = self._quality(counts)
        if coarse:
            # 只粗化成功步数。系数最大（2）的那一项粗化后对分布的推动最明显，
            # 而四项全粗化会让 q 的区间宽到接近均匀分布 —— 那测的是"模型会不会报
            # 均匀"，不是"会不会做区间推断"。
            lo = (counts["ok"] // OK_BUCKET) * OK_BUCKET
            hi = lo + OK_BUCKET - 1
        else:
            lo = hi = counts["ok"]
        # 桶内取值逐个展开成 q 的等可能列表。**不传 (lo, hi) 边界而是传这个列表**，
        # 因为桶是离散的：从边界做连续积分会把"四个整数等可能"算成"区间上连续均匀"，
        # 而 state 里印的是前者。
        qs = [self._quality(counts, ok=k) for k in range(lo, hi + 1)]
        qlo, qhi = qs[0], qs[-1]

        p = self._target(qs, t, step, sigma, reviewers)

        side = _side(rng, lang)
        subj = f"{term(rng, AGENTS, lang)} {eid(pool, 'agent', rng.randrange(64))}"
        sb = StateBuilder(lang)
        sb.add("account", self._identity(rng, lang, subj), 1)
        sb.add("history", self._history(rng, lang, counts, lo, hi, coarse), 3)
        sb.add("verification", self._verification(rng, lang, reviewers), 3)
        sb.add("grading", self._rule(rng, lang), 3)
        sb.add("policy", self._params(rng, lang, t, step, sigma, coarse), 3)

        return {
            "question": term(rng, (SCORE_QUESTIONS[qidx],), lang),
            "primitive": "score",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": "agent_score_coarse" if coarse else "agent_score_fine",
            "schema_desc": "给 agent 执行轨迹评 1–5 档",
            "positive_label": None,
            "candidates": self._candidates(rng, side),
            "target": {"kind": "soft", "p": p, "provenance": "marginalized",
                       "renormalized": False,
                       # q_true 与 q_used_* 并列。前者**只**供误差分析（"粗粒度到底
                       # 丢了什么"），目标与 oracle 上界一律读后者 —— 见模块注释。
                       "audit": {"ok": counts["ok"], "bad": counts["bad"],
                                 "retry": counts["retry"],
                                 "violation": counts["violation"],
                                 "ok_lo": lo, "ok_hi": hi,
                                 "q_true": q_true,
                                 "q_used_lo": qlo, "q_used_hi": qhi,
                                 "thresholds": list(t), "step": step,
                                 "sigma": sigma, "reviewers": list(reviewers)}},
            "meta": {"variant": "coarse" if coarse else "fine",
                     "language": lang, "question_idx": qidx, "K_full": 5},
        }

    def _candidates(self, rng, side):
        """5 个等级候选。**序号必须渲染进文本**（`[3] 一般`）—— 序关系是这个
        primitive 的全部内容，把它藏进候选顺序等于要模型从位置学序数，而位置在
        collate 里会被逐例打乱。`meta.level` 同步给出等级号，供期望分与序数 MAE。"""
        return [{"text": f"[{k + 1}] {LEVELS[k][0] if side == 'en' else LEVELS[k][1]}",
                 "label": LEVELS[k][0], "meta": {"level": k + 1}} for k in range(5)]

    # -- state 各段 ---------------------------------------------------------
    def _identity(self, rng, lang, subj):
        st = term(rng, STATUSES, lang)
        if lang == "zh":
            return f"执行体：{subj}，当前状态为 {st}"
        return f"Agent: {subj}, status {st}"

    def _history(self, rng, lang, c, lo, hi, coarse):
        """轨迹计数段。coarse 时成功步数印成区间，其余照印精确值。

        区间两端的数字都要能被 `scan_numbers` 读回来：分隔符用 ASCII 连字符，
        它不是 `\\d[\\d\\s,.]*\\d` 的词内字符，于是两端各成一个独立数字。
        """
        ok_zh = (f"成功步数 {lo}-{hi}" if coarse
                 else f"成功步数 {fmt_count(rng, c['ok'], lang)}")
        ok_en = (f"successful steps {lo}-{hi}" if coarse
                 else f"successful steps {fmt_count(rng, c['ok'], lang)}")
        tail_zh = (f"失败步数 {fmt_count(rng, c['bad'], lang)}；"
                   f"重试 {fmt_count(rng, c['retry'], lang)}；"
                   f"违规 {fmt_count(rng, c['violation'], lang)}")
        tail_en = (f"failed steps {fmt_count(rng, c['bad'], lang)}; "
                   f"retries {fmt_count(rng, c['retry'], lang)}; "
                   f"violations {fmt_count(rng, c['violation'], lang)}")
        if lang == "zh":
            return f"执行历史：{ok_zh}；{tail_zh}"
        return f"Execution history: {ok_en}; {tail_en}"

    def _verification(self, rng, lang, counts):
        """评审构成。**含 0 也照印**（`宽松 0 次`）。

        省略 0 位看起来更自然，但 `RENDERED` 是集合成员判定：0 被印出来的次数多到
        没有信息量，一省略就会让"这个数确实印在了文本里"这件事在 0 上变成空断言。
        照印则至少保证三项都有各自的出处。

        两种语言各拼一遍整句，而不是"按语言挑标签、再按语言挑计数渲染"。后者会让
        分隔符固定成中文的 `；` 却出现在英文句子里 —— 那是 `term()` 逐词选侧的同一个
        毛病（混排是**句间**切换，句内换语言只会让文本变成噪声）。
        """
        n = sum(counts)
        if lang == "zh":
            items = "；".join(f"{zh} {fmt_count(rng, c, 'zh')}"
                             for (_, zh), c in zip(STRICTNESS, counts))
            return f"评审构成：共 {n} 位 —— {items}（本次由谁评定未指定）"
        items = ", ".join(f"{en} {c}" for (en, _), c in zip(STRICTNESS, counts))
        return (f"Review panel: {n} member{'s' if n != 1 else ''} — {items} "
                f"(this run's assigned reviewer is unspecified)")

    def _rule(self, rng, lang):
        """算式句。**单独成段**（seg=`grading`），理由见 `_params`。

        措辞轮换但不改数字 —— 轮换是为了不给模板匹配器一个稳定的长锚，而
        `RULE_MARKERS` 保证无论哪种措辞都在。算式本身来自 `_formula`，不手写。
        """
        f = _formula(lang)
        if lang == "zh":
            return rng.choice((
                f"评分规则：质量分 = {f}。",
                f"评分规则：先算质量分（{f}），再对照等级门槛。",
            ))
        return rng.choice((
            f"Grading rule: quality = {f}.",
            f"Grading rule: first compute the quality score ({f}), "
            "then compare it against the level thresholds.",
        ))

    def _params(self, rng, lang, t, step, sigma, coarse):
        """门槛与浮动参数。目标赖以计算的每个量都在这里成文。

        **和算式分两段不是排版洁癖。** `scan_numbers` 返回的是数字**集合**，而算式
        的系数恰好是 1/2/3、`BASE` 是 20 —— 同段时 `step`（∈{2,3}）与 `sigma`
        （∈{1,2,3}）的回读检查会**永远通过**，无论它们是否真的印出来。一个恒真的
        检查比没有检查更糟：它给出的是虚假的保证。

        段内也刻意不出现其它数字：等级归属写成"相邻两门槛之间取较高的一档"而不是
        `第 k+1 档` —— 后者里的那个 `1` 会让 `sigma=1` 同样变成恒真。

        `coarse` 时多一句**桶内等可能**。这句不是修饰：目标是在桶内各整数上取平均，
        而"平均"这个假设若不在文本里，模型就只能猜桶内的分布形状 —— 猜错不报错，
        只让 ECE 变差，正是 R1 要挡的那种缺口。
        """
        t1, t2, t3, t4 = t
        if lang == "zh":
            tail = ("分桶区间的每个取值等可能，无先后偏好。" if coarse else "")
            return (f"等级门槛依次为 {t1}、{t2}、{t3}、{t4}；"
                    f"质量分落在相邻两门槛之间即取较高的那一档。{tail}"
                    f"严格评审的门槛整体上移 {step} 分；"
                    f"单个评审在门槛附近的判定存在 {sigma} 分的随机浮动。")
        tail = ("Every value inside a bucketed range is equally likely; no "
                "positional preference applies. " if coarse else "")
        return (f"The thresholds are {t1}, {t2}, {t3}, {t4} in order; a quality "
                f"score landing between two adjacent thresholds takes the higher "
                f"band. {tail}"
                f"A strict reviewer shifts every threshold up by {step}; an "
                f"individual reviewer's judgement fluctuates by {sigma} points "
                f"around the thresholds.")

    # -- 复述 ---------------------------------------------------------------
    def paraphrases(self, rng, question, template_id, lang):
        """**必须与 `question` 同语言** —— 换语言测的是语言切换，不是"学的是问题
        还是模板"，那正是这个套件要隔离的东西。"""
        order = list(range(len(SCORE_QUESTIONS)))
        rng.shuffle(order)
        out = []
        for j in order:
            text = term(rng, (SCORE_QUESTIONS[j],), lang)
            if text != question and text not in out:
                out.append(text)
            if len(out) == 3:
                break
        return out
