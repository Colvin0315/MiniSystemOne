"""
G5 `refund_policy` —— **从 state 里读出一条概率，再与另一条证据相乘**。

任务：判断一笔退款申请会不会被批准。它与其他生成器的区别不在难度，在**信息的形状**：
印进 state 的不是计数（G3/G4）、不是待求值的算式（G1）、也不是状态词（G2），而是
**概率本身** —— 商户等级的历史赔付率，以及核验的两个条件概率。

于是正确行为是纯粹的贝叶斯组合：

    先验胜算 = 赔付率 / (1 − 赔付率)
    似然比   = P(核验结果 | 确有缺损) / P(核验结果 | 订单完好)
    后验概率 = 先验胜算 × 似然比，再折回概率

不确定性在**世界里**（这笔订单究竟有没有缺损，无人观测），不在 state 里；而算它所需的
每一个量都在 state 里。这正是 `docs/DESIGN.md` 的 (a) 号软目标，与 G1 同构而形状不同。

**为什么值得单独一个生成器。** 上面三行里没有一步是"把一个数字代进公式"（G1 是），
也没有一步是"对区间求长度占比"（G4 是）。模型必须 (a) 认出 state 里印的是概率而非
计数，(b) 走一遍胜算 → 概率的两次换算，(c) 按核验**结果**是阳性还是阴性挑对那一支
似然比。漏掉任一项，p 都会偏，而偏的方向会被读成"模型没校准"。

两个变体，问法完全相同，**只改先验的报告形式**：

  - `refund_point` —— 历史赔付率给一个点值。目标尖锐，provenance = `explicit_rng`。
  - `refund_range` —— 历史赔付率只给一个区间（"介于 30% 与 70% 之间"）。目标是对区间内
    各百分点取平均的结果，provenance = `marginalized`。区间内每个百分点等可能这个
    假设**必须成文**（见 `sufficiency`），否则"区间内怎么取"就只是我的约定，模型无从
    推断，R1 的口子随之打开。

这一对变体是本生成器的教学要点：**同一份证据，粗粒度报告会留下更宽的后验**。模型在
`range` 与 `point` 上的可靠性差距，就是"它在做推断还是在背点值"的直接读数 —— 与 G3 的
fine/coarse 对照是同一个实验设计，只是作用在最外层的一个概率上。

核验结果**按 50/50 抽**，不按其真实边缘分布抽。理由：结果只是 x 的一部分，抽法改变的是
x 的分布，不是给定 x 后 P(y|x) 的正确性；而 50/50 让阳性与阴性两支都拿到足够样本 ——
阴性那一支的似然比是二者**补数**之比，形状与阳性支完全不同，样本少了它会成为整条链上
最弱的一环。

量纲上有一处必须钉死：下面所有概率都以**百分点**为单位抽取，再乘 100 存成 bps
（万分之一）。`fmt_pct` 只渲染一位小数，若 bps 不是 100 的整数倍，`62.17%` 会被印成
`62.2%` 而 `scan_numbers` 读回 6217 —— 那是**误报**，"这个量没渲染出来"会指向一个
本来正确的生成器。整数百分点让渲染与回读逐位相等。
"""
from dataset.synth.base import Generator, StateBuilder, pick_text as _pick
from dataset.synth.lexicon import STATUSES, eid, fmt_pct, term

ORDINALS = (
    ("merchant", "商户"), ("tenant", "租户"), ("organization", "组织"),
    ("workspace", "工作区"), ("account", "账户"),
)

TIERS = (
    ("standard", "标准"), ("silver", "白银"),
    ("gold", "黄金"), ("platinum", "白金"),
)

# 全部以**百分点**为单位（见模块 docstring 末尾关于量纲的那一段）。
PRIOR_PP = (5, 95)             # 历史赔付率点值
BUCKET_LO_PP = (25, 45)        # 区间下沿
BUCKET_WIDTH_PP = (15, 45)     # 区间宽度（百分点），至少 15
DEF_PP = (60, 90)              # 检出率 P(阳性 | 缺损)
OK_PP = (10, 40)               # 误报率 P(阳性 | 完好)

# 两个区间互不相交，因此 检出率 / 误报率 > 1 与 (1−检出率)/(1−误报率) < 1
# **结构性成立**，不需要在运行时判定（下面的范围是实测的，不是推的）：
#   阳性支似然比 = 检出率 / 误报率      ∈ [0.60/0.39, 0.89/0.10] = [1.54, 8.90]
#   阴性支似然比 = 补数之比 = 0.40/0.61 … 0.11/0.90              = [0.12, 0.66]
# 若哪天把上面两个区间改到重叠，这两条会一起失效，而失效的后果是"证据反向支持
# 结论"—— 模型仍然学得会，但学到的是反的，且没有任何检查会报出来。改前先回到这里。
MIN_BUCKET_PP = 15

# 区间内等可能的成文依据。这条不在文本里，`refund_range` 的目标就无从推断。
EQUIP_MARKERS = ("等可能", "equally likely")

REFUND_QUESTIONS = (
    ("这笔退款申请会通过吗？", "Will this refund request be approved?"),
    ("该申请最终会被批准吗？", "Will this claim ultimately be approved?"),
    ("平台会同意退款吗？", "Will the platform grant the refund?"),
    ("这笔款项能退回吗？", "Will this payment be refunded?"),
    ("审核结果是准予退款吗？", "Is the review outcome a refund approval?"),
    ("该请求会不会被驳回？", "Will this request be rejected?"),
    ("这笔退款能批下来吗？", "Will this refund go through?"),
    ("客服会放行这笔退款吗？", "Will support release this refund?"),
    ("该申请能否获得赔付？", "Does this claim qualify for compensation?"),
    ("退款审核的结果是批准吗？", "Is the refund review an approval?"),
    ("这笔申诉会被受理吗？", "Will this dispute be upheld?"),
    ("最终会下发退款吗？", "Will a refund be issued in the end?"),
)


class RefundPolicy(Generator):
    name = "refund_policy"
    prefix = "rp"
    n_templates = 24

    # 目标赖以计算的量必须能在它所在的段里读回来，否则 R1 的口子就开了。
    #
    # 各量**各自独占一个段**（赔付率在 history，两个条件概率在 verification），
    # 这不是排版偏好：`scan_numbers` 返回的是数字**集合**，若把不同量堆进同一段，
    # 某个量恰好等于同段另一个量时就再也不会被检出缺失 —— 一个恒真的检查比没有
    # 检查更糟，它给出的是虚假的保证。独占之后，每个段的数字集合里只有它自己。
    RENDERED = {
        "refund_point": (
            ("prior_bps", "money", "history"),
            ("def_bps", "money", "verification"),
            ("ok_bps", "money", "verification"),
        ),
        "refund_range": (
            ("prior_lo_bps", "money", "history"),
            ("prior_hi_bps", "money", "history"),
            ("def_bps", "money", "verification"),
            ("ok_bps", "money", "verification"),
        ),
    }
    # 措辞轮换后都必须命中其中之一 —— 否则改一句措辞就会让这条检查静默失效。
    RULE_MARKERS = {
        "refund_point": ("赔付规则：", "判定方法：", "后验胜算",
                         "Refund rule:", "Decision method:", "posterior odds"),
        "refund_range": ("赔付规则：", "判定方法：", "后验胜算",
                         "Refund rule:", "Decision method:", "posterior odds"),
    }

    def sufficiency(self, rec):
        """在基类的数字回读之外，补一条 R1 必需的检查。

        它挡的是"目标里含一个 state 无从推断的假设"这一类缺口：不报错，只让 ECE
        变差，而变差的方向会被读成模型的问题（R1）。
        """
        out = super().sufficiency(rec)
        a = rec["target"]["audit"]

        # 区间必须真的是一条**区间**。宽度为 0 时不确定性消失、目标退化成 one-hot，
        # 而 provenance 仍写着 marginalized —— 标签会说谎。
        if rec["schema"]["name"] == "refund_range":
            if a["prior_hi_bps"] - a["prior_lo_bps"] < MIN_BUCKET_PP * 100:
                out.append(f"赔付率区间宽度 {(a['prior_hi_bps'] - a['prior_lo_bps']) // 100} "
                           f"个百分点，小于下限 {MIN_BUCKET_PP}")

        # 区间内等可能这个假设必须成文。目标是按它对区间取的平均，假设不在文本里，
        # 模型就只能猜区间内的分布形状 —— 猜错不报错，只让 ECE 变差。
        if not any(m in rec["state"] for m in EQUIP_MARKERS):
            out.append("区间内等可能的成文依据缺失")
        return out

    # -- 目标 ---------------------------------------------------------------
    @staticmethod
    def _posterior(prior, lr):
        """胜算域里乘，再折回概率。先在概率域相乘是错的 —— 那会把两条证据的
        强度算歪，且偏的方向随 prior 变化，模型学不到一个固定的修正。"""
        o = (prior / (1.0 - prior)) * lr
        return o / (1.0 + o)

    @classmethod
    def _posterior_range(cls, lo_pp, hi_pp, lr):
        """区间内每个百分点等可能，取平均。

        **是对后验取平均，不是对赔付率取平均再算一次后验。** 后验是赔付率的非线性
        函数（胜算域才是线性的），E[f(x)] ≠ f(E[x])；两者之差在小 LR、宽区间时最大，
        而那时正好是这套变体最该显出"宽区间 ⇒ 宽分布"的地方。
        """
        ps = [cls._posterior(pp / 100.0, lr) for pp in range(lo_pp, hi_pp + 1)]
        return sum(ps) / len(ps)

    # -- 语义 ---------------------------------------------------------------
    def _item(self, rng, pool, template_id, lang):
        idx = int(template_id[len(self.prefix) + 1:])
        qidx = idx % len(REFUND_QUESTIONS)
        subj = f"{term(rng, ORDINALS, lang)} {eid(pool, 'org', rng.randrange(64))}"
        tier = term(rng, TIERS, lang)

        def_bps = rng.randrange(*DEF_PP) * 100
        ok_bps = rng.randrange(*OK_PP) * 100
        positive = rng.random() < 0.5
        # 见文件顶部：两个区间不相交 ⇒ 阳性支 > 1、阴性支 < 1，结构性成立。
        lr = (def_bps / ok_bps if positive
              else (10000 - def_bps) / (10000 - ok_bps))

        sb = StateBuilder(lang)
        sb.add("account", self._identity(rng, lang, subj, tier), 1)

        if idx % 2 == 1:
            name, prov = "refund_range", "marginalized"
            lo_pp = rng.randrange(*BUCKET_LO_PP)
            hi_pp = lo_pp + rng.randrange(*BUCKET_WIDTH_PP)
            p = self._posterior_range(lo_pp, hi_pp, lr)
            sb.add("history", self._history_range(rng, lang, lo_pp, hi_pp), 3)
            audit = {"prior_lo_bps": lo_pp * 100, "prior_hi_bps": hi_pp * 100}
        else:
            name, prov = "refund_point", "explicit_rng"
            prior_bps = rng.randrange(*PRIOR_PP) * 100
            p = self._posterior(prior_bps / 10000.0, lr)
            sb.add("history", self._history_point(rng, lang, prior_bps), 3)
            audit = {"prior_bps": prior_bps}

        sb.add("verification", self._verification(rng, lang, positive, def_bps, ok_bps), 3)
        sb.add("policy", self._rule(rng, lang), 3)

        audit.update(def_bps=def_bps, ok_bps=ok_bps, positive=positive,
                     lr=lr, p_raw=p)

        return {
            "question": _pick(rng, lang, *REFUND_QUESTIONS[qidx]),
            "primitive": "noul",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": name,
            "schema_desc": "退款申请是否会获批",
            "positive_label": "yes",
            "candidates": [{"text": "<yes>", "label": "yes", "meta": {"level": None}},
                           {"text": "<no>", "label": "no", "meta": {"level": None}}],
            "target": {"kind": "soft", "p": [p, 1.0 - p], "provenance": prov,
                       "renormalized": False, "audit": audit},
            "meta": {"variant": name, "language": lang, "question_idx": qidx},
        }

    # -- state 各段 ---------------------------------------------------------
    def _identity(self, rng, lang, subj, tier):
        return _pick(rng, lang,
                     f"商户：{subj}，等级为 {tier}，账户状态为 "
                     f"{term(rng, STATUSES, lang)}",
                     f"Merchant: {subj}, tier {tier}, account status "
                     f"{term(rng, STATUSES, lang)}")

    def _history_point(self, rng, lang, prior_bps):
        return _pick(rng, lang,
                     f"该等级的历史赔付率为 {fmt_pct(rng, prior_bps)}",
                     f"Historical claim rate for this tier: {fmt_pct(rng, prior_bps)}")

    def _history_range(self, rng, lang, lo_pp, hi_pp):
        return _pick(rng, lang,
                     f"该等级的历史赔付率介于 {fmt_pct(rng, lo_pp * 100)} 与 "
                     f"{fmt_pct(rng, hi_pp * 100)} 之间",
                     f"Historical claim rate for this tier is between "
                     f"{fmt_pct(rng, lo_pp * 100)} and {fmt_pct(rng, hi_pp * 100)}")

    def _verification(self, rng, lang, positive, def_bps, ok_bps):
        """核验结果 + 两个条件概率。

        两个概率用**自然语言名字**（检出率 / 误报率）而不是 `P(阳性|缺损)` 这类记号
        写出，好让 `_rule` 能用同样的词指代它们而无需重复数字 —— 规则句里一旦出现
        数字，就会与这两个量同段碰撞，把上面的回读检查变成恒真。
        """
        res_zh = "阳性" if positive else "阴性"
        res_en = "positive" if positive else "negative"
        return _pick(rng, lang,
                     f"核验结果：{res_zh}。该核验对确有缺损的订单呈阳性的检出率为 "
                     f"{fmt_pct(rng, def_bps)}，对订单完好时呈阳性的误报率为 "
                     f"{fmt_pct(rng, ok_bps)}",
                     f"Verification result: {res_en}. The check returns positive for "
                     f"{fmt_pct(rng, def_bps)} of genuinely defective orders "
                     f"(detection rate) and for {fmt_pct(rng, ok_bps)} of intact "
                     f"orders (false-positive rate)")

    def _rule(self, rng, lang):
        """规则句。**它不能省** —— 它是贝叶斯组合的成文依据。

        没有它，"先验胜算乘似然比"就只是我的约定：模型无从知道 state 里那两个概率
        是要相乘、相加还是取较大者。R1 要挡的正是这种缺口 —— 目标由我的内部逻辑算
        出，而文本没给出同等的信息。

        三种措辞轮换（同时是两种变体共用的最长恒定块，逐字不变会让每个样本都背上
        一段常数）。三者的措辞都同时覆盖点值与区间两种情形，因此在文本层面**读不出
        变体是哪一个** —— 模型必须真的去读 history 段印的是点还是区间。
        """
        forms = (
            ("赔付规则：先把历史赔付率换算成先验胜算（赔付率除以它的补数）。核验为"
             "阳性时，似然比 = 检出率除以误报率；为阴性时，似然比 = 两者补数之比。"
             "后验胜算 = 先验胜算 × 似然比，折回概率即为批准概率。若赔付率只给出"
             "区间，则区间内每个百分点等可能，取其平均。",
             "Refund rule: first convert the historical claim rate into prior odds "
             "(rate divided by its complement). If the check is positive the "
             "likelihood ratio is the detection rate divided by the false-positive "
             "rate; if negative it is the ratio of their complements. The posterior "
             "odds are the prior odds times that ratio, and converting back gives "
             "the approval probability. When the claim rate is given only as a "
             "range, every percentage point inside it is equally likely and their "
             "average is taken."),
            ("判定方法：历史赔付率先折成胜算 —— 赔付率除以它的补数。似然比随核验"
             "结果取一支：阳性取检出率与误报率之比，阴性取二者补数之比。相乘得到"
             "后验胜算，再折回概率就是批准概率。赔付率若以区间给出，区间内各百分点"
             "等可能，按平均处理。",
             "Decision method: the historical claim rate is first folded into odds "
             "— the rate divided by its complement. The likelihood ratio takes one "
             "of two branches depending on the check: for a positive result, the "
             "detection rate over the false-positive rate; for a negative result, "
             "the ratio of their complements. Multiplying gives the posterior odds, "
             "and folding back yields the approval probability. If the claim rate "
             "comes as a range, every percentage point inside is equally likely and "
             "the average is used."),
            ("后验胜算 = 先验胜算 × 似然比。先验胜算由历史赔付率折算（赔付率除以"
             "补数）；似然比在阳性时是检出率与误报率之比，阴性时是二者补数之比。"
             "区间形式的赔付率按区间内等可能即取平均。",
             "posterior odds = prior odds x likelihood ratio. The prior odds come "
             "from folding the historical claim rate (rate over its complement); "
             "the likelihood ratio is the detection rate over the false-positive "
             "rate for a positive result, or the ratio of their complements for a "
             "negative one. A claim rate given as a range is treated as equally "
             "likely throughout and averaged."),
        )
        zh, en = forms[rng.randrange(len(forms))]
        return _pick(rng, lang, zh, en)

    # -- 复述 ---------------------------------------------------------------
    def paraphrases(self, rng, question, template_id, lang):
        """复述 = 问法表里的其它问法。**必须与 `question` 同语言** —— 拿英文复述去测
        中文问法的鲁棒性，测的是语言切换而不是"学的是问题还是模板"。"""
        order = list(range(len(REFUND_QUESTIONS)))
        rng.shuffle(order)
        out = []
        for j in order:
            zh, en = REFUND_QUESTIONS[j]
            text = _pick(rng, lang, zh, en)
            if text != question and text not in out:
                out.append(text)
            if len(out) == 3:
                break
        return out
