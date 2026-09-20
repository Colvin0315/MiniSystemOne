"""
G4 `security_gate` —— **全项目唯一的弃权监督来源**，候选恒为 allow / deny / abstain。

`abstain` 的监督信号**全项目只有这一个来源**。没有它，`abstain` 在训练里从未是
正确答案，softmax 自然会把它的质量压到 0，于是"模型从不弃权"——那不是模型保守，
是它根本没被教过。

（本仓库**没有产出**风险-覆盖率曲线：`eval/eval_metrics.risk_coverage_curve`
提供了函数，但没有任何脚本调用它。本生成器的作用到"让 abstain 拿到非零质量"为止。）

## 为什么弃权必须是被教出来的

弃权**不是**第二个 head、不是阈值后处理，而是三个候选之一（见方案 §2）。这意味着
它的质量只能来自训练目标。所以这里必须构造出"正确行为就是弃权"的样本，而它们
不能是我随手指定的 —— 那会让弃权变成一条背诵规则。两个变体各提供一种**可验证**的
来源，对应 DATA_SCHEMA 里两种 provenance：

  - **`marginalized`**（偶数模板）—— 异常指数只上报一个**区间**，区间内均匀。
    门槛把风险轴切成三段：低段允许、中间灰区弃权、高段拒绝。于是

        P(动作 | x) = 区间中落在该动作区间上的长度占比

    区间窄或远离门槛 → 分布尖锐；区间跨过门槛 → 分布变宽，弃权拿到质量。
    弃权在这里是**推断出来的**，不是被指定的。

  - **`tie_set`**（奇数模板）—— 两位同级评委给出**分歧**的风险评分。state 客观地
    不决定唯一答案，valid set 就是两人的动作，目标为均匀分布。三种配对
    {允许,拒绝}、{允许,弃权}、{拒绝,弃权} 里两种含弃权。

两者合起来还提供了**负例**：tie 的 {允许,拒绝} 配对里 `abstain` 的目标恰为 0。
只教"何时弃权"而不教"何时不弃权"，模型会退化成见难题就弃权 —— 而风险-覆盖率
曲线要看的正是这条权衡。

## 候选文本不按语言翻译（R3 的一条真实信道）

候选恒为英文动作词 + `<abstain>` 特殊 token，中文 state 下也是。这一约定在
`tool_router` 里已经成立（中文 state 配 `cancel_order`）。这里额外是**必需**的：
若按语言渲染，候选长度会变成 en 侧 (allow=2, deny=3, abstain=1) 而 zh 侧
(允许=1, 拒绝=1, 弃权=2) —— 长度与语言相关，模型可以靠"最长的那项是拒绝"绕开
读文本。它不是先验风险：实测过当前 tokenizer 对这几个词的切分。
把候选换成不透明串就能检验这条信道是否真的关上了 —— 本仓库没有跑这个检验。

`<abstain>` 用特殊 token 而 allow/deny 用普通词，是**刻意的不对称**：弃权是这个
项目里唯一需要干净 pooled 向量的第三选项（整条风险-覆盖率曲线都建立在它身上），
而词表为它留了 id 8。允许/拒绝是普通词，没必要为它们加特殊 token。

## K 增广不做特例

`score` 那条"不得 K 子采样"的规则**不适用于这里**：等级的序关系是轴，丢掉中间等级
会让 `cumsum` 说谎；而闸门的三个动作之间没有序关系，丢掉 `abstain` 只是一个合法的
K 增广（记 `renormalized`），正好教模型"弃权并非总是可选"。所以 `_draw_k` 不动。

唯一要守住的是 `keep_p_min`：灰区样本里 `abstain` 的质量常超过 5%，会被强制保留，
于是"该弃权时弃权"的监督不会被采样漏掉。
"""
from dataset.synth.base import Generator, StateBuilder, pick_text
from dataset.synth.lexicon import STATUSES, eid, term

# 风险分基线。与 G3 同理：加常数不改规则，只把所有量平移进非负区间。
BASE = 20

# 风险分权重。**规则的一部分，印在 state 里**（R1 的前提）。
# 敏感度推高风险、异常指数推高风险、可信度拉低风险。
RISK_WEIGHTS = {"sensitivity": 3, "anomaly": 2, "trust": -2}

# 算式项表 —— 规则句由它拼出，见 `_formula`。与权重表是同一件事的两半。
RISK_TERMS = (
    ("sensitivity", "资源敏感度", "resource sensitivity"),
    ("anomaly", "异常指数", "anomaly index"),
    ("trust", "请求方可信度", "requester trust"),
)

SENS_RANGE = (1, 6)        # 资源敏感度取值
TRUST_RANGE = (0, 5)       # 可信度取值
ANOM_MAX = 12              # 异常指数上界
BAND_RANGE = (2, 8)        # 上报区间宽度（连续均匀）

# 候选。**语言中立**，理由见模块注释。
ACTIONS = ("allow", "deny", "abstain")
ABSTAIN_TEXT = "<abstain>"
ACTION_TEXT = {"allow": "allow", "deny": "deny", "abstain": ABSTAIN_TEXT}

GATE_QUESTIONS = (
    ("该请求应当如何处理？", "How should this request be handled?"),
    ("安全闸门的判定结果是？", "What is the security gate's verdict?"),
    ("这一请求应被放行、拒绝还是弃权？",
     "Should this request be allowed, denied, or abstained on?"),
    ("门禁给出的决定是？", "What decision does the access gate give?"),
    ("针对该请求的处置动作是？", "What action should be taken on this request?"),
    ("应当对该请求作出何种判定？",
     "What determination should be made about this request?"),
    ("这次访问控制的结论是？", "What is the outcome of this access control check?"),
    ("请求的处置结论为？", "What is the disposition of the request?"),
    ("闸门应输出哪个动作？", "Which action should the gate output?"),
    ("该访问请求的裁定是？", "What is the ruling on this access request?"),
    ("请给出该请求的处置决定。", "Give the disposition decision for this request."),
    ("安全校验的结果是？", "What is the result of the security check?"),
)

REQUESTS = (
    ("service account", "服务账号"), ("automation job", "自动化作业"),
    ("field engineer", "现场工程师"), ("contractor", "外包人员"),
    ("on-call operator", "值班运维"), ("third-party integration", "第三方集成"),
    ("internal auditor", "内审员"), ("release pipeline", "发布流水线"),
)

RESOURCES = (
    ("production database", "生产数据库"), ("billing export", "账单导出"),
    ("customer records", "客户记录"), ("deployment keys", "部署密钥"),
    ("audit log archive", "审计日志归档"), ("payment gateway", "支付网关"),
    ("source repository", "代码仓库"), ("backup snapshot", "备份快照"),
)


def _clamp01(x):
    return 0.0 if x < 0.0 else (1.0 if x > 1.0 else x)


def _formula(lang):
    """把风险分算式拼成字符串。数字的唯一来源是 `RISK_WEIGHTS` / `BASE`。

    与 G3 的 `_formula` 同源：手写规则句会让权重存两份，漂移后 state 描述的是一条
    模型算不出来的规则，而没有任何检查会报出来。系数恒写出（含 `× 1`）。
    """
    parts = []
    for i, (key, zh, en) in enumerate(RISK_TERMS):
        w = RISK_WEIGHTS[key]
        name = zh if lang == "zh" else en
        if i == 0:
            parts.append(f"{name} × {w}" if w >= 0 else f"−{name} × {abs(w)}")
        else:
            parts.append(f"{' + ' if w >= 0 else ' − '}{name} × {abs(w)}")
    return "".join(parts) + f" + {BASE}"


class SecurityGate(Generator):
    name = "security_gate"
    prefix = "sg"
    n_templates = 24

    # 目标赖以计算的量必须能在它所在的段里读回来。逐变体声明。
    #
    # `allow`/`deny`/`abstain` 的动作归属由门槛算出，所以**门槛本身**是最关键的两个
    # 待回读量 —— 少了它们，风险轴怎么切就只是我的约定。
    RENDERED = {
        "gate_band": (
            ("sensitivity", "count", "limits"), ("trust", "count", "limits"),
            ("anom_lo", "count", "risk"), ("anom_hi", "count", "risk"),
            ("theta_low", "count", "policy"), ("theta_high", "count", "policy"),
        ),
        "gate_split": (
            ("score_a", "count", "risk"), ("score_b", "count", "risk"),
            ("theta_low", "count", "policy"), ("theta_high", "count", "policy"),
        ),
    }
    # 措辞轮换后都必须命中其中之一 —— 否则改一句措辞就会让这条检查静默失效。
    RULE_MARKERS = {
        "gate_band": ("闸门规则：", "判定依据：", "Gate rule:", "Decision basis:"),
        "gate_split": ("闸门规则：", "判定依据：", "Gate rule:", "Decision basis:"),
    }

    def sufficiency(self, rec):
        """在基类的数字回读之外，补两条 R1 必需的检查。

        都是"目标里含一个 state 无从推断的假设"这一类缺口。它们不报错，只让 ECE
        变差，而变差的方向会被读成模型的问题。
        """
        out = super().sufficiency(rec)
        name = rec["schema"]["name"]
        a = rec["target"]["audit"]

        if name == "gate_band":
            # ① 区间必须真的是一条**区间**。宽度 0 时不确定性消失，目标退化成
            #    one-hot，而 provenance 仍写着 marginalized —— 标签会说谎。
            if a["anom_hi"] - a["anom_lo"] < BAND_RANGE[0]:
                out.append(f"异常区间宽度 {a['anom_hi'] - a['anom_lo']} "
                           f"小于下限 {BAND_RANGE[0]}")
            # ② 区间内均匀这个假设必须成文。目标是区间上取长度占比算出来的，
            #    假设不在文本里，模型就只能猜桶内分布形状。
            if not any(m in rec["state"] for m in
                       ("区间内均匀", "uniformly distributed")):
                out.append("异常区间内均匀分布的假设未成文")
        else:
            # tie 的成文依据：两位评委同级且分歧时两者同等有效。没有它，
            # "分歧时该不该给平的分布"就只是我的约定。
            if not any(m in rec["state"] for m in
                       ("两位评委同级", "assessors have equal standing")):
                out.append("评委同级的成文依据缺失")
            if a["score_a"] == a["score_b"]:
                out.append("两位评委评分相同，那就不是 tie")
        return out

    # -- 语义 ---------------------------------------------------------------
    def _draw_request(self, rng):
        return {"sensitivity": rng.randint(*SENS_RANGE),
                "trust": rng.randint(*TRUST_RANGE)}

    def _risk_at(self, s, t, a):
        """风险分。`a` 是异常指数的**一个具体取值**（区间端点或中点）。"""
        return (BASE + RISK_WEIGHTS["sensitivity"] * s
                + RISK_WEIGHTS["anomaly"] * a + RISK_WEIGHTS["trust"] * t)

    def _risk_mid(self, req, a_mid):
        return self._risk_at(req["sensitivity"], req["trust"], a_mid)

    def _thresholds(self, rng):
        """风险轴上的两个门槛，取分布的三分位。

        **手写固定门槛行不通**：风险分布的形状依赖上面那几个取值区间与权重，改一处
        门槛就错位，而错位不报错 —— 它只会让某一段（最常见的是中间的灰区）几乎
        收不到样本，于是 abstain 的质量恒为 0，整套弃权监督静默失效。

        取整数：state 里印的是整数，`RENDERED` 按集合成员回读，若门槛是 31.5 而
        渲染成 `31.5`，`scan_numbers` 的 plain 读法是 `int(31.5) == 31`，回读会
        **误报缺失**（G3 踩过这一类的坑）。
        """
        pilot = []
        for _ in range(500):
            req = self._draw_request(rng)
            w = rng.randint(*BAND_RANGE)
            a_lo = rng.randint(0, ANOM_MAX - w)
            pilot.append(self._risk_mid(req, a_lo + w / 2.0))
        pilot.sort()
        low = int(round(pilot[len(pilot) // 3]))
        high = int(round(pilot[2 * len(pilot) // 3]))
        # 灰区太窄会让 abstain 拿不到质量。4 分是最小的有意义宽度（权重量级为 2~3）。
        if high < low + 4:
            high = low + 4
        return low, high

    def _action_of(self, score, theta_low, theta_high):
        if score < theta_low:
            return "allow"
        if score > theta_high:
            return "deny"
        return "abstain"

    def _target_band(self, req, a_lo, a_hi, theta_low, theta_high):
        """区间积分：P(动作) = 区间中落在该动作段上的长度占比。

        异常指数在区间上**连续均匀**，风险是它的线性函数，所以占比就是长度比 ——
        不需要数值积分，也不需要 Φ。这与 G3 的离散分桶相对：那里的成功步数是计数
        （离散），这里的异常指数是连续读数（区间的存在是因为遥测只上报范围）。
        两处的假设形状不同，所以两处都必须在 state 里写清。
        """
        c = BASE + RISK_WEIGHTS["sensitivity"] * req["sensitivity"] \
            + RISK_WEIGHTS["trust"] * req["trust"]
        wa = RISK_WEIGHTS["anomaly"]
        w = float(a_hi - a_lo)
        # 风险 < θ_low  ⟺  异常指数 < (θ_low − c) / wa
        p_allow = _clamp01(((theta_low - c) / wa - a_lo) / w)
        # 风险 > θ_high ⟺  异常指数 > (θ_high − c) / wa
        p_deny = _clamp01((a_hi - (theta_high - c) / wa) / w)
        p_abstain = max(0.0, 1.0 - p_allow - p_deny)
        return [p_allow, p_deny, p_abstain]

    # -- 组装 ---------------------------------------------------------------
    def _item(self, rng, pool, template_id, lang):
        idx = int(template_id[len(self.prefix) + 1:])
        qidx = idx // 2
        split = idx % 2 == 1
        theta_low, theta_high = self._thresholds(rng)

        subj = f"{term(rng, REQUESTS, lang)} {eid(pool, 'req', rng.randrange(64))}"
        sb = StateBuilder(lang)
        sb.add("account", self._identity(rng, lang, subj), 1)

        if split:
            scores = self._draw_split(rng, theta_low, theta_high)
            p = self._target_split(scores, theta_low, theta_high)
            name, prov = "gate_split", "tie_set"
            sb.add("risk", self._assessor_block(rng, lang, scores), 3)
            sb.add("policy", self._policy_split(rng, lang, theta_low, theta_high), 3)
            audit = {"theta_low": theta_low, "theta_high": theta_high,
                     "score_a": scores[0], "score_b": scores[1],
                     "action_a": self._action_of(scores[0], theta_low, theta_high),
                     "action_b": self._action_of(scores[1], theta_low, theta_high)}
            variant = "split"
        else:
            req = self._draw_request(rng)
            w = rng.randint(*BAND_RANGE)
            a_lo = rng.randint(0, ANOM_MAX - w)
            a_hi = a_lo + w
            p = self._target_band(req, a_lo, a_hi, theta_low, theta_high)
            name, prov = "gate_band", "marginalized"
            sb.add("limits", self._request_block(rng, lang, req), 3)
            sb.add("risk", self._anomaly_block(rng, lang, a_lo, a_hi), 3)
            sb.add("grading", self._rule(rng, lang), 3)
            sb.add("policy", self._policy_band(rng, lang, theta_low, theta_high), 3)
            audit = {"theta_low": theta_low, "theta_high": theta_high,
                     "sensitivity": req["sensitivity"], "trust": req["trust"],
                     "anom_lo": a_lo, "anom_hi": a_hi}
            variant = "band"

        return {
            "question": term(rng, (GATE_QUESTIONS[qidx],), lang),
            "primitive": "choice",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": name,
            "schema_desc": "安全闸门判定：允许 / 拒绝 / 弃权",
            "positive_label": None,
            "candidates": self._candidates(),
            "target": {"kind": "soft", "p": self._normalize(p),
                       "provenance": prov, "renormalized": False, "audit": audit},
            "meta": {"variant": variant, "language": lang, "question_idx": qidx,
                     "K_full": len(ACTIONS)},
        }

    # -- 两个变体的抽样 -----------------------------------------------------
    def _draw_split(self, rng, theta_low, theta_high):
        """抽出两位**分歧**的同级评委评分。

        两人评的是**同一份请求**（资源敏感度与可信度共用），但各自跑的是独立的稀疏
        采样，所以异常读数彼此独立。这一条是必需的，不是随手取的：若两人的读数取自
        同一个窄区间，能造出的分歧就几乎只有"一个落在灰区、一个贴着灰区"，于是
        {允许, 拒绝} 这个配对几乎抽不到 —— 而它恰是**不弃权**那批负例的唯一来源
        （实测同区间版本里它只占 2%，等于整套负例缺席）。

        拒绝采样而不是构造性地错开：构造会让分歧的模式变成一条固定形状，模型能靠
        它反推动作对。上限兜底防死循环，兜底直接跨门槛取一低一高。
        """
        for _ in range(300):
            req = self._draw_request(rng)
            s, t = req["sensitivity"], req["trust"]
            r1 = self._risk_at(s, t, rng.randint(0, ANOM_MAX))
            r2 = self._risk_at(s, t, rng.randint(0, ANOM_MAX))
            if self._action_of(r1, theta_low, theta_high) != \
               self._action_of(r2, theta_low, theta_high):
                return (r1, r2)
        return (theta_low - 1, theta_high + 1)

    def _target_split(self, scores, theta_low, theta_high):
        """两位评委的动作上均匀。第三种动作恒为 0 —— 这是弃权的**负例**。"""
        a1 = self._action_of(scores[0], theta_low, theta_high)
        a2 = self._action_of(scores[1], theta_low, theta_high)
        p = [0.0, 0.0, 0.0]
        if a1 == a2:                      # 构造上不该发生，兜底给 one-hot
            p[ACTIONS.index(a1)] = 1.0
            return p
        p[ACTIONS.index(a1)] = p[ACTIONS.index(a2)] = 0.5
        return p

    @staticmethod
    def _normalize(p):
        s = sum(p)
        if s <= 0:
            return [1.0 / len(p)] * len(p)
        return [x / s for x in p]

    def _candidates(self):
        """恒三个候选，顺序由基类逐例打乱（连同目标一起置换）。"""
        return [{"text": ACTION_TEXT[a], "label": a, "meta": {"level": None}}
                for a in ACTIONS]

    # -- state 各段 ---------------------------------------------------------
    def _identity(self, rng, lang, subj):
        st = term(rng, STATUSES, lang)
        return pick_text(rng, lang,
                         f"请求方：{subj}，当前状态为 {st}",
                         f"Requester: {subj}, status {st}")

    def _request_block(self, rng, lang, req):
        """敏感度与可信度 —— 两个**精确**读数，进风险算式的一部分。"""
        res = term(rng, RESOURCES, lang)
        return pick_text(
            rng, lang,
            f"目标资源：{res}；资源敏感度 {req['sensitivity']}（越高越敏感）；"
            f"请求方可信度 {req['trust']}（越高越可信）",
            f"Target resource: {res}; resource sensitivity "
            f"{req['sensitivity']} (higher is more sensitive); requester trust "
            f"{req['trust']} (higher is more trusted)")

    def _anomaly_block(self, rng, lang, a_lo, a_hi):
        """异常指数 —— **只上报区间**。这是 marginalized 那一层的来源。"""
        return pick_text(
            rng, lang,
            f"异常指数：本次采样率过低，只上报区间 {a_lo}-{a_hi}，"
            f"区间内均匀分布",
            f"Anomaly index: sampling was too sparse this run, so only the range "
            f"{a_lo}-{a_hi} was reported; it is uniformly distributed within "
            f"that range")

    def _assessor_block(self, rng, lang, scores):
        """两位同级评委的风险评分。都是精确读数。"""
        return pick_text(
            rng, lang,
            f"风险评估：评委甲给出风险分 {scores[0]}；评委乙给出风险分 {scores[1]}",
            f"Risk assessment: assessor A scored {scores[0]}; "
            f"assessor B scored {scores[1]}")

    def _rule(self, rng, lang):
        """算式句。单独成段（seg=`grading`），理由同 G3：与门槛同段会让回读恒真。"""
        f = _formula(lang)
        if lang == "zh":
            return rng.choice((
                f"闸门规则：风险分 = {f}。",
                f"闸门规则：先算风险分（{f}），再对照判定门槛。",
            ))
        return rng.choice((
            f"Gate rule: risk = {f}.",
            f"Gate rule: first compute the risk score ({f}), then compare it "
            "against the decision thresholds.",
        ))

    def _policy_band(self, rng, lang, theta_low, theta_high):
        """门槛 + 三段判定 + 区间均匀的成文依据。

        段内不出现算式系数（它们在 `grading` 段），也不出现区间宽度 —— 宽度可由
        `risk` 段两端的差得到，再写一遍只会让段内的数字集合多一个可撞的成员，
        而 `RENDERED` 是按集合成员判定的。措辞轮换但不改数字。
        """
        if lang == "zh":
            head = rng.choice((
                "闸门规则：",
                "判定依据：",
                "闸门规则（按风险分三段判定）：",
            ))
            return (f"{head}风险分低于 {theta_low} 则允许，高于 {theta_high} 则拒绝，"
                    f"落在 {theta_low} 与 {theta_high} 之间（含两端）则弃权。"
                    f"异常指数在上报区间内均匀分布，风险分随之在该区间上均匀取值。")
        head = rng.choice((
            "Gate rule: ",
            "Decision basis: ",
            "Gate rule (three bands over the risk score): ",
        ))
        return (f"{head}a risk score below {theta_low} means allow, above "
                f"{theta_high} means deny, and between {theta_low} and "
                f"{theta_high} inclusive means abstain. The anomaly index is "
                f"uniformly distributed over the reported range, so the risk "
                f"score is uniform over that interval as well.")

    def _policy_split(self, rng, lang, theta_low, theta_high):
        """门槛 + 三段判定 + **两位评委同级**的成文依据。

        同级那句是 tie 语义的全部依据。它把"两位评委分歧时该不该给平的分布"
        从我的约定变成 state 里写着的事实 —— 少了它，R1 的口子就开了。
        """
        if lang == "zh":
            return rng.choice((
                f"闸门规则：风险分低于 {theta_low} 则允许，高于 {theta_high} 则拒绝，"
                f"居中（含两端）则弃权。两位评委同级，各自评分独立生效；"
                f"两人动作不一致时，两个动作同样有效。",
                f"判定依据：风险分三段 —— 低于 {theta_low} 允许，高于 "
                f"{theta_high} 拒绝，其间弃权。两位评委同级，分歧时二者并行有效。",
            ))
        return rng.choice((
            f"Gate rule: a risk score below {theta_low} means allow, above "
            f"{theta_high} means deny, and in between inclusive means abstain. "
            f"The two assessors have equal standing and each score takes effect "
            f"independently; when their actions disagree, both actions are "
            f"equally valid.",
            f"Decision basis: three bands over the risk score — below "
            f"{theta_low} allow, above {theta_high} deny, in between abstain. "
            f"The two assessors have equal standing, so their disagreement "
            f"leaves both actions equally valid.",
        ))

    # -- 复述 ---------------------------------------------------------------
    def paraphrases(self, rng, question, template_id, lang):
        order = list(range(len(GATE_QUESTIONS)))
        rng.shuffle(order)
        out = []
        for j in order:
            text = term(rng, (GATE_QUESTIONS[j],), lang)
            if text != question and text not in out:
                out.append(text)
            if len(out) == 3:
                break
        return out
