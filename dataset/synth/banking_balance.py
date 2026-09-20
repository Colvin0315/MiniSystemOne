"""
G1 `banking_balance` —— **旗舰校准来源**，provenance = `explicit_rng`。

核心构造（`docs/DESIGN.md` 的 (a) 号软目标）：银行有一个 **state 里看不见的 RNG**，
但它的**规则与全部参数都印在 state 里**。于是

    P(批准 | x) = σ(margin / τ)     精确成立

其中 `margin = 余额 − 金额 − 手续费 + 信用加分`，`τ = 审核容差`，四个量都以自然
语言渲染进 state。不确定在世界里（银行的随机数生成器），不在 state 里 —— 这正是
最干净的校准设定：目标由生成器自身逻辑、从它明确知道"已渲染进 state"的量算出，
`audit` 里留下 `margin` / `q_raw` / `tau` 供 oracle 上界与充分性测试核对。

同一份 state 支持两种 schema：
  - 偶数模板 → **Noul**：是否批准。候选恒为 `<yes>`/`<no>` 两个单 token 特殊 token。
  - 奇数模板 → **Choice**：路由到哪个部门。4 个部门按**剩余处理能力加权**随机分配，
    能力值印在 state 里，所以 `p_d = capacity_d / Σ capacity` 同样精确成立。

两种变体的目标都能从 state 精确重算 —— 这是 R1 缓解措施①（oracle 上界）成立的前提。
往 state 里**加**一个不影响目标的字段无害；**拿掉**一个影响目标的字段会被
`audit_synthetic.py` 的特征充分性测试抓住。

模板 id 的奇偶决定变体、`idx // 2` 决定问法，所以**问法完全由 template_id 决定**。
这一点不能松：split 按 template_id 划分，若问法是随机挑的，"留出模板"就不再意味着
"留出问法"，`test_known` 的泛化结论会被污染成一句空话。
"""
from dataset.synth.base import Generator, StateBuilder, sigmoid
from dataset.synth.lexicon import (
    DEPARTMENTS, STATUSES, eid, fmt_amount, fmt_count, term,
)

ORDINALS = (
    ("organization", "组织"), ("merchant", "商户"), ("tenant", "租户"),
    ("workspace", "工作区"), ("endpoint", "接口"),
)

APPROVAL_QUESTIONS = (
    ("银行是否会批准这笔交易？", "Will the bank approve this transaction?"),
    ("这笔申请会不会被通过？", "Will this request be approved?"),
    ("该笔转账能否获批？", "Can this transfer be authorized?"),
    ("审核结果是批准吗？", "Is the review outcome an approval?"),
    ("这笔款项会被放行吗？", "Will these funds be released?"),
    ("风控会同意这次扣款吗？", "Will risk control approve this charge?"),
    ("这笔交易最终能成交吗？", "Will this transaction ultimately go through?"),
    ("申请是否会被驳回？", "Will the application be rejected?"),
    ("该请求能否通过审核？", "Does this request pass review?"),
    ("这笔支出会被受理吗？", "Will this payment be accepted?"),
    ("审核员会给这笔单子放行吗？", "Will the reviewer clear this item?"),
    ("该笔划账是否获批？", "Is this debit authorized?"),
)

ROUTING_QUESTIONS = (
    ("这个工单会被路由到哪个部门？", "Which department will this case be routed to?"),
    ("哪个团队会接手这个案件？", "Which team will take this case?"),
    ("该案件由哪个部门处理？", "Which department handles this case?"),
    ("这单会分派给谁？", "Who will this ticket be assigned to?"),
    ("案件将进入哪个处理队列？", "Which queue will this case enter?"),
    ("哪个部门会受理这笔申诉？", "Which department will process this claim?"),
    ("该请求被转交给哪个团队？", "Which team is this request forwarded to?"),
    ("这笔争议归谁处理？", "Who owns this dispute?"),
    ("工单的目的地是哪个部门？", "What is the destination department for this ticket?"),
    ("该申请会被分配到哪个队列？", "Which queue will this application be assigned to?"),
    ("哪个小组会跟进这个案件？", "Which group will follow up on this case?"),
    ("案件最终落到哪个部门？", "Which department does this case end up with?"),
)


def _pair(rng, lang, table):
    """从 (zh, en) 表里按语言取一条；混排时随机取一侧。"""
    zh, en = table[rng.randrange(len(table))]
    return zh if lang == "zh" else en if lang == "en" else (en if rng.random() < 0.5 else zh)


def _txt(rng, lang, zh, en):
    return zh if lang == "zh" else en if lang == "en" else (en if rng.random() < 0.5 else zh)


class BankingBalance(Generator):
    name = "banking_balance"
    prefix = "bb"
    n_templates = 24

    # 目标赖以计算的量必须能在它所在的段里被读回来，否则 R1 的口子就开了。
    RENDERED = {
        "banking_approval": (("balance", "money", "risk"), ("credit", "money", "risk"),
                             ("amount", "money", "billing"), ("fees", "money", "billing"),
                             ("tau", "money", "policy")),
        "banking_routing": (("capacities", "count", "limits"),),
    }
    RULE_MARKERS = {
        "banking_approval": ("审核容差为", "review tolerance is"),
        "banking_routing": ("路由策略", "Routing policy"),
    }


    def _item(self, rng, pool, template_id, lang):
        idx = int(template_id[len(self.prefix) + 1:])
        qidx = idx // 2                      # 每个变体各分到 12 条互不相同的问法
        subj = f"{term(rng, ORDINALS, lang)} {eid(pool, 'org', rng.randrange(64))}"
        status = term(rng, STATUSES, lang)

        if idx % 2 == 0:
            item = self._approval(rng, lang, subj, status)
            item["question"] = _pair(rng, lang, (APPROVAL_QUESTIONS[qidx],))
        else:
            item = self._routing(rng, lang, subj, status)
            item["question"] = _pair(rng, lang, (ROUTING_QUESTIONS[qidx],))

        item["meta"].update(variant=item["primitive"], language=lang, question_idx=qidx)
        return item

    # 段落优先级：干扰项 0 < 账户 1 < 号码/策略 3。截断时先丢**不影响目标**的段，
    # 所以 account（只是身份信息）比 policy（承载 τ）先走。
    def _base_sections(self, rng, lang, subj, status):
        sb = StateBuilder(lang)
        sb.add("account", _txt(rng, lang,
                               f"账户：{subj}，状态为 {status}",
                               f"Account: {subj}, status {status}"), 1)
        return sb

    # -- 变体 A：Noul 批准 -------------------------------------------------
    def _approval(self, rng, lang, subj, status):
        # τ 是审核容差，其余四个量按 τ 缩放，使 margin/τ 落在 σ 的敏感区
        tau = rng.randrange(5000, 40000)
        balance = rng.randrange(2 * tau, 6 * tau)
        amount = rng.randrange(int(1.5 * tau), int(6.5 * tau))
        fees = rng.randrange(0, int(0.4 * tau) + 1)
        credit = rng.randrange(-tau, tau + 1)

        margin = balance - amount - fees + credit
        q = sigmoid(margin / tau)

        sb = self._base_sections(rng, lang, subj, status)
        sb.add("billing", _txt(rng, lang,
               f"申请金额 {fmt_amount(rng, amount, lang)}；手续费 {fmt_amount(rng, fees, lang)}",
               f"Requested amount {fmt_amount(rng, amount, lang)}; "
               f"fees {fmt_amount(rng, fees, lang)}"), 3)
        sb.add("risk", _txt(rng, lang,
               f"账户余额 {fmt_amount(rng, balance, lang)}；"
               f"信用加分 {fmt_amount(rng, credit, lang)}",
               f"Account balance {fmt_amount(rng, balance, lang)}; "
               f"credit adjustment {fmt_amount(rng, credit, lang)}"), 3)
        sb.add("policy", _txt(rng, lang,
               f"审核采用标准风险模型：净差额相对审核容差决定批准概率。"
               f"净差额 = 余额 − 金额 − 手续费 + 信用加分，审核容差为 {fmt_amount(rng, tau, lang)}。",
               f"Review uses the standard risk model: the net margin relative to the "
               f"review tolerance determines the approval probability. Net margin = "
               f"balance − amount − fees + credit adjustment; the review tolerance is "
               f"{fmt_amount(rng, tau, lang)}."), 3)

        return {
            "primitive": "noul",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": "banking_approval",
            "schema_desc": "是否批准一笔交易",
            "positive_label": "yes",
            "candidates": [{"text": "<yes>", "label": "yes", "meta": {"level": None}},
                           {"text": "<no>", "label": "no", "meta": {"level": None}}],
            # audit 里同时留**分量**和推导量：分量（balance/amount/fees/credit/tau）
            # 由 `RENDERED` 声明为必须在 state 里读得回来；margin/q_raw 是推导量，
            # 供 oracle 上界直接取用，不要求出现在文本里。
            "target": {"kind": "soft", "p": [q, 1.0 - q], "provenance": "explicit_rng",
                       "renormalized": False,
                       "audit": {"balance": balance, "amount": amount, "fees": fees,
                                 "credit": credit, "tau": tau,
                                 "margin": margin, "q_raw": q}},
            "meta": {"K_full": 2},
        }

    # -- 变体 B：Choice 路由 -----------------------------------------------
    def _routing(self, rng, lang, subj, status):
        depts = rng.sample(DEPARTMENTS, 4)
        caps = [rng.randrange(5, 200) for _ in depts]
        if rng.random() < 0.3:                      # 偶尔制造一个明显占优的队列
            caps[rng.randrange(4)] *= 3
        total = sum(caps)
        p = [c / total for c in caps]

        sb = self._base_sections(rng, lang, subj, status)
        listed = "；".join(f"{zh if lang == 'zh' else en} {fmt_count(rng, c, lang)}"
                          for (en, zh), c in zip(depts, caps))
        sb.add("limits", _txt(rng, lang,
               f"各部门剩余处理能力：{listed}",
               f"Remaining handling capacity by department: {listed}"), 3)
        sb.add("policy", _txt(rng, lang,
               "路由策略：案件按各部门剩余处理能力加权随机分配，能力越高被选中的概率越大。",
               "Routing policy: cases are assigned at random weighted by each "
               "department's remaining handling capacity; higher capacity means a "
               "higher chance of being selected."), 3)

        return {
            "primitive": "choice",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": "banking_routing",
            "schema_desc": "案件路由到哪个部门",
            "candidates": [{"text": (zh if lang == "zh" else en), "label": en,
                            "meta": {"level": None}} for en, zh in depts],
            # `capacities` 是**值**的列表，基类 `_one` 打乱候选后它仍然对得上 ——
            # RENDERED 的回读只问"这些值在 limits 段里出没出现过"，与顺序无关。
            # 但"哪个部门是多大能力"是**逐位对齐**的关系，而 audit **不会**随候选
            # 一起打乱（见 G6 的同一条注释），所以那份对应关系必须按键存，不能按下标。
            "target": {"kind": "soft", "p": p, "provenance": "explicit_rng",
                       "renormalized": False,
                       "audit": {"capacities": caps, "total_capacity": total,
                                 "capacity_by_dept": {en: c for (en, _), c
                                                      in zip(depts, caps)}}},
            "meta": {"K_full": 4},
        }

    # -- 复述 ---------------------------------------------------------------
    def paraphrases(self, rng, question, template_id, lang):
        """复述 = 同变体问法表里的其它问法。它们问的是同一件事，所以是真正等价的复述。

        **必须与 `question` 同语言** —— 拿英文复述去测中文问法的鲁棒性，测的是
        语言切换而不是"学的是问题还是模板"，那正是这个套件要隔离的东西。
        """
        idx = int(template_id[len(self.prefix) + 1:])
        table = APPROVAL_QUESTIONS if idx % 2 == 0 else ROUTING_QUESTIONS
        order = list(range(len(table)))
        rng.shuffle(order)
        out = []
        for j in order:
            zh, en = table[j]
            text = zh if lang == "zh" else en if lang == "en" else (en if rng.random() < 0.5 else zh)
            if text != question and text not in out:
                out.append(text)
            if len(out) == 3:
                break
        return out
