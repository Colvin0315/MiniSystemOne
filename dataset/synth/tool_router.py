"""
G2 `tool_router` —— **准确率主力**，provenance = `hard`（唯一答案）或 `tie_set`（并列集）。

任务：给定一段服务台工单 state，选出应当执行的那个工具。难度不来自词表大小，
而来自**难负例**：候选里同时放进「同动作换对象」（`cancel_order` vs `cancel_subscription`）
与「同对象换动作」（`create_user` vs `delete_user`）。只匹配动作词、或只匹配对象词的
模型，在这两族之间必然二选一错，而两族的正确项长得一样像 —— 这是"读候选文本"的
直接压力测试。

两种目标来源，对应 state 到底说不说得清：

  - `hard` —— notes 段点名了目标对象（`取消` + `订阅`），答案唯一，one-hot。
    **这类样本不进任何校准指标**（`hard` 在 DATA_SCHEMA 里被排除在 ECE/Brier 之外），
    它只贡献准确率与干扰上下文鲁棒性。
  - `tie_set` —— notes 段只给动作、并明确说对象未指明，而 state 另一段列出了用户名下
    该动作可作用的**全部**对象。于是有效答案就是这几个对象各自的工具，目标为均匀分布。
    并列集的**大小随 state 变化**（2～4 个对象），所以模型必须真的去读那段清单：
    两个并列时各 0.5，三个时各 1/3，四个时各 1/4。

最后一条是本生成器的教学要点：**不确定性可以说出口**。当 state 客观上不决定唯一答案
时，正确行为是给出平的分布，而不是假装自信地挑一个。这正是校准要教的东西，也是它
必须与 `hard` 样本分开报告指标的原因。

两类样本的清单段用**同一种形状**（"该用户可<动作>的对象：…"），只在条数与 notes 上
不同：`hard` 的清单里混入 1～3 个同动作的诱饵对象，`tie` 的清单就是并列集本身。若让
"清单出现 ⇒ tie"，模型会去背这个相关性而不是读内容。
"""
from dataset.synth.base import (
    Generator, StateBuilder, one_hot, pick_text as _pick, uniform,
)
from dataset.synth.lexicon import STATUSES, TOOLS, eid, term

TOOLSET = frozenset(TOOLS)

# 动作 → 可作用的对象。**手写而非从 TOOLS 猜**：`transfer_partial` 这种切出来的
# "对象"不是对象，"refund_partial" 也一样，自动解析会把它们混进并列集，使 tie 的
# 有效答案集算错。手写表由下面的断言钉住，不会与 lexicon 漂移。
VERB_NOUNS = {
    "transfer": ("ownership", "funds", "ticket", "domain"),
    "cancel": ("order", "subscription", "booking", "invoice"),
    "update": ("email", "address", "payment_method", "profile"),
    "create": ("user", "team", "webhook", "api_key"),
    "delete": ("user", "team", "webhook"),
    "list": ("orders", "invoices", "members", "webhooks"),
    "get": ("balance", "invoice", "usage_report", "audit_log"),
}

# 反向索引则**从 TOOLS 全表解析**，这样难负例能覆盖到手写表之外的近亲
# （`escalate_ticket` / `merge_ticket` / `reassign_ticket` / `close_ticket` 都作用在
# ticket 上，是最好的"同对象换动作"负例）。
NOUN_VERBS = {}
for _t in TOOLS:
    _v, _n = _t.split("_", 1)
    NOUN_VERBS.setdefault(_n, []).append(_v)

for _v, _ns in VERB_NOUNS.items():
    for _n in _ns:
        assert f"{_v}_{_n}" in TOOLSET, f"手写动作/对象表与 lexicon 漂移：{_v}_{_n}"

ACTIONS = {
    "transfer": ("转移", "transfer"), "cancel": ("取消", "cancel"),
    "update": ("更新", "update"), "create": ("创建", "create"),
    "delete": ("删除", "delete"), "list": ("列出", "list"),
    "get": ("查询", "get"),
}

OBJECTS = {
    "ownership": ("所有权", "ownership"), "funds": ("资金", "funds"),
    "ticket": ("工单", "ticket"), "domain": ("域名", "domain"),
    "order": ("订单", "order"), "subscription": ("订阅", "subscription"),
    "booking": ("预约", "booking"), "invoice": ("发票", "invoice"),
    "email": ("邮箱", "email"), "address": ("地址", "address"),
    "payment_method": ("支付方式", "payment method"), "profile": ("资料", "profile"),
    "user": ("用户", "user"), "team": ("团队", "team"), "webhook": ("回调", "webhook"),
    "api_key": ("密钥", "API key"), "orders": ("订单列表", "orders"),
    "invoices": ("发票列表", "invoices"), "members": ("成员", "members"),
    "webhooks": ("回调列表", "webhooks"), "balance": ("余额", "balance"),
    "usage_report": ("用量报告", "usage report"), "audit_log": ("审计日志", "audit log"),
}

ORDINALS = (
    ("account", "账户"), ("tenant", "租户"), ("workspace", "工作区"),
    ("organization", "组织"), ("endpoint", "接口"),
)

ROUTING_QUESTIONS = (
    ("应该执行哪个操作？", "Which operation should be performed?"),
    ("该执行哪个工具？", "Which tool should be invoked?"),
    ("这个工单应该走哪个操作？", "Which operation does this ticket require?"),
    ("应当调用哪个函数？", "Which function should be called?"),
    ("系统应该执行什么操作？", "What operation should the system perform?"),
    ("该请求对应哪个工具调用？", "Which tool call does this request map to?"),
    ("请选择要执行的操作。", "Choose the operation to perform."),
    ("哪个操作是正确的？", "Which operation is the correct one?"),
    ("这单应该派发到哪个工具？", "Which tool should this case be dispatched to?"),
    ("需要调用哪个操作？", "Which operation is needed here?"),
    ("应当运行哪个工具？", "Which tool should be run?"),
    ("哪项操作适用？", "Which operation applies?"),
    ("执行哪个动作？", "Which action should be taken?"),
    ("这个请求该用哪个工具？", "Which tool fits this request?"),
    ("选出正确的操作。", "Select the correct operation."),
    ("对应哪个工具？", "Which tool corresponds to this?"),
    ("应该触发哪个操作？", "Which operation should be triggered?"),
    ("该工单匹配哪个工具？", "Which tool matches this ticket?"),
    ("要执行的操作是哪个？", "What is the operation to be performed?"),
    ("哪个工具能处理它？", "Which tool can handle it?"),
    ("请指出应调用的操作。", "Indicate the operation to be invoked."),
    ("此请求应由哪个工具处理？", "Which tool should handle this request?"),
    ("操作选择：哪一个？", "Operation selection: which one?"),
    ("该用哪个操作？", "Which operation should be used?"),
)


class ToolRouter(Generator):
    name = "tool_router"
    prefix = "tr"
    n_templates = 24

    # 这条链上的量不是数字而是**对象词**，所以不走 RENDERED 的数字回读，改在
    # `sufficiency()` 里查：清单段必须逐字出现 audit 记下的每一个对象。
    RULE_MARKERS = {
        "tool_routing": ("工具选择规则", "Tool selection", "选用工具时",
                         "A tool qualifies", "路由依据", "Routing basis"),
    }

    def sufficiency(self, rec):
        out = super().sufficiency(rec)
        if rec["target"].get("provenance") == "hard":
            return out
        frag = "".join(s["text"] for s in rec["state_sections"] if s["seg"] == "order")
        for n in rec["target"]["audit"]["listed_nouns"]:
            zh, en = OBJECTS[n]
            if zh not in frag and en not in frag:
                out.append(f"并列集对象 {n} 未出现在清单段里")
        return out

    def _item(self, rng, pool, template_id, lang):
        idx = int(template_id[len(self.prefix) + 1:])
        verb = rng.choice(list(VERB_NOUNS))
        nouns = VERB_NOUNS[verb]
        act_zh, act_en = ACTIONS[verb]

        if rng.random() < 0.45:
            # tie：动作已知、对象未知，并列集 = 该动作下用户拥有的全部对象
            mode = "tie"
            valid = rng.sample(nouns, rng.randrange(2, len(nouns) + 1))
            listed = list(valid)
        else:
            mode = "hard"
            valid = [rng.choice(nouns)]
            decoys = rng.sample([n for n in nouns if n != valid[0]],
                                rng.randrange(1, len(nouns)))
            listed = valid + decoys
        rng.shuffle(listed)

        # 清单项的 id 必须**逐例变化**：若 id 由 (pool, 对象, 序号) 决定，同一个对象
        # 在每个样本里都挂着同一个 id，BoW 探针就能靠 id 反查对象，答案泄漏。
        objs = [(n, eid(pool, "obj", rng.randrange(65536), salt=n)) for n in listed]

        sb = StateBuilder(lang)
        subj = f"{term(rng, ORDINALS, lang)} {eid(pool, 'org', rng.randrange(64))}"
        sb.add("account", _pick(rng, lang,
                               f"账户：{subj}，状态为 {term(rng, STATUSES, lang)}",
                               f"Account: {subj}, status {term(rng, STATUSES, lang)}"), 1)
        sb.add("notes", self._notes(rng, lang, mode, act_zh, act_en, valid, len(listed)), 3)
        sb.add("order", self._object_list(rng, lang, act_zh, act_en, objs), 3)
        sb.add("policy", self._policy(rng, lang), 3)

        tools, p = self._candidates(rng, verb, nouns, valid)
        return {
            "question": _pick(rng, lang, *ROUTING_QUESTIONS[idx]),
            "primitive": "choice",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": "tool_routing",
            "schema_desc": "选择应当执行的工具",
            "candidates": [{"text": t, "label": t, "meta": {"level": None}} for t in tools],
            "target": {"kind": "soft", "p": p,
                       "provenance": "tie_set" if mode == "tie" else "hard",
                       "renormalized": False,
                       "audit": {"verb": verb, "valid_nouns": list(valid),
                                 "listed_nouns": list(listed), "n_valid": len(valid)}},
            "meta": {"variant": mode, "language": lang, "question_idx": idx},
        }

    # -- state 各段 ---------------------------------------------------------
    def _notes(self, rng, lang, mode, act_zh, act_en, valid, n_listed):
        if mode == "hard":
            o_zh, o_en = OBJECTS[valid[0]]
            forms = (
                (f"请求动作：{act_zh}；目标对象：{o_zh}",
                 f"Requested action: {act_en}; target object: {o_en}"),
                (f"用户要求{act_zh}指定的{o_zh}。",
                 f"The user asked to {act_en} the specified {o_en}."),
                (f"受理请求 —— {act_zh} {o_zh}。",
                 f"Inbound request — {act_en} {o_en}."),
            )
        else:
            forms = (
                (f"请求动作：{act_zh}；目标对象：未指明（名下符合该动作的对象共 {n_listed} 个）。",
                 f"Requested action: {act_en}; target object: unspecified "
                 f"({n_listed} objects match this action)."),
                (f"用户要求{act_zh}，但未说明具体是哪一项，名下符合的对象见清单。",
                 f"The user asked to {act_en} but did not say which one; "
                 f"the matching objects are listed."),
                (f"受理请求 —— {act_zh}，对象待确认（清单中共 {n_listed} 项）。",
                 f"Inbound request — {act_en}, object to be confirmed "
                 f"({n_listed} entries in the list)."),
            )
        zh, en = forms[rng.randrange(len(forms))]
        return _pick(rng, lang, zh, en)

    def _object_list(self, rng, lang, act_zh, act_en, objs):
        zh_items = "；".join(f"{OBJECTS[n][0]} {i}" for n, i in objs)
        en_items = "; ".join(f"{OBJECTS[n][1]} {i}" for n, i in objs)
        return _pick(rng, lang,
                     f"该用户可{act_zh}的对象：{zh_items}",
                     f"Objects this user can {act_en}: {en_items}")

    def _policy(self, rng, lang):
        """规则照实写：动作与对象**都**要对上，且未指明对象时并列有效。

        这句不能省 —— 它是 tie 语义的**成文依据**。没有它，"对象未指明" 时该不该
        均匀分就只是我的约定，模型无从推断，R1 的口子就开了。

        三种措辞轮换：它同时是两种模式共用的**最长恒定块**，逐字不变会让每个样本
        都背上一段几十 token 的常数，既浪费上下文又给模板匹配器一个稳定的锚。
        """
        forms = (
            ("工具选择规则：动作与对象须同时与工具名相符，任一不符者不得选用；"
             "若请求未指明对象，则所有与动作相符的工具同等有效。",
             "Tool selection: the action and object must both match the tool name, and "
             "a tool differing in either must not be chosen; if the object is "
             "unspecified, all tools matching the action are equally valid."),
            ("选用工具时，动作与目标对象都要对上；未指明对象的情况下，凡动作相符的"
             "工具一律同样有效。",
             "A tool qualifies only if both the action and the target object match; "
             "when no object is given, every tool with the matching action qualifies "
             "equally."),
            ("路由依据：动作 + 对象。对象缺失时按动作取并列集，不作猜测。",
             "Routing basis: action + object. When the object is missing, take the "
             "tied set over the action rather than guessing."),
        )
        zh, en = forms[rng.randrange(len(forms))]
        return _pick(rng, lang, zh, en)

    # -- 候选与目标 ---------------------------------------------------------
    def _candidates(self, rng, verb, nouns, valid):
        """正确项 + 两族难负例 + 无关填充。返回 (工具名列表, 目标分布)。

        `tie` 时正确项是多个，目标均匀；`hard` 时唯一，one-hot。候选**必定多于正确项**
        ——否则"对所有候选均匀"这种不读 state 的策略就能拿满分，任务也就不再是
        「读 state 再决定」，而是「数一数候选有几个」。
        """
        valid_tools = [f"{verb}_{n}" for n in valid]
        negs = [f"{verb}_{n}" for n in nouns if n not in valid]          # 同动作换对象
        for n in valid:                                                   # 同对象换动作
            for v in NOUN_VERBS.get(n, ()):
                if v != verb and f"{v}_{n}" in TOOLSET:
                    negs.append(f"{v}_{n}")
        negs = list(dict.fromkeys(t for t in negs if t not in valid_tools))
        rng.shuffle(negs)

        k = max(rng.choice((4, 6, 8)), len(valid_tools) + 1)
        tools = valid_tools + negs[:k - len(valid_tools)]
        while len(tools) < k:
            cand = rng.choice(TOOLS)
            if cand not in tools:
                tools.append(cand)

        # 均匀**只在有效集上**，无效候选恒为 0 —— `uniform(len(tools))` 会把质量
        # 摊给难负例，那等于告诉模型"这些也有可能是答案"，与 tie 的语义正相反。
        if len(valid_tools) > 1:
            q = 1.0 / len(valid_tools)
            p = [q if t in valid_tools else 0.0 for t in tools]
        else:
            p = one_hot(len(tools), 0)
        return tools, p

    # -- 复述 ---------------------------------------------------------------
    def paraphrases(self, rng, question, template_id, lang):
        order = list(range(len(ROUTING_QUESTIONS)))
        rng.shuffle(order)
        out = []
        for j in order:
            zh, en = ROUTING_QUESTIONS[j]
            text = _pick(rng, lang, zh, en)
            if text != question and text not in out:
                out.append(text)
            if len(out) == 3:
                break
        return out
