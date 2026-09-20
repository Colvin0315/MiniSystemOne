"""
决策域词表 —— 生成器与 tokenizer 训练语料的共同事实来源。

这里只放**词表和格式化工具**，不放模板和决策逻辑（那些在 synth/*.py）。
分开是必需的：数据集 split 按「实体池 × 模板」划分，如果各生成器自己造
一套名字，各池就不再不相交，"留出实体池"就不再是真正的留出。

词表覆盖决定决策质量的四类词：**工具名、部门/状态标签、等级标签、金额与日期**。
tokenizer 若没见过 `transfer_ownership` / `neutral` / `¥1,240.00`，它们会碎成
多个 token，候选变长、pooled 向量变噪，而候选长度直接进入被测量的指标。

语言配比目标 CN / EN / 混排 ≈ 35 / 35 / 30（由 LANG_WEIGHTS 控制）。
"""
import hashlib
import random

# ---------------------------------------------------------------------------
# 语言与实体池
# ---------------------------------------------------------------------------

LANG_WEIGHTS = (("zh", 35), ("en", 35), ("mix", 30))

# 5 个不相交实体池。split 按 pool 划分，所以池之间必须完全无重叠。
POOLS = ("p1", "p2", "p3", "p4", "p5")

# ---------------------------------------------------------------------------
# 工具名（tool_router 的候选与难负例来源）
# 刻意成对出现：难负例的"难"来自前缀/语义高度相似但不等价。
# ---------------------------------------------------------------------------
TOOLS = (
    "transfer_ownership", "transfer_funds", "transfer_ticket", "transfer_domain",
    "cancel_order", "cancel_subscription", "cancel_booking", "cancel_invoice",
    "refund_payment", "refund_shipping", "refund_partial", "reverse_charge",
    "update_email", "update_address", "update_payment_method", "update_profile",
    "create_user", "create_team", "create_webhook", "create_api_key",
    "delete_user", "delete_team", "delete_webhook", "revoke_api_key",
    "list_orders", "list_invoices", "list_members", "list_webhooks",
    "get_balance", "get_invoice", "get_usage_report", "get_audit_log",
    "enable_2fa", "disable_2fa", "reset_password", "rotate_credentials",
    "export_data", "import_data", "archive_project", "restore_project",
    "pause_sync", "resume_sync", "retry_sync", "reset_sync_cursor",
    "apply_coupon", "remove_coupon", "extend_trial", "convert_trial",
    "escalate_ticket", "merge_ticket", "reassign_ticket", "close_ticket",
    "grant_role", "revoke_role", "bind_phone", "unbind_phone",
    "schedule_report", "unschedule_report", "test_webhook", "verify_domain",
)

# ---------------------------------------------------------------------------
# 标签：部门 / 状态 / 序数等级 / 闸门动作
# ---------------------------------------------------------------------------

DEPARTMENTS = (
    ("engineering", "工程"), ("billing", "账单"), ("finance", "财务"),
    ("support", "客服"), ("logistics", "物流"), ("compliance", "合规"),
    ("marketing", "市场"), ("security", "安全"), ("procurement", "采购"),
    ("infrastructure", "基础设施"),
)

# 状态词：决策 state 里反复出现，且两类（好/坏）必须都进词表
STATUSES = (
    ("active", "生效中"), ("suspended", "已暂停"), ("pending", "待处理"),
    ("expired", "已过期"), ("voided", "已作废"), ("archived", "已归档"),
    ("frozen", "已冻结"), ("verified", "已验证"), ("disputed", "有争议"),
)

# Score primitive 的序数等级。序号同时渲染进候选文本，让编码器能看见序关系。
LEVELS = (
    ("very poor", "很差"), ("poor", "较差"), ("neutral", "一般"),
    ("good", "较好"), ("excellent", "很好"),
)

# security_gate 的弃权候选 —— 全项目唯一的弃权监督来源
GATES = (("allow", "允许"), ("deny", "拒绝"), ("abstain", "弃权"))

# state 段落名。段落顺序随机化是反模板化措施之一。
#
# `distractor` 是**唯一不作为渲染标签使用**的段名：干扰项在文本里伪装成某个真实
# 段落（前缀取真实段名），但 seg 字段恒为它，好让工具链能把干扰项与真段落分开。
# 模型看不到 seg —— seg 只用于 state_sections，不进打包序列的 seg_id。
DISTRACTOR_SEG = "distractor"
SECTIONS = (
    ("account", "账户"), ("order", "订单"), ("policy", "条款"),
    ("history", "历史"), ("limits", "限额"), ("notes", "备注"),
    ("verification", "核验"), ("billing", "计费"), ("risk", "风险"),
    (DISTRACTOR_SEG, "干扰"),
)

# 可以用作渲染标签的段名（排除 distractor 自己）
LABELS = tuple(s for s in SECTIONS if s[0] != DISTRACTOR_SEG)

# 主语 / 名词，供 state 渲染出自然语言
SUBJECTS = (
    ("workspace", "工作区"), ("organization", "组织"), ("merchant", "商户"),
    ("tenant", "租户"), ("repository", "代码库"), ("campaign", "投放计划"),
    ("shipment", "运单"), ("subscription", "订阅"), ("invoice", "发票"),
    ("endpoint", "接口"),
)

# 表面干扰项：无关账户 / 作废条款 / 过期历史。它们的唯一作用是让模板匹配器失效。
DISTRACTORS = (
    ("an unrelated account in the same billing group", "同一账单组下的一个无关账户"),
    ("a clause that was voided in amendment 4", "第 4 号修正案中已作废的条款"),
    ("a historical entry from two fiscal years ago", "两个财年之前的一条历史记录"),
    ("a duplicate record created by the nightly sync", "夜间同步产生的一条重复记录"),
    ("a sandbox tenant that is never billed", "一个从不计费的沙箱租户"),
    ("a pending change that was never approved", "一项从未获批的待定变更"),
    ("a rate limit that applies only to the legacy API", "仅适用于旧版接口的限流"),
    ("an internal note unrelated to the request", "一条与请求无关的内部备注"),
)


def pick_lang(rng: random.Random) -> str:
    r = rng.uniform(0, sum(w for _, w in LANG_WEIGHTS))
    acc = 0.0
    for lang, w in LANG_WEIGHTS:
        acc += w
        if r <= acc:
            return lang
    return "en"


def term(rng: random.Random, table, lang: str) -> str:
    """从 (en, zh) 表里按语言取词；混排时随机取一侧。"""
    en, zh = rng.choice(table)
    if lang == "zh":
        return zh
    if lang == "en":
        return en
    return en if rng.random() < 0.5 else zh


# ---------------------------------------------------------------------------
# 实体 id：sha1(seed||counter)[:8]
# 确定性生成，因此"留出实体池"可以被断言，而不是靠人工避免重名。
# ---------------------------------------------------------------------------

def eid(pool: str, kind: str, counter: int, salt: str = "") -> str:
    raw = f"{pool}|{kind}|{counter}|{salt}".encode()
    return hashlib.sha1(raw).hexdigest()[:8]


def entities(pool: str, kind: str, n: int, salt: str = ""):
    return [eid(pool, kind, i, salt) for i in range(n)]


# ---------------------------------------------------------------------------
# 数值 / 日期格式化：同义不同形，迫使模型读数值而不是记字符串
# ---------------------------------------------------------------------------

def fmt_amount(rng: random.Random, cents: int, lang: str) -> str:
    """金额的多种写法。cents 是整数分，避免浮点漂移。"""
    whole, frac = divmod(abs(cents), 100)
    sign = "-" if cents < 0 else ""
    grouped = f"{whole:,}"
    style = rng.randrange(5)
    if lang == "zh" or (lang == "mix" and rng.random() < 0.5):
        return rng.choice([
            f"{sign}¥{grouped}.{frac:02d}",
            f"{sign}{grouped}.{frac:02d} 元",
            f"{sign}人民币 {grouped}.{frac:02d}",
        ])
    if style == 0:
        return f"{sign}${grouped}.{frac:02d}"
    if style == 1:
        return f"{sign}USD {grouped}.{frac:02d}"
    if style == 2:
        return f"{sign}{grouped}.{frac:02d} USD"
    if style == 3:
        # 欧式：**空格**千分位 + 逗号小数点。不能直接用 `grouped`（逗号千分位），
        # 否则 1202.53 会渲染成 `€1,202,53` —— 一串里两个逗号，人和模型都无法
        # 可靠地判读。R1 要求金额能从 state 文本恢复，这种渲染会让它失败。
        return f"{sign}€{grouped.replace(',', ' ')},{frac:02d}"
    return f"{sign}£{grouped}.{frac:02d}"


def fmt_date(rng: random.Random, day: int, lang: str) -> str:
    """把一个日期序号渲染成多种格式。day 以 2026-01-01 为 0。"""
    y = 2026 + day // 365
    rem = day % 365
    m = rem // 31 + 1
    d = rem % 31 + 1
    if lang == "zh":
        return rng.choice([f"{y}年{m}月{d}日", f"{y}-{m:02d}-{d:02d}", f"{m}月{d}日"])
    return rng.choice([f"{y}-{m:02d}-{d:02d}", f"{m:02d}/{d:02d}/{y}", f"{d} {MONTHS[m - 1]} {y}"])


MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def fmt_pct(rng: random.Random, bps: int) -> str:
    """bps 是万分之一。渲染成 12.5% / 12.5 percent / 百分之 12.5。"""
    v = bps / 100
    style = rng.randrange(3)
    if style == 0:
        return f"{v:.1f}%"
    if style == 1:
        return f"{v:.1f} percent"
    return f"百分之 {v:.1f}"


def fmt_count(rng: random.Random, n: int, lang: str) -> str:
    if lang == "zh":
        return rng.choice([f"{n:,} 次", f"{n:,}", f"共 {n:,} 条"])
    return rng.choice([f"{n:,}", f"{n:,} times", f"{n:,} records"])
