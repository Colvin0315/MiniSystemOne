"""
合成生成器的公共骨架：split 划分、state 渲染、候选重排、schema 组装。

生成器只负责**语义**（这题的正确答案是什么、为什么），这里负责**表面**
（用什么语言、哪个模板、段落什么顺序、候选怎么排）。分开是因为反模板化措施
必须由一处统一执行 —— 各生成器自己造一套，就会有的加了 ≥20 模板有的忘了，
而 BoW 门禁只会告诉你"整体过度模板化"，不会告诉你漏在哪一个。

三条由这里强制的硬约束：
  1. **split 按 (template_id, entity_pool) 组合划分**，绝不逐条随机。
  2. **候选逐例重排**（连同目标与等级一起置换），位置不携带信息。
  3. `state` 必须**恰好等于** `state_sections` 按呈现顺序用 "\\n" 拼接 —— 因为
     `decision_dataset.fit_state` 会按 priority 丢段后重新拼接，两者一旦不一致，
     裁剪后的 state 就与训练时看到的分布不同，而这种漂移不会报错。
"""
import math
import random
import re

from dataset.synth.lexicon import (
    DISTRACTORS, DISTRACTOR_SEG, LABELS, POOLS, SUBJECTS, eid, pick_lang, term,
)

SPLIT_NAMES = ("train", "val", "calib", "test_known", "test_ood")

# 所有生成器共用的版本号。**改动即测试集失效** —— 结果 JSON 里会记录它，
# 测试集只允许用冻结版本重生成。
GEN_VERSION = "1.0.0"


# ---------------------------------------------------------------------------
def build_split_map(template_ids, pools=POOLS, val_frac=0.10, calib_frac=0.10,
                    holdout_template_frac=0.15, holdout_pools=1, seed=0):
    """把 (template_id, entity_pool) 网格划分给各 split。返回 {(tid, pool): split}。

    划分的语义（这是全部泛化评测的基础，所以逐条写清楚）：

      - `train` / `val` / `calib` 都落在**见过的模板 × 见过的实体池**里，三者互不相交。
        val 与 calib 只是没被训练的**同分布组合** —— 温度必须在同分布上拟合，
        在 OOD 上拟合出来的温度没有指称对象。
      - `test_known` 是**任何碰到留出模板或留出实体池**的组合。它测的是
        "同一个生成器、换了问法/换了实体" 的泛化。
      - `test_ood` 不在这里划分 —— 它由 `build_dataset.py` 整族留出生成器得到。

    留出池整池留出而非按比例抽样，是因为词表本身要按池不重叠：`lexicon.eid()` 用
    池名做 salt，所以池不重叠是可断言的，而不是靠人工避免重名。
    """
    rng = random.Random(seed)
    tids = sorted(template_ids)
    n_ho = max(1, int(round(len(tids) * holdout_template_frac)))
    ho_templates = set(tids[len(tids) - n_ho:])
    seen_templates = [t for t in tids if t not in ho_templates]

    plist = sorted(pools)
    ho_pools = set(plist[len(plist) - holdout_pools:])
    seen_pools = [p for p in plist if p not in ho_pools]

    mapping = {}
    for t in tids:
        for p in plist:
            if t in ho_templates or p in ho_pools:
                mapping[(t, p)] = "test_known"

    in_dist = [(t, p) for t in seen_templates for p in seen_pools]
    rng.shuffle(in_dist)
    n_val = max(1, int(round(len(in_dist) * val_frac)))
    n_calib = max(1, int(round(len(in_dist) * calib_frac)))
    for combo in in_dist[:n_val]:
        mapping[combo] = "val"
    for combo in in_dist[n_val:n_val + n_calib]:
        mapping[combo] = "calib"
    for combo in in_dist[n_val + n_calib:]:
        mapping[combo] = "train"

    return mapping


def assert_split_disjoint(maps):
    """断言多个生成器的 split 划分不共享 train 组合。

    生成器之间共享 entity pool，但**模板 id 必须各自带前缀**，否则两个生成器会把
    同一个 (template_id, pool) 一个划进 train 一个划进 test_known。这个断言就是
    防止那种串味 —— 它一旦发生，test_known 就不再是留出集，而所有指标都会变好看。
    """
    seen = {}
    for name, m in maps.items():
        for combo, split in m.items():
            if combo in seen:
                raise AssertionError(
                    f"生成器 {name} 与 {seen[combo]} 共用了组合 {combo}；"
                    f"template_id 必须带生成器前缀"
                )
            seen[combo] = name


# ---------------------------------------------------------------------------
class StateBuilder:
    """把结构化 state 攒起来，再按随机顺序渲染。

    段落顺序随机化是反模板化措施之一：模板匹配器靠的是固定顺序的固定字段。
    """

    def __init__(self, lang):
        self.lang = lang
        self.sections = []

    def add(self, seg, text, priority):
        self.sections.append({"seg": seg, "text": text, "priority": priority})
        return self

    def render(self, rng, distractors=0, distractor_texts=None):
        """返回 (state 文本, sections)。两者严格一致，由 `Generator._check` 再断言一次。

        干扰项**带上真实段落前缀**（`历史：两个财年之前的一条历史记录`）而不是光秃秃
        一句短语。这是必须的：真实段落都是「标签：内容」的形状，干扰项若没有标签，
        模型可以靠"有没有冒号"把它们筛掉 —— 那样一来，任何"加干扰项会不会掉分"的
        测试度量的就是格式识别，而不是"是否真的在读内容"。
        seg 恒为 `DISTRACTOR_SEG`，让工具链仍能把它们认出来。
        """
        secs = [dict(s) for s in self.sections]
        pool = distractor_texts if distractor_texts is not None else DISTRACTORS
        for _ in range(distractors):
            # priority=0：截断时第一个丢
            label = term(rng, LABELS, self.lang)
            secs.append({"seg": DISTRACTOR_SEG,
                         "text": f"{label}: {term(rng, pool, self.lang)}",
                         "priority": 0})
        rng.shuffle(secs)
        return "\n".join(s["text"] for s in secs), secs


# ---------------------------------------------------------------------------
# 渲染回读 —— 给 R1 的"渲染充分性"检查用
# ---------------------------------------------------------------------------
# 生成器把金额渲染成多种样式（¥ / $ / USD / € / £；逗号或空格千分位；点或逗号小
# 数点；负号在货币符**之前**）。要证明某个量真的印进了 state，就得能把它读回来。
# 这个检查存在的理由就是 R1：`P*` 由生成器**内部**的量算出，若其中某个量没渲染
# 进 state，模型学不到，而报告出来的 ECE 量的是生成器 bug 而不是模型校准。
#
# 数字 token 的形状只有四种：`1234` / `1234.56` / `1,234,567.89`（英美千分位）/
# `1 234 567,89`（欧式）。所以分隔符只允许以**成组**的方式出现：
#   `[ ,]\d{3}`  —— 千分位组，逗号或空格后恰好三位
#   `[.,]\d{1,2}` —— 小数部分，最多两位
#
# **不能写成 `\d[\d\s,.]*\d`。** 那个写法（本文件早期的版本）会把
# `13, 16, 20, 24` 整串吞成一个"数字"，随后 replace 成 `13.16.20.24` 解析失败、
# 被 `continue` 静默丢弃 —— 于是一份**照实渲染了**的阈值表被报告成"没渲染出来"。
# 它同时把 `1,234` 当成欧式小数读成 1。两个方向都会让 R1 的充分性检查失真，
# 而失真的方向是**误报**，会让人去改一个本来正确的生成器。
# `(?<![\d])` 是必需的：没有它，区间 `3-5` 会被读成 `3` 和 **`-5`** ——
# finditer 匹配完 `3` 后从 `-` 重新开始，而 `(-?)` 乐意把那个连字符当成负号，
# 于是 5 被消费掉、改头换面成 -5 进了集合，"5 没渲染出来"的误报随之出现。
# 紧跟在数字后面的连字符是区间分隔符，不是负号。
#
# 小数分支的 `(?!\d)` 是关键：没有它，`1,240.00` 会被读成 `1,24` + `0.00`
# （`,24` 先被当成了小数部分，因为交替里小数分支排在千分位分支前面），
# 于是千分位金额一律错。加上前瞻后 `,240` 走不通小数分支，落到千分位分支。
_NUM_RE = re.compile(
    r"(?<![\d])(-?)\s*(?:¥|€|£|\$|USD|RMB|人民币)?\s*"
    r"(\d+(?:[.,]\d{1,2}(?!\d)|[ ,]\d{3})*)"
)


def _canonical_number(num):
    """把一个数字 token 规整成 `float()` 读得懂的形式。"""
    s = re.sub(r"\s", "", num)
    if "." in s and "," in s:                     # 谁在后面谁是小数点
        return (s.replace(".", "").replace(",", ".") if s.rfind(",") > s.rfind(".")
                else s.replace(",", ""))
    if "," in s:
        # 逗号后恰好成 3 位一组 → 千分位（`1,234`）；否则（1–2 位）→ 欧式小数点（`1,5`）。
        return (s.replace(",", "") if re.fullmatch(r"\d{1,3}(?:,\d{3})+", s)
                else s.replace(",", "."))
    return s


def scan_numbers(text):
    """扫出文本里的数字，返回 (cents 集合, 原值集合)。

    两种读法都要，因为两类量用两套渲染器：金额以**分**为单位渲染成两位小数
    （`fmt_amount`），计数类量按整数渲染（`fmt_count`，带千分位）。
    """
    cents, plain = set(), set()
    for m in _NUM_RE.finditer(text):
        # 负号在货币符**之前**（`-¥56.66` / `-100.00 元`），所以它落在 group(1)，
        # 不在数字串里。忘了带上它，负的信用加分就永远读不回来。
        sign = -1 if m.group(1) else 1
        try:
            f = float(_canonical_number(m.group(2)))
        except ValueError:
            continue
        cents.add(sign * round(f * 100))
        plain.add(sign * int(f))
    return cents, plain


# ---------------------------------------------------------------------------
class Generator:
    """生成器基类。子类实现 `_item(...)` 产出语义项，基类负责表面与组装。"""

    name = ""            # 必须是唯一的短名，会进 id 与 source
    prefix = ""          # template_id 的前缀，必须唯一（见 assert_split_disjoint）
    n_templates = 24     # 每问的复述模板数；方案要求 ≥20

    # 渲染充分性声明，按 `schema.name` 分变体（同一生成器的不同变体渲染不同的量）。
    #   RENDERED: {schema_name: ((audit 字段, "money"|"count", 所在段 seg), ...)}
    #   RULE_MARKERS: {schema_name: (子串, ...)}，至少命中一个
    # 两者都是**声明**，检查在 `sufficiency()` 里做，在 `audit_synthetic.py` 里批量跑。
    RENDERED = {}
    RULE_MARKERS = {}

    def __init__(self, seed=0):
        self.seed = seed
        self.template_ids = [f"{self.prefix}t{i:02d}" for i in range(self.n_templates)]
        self.split_map = build_split_map(self.template_ids, seed=seed)

    # -- 子类实现 ----------------------------------------------------------
    def _item(self, rng, pool, template_id, lang):
        """产出一个语义项 dict。必须含：

            sections   : [(seg, text, priority), ...]（优先级语义见 DATA_SCHEMA）
            question   : str
            candidates : [{"text":..., "label":..., "meta": {"level": None}}]
            target     : {"kind","p","provenance","audit"}  # p 与 candidates 同序
            primitive  : "noul" | "choice" | "score"
            distractors: int（可选，注入几个无关 state 段）

        语言由 `lang` 决定（"zh" / "en" / "mix"），用 `lexicon.term()` 取词。
        `template_id` 决定问法，同一语义换模板必须给出真正等价的复述。
        """
        raise NotImplementedError

    def paraphrases(self, rng, question, template_id, lang):
        """同语言的其它问法。子类必须覆盖 —— 默认原样重复，只够跑通。"""
        return [question] * 3

    # -- 组装 --------------------------------------------------------------
    def generate(self, split, n):
        """产出 `n` 条 record。split 决定用哪些 (模板, 池) 组合。

        组合轮转而非随机抽：保证每个 (template, pool) 组合都被用到，否则留出
        `test_known` 的泛化结论会被"训练时压根没见过某个组合"污染。
        """
        combos = [c for c, s in self.split_map.items() if s == split]
        if not combos:
            raise ValueError(f"{self.name} 没有划分到 {split} 的组合")
        rng = random.Random(f"{self.seed}|{self.name}|{split}")
        return [self._one(rng, *combos[i % len(combos)], index=i) for i in range(n)]

    def _one(self, rng, template_id, pool, index):
        lang = pick_lang(rng)
        item = self._item(rng, pool, template_id, lang)
        state, sections = self._render(rng, item, lang)

        cands = item["candidates"]
        p = list(item["target"]["p"])
        assert len(cands) == len(p), f"{self.name}: 候选数与目标长度不符"

        # 候选逐例重排 —— 连同目标、等级、标签一起置换
        order = list(range(len(cands)))
        rng.shuffle(order)
        cands = [cands[j] for j in order]
        p = [p[j] for j in order]

        total = sum(p)
        assert abs(total - 1.0) < 1e-9, f"{self.name}: 目标和为 {total}，应为 1"
        assert all(x >= 0 for x in p), f"{self.name}: 目标含负概率"

        question = item["question"]
        split = self.split_of(template_id, pool)
        rec = {
            "id": f"{self.name}::{split}::{index:07d}",
            "source": f"synth:{self.name}",
            "gen_version": GEN_VERSION,
            "split": split,
            "schema": {
                "primitive": item["primitive"],
                "name": item.get("schema_name", self.name),
                "desc": item.get("schema_desc", ""),
                "positive_label": item.get("positive_label"),
            },
            "state": state,
            "state_sections": sections,
            "question": question,
            "question_paraphrases": self.paraphrases(rng, question, template_id, lang),
            "candidates": cands,
            "target": dict(item["target"], p=p),
            "meta": dict(item.get("meta", {}), template_id=template_id,
                         entity_pool=pool, K_full=len(cands)),
        }
        self._check(rec)
        return rec

    def _render(self, rng, item, lang):
        """把语义项的 sections 交给 StateBuilder 渲染（随机段序 + 注入干扰项）。"""
        sb = StateBuilder(lang)
        for s in item["sections"]:
            if isinstance(s, dict):
                sb.add(s["seg"], s["text"], s["priority"])
            else:
                sb.add(*s)
        return sb.render(rng, distractors=item.get("distractors", 0),
                         distractor_texts=item.get("distractor_texts"))

    # -- 渲染充分性（R1 缓解措施 ②） ---------------------------------------
    def sufficiency(self, rec):
        """检查目标赖以计算的量是否真的渲染进了 state。返回问题列表，空 = 通过。

        只做最便宜的那一半 —— **量在不在文本里**。另一半"用这些量能否恢复 P*"由
        oracle 上界负责（`eval/`）。两半都不能省：这里会漏掉"量在、但规则写错了"，
        oracle 会漏掉"规则对、但某个量没印出来所以模型无从下手"。
        """
        if rec["target"].get("provenance") == "hard":
            return []                      # 硬目标没有可恢复的 P*，无此口子
        name = rec["schema"]["name"]
        frag = {}
        for s in rec["state_sections"]:
            frag[s["seg"]] = frag.get(s["seg"], "") + s["text"] + "\n"
        out = []

        markers = self.RULE_MARKERS.get(name, ())
        if markers and not any(m in rec["state"] for m in markers):
            out.append(f"规则句缺失，{markers} 一个都没出现")

        scan = {seg: scan_numbers(txt) for seg, txt in frag.items()}
        audit = rec["target"].get("audit", {})
        for key, kind, seg in self.RENDERED.get(name, ()):
            val = audit.get(key)
            if val is None:
                out.append(f"audit 缺字段 {key}")
                continue
            found = scan.get(seg, (set(), set()))[0 if kind == "money" else 1]
            for v in (val if isinstance(val, (list, tuple)) else [val]):
                if v not in found:
                    out.append(f"{key}={v} 未在 {seg} 段渲染出来")
        return out

    @staticmethod
    def _check(rec):
        """把 schema 契约钉死在生成处，而不是等下游某个不报错的地方炸掉。"""
        joined = "\n".join(s["text"] for s in rec["state_sections"])
        assert joined == rec["state"], f"{rec['id']}: state 与 state_sections 不一致"
        assert len(rec["target"]["p"]) == len(rec["candidates"])
        assert rec["target"]["provenance"] in (
            "explicit_rng", "marginalized", "tie_set", "human_annotators", "hard")
        assert rec["schema"]["primitive"] in ("noul", "choice", "score")
        if rec["schema"]["primitive"] == "score":
            assert all(c["meta"]["level"] is not None for c in rec["candidates"])

    def split_of(self, template_id, pool):
        return self.split_map[(template_id, pool)]


# ---------------------------------------------------------------------------
# 语言无关的小工具，供各生成器复用
# ---------------------------------------------------------------------------

def pick_text(rng, lang, zh, en):
    """按语言整句取一侧。

    **必须整句取，不能逐词取。** 逐词选会造出 `该用户可transfer的对象` 这种句子 ——
    混排（code-switching）是**句间**切换，句内换语言只会让 state 变成噪声。

    这条约束在每个生成器里都出现过，所以放在基类而不是各写一份：写漏一次的后果不是
    报错，而是有一批样本的 state 变成夹生句，而它们照样进训练集。
    """
    if lang == "zh":
        return zh
    if lang == "en":
        return en
    return zh if rng.random() < 0.5 else en


def subject_phrase(rng, pool, lang, salt="subject"):
    """一个带确定性 id 的主语，如 `租户 8f3a91c2`。"""
    return f"{term(rng, SUBJECTS, lang)} {eid(pool, salt, rng.randrange(64))}"


def uniform(n):
    return [1.0 / n] * n


def one_hot(n, index):
    p = [0.0] * n
    p[index] = 1.0
    return p


def softmax(xs, tau=1.0):
    m = max(xs)
    ex = [math.exp((x - m) / tau) for x in xs]
    s = sum(ex)
    return [e / s for e in ex]


def sigmoid(x):
    """数值稳定版。决策规则里大量用到 σ(margin/τ)，τ 小时容易溢出。"""
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)
