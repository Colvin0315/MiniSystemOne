"""
G6 `calendar_slot` —— **大 K 路径**，candidates 上到 255，目标多峰且大量近并列。

任务：把一场会议排进哪个时段。它是全套里唯一 K 能上到 255 的生成器，因此承担三件事：

  1. **大 K 单次前向** —— 255 个候选挤在一条打包序列里，直接测 `S ≤ 1024` 之外的
     实际容量与分块路径（`prefix_blocked` + `crosstalk=False` 使打包序列**精确可划分**，
     前缀 K/V 可缓存复用）。
  2. **顺序不变性的最敏感测试** —— 目标里大量近并列的候选意味着 logits 彼此接近，
     任何位置相关的噪声都会在排序上显形。`scripts/smoke_test.py` 的 [B] 置换不变性
     主要就是冲它来的。
  3. **"不确定时给平的分布"的大规模版本** —— 同一组内权重**完全相等**，正确行为是
     组内均匀。这是 G2 的 tie 语义在 255 候选规模上的复现。

构造：每个与会者有一张「日 × 区段」权重表（成文印在 state 里），时段 s 的联合权重

    w(s) = Π_a  日权重_a(day(s)) × 区段权重_a(band(s))

目标为 `w` 在候选集上的归一化。乘积可分解成 `D(day) × P(band)`，于是权重**完全相同**的
候选严格并列（同一天同一区段的全部时刻），而不同 (日, 区段) 组合之间按乘积渐变 ——
目标因此是多峰、大量近并列的软分布，而不是 one-hot。权重全在 **state** 里，候选文本
只提供"哪一天、哪一刻"，所以"把候选改成不透明串会不会掉分"这一测试在这里最有
含义 —— 本仓库没有产出它，但数据是按能被它区分的方式造的。

三处必须成文，否则 R1 的口子就开（目标由我的内部逻辑算出，而文本没给出同等信息）：

  - **联合概率是各与会者权重之积**，且与会者权重 = 日权重 × 区段权重。没有这句，
    "相乘还是取最小还是取平均"就只是我的约定。
  - **权重相同则概率相同。** 组内并列不是近似而是精确相等，不说清楚，模型只能猜
    是否还有别的偏好序，而组内均匀是这批样本上唯一正确的答案。
  - **区段边界**（08:00–11:45 等）。候选文本给的是"周一 09:15"，模型要靠边界才能
    把它映射到一个区段。

前两条由 `sufficiency` 单独检查 —— 它们最容易在改写措辞时被顺手丢掉，而丢掉不报错。

权重表的量纲沿 G5 的做法：以**百分点**为整数抽取（55–98），乘 100 存成 bps，经
`fmt_pct` 渲染成一位小数。整数百分点让渲染与 `scan_numbers` 的回读逐位相等 —— 若
出现 `82.17%`，印出来是 `82.2%` 而回读是 8217，那会**误报**成"这个量没渲染出来"。

反空洞（vacuous check）的构造：每个与会者的 10 个权重在同一张表里**互不相等**
（`rng.sample` 无放回），且该段内除这 10 个百分数外**不出现任何其它数字**（与会者标签
用字母 A–E，日名与区段名都不含阿拉伯数字，实体 id 一律不进这张表 —— 十六进制 id 里的
`82` 会与权重 `0.82` 撞值）。两条合起来保证「声明值出现在该段数字集合里」等价于
「它真的被印出来了」。少任何一条，检查都会退化成恒真。
"""
from dataset.synth.base import Generator, StateBuilder, pick_text as _pick
from dataset.synth.lexicon import STATUSES, eid, fmt_pct, term

ORDINALS = (
    ("workspace", "工作区"), ("organization", "组织"), ("team", "团队"),
    ("department", "部门"), ("tenant", "租户"),
)

DAYS = (
    ("Mon", "周一"), ("Tue", "周二"), ("Wed", "周三"),
    ("Thu", "周四"), ("Fri", "周五"), ("Sat", "周六"),
)

# 区段 → (键, 中文名, 英文名, 起始小时, 含多少个刻钟)。四个区段拼成 08:00–19:45。
PARTS = (
    ("morning", "上午", "morning", 8, 16),
    ("noon", "中午", "noon", 12, 8),
    ("afternoon", "下午", "afternoon", 14, 16),
    ("evening", "晚间", "evening", 18, 8),
)

FIRST_HOUR = 8
SLOT_MINUTES = 15
N_SLOTS_PER_DAY = sum(p[4] for p in PARTS)          # 48

# 与会者权重以百分点为整数抽取。**同一张表内各不相同**（无放回），见模块 docstring。
WEIGHT_GRID = tuple(range(55, 99))
N_WEIGHTS_PER_ATTENDEE = len(DAYS) + len(PARTS)     # 6 个日权重 + 4 个区段权重

ATT_MIN, ATT_MAX = 2, 5
ATTENDEE_LABELS = ("A", "B", "C", "D", "E")

# 每个与会者**独占一个 seg**。这是非空洞检查成立的前提：`scan_numbers` 返回的是
# 集合，把两张权重表放进同一段，两张表里等值的权重就再也分不清谁渲没渲染。
# 模型看不到 seg（它只进 state_sections），所以这几个词的语义只是内部记号。
ATTENDEE_SEGS = ("limits", "verification", "history", "notes", "billing")

# K 的取法：从 3 一路铺到 255，让大 K 与分块路径拿到足够样本，而不是偶发几次。
# 训练侧 `DecisionDataset` 会再按 K ≤ 32 子采样（`subsample_candidates`），
# 这里的 K_full 决定的是**超集**与评测时的变 K 能力。
K_CHOICES = (3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 255)

SLOT_QUESTIONS = (
    ("会议会被安排在哪个时段？", "Which slot will the meeting be booked into?"),
    ("这场会议最终定在什么时间？", "What time will this meeting end up being held?"),
    ("系统会把会议排进哪个时段？", "Which slot will the system schedule the meeting for?"),
    ("哪个时段会被选中？", "Which slot will be selected?"),
    ("会议的时间落在哪一档？", "Which time slot does the meeting fall into?"),
    ("排期结果是哪个时段？", "What is the scheduled time slot?"),
    ("这场会议将占用哪个时段？", "Which slot will this meeting occupy?"),
    ("最终预约到哪个时间？", "Which time is the meeting finally booked for?"),
    ("会议被安排在哪一刻？", "At which time is the meeting placed?"),
    ("哪个时段能排上这场会？", "Which slot will host this meeting?"),
    ("预约落在哪个格子？", "Which cell does the booking land in?"),
    ("这场会议该排在什么时候？", "When should this meeting be scheduled?"),
    ("排期选中的是哪个时段？", "Which slot was chosen by the scheduler?"),
    ("会议时间最终敲定在哪？", "Which slot is the meeting finalized at?"),
    ("这场会排在哪个时段？", "Which slot is this meeting placed in?"),
    ("系统选定的时段是哪个？", "Which slot did the system pick?"),
    ("会议被预定到哪个时段？", "Which slot is the meeting reserved for?"),
    ("哪个时段会分配给这场会？", "Which slot will be allocated to this meeting?"),
    ("这场会议的档期是？", "What is the meeting's slot?"),
    ("排期结果落在哪个时间？", "Which time does the schedule land on?"),
    ("会议将定在哪个时段？", "Which slot will the meeting be set to?"),
    ("选定的是哪个时段？", "Which slot was chosen?"),
    ("这场会最终排进哪个格子？", "Which cell does this meeting end up in?"),
    ("会议时间被安排在哪一档？", "Which slot is the meeting time assigned to?"),
)


def _band_bounds(part):
    """区段的 [起, 止] 分钟（止 = 最后一个刻钟的**起点**，不是结束时刻）。

    从 `PARTS` 推出来而不是手写常量：手写一份"08:00–11:45"就等于把同一件事存两处，
    改一处忘一处会让模型拿到错的边界，而目标仍按对的那份算 —— 坏 ECE 度量的是那句话。
    """
    start = part[3] * 60
    return start, start + (part[4] - 1) * SLOT_MINUTES


def _band_of_minutes(mins):
    """当日**绝对**分钟 → 区段序号。与 state 里印的边界同源。

    参数是"从零点起算的分钟"（08:00 = 480），不是"从开工起算的偏移"。两者混用不会
    报错，只会让每个时刻落进错误的区段 —— 而目标随之按错的区段算，state 里印的却是
    正确的边界。那是一份**模型无从复现**的目标，正是 R1 要挡的东西。
    """
    for i, part in enumerate(PARTS):
        lo, hi = _band_bounds(part)
        if lo <= mins <= hi:
            return i
    return len(PARTS) - 1


def _slot_minutes(q):
    """刻钟序号 → 当日绝对分钟。区段查找的唯一入口，避免各处自行换算。"""
    return FIRST_HOUR * 60 + q * SLOT_MINUTES


class CalendarSlot(Generator):
    name = "calendar_slot"
    prefix = "cs"
    n_templates = 24

    # 每位与会者一张权重表，各自独占一段。见模块 docstring。
    #
    # 读法是 `count` 而不是 `money`：权重以百分点存整数（68），渲染成 `68.0%` 后
    # **分**读数是 6800、**整**读数才是 68。声明成 money 会让每一个权重都报"没渲染
    # 出来" —— 而它们全都印得好好的。量纲与读法的对应关系是这里唯一容易搞错的地方。
    RENDERED = {name: tuple(
        (f"w{i}", "count", seg) for i, seg in enumerate(ATTENDEE_SEGS))}
    RULE_MARKERS = {
        name: ("排期规则：", "会议规则：", "Scheduling rule:", "Meeting rule:")}

    def sufficiency(self, rec):
        """在基类的数字回读之外，补两条 R1 必需的检查 —— 都是"目标含一个 state
        无从推断的假设"这一类缺口：不报错，只让 ECE 变差。"""
        out = super().sufficiency(rec)

        # 相乘语义必须成文。否则"联合"是取积、取最小还是取平均，模型只能猜。
        if not any(m in rec["state"] for m in ("之积", "相乘", "product")):
            out.append("各与会者权重相乘的成文依据缺失")

        # 组内**精确**并列。不说清"权重相同则概率相同"，模型只能猜是否还有别的偏好序，
        # 而组内均匀是这批样本上唯一正确的答案。
        #
        # 大小写各写一遍：标记是**区分大小写**的子串匹配，而英文侧有一句以 "Equal"
        # 开头。漏掉大写形式不会让检查失效，而是让它对每个英文样本都误报 —— 误报的
        # 代价是让人去改一个本来正确的生成器。
        if not any(m in rec["state"] for m in
                   ("权重相同", "equal weights", "Equal weights")):
            out.append("权重相同则概率相同的成文依据缺失")
        return out

    # -- 语义 ---------------------------------------------------------------
    def _item(self, rng, pool, template_id, lang):
        idx = int(template_id[len(self.prefix) + 1:])
        qidx = idx % len(SLOT_QUESTIONS)

        n_att = rng.randint(ATT_MIN, ATT_MAX)
        # 每人的 10 个权重互不相等（无放回），见模块 docstring 末段。
        tables = [rng.sample(WEIGHT_GRID, N_WEIGHTS_PER_ATTENDEE) for _ in range(n_att)]

        # 联合权重可分解：w(day, band) = D(day) × P(band)。同一 (日, 区段) 的所有候选
        # 权重**严格相等**，这正是组内并列的来源。
        day_w = [1.0] * len(DAYS)
        band_w = [1.0] * len(PARTS)
        for t in tables:
            for d in range(len(DAYS)):
                day_w[d] *= t[d] / 100.0
            for b in range(len(PARTS)):
                band_w[b] *= t[len(DAYS) + b] / 100.0

        pool_size = len(DAYS) * N_SLOTS_PER_DAY
        K = min(rng.choice(K_CHOICES), pool_size)
        slots = rng.sample(range(pool_size), K)
        weights = [day_w[s // N_SLOTS_PER_DAY]
                   * band_w[_band_of_minutes(_slot_minutes(s % N_SLOTS_PER_DAY))]
                   for s in slots]
        total = sum(weights)
        p = [x / total for x in weights]

        sb = StateBuilder(lang)
        sb.add("account", self._meeting(rng, lang, pool), 1)
        for i in range(n_att):
            sb.add(ATTENDEE_SEGS[i], self._availability(rng, lang, i, tables[i]), 3)
        sb.add("policy", self._rule(rng, lang), 3)

        # audit 只有**与候选顺序无关**的量。基类 `_one` 会打乱候选与目标，**但不打乱
        # audit** —— 任何"与 candidates 逐位对齐"的列表在这里都会静默错位。
        # oracle 若需要逐候选权重，用 day_w / band_w 配候选文本里的日与时刻重算。
        audit = {f"w{i}": (list(tables[i]) if i < n_att else [])
                 for i in range(ATT_MAX)}
        audit.update(n_att=n_att, day_w=day_w, band_w=band_w)

        texts = [self._slot_text(rng, lang, s) for s in slots]
        return {
            "question": _pick(rng, lang, *SLOT_QUESTIONS[qidx]),
            "primitive": "choice",
            "sections": sb.sections,
            "distractors": rng.randrange(0, 3),
            "schema_name": self.name,
            "schema_desc": "会议排进哪个时段",
            "candidates": [{"text": t, "label": t, "meta": {"level": None}}
                           for t in texts],
            "target": {"kind": "soft", "p": p, "provenance": "explicit_rng",
                       "renormalized": False, "audit": audit},
            "meta": {"variant": self.name, "language": lang, "question_idx": qidx,
                     "n_attendees": n_att, "K_drawn": K},
        }

    # -- state 各段 ---------------------------------------------------------
    def _meeting(self, rng, lang, pool):
        subj = f"{term(rng, ORDINALS, lang)} {eid(pool, 'org', rng.randrange(64))}"
        return _pick(rng, lang,
                     f"会议：{subj}，状态为 {term(rng, STATUSES, lang)}",
                     f"Meeting: {subj}, status {term(rng, STATUSES, lang)}")

    def _availability(self, rng, lang, i, table):
        """一位与会者的权重表。**只渲染选中的那一侧语言** —— 两侧都拼会多消耗随机数，
        让同一条样本的其它字段随语言而变，那是无谓的耦合。"""
        label = ATTENDEE_LABELS[i]
        if lang == "zh":
            days = "，".join(f"{DAYS[d][1]} {fmt_pct(rng, table[d] * 100)}"
                             for d in range(len(DAYS)))
            bands = "，".join(f"{PARTS[b][1]} {fmt_pct(rng, table[len(DAYS) + b] * 100)}"
                              for b in range(len(PARTS)))
            return f"与会者 {label} 的可用权重 —— 按日：{days}；按区段：{bands}"
        days = ", ".join(f"{DAYS[d][0]} {fmt_pct(rng, table[d] * 100)}"
                         for d in range(len(DAYS)))
        bands = ", ".join(f"{PARTS[b][2]} {fmt_pct(rng, table[len(DAYS) + b] * 100)}"
                          for b in range(len(PARTS)))
        return f"Attendee {label} availability weights — by day: {days}; by band: {bands}"

    def _rule(self, rng, lang):
        """规则句。乘积语义、并列语义、区段边界三件事都在这里，缺一不可：

          - 乘积语义缺了，模型不知道"联合"怎么算；
          - 并列语义缺了，组内均匀就只是我的约定；
          - 边界缺了，模型无法把候选文本里的"周一 09:15"映射到一个区段。

        前两条由 `sufficiency` 单独检查 —— 它们在改写措辞时最容易被顺手丢掉。
        """
        bands_zh = "、".join(
            f"{p[1]} {_fmt_hm(_band_bounds(p)[0])}–{_fmt_hm(_band_bounds(p)[1])}"
            for p in PARTS)
        bands_en = ", ".join(
            f"{p[2]} {_fmt_hm(_band_bounds(p)[0])}-{_fmt_hm(_band_bounds(p)[1])}"
            for p in PARTS)
        forms = (
            (f"排期规则：会议时段按各时段全体与会者可出席的联合概率加权随机选定；"
             f"联合概率 = 各与会者在该时段的权重之积，而单个与会者在该时段的权重 = "
             f"其当日权重 × 该时刻所属区段的权重。权重相同则概率相同，不另作偏好。"
             f"区段划分：{bands_zh}。候选时段以 15 分钟为间隔。",
             f"Scheduling rule: the meeting slot is drawn at random with probability "
             f"proportional to the joint availability of all attendees at that slot. "
             f"The joint value is the product of every attendee's weight at that slot, "
             f"and a single attendee's weight at a slot is their weight for that day "
             f"times their weight for the band the time falls in. Equal weights mean "
             f"equal probability, with no further preference. Bands: {bands_en}. "
             f"Slots are spaced 15 minutes apart."),
            (f"会议规则：每个候选时段有一个联合权重，等于各与会者在该时段的权重相乘；"
             f"与会者权重又等于其当日权重乘以该时刻所属区段的权重。时段按联合权重"
             f"加权随机选定，权重相同则概率相同。区段划分：{bands_zh}。",
             f"Meeting rule: every candidate slot carries a joint weight equal to the "
             f"product of each attendee's weight at that slot, and an attendee's weight "
             f"is that attendee's weight for the day times the weight for the band the "
             f"time falls in. Slots are drawn at random weighted by the joint value, "
             f"and equal weights mean equal probability. Bands: {bands_en}."),
        )
        zh, en = forms[rng.randrange(len(forms))]
        return _pick(rng, lang, zh, en)

    def _slot_text(self, rng, lang, s):
        """候选文本：哪一天、哪一刻。格式轮换，迫使模型读时刻而不是记字符串。

        **区段不写进候选** —— 模型要靠 state 里的边界把它推出来。若把区段印在候选上，
        "候选文本是否真的被读"就永远测不出来了。
        """
        d, q = divmod(s, N_SLOTS_PER_DAY)
        h = FIRST_HOUR + q * SLOT_MINUTES // 60
        m = q * SLOT_MINUTES % 60
        day_zh, day_en = DAYS[d][1], DAYS[d][0]
        if lang == "en" or (lang == "mix" and rng.random() < 0.5):
            if rng.random() < 0.5:
                return f"{day_en} {h % 12 or 12}:{m:02d} {'AM' if h < 12 else 'PM'}"
            return f"{day_en} {h:02d}:{m:02d}"
        return f"{day_zh} {h:02d}:{m:02d}"

    # -- 复述 ---------------------------------------------------------------
    def paraphrases(self, rng, question, template_id, lang):
        """复述 = 问法表里的其它问法，**必须同语言** —— 拿英文复述去测中文问法，测的
        是语言切换而不是"学的是问题还是模板"。"""
        order = list(range(len(SLOT_QUESTIONS)))
        rng.shuffle(order)
        out = []
        for j in order:
            zh, en = SLOT_QUESTIONS[j]
            text = _pick(rng, lang, zh, en)
            if text != question and text not in out:
                out.append(text)
            if len(out) == 3:
                break
        return out


def _fmt_hm(mins):
    """当日分钟 → `HH:MM`。边界文本与 `_band_of_minutes` 共用 `PARTS`，不会漂移。"""
    return f"{mins // 60:02d}:{mins % 60:02d}"
