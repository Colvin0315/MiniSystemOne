"""
合成数据集审计 —— **全仓库最有价值的脚本**。

它回答的问题是：这个数据集的分数，是模型真在读 state，还是词法表面上的假象？

四道检查，从便宜到贵：

1. **schema 契约**：逐条核对生成时钉死的那些不变量（state 与 state_sections 一致、
   目标和为 1、provenance 合法、K_full 对得上）。这些在 `Generator._check` 里
   生成时就查过一遍；这里查的是**落盘之后**的那份，因为 JSON 往返会丢掉类型
   （tuple→list）与精度，而训练只看得见落盘的那份。

2. **split 不相交（数据层复核）**：不看内存里的 split_map，直接按 jsonl 里每条记录的
   `(source, template_id, entity_pool)` 重建集合，断言 train / val / calib /
   test_known 两两不相交。生成时的那次断言只覆盖了生成器自己声明的映射；这一条
   覆盖的是**实际写出来的数据**，能抓到"映射对了但采样串了"这类错误。

3. **渲染充分性**（R1 缓解措施 ②）：目标赖以计算的量必须真的渲染进了 state。
   实现委托给各生成器的 `sufficiency()`，声明在生成器上（它才知道自己渲染了什么）。

4. **BoW 门禁**（R2）：在训练集上训一个**词法模型**，在留出集上评测。
   - 门禁探针（`--model_eval` 给了结果 JSON 时强制）：词法模型能看到的东西是
     "候选自身的词" + "候选的词是否也出现在 state / question 里"，够它做字符串
     匹配，但做不了算术，也做不了跨语言的语义映射。要求**真实模型比它高出至少
     `--min_gap` 个点**；达不到就说明生成器过度模板化，先多样化再报结果。
   - 答案泄漏探针：只给 `state + question`（**不给候选**）的多分类词法模型。
     它高 = state 把答案写得太直白。这一条不设硬门槛，因为对 `tool_router` 这
     类**本质上就是把意图词映射到工具名**的任务，词法映射本身就是任务的一部分；
     但对 `banking_balance` 这类要算数的任务，它**必须**接近随机，否则就是泄漏。

跑法：
    python scripts/audit_synthetic.py                       # 审计 dataset/synth
    python scripts/audit_synthetic.py --model_eval out/eval/xxx/test_known.json
"""
import argparse
import collections
import json
import os
import random
import re
import sys
import zlib

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from trainer.trainer_utils import unbuffer_stdout

SPLITS = ("train", "val", "calib", "test_known", "test_ood")
# 中英混排的分词：拉丁词整词，负号并入数字，CJK 逐字。
#
# **下划线必须切开。** 候选文本是标识符（`cancel_order`），而 state 把动作与对象
# 分开写（"Requested action: cancel; target object: order"）。若 `[a-z_]+` 把
# `cancel_order` 吞成一个 token，它就与 state 的词表永不相交，`shared` 恒为 0，
# 探针退化成只会报类别先验 —— 门禁看上去在跑，实际什么都没测。这是最坏的一种
# 失败：BoW 上界被压到随机水平，于是"模型高出 25 点"这个门槛自动成立，
# 过度模板化就再也拦不住了。切开后 `cancel_order` → {cancel, order}，
# 与 `cancel_subscription` → {cancel, subscription} 只共享动作词，
# 正是"读候选文本"该有的压力。
TOKEN_RE = re.compile(r"-?\d[\d,.]*|[a-z]+|<[a-z]+>|[一-鿿]")
NUM_TOKEN_RE = re.compile(r"-?\d[\d,.]*$")
HASH_BUCKETS = 1 << 14


def tokens(text, drop_numbers=False):
    """`drop_numbers` 只给答案泄漏探针用，见 `state_only_probe`。

    数字是**模型该去算的量**，不是该去背的串。词法探针能看见数字时，它可以绕开
    算术、直接记住"这个数字串配这个词 ⇒ 这个标签"，于是量出来的是**输入空间的
    基数**，不是"文字有没有把答案写出来"。把数字摘掉，剩下的才是措辞本身泄漏了
    多少 —— 那正是这个探针要问的问题。
    """
    ws = TOKEN_RE.findall(text.lower())
    if drop_numbers:
        ws = [w for w in ws if not NUM_TOKEN_RE.match(w)]
    return ws


def hash_ids(words, offsets=(0,)):
    """把词表哈希到固定桶，避免为探针建词表（探针不该比模型多知道什么）。

    用 crc32 而不是内置 `hash()`：后者对 str 加盐，每个进程的结果都不同，
    探针的分数会随机漂移，审计就不可复现了。
    """
    return [(zlib.crc32(w.encode()) % HASH_BUCKETS) + off
            for off in offsets for w in words]


def load(path):
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stratify(recs, n, seed=0):
    """按 `source` 分层抽样到 n 条。n <= 0 表示不抽。

    **不能写成 `recs[:n]`。** `build_dataset.py` 是按生成器逐族写出记录的
    （`for g in train_gens:`），所以前缀切片拿到的全是**第一个生成器** ——
    探针在一族上学出一个上界，再拿去评价另外五族，门禁的数字就没有意义了。

    抽样的代价要说清楚：探针训练集变小，词法上界会**略低**，门禁因此略微变松。
    接受这个代价是因为探针的瓶颈是**容量**（3×16384 桶的 bag-of-words + 4 个标量）
    而不是数据量，3 万条对它是饱和的；而全量 18 万条里 calendar_slot 那一族
    每条的 K 可以到 255，逐候选建特征会把审计撑到 6.5 GB 内存且跑不完。
    """
    if n <= 0 or len(recs) <= n:
        return recs
    rng = random.Random(seed)
    by = collections.defaultdict(list)
    for r in recs:
        by[r["source"]].append(r)
    out = []
    for src, group in sorted(by.items()):
        k = max(1, round(n * len(group) / len(recs)))
        out.extend(rng.sample(group, min(k, len(group))))
    return out


# ---------------------------------------------------------------------------
def check_contract(recs, split, problems):
    for r in recs:
        rid = r.get("id", "?")
        joined = "\n".join(s["text"] for s in r["state_sections"])
        if joined != r["state"]:
            problems.append(f"{rid}: state 与 state_sections 不一致")
        p = r["target"]["p"]
        if abs(sum(p) - 1.0) > 1e-9:
            problems.append(f"{rid}: 目标和为 {sum(p)}")
        if len(p) != len(r["candidates"]):
            problems.append(f"{rid}: 目标长度 {len(p)} ≠ 候选数 {len(r['candidates'])}")
        if r["target"]["provenance"] not in ("explicit_rng", "marginalized", "tie_set",
                                             "human_annotators", "hard"):
            problems.append(f"{rid}: 非法 provenance {r['target']['provenance']}")
        if r["schema"]["primitive"] not in ("noul", "choice", "score"):
            problems.append(f"{rid}: 非法 primitive")
        if r["meta"]["K_full"] != len(r["candidates"]):
            problems.append(f"{rid}: K_full 与候选数不符")
        if r["split"] != split:
            problems.append(f"{rid}: split 字段 {r['split']} ≠ 所在文件 {split}")
        if not r.get("meta", {}).get("approx_tokens"):
            problems.append(f"{rid}: 缺 meta.approx_tokens")
        if r["schema"]["primitive"] == "score":
            if any(c["meta"]["level"] is None for c in r["candidates"]):
                problems.append(f"{rid}: score 样本有候选缺 level")


def check_split_overlap(data):
    """按落盘记录的 (source, template_id, entity_pool) 重建组合集合并查相交。"""
    combos = {s: set() for s in SPLITS}
    for s, recs in data.items():
        for r in recs:
            combos[s].add((r["source"], r["meta"]["template_id"], r["meta"]["entity_pool"]))
    problems = []
    base = [s for s in ("train", "val", "calib") if combos[s]]
    for i, a in enumerate(base):
        for b in base[i + 1:]:
            inter = combos[a] & combos[b]
            if inter:
                problems.append(f"{a} 与 {b} 共用 {len(inter)} 个组合，例如 {sorted(inter)[:2]}")
    for s in ("test_known", "test_ood"):
        if not combos[s]:
            continue
        # test_known 本身就是"碰留出"的组合，所以只要求它与 train 不相交
        inter = combos["train"] & combos[s]
        if inter:
            problems.append(f"train 与 {s} 共用 {len(inter)} 个组合，例如 {sorted(inter)[:2]}")
    return combos, problems


# ---------------------------------------------------------------------------
# 词法模型（两个探针共用同一套基础设施）
# ---------------------------------------------------------------------------
class BagScorer(torch.nn.Module):
    """候选打分：Σ 桶权重 + 标量项。特征只看得见词的**存在**，看不见顺序，
    更看不见任何算术 —— 这正是"词法上界"该有的能力边界。"""

    def __init__(self, dim, n_scalar):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(dim, 1, mode="sum")
        self.scalar = torch.nn.Linear(n_scalar, 1)
        torch.nn.init.zeros_(self.bag.weight)
        torch.nn.init.zeros_(self.scalar.weight)
        torch.nn.init.zeros_(self.scalar.bias)

    def forward(self, idx, off, scal):
        return (self.bag(idx, off).squeeze(-1) + self.scalar(scal).squeeze(-1))


def _batch_tensors(records, kmax, device):
    """records: 一批记录，每条形如 [(候选 token id 列表, 候选标量), ...]。

    不足 kmax 的记录补**空袋**（offset 不前进 → 求和为 0），这样同一批里 K 不齐也
    能一次前向；padding 槽的分数随后由 mask 置 -inf，不参与 softmax。
    """
    flat, offs, scal = [], [0], []
    n_scalar = len(records[0][0][1])
    for rec in records:
        for j in range(kmax):
            if j < len(rec):
                flat.extend(rec[j][0])
                scal.append(rec[j][1])
            else:
                scal.append([0.0] * n_scalar)
            offs.append(len(flat))
    return (torch.tensor(flat, dtype=torch.long, device=device),
            torch.tensor(offs[:-1], dtype=torch.long, device=device),
            torch.tensor(scal, dtype=torch.float32, device=device))


def _fit(model, rows, labels, epochs, lr, device, batch=512):
    """按记录分批训打分器。同一批内 K 必须对齐，不足者补空袋并 mask 掉。"""
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = len(rows)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for s in range(0, n, batch):
            sel = perm[s:s + batch].tolist()
            sub = [rows[i] for i in sel]
            kmax = max(len(r) for r in sub)
            idx, off, scal = _batch_tensors(sub, kmax, device)
            logits = model(idx, off, scal).view(len(sub), kmax)
            # mask 必须补到 kmax，否则 (B,Ki) 与 (B,kmax) 无法广播
            keep = torch.zeros((len(sub), kmax), dtype=torch.bool, device=device)
            for j, i in enumerate(sel):
                keep[j, :len(rows[i])] = True
            logits = logits.masked_fill(~keep, float("-inf"))
            tgt = torch.tensor([labels[i] for i in sel], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(logits, tgt)
            opt.zero_grad(); loss.backward(); opt.step()


def bow_gate(train, device, epochs):
    """门禁探针：候选自身的词 + 候选词是否也出现在 state / question 里。

    特征刻意包含"共现"这一项，因为**没有它，这个探针就是个废物**：候选只依赖自身
    的特征在 softmax 里只能学到"哪些词更像答案"的全局先验，与 state 无关，任何
    生成器都能轻松"打败"它，门禁就成了走过场。加上共现，它才真正是那条"字符串
    匹配能做到的上界"。

    返回 `evaluate(recs) -> 准确率`，只训一次、任意子集复用 —— 门禁的价值恰恰在于
    **分语言/分来源**看，一个总平均值会让"跨语言对齐"这种浅层能力顶掉门禁
    （中英混排下候选是英文标识符、state 是中文时词法探针必然失手，见 `TOKEN_RE`）。
    """
    stats = {}
    for r in train:
        st, qt = set(tokens(r["state"])), set(tokens(r["question"]))
        stats[r["id"]] = (st, qt)

    def rows_of(recs):
        out = []
        for r in recs:
            st, qt = stats.get(r["id"], (set(tokens(r["state"])), set(tokens(r["question"]))))
            rows = []
            for c in r["candidates"]:
                ct = tokens(c["text"])
                shared = sum(1 for t in ct if t in st)
                ids = (hash_ids(ct, (0,))
                       + hash_ids([t for t in ct if t in st], (HASH_BUCKETS,))
                       + hash_ids([t for t in ct if t in qt], (2 * HASH_BUCKETS,)))
                rows.append((ids, [len(ct), shared / max(len(ct), 1), len(r["candidates"]),
                                   1.0 if shared else 0.0]))
            out.append((rows, r))
        return out

    tr = rows_of(train)
    model = BagScorer(3 * HASH_BUCKETS, 4).to(device)
    _fit(model, [x[0] for x in tr], [gold_index(r) for _, r in tr], epochs, 0.05, device)

    def evaluate(recs):
        if not recs:
            return 0.0
        ev = rows_of(recs)
        correct = 0
        with torch.no_grad():
            for rows, r in ev:
                kmax = len(rows)
                idx, off, scal = _batch_tensors([rows], kmax, device)
                logits = model(idx, off, scal).view(1, kmax)
                correct += int(int(logits.argmax(-1)) == gold_index(r))
        return correct / len(ev)

    return evaluate


def gold_index(rec):
    """软目标的"正确项"取 argmax。探针只用得上一个答案，用 argmax 是最自然的选择
    （也是准确率的定义）；并列样本上它对词法模型和真实模型一样不利，不构成偏向。"""
    p = rec["target"]["p"]
    return max(range(len(p)), key=lambda i: p[i])


def gold_label(rec):
    return rec["candidates"][gold_index(rec)]["label"]


class StateOnlyProbe(torch.nn.Module):
    def __init__(self, dim, n_class):
        super().__init__()
        self.bag = torch.nn.EmbeddingBag(dim, n_class, mode="sum")
        torch.nn.init.zeros_(self.bag.weight)

    def forward(self, idx, off):
        return self.bag(idx, off)


def state_only_probe(train, device, epochs, drop_numbers=False):
    """泄漏探针：**不给候选**，只凭 state + question 预测答案标签。

    返回 (predict, 类别数)。`predict(recs)` 给出 (准确率, 预测落在候选集内的比例)。
    探针只训一次，各子集复用 —— 按来源在循环里重训一遍会把审计时间乘上来源个数，
    而"词法模型从同一批数据里学到了什么"这件事与评测子集无关。

    **两个变体都要跑，差值本身就是结论。** 允许看数字时，探针可以跳过算术、直接
    把"这个数字串配紧邻的那个词 ⇒ 这个标签"背下来 —— 像 `refund_policy` 那样输入
    是四五个**低基数**的渲染值（赔付率 90 个取值、检出率/误报率各 30 个）时，它
    能背出一张两三百项的查找表。那测到的是**输入空间的基数**，不是泄漏。
    摘掉数字之后剩下的读数，才是"措辞有没有把答案写出来"。
    两者都高 ⇒ 真的泄漏；只有前者高 ⇒ 任务输入太离散，该记在任务的难度上。
    """
    classes = sorted({c["label"] for r in train for c in r["candidates"]})
    cidx = {c: i for i, c in enumerate(classes)}

    def feat(r):
        w = tokens(r["state"], drop_numbers) + tokens(r["question"], drop_numbers)
        big = [f"{a}|{b}" for a, b in zip(w, w[1:])]
        return hash_ids(w, (0,)) + hash_ids(big, (2 * HASH_BUCKETS,))

    feats = [feat(r) for r in train]
    labels = [cidx[gold_label(r)] for r in train]
    model = StateOnlyProbe(4 * HASH_BUCKETS, len(classes)).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(epochs):
        perm = torch.randperm(len(feats))
        for s in range(0, len(feats), 256):
            sel = perm[s:s + 256].tolist()
            flat, offs = [], [0]
            for i in sel:
                flat.extend(feats[i]); offs.append(len(flat))
            idx = torch.tensor(flat, dtype=torch.long, device=device)
            off = torch.tensor(offs[:-1], dtype=torch.long, device=device)
            tgt = torch.tensor([labels[i] for i in sel], dtype=torch.long, device=device)
            loss = torch.nn.functional.cross_entropy(model(idx, off), tgt)
            opt.zero_grad(); loss.backward(); opt.step()

    def predict(recs):
        if not recs:
            return 0.0, 0.0
        fs = [feat(r) for r in recs]
        flat, offs = [], [0]
        for f in fs:
            flat.extend(f); offs.append(len(flat))
        with torch.no_grad():
            idx = torch.tensor(flat, dtype=torch.long, device=device)
            off = torch.tensor(offs[:-1], dtype=torch.long, device=device)
            pred = model(idx, off).argmax(-1).tolist()
        hit = sum(1 for r, pi in zip(recs, pred) if classes[pi] == gold_label(r))
        in_set = sum(1 for r, pi in zip(recs, pred)
                     if classes[pi] in {c["label"] for c in r["candidates"]})
        return hit / len(recs), in_set / len(recs)

    return predict, len(classes)


# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="审计合成数据集")
    p.add_argument("--data", default="dataset/synth")
    p.add_argument("--split_dir", default=None, help="改用自定义数据目录（默认 --data）")
    p.add_argument("--model_eval", default=None,
                   help="真实模型的结果 JSON（含 metrics.accuracy），用于强制门禁")
    p.add_argument("--min_gap", type=float, default=0.25, help="门禁要求的领先点数")
    p.add_argument("--probe_epochs", type=int, default=25)
    p.add_argument("--probe_records", type=int, default=30000,
                   help="两个词法探针的训练条数上限（按来源分层抽样）；0 = 用全量")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    unbuffer_stdout()
    args = parse_args()
    torch.manual_seed(args.seed)
    root = args.split_dir or args.data

    print("========== 1. 读取 + manifest 核对 ==========")
    manifest = None
    mpath = os.path.join(root, "manifest.json")
    if os.path.exists(mpath):
        manifest = json.load(open(mpath, encoding="utf-8"))
        print(f"  gen_version {manifest['gen_version']}  tokenizer {manifest['tokenizer_sha1']}"
              f"  生成器 {manifest['generators']}")
        if manifest.get("ood_generators"):
            print(f"  整族留出 {manifest['ood_generators']} -> test_ood")
    else:
        print("  （无 manifest，跳过版本核对）")

    data = {}
    for s in SPLITS:
        p = os.path.join(root, f"{s}.jsonl")
        if os.path.exists(p):
            data[s] = load(p)
            if data[s]:
                print(f"  {s:11s} {len(data[s]):7d} 条")
            else:
                # 空 split 会被下面所有按 split 平均的地方除零。它现在唯一的成因是
                # 没配 `--ood_generators`，所以直接说清楚，而不是留一个 0.000 行
                # 让人以为探针在空集上「表现完美」。
                del data[s]
                print(f"  {s:11s}      空（未配置留出生成器，整族留出要等 G4/G6 就位）")
    if "train" not in data or not data["train"]:
        raise SystemExit(f"{root}/train.jsonl 不存在或为空")

    print("\n========== 2. schema 契约 ==========")
    problems = []
    for s, recs in data.items():
        check_contract(recs, s, problems)
    if manifest:
        for s, recs in data.items():
            bad = {r["gen_version"] for r in recs} - {manifest["gen_version"]}
            if bad:
                problems.append(f"{s}: 出现 manifest 之外的 gen_version {bad}")
    print(f"  检查 {sum(len(v) for v in data.values())} 条，问题 {len(problems)}")
    for x in problems[:5]:
        print(f"    {x}")

    print("\n========== 3. split 不相交（数据层复核） ==========")
    combos, overlap = check_split_overlap(data)
    for s in SPLITS:
        if combos[s]:
            print(f"  {s:11s} {len(combos[s]):5d} 个 (source, template, pool) 组合")
    print(f"  相交问题 {len(overlap)}")
    for x in overlap[:5]:
        print(f"    {x}")

    print("\n========== 4. 渲染充分性（R1 ②） ==========")
    from dataset.synth import build_all

    gens = {g.name: g for g in build_all(seed=manifest["seed"] if manifest else 0)}
    suff = collections.Counter()
    suff_bad = []
    for s, recs in data.items():
        for r in recs:
            g = gens.get(r["source"].split(":", 1)[-1])
            if g is None:
                continue          # 公开数据集 adapter 不在此审计范围内
            issues = g.sufficiency(r)
            suff[(s, r["schema"]["name"], bool(issues))] += 1
            if issues and len(suff_bad) < 6:
                suff_bad.append((r["id"], issues))
    for s in SPLITS:
        keys = [k for k in suff if k[0] == s]
        if not keys:
            continue
        for schema in sorted({k[1] for k in keys}):
            ok, ng = suff[(s, schema, False)], suff[(s, schema, True)]
            flag = "" if ng == 0 else f"  ← {ng} 条未通过"
            print(f"  {s:11s} {schema:18s} {ok}/{ok + ng}{flag}")
    for x in suff_bad:
        print(f"    {x}")

    print("\n========== 5. BoW 门禁（R2） ==========")
    train = stratify(data["train"], args.probe_records, args.seed)
    evals = {s: v for s, v in data.items() if s != "train" and v}

    def chance_of(recs):
        return sum(1.0 / len(r["candidates"]) for r in recs) / max(len(recs), 1)

    bow = bow_gate(train, args.device, args.probe_epochs)
    acc = {}
    for s, recs in evals.items():
        ch = chance_of(recs)
        acc[s] = bow(recs)
        print(f"  {s:11s} 词法上界 {acc[s]:.3f}   随机水平 {ch:.3f}   领先 {acc[s] - ch:+.3f}")
    print(f"  探针训练集 {len(train)}/{len(data['train'])} 条（按来源分层）；"
          f"特征 = 候选词 + 候选词在 state/question 中的共现")

    if "test_known" in evals:
        tk = evals["test_known"]
        print(f"\n  test_known 细分 —— **判读以这里为准**：总平均值会把两种完全不同的"
              f"难度混在一起。")
        def groups(recs, key):
            out = collections.defaultdict(list)
            for r in recs:
                out[key(r)].append(r)
            return out

        for label, key in (("语言", lambda r: r["meta"]["language"]),
                           ("来源", lambda r: r["source"].split(":")[-1]),
                           ("来源×语言", lambda r: f"{r['source'].split(':')[-1]}/"
                                                  f"{r['meta']['language']}")):
            for k, sub in sorted(groups(tk, key).items()):
                ch = chance_of(sub)
                a = bow(sub)
                print(f"    {label:8s} {k:20s} n={len(sub):5d}  词法上界 {a:.3f}  "
                      f"随机 {ch:.3f}  领先 {a - ch:+.3f}")
        print("    候选是英文标识符（`cancel_order`）而 state 是中文（`取消`+`订单`）时，")
        print("    词法探针**必然**失手 —— 那几行低不代表任务难，只代表字符串匹配")
        print("    跨不过语言。判门禁要看每一行，不能只看平均。")

    gate_ok = None
    if args.model_eval and os.path.exists(args.model_eval):
        res = json.load(open(args.model_eval, encoding="utf-8"))
        # `eval_harness.py` 落的是 `metrics: {all, calibration, <provenance>...}`，
        # 整体准确率在 `all` 里。**要下钻一层**：直接 `res["metrics"].get("accuracy")`
        # 恒为 None，于是 `f"{None:.3f}"` 抛 TypeError —— 也就是说这条门禁在真的
        # 拿到一份评测 JSON 之前从来没跑通过，只在没给 `--model_eval` 时打印过
        # "只报数不判定"。缺了这一步，门禁的形式在、判定不在。
        m = res.get("metrics", res)
        if "all" in m:
            m = m["all"]
        model_acc = m.get("accuracy")
        if model_acc is None:
            raise SystemExit(
                f"{args.model_eval} 里找不到整体准确率（期望 metrics.all.accuracy）；"
                f"实际顶层键 {sorted(res)[:8]}，metrics 的键 {sorted(res.get('metrics', {}))[:8]}")
        print(f"\n  真实模型 test_known 准确率 {model_acc:.3f}（{args.model_eval}）")
        tk_acc = acc.get("test_known", 0.0)
        gate_ok = model_acc - tk_acc >= args.min_gap
        print(f"  模型 − 词法上界 = {model_acc - tk_acc:+.3f}，门槛 ≥{args.min_gap:.2f}  "
              f"→ {'通过' if gate_ok else '**未通过：生成器过度模板化**'}")
        print("  注意：该判定用的是**平均值**。若上表里有哪一行的词法上界已经很高，")
        print("  平均门槛就是被其它行抬过去的 —— 那一行才是真正需要加难的地方。")
    else:
        print(f"\n  （未提供 --model_eval，门禁只报数不判定。训练出模型后必须回来跑一次："
              f"\n   真实模型必须比上面这个上界高出 ≥{args.min_gap:.2f}，否则先多样化生成器。）")

    print("\n========== 6. 答案泄漏探针：只读 state + question ==========")
    predict, n_class = state_only_probe(train, args.device, args.probe_epochs)
    predict_wn, _ = state_only_probe(train, args.device, args.probe_epochs,
                                     drop_numbers=True)
    print(f"  两列对比：**全部词**（含数字串）与**去数字**。差值就是探针不靠算术、"
          f"纯靠背数字串拿到的分。类别空间 {n_class}")
    for s in ("val", "calib", "test_known", "test_ood"):
        if s not in evals:
            continue
        ch = chance_of(evals[s])
        a, inset = predict(evals[s])
        w, _ = predict_wn(evals[s])
        print(f"  {s:11s} 随机 {ch:.3f}   全部词 {a:.3f}   去数字 {w:.3f}"
              f"   预测落在候选集内 {inset:.3f}")
    print("  逐来源（test_known）—— 判读看**去数字**那一列：")
    print("    · 去数字仍接近随机 ⇒ 没泄漏（算术型任务本该如此：banking_balance）")
    print("    · 去数字也高      ⇒ 措辞把答案写出来了，该加难该生成器")
    print("    · 只有全部词高    ⇒ 输入空间太离散，探针背了一张查找表。")
    print("      这不是泄漏，但记在任务难度上 —— refund_policy 就属于这一类。")
    print("    · tool_router 是例外：把意图词映射到工具名**就是**它的任务，")
    print("      两列都高是设计如此，不是缺陷。")
    if "test_known" in evals:
        for src in sorted({r["source"] for r in evals["test_known"]}):
            sub = [r for r in evals["test_known"] if r["source"] == src]
            if not sub:
                continue
            ch = chance_of(sub)
            a, _ = predict(sub)
            w, _ = predict_wn(sub)
            print(f"    {src.split(':')[-1]:16s} n={len(sub):5d}  随机 {ch:.3f}  "
                  f"全部词 {a:.3f}（{a - ch:+.3f}）  去数字 {w:.3f}（{w - ch:+.3f}）")

    print("\n========== 结论 ==========")
    hard_fail = len(problems) + len(overlap) + len(suff_bad)
    print(f"  契约 {len(problems)} + split {len(overlap)} + 渲染充分性 {len(suff_bad)} "
          f"= {hard_fail} 个硬问题")
    if gate_ok is False:
        print("  **BoW 门禁未通过** —— 按方案约定，此时不得报告任何模型指标。")
    elif hard_fail:
        print("  **有硬问题** —— 修好之前不要往下训。")
    else:
        print("  硬检查全过。" + ("" if gate_ok is None else " BoW 门禁通过。"))


if __name__ == "__main__":
    main()
