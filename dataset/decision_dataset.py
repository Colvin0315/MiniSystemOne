"""
决策数据集 —— 读 `docs/DATA_SCHEMA.md` 冻结的 schema，产出打包好的训练样本。

三件事在这里发生，每件都对应一个被测量的东西：

1. **state 截断**：段落感知丢弃 → 头尾切片。`state_sections` 带 priority，所以
   结构化 state 可以**语义截断**而非盲目切。语料是程序生成的，所以模型在测试集上
   永远看到完整 state —— 截断策略的实际代价**没有被测过**（本仓库没有任何评测
   产出这条曲线），所以它是一个已知缺口，不是已验证的结论。

2. **K 子采样**（仅训练）：`K ~ Uniform{2..32}`，但 `p*_k ≥ keep_p_min` 的候选恒被保留。
   一举两得 —— 免费的 K 增广，以及变 K 评测所需的超集。目标重新归一化并记
   `renormalized`，让下游知道它不再是世界的真实分布。

3. **K 分桶采样**：固定 `K_max` padding 会浪费约一半候选槽位。按 K 分组、组内按
   长度细分，保证每 batch K 一致且长度差 < 一个桶宽。

**`__getitem__` 对 (index, epoch) 完全确定**（自带 rng，不用全局随机）。这不是洁癖：
分桶采样器要在不开 tokenizer 的前提下复现同一个 K，否则 batch 里 K 不一致，
`cand_mask` 会退化成大量 padding 槽，而 mask 的正确性是这个项目里最贵的东西。
"""
import json
import random
from collections import defaultdict

import torch
from torch.utils.data import Dataset, Sampler

from model.serialize import encode_text, pack_example, collate_packed, truncate_head_tail

# 分桶采样的长度细分粒度（token）。桶越窄 padding 越省，但每个桶里的样本越少、
# 批次内越同质。32 是实测的折中：padding 浪费 <4%，桶内仍有数百样本。
LENGTH_BUCKET = 32


# ---------------------------------------------------------------------------
def fit_state(sections, budget, tokenizer, trunc_id, nl_ids):
    """把 state 塞进 budget 个 token。返回 token id 列表。

    两级：① 按 priority 从小到大**整段丢弃**，重渲染；② 仍然超出才头尾切片。
    第一级是这里存在的理由 —— state 是结构化的，丢"营销备注"比丢"账户状态"便宜得多，
    而盲目头尾切会把两者一视同仁。
    """
    enc = [(s.get("priority", 0), encode_text(tokenizer, s["text"])) for s in sections]

    def render(kept):
        ids = []
        for i in kept:
            if ids:
                ids.extend(nl_ids)
            ids.extend(enc[i][1])
        return ids

    # ① 按 priority 从小到大整段丢弃，但**至少留下最后一段**。全丢光等于把 state
    #    变成空串：模型连问题在问什么都无从判断，那比截断更糟。剩下的交给 ②。
    kept = list(range(len(enc)))
    for i in sorted(range(len(enc)), key=lambda j: enc[j][0]):
        if len(render(kept)) <= budget or len(kept) == 1:
            break
        kept = [k for k in kept if k != i]

    ids = render(kept)
    # ② 留下的段仍然超预算：头尾切片（近期偏置 —— 当前处境在末尾）
    if len(ids) > budget:
        ids = truncate_head_tail(ids, budget, trunc_id)
    return ids


def subsample_candidates(p, k_draw, keep_p_min, k_max, rng):
    """按 K 子采样候选。返回 (kept 下标, 重归一化后的目标, 是否重归一化过)。

    `p*_k ≥ keep_p_min` 的候选恒被保留 —— 训练里丢掉一个真实概率 5% 以上的答案，
    等于教模型它对某些明显可能的选项可以完全无视。
    """
    n = len(p)
    if k_draw >= n and n <= k_max:
        return list(range(n)), list(p), False

    mandatory = [i for i in range(n) if p[i] >= keep_p_min]
    if len(mandatory) > k_max:
        # 超集里高概率候选本身就超过训练上限（calendar_slot 的近并列会有）。
        # 训练必须守住 K ≤ k_max 的显存预算，所以这里只能降采样 —— 取概率最高的，
        # 并把这件事留在返回的标志里让上游能统计。
        mandatory = sorted(mandatory, key=lambda i: -p[i])[:k_max]

    mandatory_set = set(mandatory)
    rest = [i for i in range(n) if i not in mandatory_set]
    k = min(max(k_draw, len(mandatory)), n)
    need = k - len(mandatory)
    chosen = mandatory + (rng.sample(rest, need) if need > 0 else [])
    # 再打乱一次：mandatory 是按概率选的，保持原序会让高概率候选系统性地靠前
    rng.shuffle(chosen)

    mass = sum(p[i] for i in chosen)
    renorm = len(chosen) < n
    q = [p[i] / mass for i in chosen] if mass > 0 else [1.0 / len(chosen)] * len(chosen)
    return chosen, q, renorm


# ---------------------------------------------------------------------------
class DecisionDataset(Dataset):
    """一条 jsonl 一个 split。训练时开 K 子采样与截断，评测时用全集。"""

    def __init__(self, path, tokenizer, max_len=1024, k_min=2, k_max=32,
                 keep_p_min=0.05, augment_k=True, seed=0, epoch=0,
                 gen_version=None, limit=0):
        self.path = path
        self.tok = tokenizer
        self.max_len = max_len
        self.k_min, self.k_max = k_min, k_max
        self.keep_p_min = keep_p_min
        self.augment_k = augment_k
        self.seed = seed
        self.epoch = epoch
        # 评测期开关，训练一律为 None / 1.0。放这里而不是各写一个包装数据集：
        # 包装会复制 __getitem__ 的全部逻辑，而"候选顺序"与"state 预算"恰好是
        # 鲁棒性套件①⑦要单独拧的两个旋钮，复制一份必然与主路径漂开。
        self.perm_seed = None   # 非 None 时对候选额外做一次确定性重排（套件①）
        self.state_scale = 1.0  # state 预算的缩放（套件⑦截断敏感性）

        self.pad_id = tokenizer.convert_tokens_to_ids("<pad>")
        self.sep_id = tokenizer.convert_tokens_to_ids("<sep>")
        self.trunc_id = tokenizer.convert_tokens_to_ids("<trunc>")
        self.nl_ids = encode_text(tokenizer, "\n")

        self.records = []
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                rec = json.loads(line)
                if gen_version is not None and rec.get("gen_version") != gen_version:
                    raise ValueError(
                        f"{rec.get('id')} 的 gen_version={rec.get('gen_version')} "
                        f"与要求的 {gen_version} 不符；测试集必须用冻结版本重生成"
                    )
                self.records.append(rec)
        # 读完整份再截断，而不是提前 break：`gen_version` 断言必须覆盖每一行，
        # 否则一个混版本的文件能在前 N 条之外悄悄溜过去。
        if limit:
            self.records = self.records[:limit]

    # -- 采样器要用的两个键：都**不需要 tokenizer**，所以分桶是廉价的 ------------
    def bucket_key(self, index):
        """K 桶。用**实际会用的 K**，不是 K_full，否则同批 K 不一致。"""
        return self._draw_k(index)[1] if self.augment_k else len(self.records[index]["candidates"])

    def length_key(self, index):
        return self.records[index].get("meta", {}).get("approx_tokens", 256) // LENGTH_BUCKET

    def _seed_for(self, index):
        """确定性种子。避免用全局 rng —— 采样器与 __getitem__ 必须算出同一个 K。"""
        return (self.seed * 1000003 + self.epoch * 10007 + index) & 0x7FFFFFFF

    def _draw_k(self, index):
        """返回 (K_full, 实际 K)。`augment_k=False` 或 score 时恒为全集。

        **Score 一律不参与 K 增广。** 它的候选**就是**等级轴：序数 CDF 损失
        （`is_ord` 门控那条）与 `ordinal_mae` 都定义在这条轴上。丢掉中间的等级会让
        `cumsum` 把两个不相邻的等级当成相邻 —— 不报错，只让序数信号变成一句错话。
        这里而不是在 `__getitem__` 里挡，是因为 `bucket_key` 也调它：两处若各判一次，
        采样器就会按一个用不到的 K 分组，批次里全是 padding 槽。
        """
        rec = self.records[index]
        n = len(rec["candidates"])
        if (not self.augment_k or n <= self.k_min
                or rec["schema"]["primitive"] == "score"):
            return n, n
        rng = random.Random(self._seed_for(index))
        return n, min(rng.randint(self.k_min, self.k_max), n)

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.records)

    # -----------------------------------------------------------------------
    def __getitem__(self, index):
        rec = self.records[index]
        rng = random.Random(self._seed_for(index))

        cands = rec["candidates"]
        p_full = rec["target"]["p"]
        levels_full = [c.get("meta", {}).get("level") for c in cands]

        if self.augment_k:
            k_draw = self._draw_k(index)[1]
            order, target, renorm = subsample_candidates(
                p_full, k_draw, self.keep_p_min, self.k_max, rng)
        else:
            order, target, renorm = list(range(len(cands))), list(p_full), False

        # 套件①：只看候选顺序的影响，其余一字不改。生成时已重排过一次，这里再排一次
        # 是为了拿到「同一道题、不同呈现顺序」的配对样本 —— 变的是顺序，不是内容。
        if self.perm_seed is not None:
            pos = list(range(len(order)))
            random.Random(f"{self.perm_seed}|{index}").shuffle(pos)
            order = [order[j] for j in pos]
            target = [target[j] for j in pos]

        cand_ids = []
        levels = []
        for i in order:
            ids = encode_text(self.tok, cands[i]["text"]) or [self.sep_id]
            cand_ids.append(ids)
            levels.append(-1 if levels_full[i] is None else float(levels_full[i]))

        question_ids = encode_text(self.tok, rec["question"])

        # state 的预算 = 总预算 − 固定开销。先算固定部分，剩下的全给 state。
        fixed = len(question_ids) + 2 + sum(len(c) + 1 for c in cand_ids)
        # 套件⑦要的是「把截断启发式变成实测曲线」，所以缩放的是**给 state 的预算**
        # 而不是 max_len —— 后者会顺带砍掉候选，量到的就不是截断敏感性了。
        state_budget = max(int((self.max_len - fixed) * self.state_scale), 8)

        sections = rec.get("state_sections")
        if sections:
            state_ids = fit_state(sections, state_budget, self.tok, self.trunc_id, self.nl_ids)
        else:
            state_ids = encode_text(self.tok, rec["state"])
            if len(state_ids) > state_budget:
                state_ids = truncate_head_tail(state_ids, state_budget, self.trunc_id)

        ids, seg, cid, spans = pack_example(state_ids, question_ids, cand_ids, self.sep_id)

        return {
            "input_ids": ids,
            "seg_id": seg,
            "cand_id": cid,
            "cand_spans": spans,
            "target": target,
            "levels": levels,
            "primitive": rec["schema"]["primitive"],
            # 下面这些不进张量，只供 per-provenance 指标与误差分析
            "id": rec.get("id"),
            "source": rec.get("source"),
            "provenance": rec["target"].get("provenance"),
            # 标注数用于假设标注模型下的 MC 参考量，不参与训练。
            "counts": rec["target"].get("counts"),
            "gen_version": rec.get("gen_version"),
            "template_id": rec.get("meta", {}).get("template_id"),
            "entity_pool": rec.get("meta", {}).get("entity_pool"),
            "K_full": len(cands),
            "K_used": len(cand_ids),
            "renormalized": renorm,
            "labels": [cands[i].get("label") for i in order],
        }


# ---------------------------------------------------------------------------
class CandidateBucketSampler(Sampler):
    """每 batch K 一致、长度差 < 一个桶宽的批采样器。

    两段分组：先按 K（**精确**，不需要 tokenizer），再按 `approx_tokens // LENGTH_BUCKET`。
    组内打乱保证组成不固定，组间打乱保证 epoch 顺序不固定。
    """

    def __init__(self, dataset, batch_size, shuffle=True, seed=0, drop_last=False):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = epoch
        self.dataset.set_epoch(epoch)

    def __len__(self):
        return len(list(self._batches()))

    def _batches(self):
        by_k = defaultdict(lambda: defaultdict(list))
        for i in range(len(self.dataset)):
            by_k[self.dataset.bucket_key(i)][self.dataset.length_key(i)].append(i)

        rng = random.Random(self.seed * 7919 + self.epoch)
        batches = []
        for k in sorted(by_k):
            idxs = []
            for length_key in sorted(by_k[k]):
                group = by_k[k][length_key]
                if self.shuffle:
                    rng.shuffle(group)
                idxs.extend(group)
            for s in range(0, len(idxs), self.batch_size):
                chunk = idxs[s:s + self.batch_size]
                if len(chunk) == self.batch_size or not self.drop_last:
                    batches.append(chunk)
        if self.shuffle:
            rng.shuffle(batches)
        return batches

    def __iter__(self):
        yield from self._batches()


def collate_decision(examples, pad_id=0, sep_id=3, device=None):
    """把一批样本对齐成 batch，并带上 per-provenance 指标需要的元信息。"""
    batch = collate_packed(examples, pad_id=pad_id, sep_id=sep_id, device=device)
    for key in ("provenance", "source", "template_id", "entity_pool", "id",
                "K_full", "K_used", "renormalized", "labels", "counts", "gen_version"):
        batch[key] = [e.get(key) for e in examples]
    return batch
