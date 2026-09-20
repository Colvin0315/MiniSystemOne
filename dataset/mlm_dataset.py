r"""
Stage 1 的 MLM 数据集：span masking + 80/10/10，与决策格式**完全分开**。

为什么不把预训练和决策训练混在一个 sampler 里：

1. **结构 token 必须永不被破坏。** `<sep>`、`<trunc>`、`<yes>`、`<no>` 这些在决策
   格式里是承重的；MLM 若 mask 它们，就是在教模型"结构可以猜"。但结构 token 一旦
   永不 mask，它们又成了 MLM 里的常数，白白占住 15% 的 mask 预算。两个目标直接
   冲突 —— 分开阶段就不冲突了。
2. **长度分布差太远**（本文档 ~150–340 token，决策打包 ~1024）。塞进一个 sampler，
   每个 batch 的 padding 浪费会按较长的那个算。
3. **最优学习率与序列长度都不同**（决策阶段要长序列以容纳多候选）。

span 而不是 whole-word：语料是中英混排，tokenizer 是 ByteLevel + GPT-2 正则，
`\p{L}+` 会把一整串汉字匹配成**单个** pretoken —— whole-word 会一次 mask 掉整个
句子片段，对无空格语言也本就定义不清。span 在两种语言上都优雅退化。
"""
import random
import sys

import numpy as np
import torch

sys.path.append(".")

# 永不 mask 的位置。全部 11 个特殊 token 都在此列：它们没有内容语义，mask 掉
# 只会让模型去猜一个不可猜的结构标记（`<unk>` 尤其退化 —— 预测"未知"无意义）。
NEVER_MASK = frozenset(range(11))

MASK_RATIO = 0.15            # BERT 标准。语料不大，更高比例会让上下文变得不干净
MAX_SPAN = 4
SPAN_KEEP_P = 0.8            # 见 _span_len：均值落在 3
MASK_PROB, RANDOM_PROB = 0.8, 0.1      # 剩下 0.1 是"保持不变"


def _span_len(rng):
    """几何分布截断到 [1, MAX_SPAN]，实测均值 2.95 ≈ 方案要求的 3。

    `while rng.random() < SPAN_KEEP_P` 的语义是"以 0.8 的概率继续延长"，
    所以 P(l=1)=0.2、P(l=2)=0.16、P(l=3)=0.128、P(l=4)=0.8³=0.512。
    反过来（以 0.2 继续）得到的是均值 1.48 的分布，全是短 span —— 数值上
    差一点点，效果上等于没做 span masking。
    """
    length = 1
    while length < MAX_SPAN and rng.random() < SPAN_KEEP_P:
        length += 1
    return length


def span_mask(ids, rng, mask_id, vocab_size, ratio=MASK_RATIO):
    """对一条序列做 span masking，返回 (input_ids, labels)。

    labels 在未破坏处为 -100（`cross_entropy` 的 ignore_index），所以损失只算被
    mask 的位置。**先按总长度定预算，再挑 span 直到花完** —— 而不是"每个 span
    恰好 15%"：后者会让短序列的 mask 数在四舍五入后剧烈波动。
    """
    n = len(ids)
    target = max(1, int(round(n * ratio)))
    input_ids = list(ids)
    labels = [-100] * n
    used = [False] * n
    n_masked = 0

    # 守卫上限：候选位置可能几乎全是 NEVER_MASK（合成长文档），若无上限会空转。
    for _ in range(4 * n):
        if n_masked >= target:
            break
        start = rng.randrange(n)
        if used[start] or ids[start] in NEVER_MASK:
            continue
        end = start
        want = _span_len(rng)
        while (end < n and end - start < want
               and not used[end] and ids[end] not in NEVER_MASK):
            end += 1
        if end == start:
            continue
        for i in range(start, end):
            used[i] = True
            labels[i] = ids[i]
            # 80/10/10：保持预训练与决策阶段的一致 —— `<mask>` 在决策训练里
            # 永不出现，全用 `<mask>` 会让编码器过拟合到一个随后消失的 token。
            r = rng.random()
            if r < MASK_PROB:
                input_ids[i] = mask_id
            elif r < MASK_PROB + RANDOM_PROB:
                input_ids[i] = rng.randrange(vocab_size)
            # else: 原样保留
        n_masked += end - start

    return input_ids, labels


class MLMDataset(torch.utils.data.Dataset):
    """把语料**预分词一次**成一条扁平 id 数组，再按 max_len 切块。

    预分词而不是在 `__getitem__` 里现分词，是因为 MLM 要跑多个 epoch：现分词会
    每个 epoch 重付一次 BPE 的代价（400k 篇 × 8 个 epoch 是分钟级的纯浪费）。
    扁平数组用 uint16 存（vocab 6400 < 65536），100M token 约 200 MB，切块是
    零拷贝切片。

    mask 由 `(seed, epoch, index)` 决定，所以同一 epoch 内可复现、跨 epoch 变化 ——
    每个 epoch 看到不同的 mask 位置，但重跑实验得到完全相同的结果。
    """

    def __init__(self, tokenizer, corpus, max_len=1024, seed=0, min_len=None):
        self.max_len = max_len
        self.seed = seed
        self.epoch = 0
        self.mask_id = tokenizer.convert_tokens_to_ids("<mask>")
        self.sep_id = tokenizer.convert_tokens_to_ids("<sep>")
        self.vocab_size = tokenizer.vocab_size

        min_len = min_len or max_len // 2
        flat = []
        n_docs = 0
        for doc in corpus:
            enc = tokenizer(doc, add_special_tokens=False)["input_ids"]
            if len(enc) < 8:
                continue
            n_docs += 1
            flat.extend(enc)
            # 文档之间插 `<sep>`：它已在词表里，且让块边界落在文档边界上时
            # 仍有可学结构（否则长文档会被硬切，模型学到的是半句话）。
            flat.append(self.sep_id)
        self.flat = np.asarray(flat, dtype=np.uint16)
        self.n_chunks = max(1, len(self.flat) // max_len)
        print(f"  MLM 语料：{n_docs} 篇 → {len(self.flat)/1e6:.1f}M token "
              f"→ {self.n_chunks} 个 {max_len} 长块"
              f"（丢弃尾部 {len(self.flat) % max_len} token）")

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return self.n_chunks

    def __getitem__(self, index):
        if index >= self.n_chunks:
            raise IndexError(index)
        ids = self.flat[index * self.max_len:(index + 1) * self.max_len].tolist()
        # seed 只由 (seed, epoch, index) 决定，与 batch 组成无关 —— 换 batch_size
        # 或加梯度累积都不会改变任何一条样本的 mask 结果。
        rng = random.Random((self.seed << 40) ^ (self.epoch << 20) ^ index)
        input_ids, labels = span_mask(ids, rng, self.mask_id, self.vocab_size)
        return (torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(labels, dtype=torch.long))


def collate_mlm(batch):
    """返回 dict，可直接喂给 `MiniSystemOneForMaskedLM.forward`。

    **没有 attention mask，也没有 seg_id。** 所有块都被切成等长 `max_len`，尾部
    不足一块的直接丢弃，所以序列里不存在 PAD —— 而实测无 mask 是 SDPA 最快的
    路径（有 mask 要多花约 35% 时间）。`seg_id=None` 让 Encoder 走纯文本分支
    （不加 segment embedding，位置用连续 arange），这正是 MLM 想要的。

    这里因此**不需要** padding mask 的构造逻辑。变长打包的场景（AR baseline 的
    前缀）才需要 `serialize.build_padding_mask`。
    """
    return {
        "input_ids": torch.stack([b[0] for b in batch]),
        "labels": torch.stack([b[1] for b in batch]),
    }
