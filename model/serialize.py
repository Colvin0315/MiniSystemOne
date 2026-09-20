"""
MiniSystemOne 唯一的规范序列化。

数据加载与推理都必须经过这里，否则打包格式与 mask 契约会漂移。
决策模型没有 chat template —— 规范序列化是这段代码，不是 Jinja 字符串。

打包布局：
    pos:  [ state tokens ][SEP][ question tokens ][SEP][ c1 ][SEP][ c2 ][SEP] ...
    seg:   2 2 2 2 2 2 2   2      3 3 3 3 3 3      3    4 4    4    4 4    4

每个 <sep> 继承它所终止的 span 的 seg_id，并计入该 span 的 pooling
（BERT 的 [SEP]-as-summary 约定）。
"""
import math
import torch

from model.model_system_one import (
    SEG_PAD, SEG_PLAIN, SEG_STATE, SEG_QUESTION, SEG_CANDIDATE, NEG_INF,
)

STRUCT_SEGMENTS = (SEG_STATE, SEG_QUESTION, SEG_CANDIDATE)


def pack_example(state_ids, question_ids, candidates_ids, sep_id):
    """把单个 (state, question, candidates) 打包成三条等长的 id 序列。

    返回 (input_ids, seg_id, cand_id, cand_spans)
      cand_id  : 每个位置所属的候选下标，非候选位置为 -1
      cand_spans: [(start, end), ...]，end 为开区间，含尾随 <sep>
    """
    ids, seg, cid, spans = [], [], [], []

    ids.extend(state_ids)
    seg.extend([SEG_STATE] * len(state_ids))
    cid.extend([-1] * len(state_ids))
    ids.append(sep_id); seg.append(SEG_STATE); cid.append(-1)

    ids.extend(question_ids)
    seg.extend([SEG_QUESTION] * len(question_ids))
    cid.extend([-1] * len(question_ids))
    ids.append(sep_id); seg.append(SEG_QUESTION); cid.append(-1)

    for k, cand_ids in enumerate(candidates_ids):
        start = len(ids)
        ids.extend(cand_ids)
        seg.extend([SEG_CANDIDATE] * len(cand_ids))
        cid.extend([k] * len(cand_ids))
        ids.append(sep_id); seg.append(SEG_CANDIDATE); cid.append(k)
        spans.append((start, len(ids)))

    return ids, seg, cid, spans


def pack_plain(ids):
    """纯文本（MLM / AR baseline）。seg 全为 PLAIN，无候选结构。"""
    return ids, [SEG_PLAIN] * len(ids), [-1] * len(ids), []


def build_prefix_mask(seg_id):
    """state ∪ question 的位置掩码，供 AttnPool 使用。返回 bool (B, S)。"""
    return (seg_id == SEG_STATE) | (seg_id == SEG_QUESTION)


def has_structure(seg_id):
    """序列是否包含决策结构段。纯文本返回 False。"""
    return bool((seg_id >= SEG_STATE).any())


def build_attn_mask(seg_id, cand_id=None, crosstalk=False, prefix_blocked=True,
                    dtype=torch.float32):
    """构造 float additive attention mask，形状 (B, 1, S, S)。

    许可表（allowed[query][key]）：
                  STATE   QUESTION  CANDIDATE
        STATE     full    -         -
        QUESTION  full    full      -
        CANDIDATE full    full      crosstalk ? full : 仅自身 span

    返回 None 表示可以用无 mask 的快路径（纯文本，或全局放行）。

    必须用 float 而非 bool：float additive mask 走 EFFICIENT 内核，比 bool mask
    更快更省显存；一旦掉进 MATH 后端会慢一个量级。**这两个数来自 torch
    2.5.1+cu121 的实测**（float 0.886 ms / 58.7 MB，bool 1.028 ms / 75.5 MB，
    MATH 14.4 ms / 998 MB）。SDPA 的后端选择随版本变，而 requirements.txt 钉的是
    2.6.0+cu124 —— 换版本后请用 `python scripts/smoke_test.py --bench_mask` 重测，
    不要默认这个结论仍然成立。
    """
    if not has_structure(seg_id):
        return None
    if crosstalk and not prefix_blocked and not (seg_id == SEG_PAD).any():
        return None

    bsz, seq_len = seg_id.shape

    q_state = (seg_id == SEG_STATE).unsqueeze(-1)
    q_qst = (seg_id == SEG_QUESTION).unsqueeze(-1)
    q_cand = (seg_id == SEG_CANDIDATE).unsqueeze(-1)
    k_state = (seg_id == SEG_STATE).unsqueeze(1)
    k_qst = (seg_id == SEG_QUESTION).unsqueeze(1)
    k_cand = (seg_id == SEG_CANDIDATE).unsqueeze(1)
    k_real = (seg_id != SEG_PAD).unsqueeze(1)

    allowed = ((q_state & k_state)
               | (q_qst & (k_state | k_qst))
               | (q_cand & (k_state | k_qst)))

    if crosstalk:
        allowed = allowed | (q_cand & k_cand)
    else:
        if cand_id is None:
            raise ValueError("crosstalk=False 需要 cand_id 来构造 block-diagonal mask")
        same_cand = ((cand_id.unsqueeze(-1) == cand_id.unsqueeze(1))
                     & (cand_id.unsqueeze(-1) >= 0))
        allowed = allowed | (q_cand & k_cand & same_cand)

    if prefix_blocked:
        # 前缀不可 attend 到候选。这是 chunk 不变性的前提。
        allowed = allowed & ~(q_state & k_cand) & ~(q_qst & k_cand)

    allowed = allowed & k_real

    # PAD query 行必须至少有一个可 attend 的 key，否则整行被 mask。
    # 输出会被丢弃，且 attention 按 query 行独立，所以放行全部真实 key 是安全的。
    q_pad = (seg_id == SEG_PAD).unsqueeze(-1)
    allowed = torch.where(q_pad, k_real.expand_as(allowed), allowed)

    return torch.where(allowed, 0.0, NEG_INF).to(dtype).unsqueeze(1)


def build_padding_mask(seg_id, dtype=torch.float32):
    """只屏蔽 PAD 的 additive mask，形状 (B, 1, 1, S)。纯文本路径（MLM、AR 前缀）用。

    形状必须是 (B,1,1,S) 而不是 (B,S)：SDPA 把 (B,S) 当作 (B,1,1,S) 广播**只在
    S 恰好等于最后一维时**才对，一旦 query 长度与 key 长度不同就静默广播成
    完全错误的形状 —— 所以这里显式补维度，不依赖调用方。

    无 PAD 时返回 None，让调用方走无 mask 的快路径（无 mask 实测最快）。
    """
    if not (seg_id == SEG_PAD).any():
        return None
    return torch.where(seg_id == SEG_PAD, NEG_INF, 0.0).to(dtype)[:, None, None, :]


def build_causal_mask(seq_len, dtype=torch.float32, device="cpu"):
    """标准下三角 additive mask，仅供需要显式 mask 的 AR 路径使用。"""
    m = torch.full((seq_len, seq_len), NEG_INF, dtype=dtype, device=device)
    m = torch.triu(m, diagonal=1)
    return m.unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# 分词封装
# ---------------------------------------------------------------------------

def encode_text(tokenizer, text, max_tokens=None):
    """纯文本编码，不加特殊 token（结构 token 由 pack_example 插入）。"""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if max_tokens is not None:
        ids = ids[:max_tokens]
    return ids


def truncate_head_tail(ids, budget, trunc_id):
    """头尾切片：0.3 头 + 0.7 尾，中间一个 <trunc>。

    近期偏置对 agent trace 是正确先验 —— 当前处境在末尾。
    """
    if len(ids) <= budget:
        return ids
    keep = max(budget - 1, 1)
    head = int(keep * 0.3)
    tail = keep - head
    return ids[:head] + [trunc_id] + (ids[-tail:] if tail > 0 else [])


def build_ar_target(tokenizer, answer_text, ans_open_id, ans_close_id):
    """AR baseline 的目标区：<ans> ... </ans>。返回 (ids, label_mask)。"""
    body = tokenizer(answer_text, add_special_tokens=False)["input_ids"]
    ids = [ans_open_id] + body + [ans_close_id]
    # 只在 answer 区域算 loss；<ans> 本身算入（模型需要学会开启答案区）
    mask = [1] * len(ids)
    return ids, mask


def serialize_decision(tokenizer, state, question, candidates,
                       sep_id, max_state_tokens=None, trunc_id=None):
    """把一条决策样本编码成打包后的 id 序列。

    candidates: list[str]，候选文本列表（顺序即呈现顺序）。
    """
    state_ids = encode_text(tokenizer, state)
    if max_state_tokens is not None and len(state_ids) > max_state_tokens:
        if trunc_id is None:
            state_ids = state_ids[:max_state_tokens]
        else:
            state_ids = truncate_head_tail(state_ids, max_state_tokens, trunc_id)
    question_ids = encode_text(tokenizer, question)
    cand_ids = [encode_text(tokenizer, c) or [sep_id] for c in candidates]
    return pack_example(state_ids, question_ids, cand_ids, sep_id)


# ---------------------------------------------------------------------------
# collate / 批处理
# ---------------------------------------------------------------------------

def pad_to_multiple(length, multiple=8):
    return int(math.ceil(length / multiple) * multiple)


def collate_packed(examples, pad_id, sep_id, dtype=torch.float32, device=None):
    """把一批 pack_example 的输出对齐成一个 batch。

    每个 example 是 dict，至少含 input_ids / seg_id / cand_id / cand_spans，
    以及可选的 target / level_idx / is_ord。

    返回的 K 是这一批里的最大候选数，不足者用 cand_mask 标记。
    """
    bsz = len(examples)
    max_len = pad_to_multiple(max(len(e["input_ids"]) for e in examples))
    max_k = max(len(e["cand_spans"]) for e in examples)

    input_ids = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    seg_id = torch.zeros((bsz, max_len), dtype=torch.long)
    cand_id = torch.full((bsz, max_len), -1, dtype=torch.long)
    cand_span = torch.full((bsz, max_k, 2), -1, dtype=torch.long)
    cand_mask = torch.zeros((bsz, max_k), dtype=torch.bool)
    target = torch.zeros((bsz, max_k), dtype=torch.float32)
    level_idx = torch.full((bsz, max_k), -1, dtype=torch.long)
    is_ord = torch.zeros(bsz, dtype=torch.float32)

    for i, e in enumerate(examples):
        n = len(e["input_ids"])
        input_ids[i, :n] = torch.tensor(e["input_ids"], dtype=torch.long)
        seg_id[i, :n] = torch.tensor(e["seg_id"], dtype=torch.long)
        cand_id[i, :n] = torch.tensor(e["cand_id"], dtype=torch.long)
        k = len(e["cand_spans"])
        for j, (s, t) in enumerate(e["cand_spans"]):
            cand_span[i, j] = torch.tensor([s, t], dtype=torch.long)
        cand_mask[i, :k] = True
        if "target" in e:
            target[i, :k] = torch.tensor(e["target"], dtype=torch.float32)
        if "levels" in e:
            level_idx[i, :k] = torch.tensor(e["levels"], dtype=torch.long)
        is_ord[i] = 1.0 if e.get("primitive") == "score" else 0.0

    batch = {
        "input_ids": input_ids,
        "seg_id": seg_id,
        "cand_id": cand_id,
        "cand_span": cand_span,
        "cand_mask": cand_mask,
        "target": target,
        "level_idx": level_idx,
        "is_ord": is_ord,
        "prefix_mask": build_prefix_mask(seg_id),
    }
    if device is not None:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    return batch
