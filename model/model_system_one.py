"""
MiniSystemOne 模型定义 —— 决策原生模型，单文件，风格对齐 MiniMind。

核心主张：
  state + typed questions  ->  带校准概率的类型化决策，单次并行前向。

与 MiniMind 的关键差异（详见 docs/DESIGN.md）：
  1. 双向编码器，默认非 causal。决策不需要自回归生成。
  2. RoPE 按 position_ids 索引（不是 MiniMind 的连续切片），因为打包格式
     需要任意位置（分块起点、前缀偏移）。
  3. 用 float additive attention mask，不是 bool。float mask 走 EFFICIENT 内核，
     比 bool mask 更快（0.886 vs 1.028 ms）且更省显存（58.7 vs 75.5 MB）；
     而 MATH 后端慢 20 倍、显存 17 倍，任何让 attention 掉进 MATH 的写法都会
     静默毁掉训练。**这组数来自 torch 2.5.1+cu121 的实测**，而 requirements.txt
     钉的是 2.6.0+cu124 —— SDPA 后端选择随版本变，换版本后需用
     `python scripts/smoke_test.py --bench_mask` 重测。
  4. 不使用 enable_gqa=True。EFFICIENT 和 FLASH 都要求 q/k/v head 数相同，
     只有 CUDNN 接受 enable_gqa。沿用显式 repeat_kv expand。
  5. rope_theta = 1e4（MiniMind 用 1e6，因为它面向 32k 上下文）。
     我们预算 <=8192，1e4 的短程分辨率更好，而决策所需关系都是局部的。
"""
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.activations import ACT2FN
from transformers import PretrainedConfig, PreTrainedModel
from dataclasses import dataclass

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

# 段 id。PAD 必须为 0；PLAIN 供 MLM / AR baseline 的纯文本使用。
SEG_PAD = 0
SEG_PLAIN = 1
SEG_STATE = 2
SEG_QUESTION = 3
SEG_CANDIDATE = 4
NUM_SEGMENTS = 5

# mask 常量。用大的有限负数而非 -inf：整行全被 mask 时 softmax 退化为均匀分布
# 而不是产出 NaN（NaN 会经 LayerNorm 污染整个序列）。
NEG_INF = -1e9


def build_position_ids(seg_id, cand_id=None, prefix_len=None):
    """给打包序列分配 RoPE 位置。返回 (B, S) 的 long。

    纯文本（MLM / AR）用自然位置。打包序列里：
      前缀（state ∪ question，含各自的 <sep>）用自然位置 0..P-1；
      **每个候选 span（含其尾随 <sep>）的位置从 P 重新开始** —— 也就是每个
      候选都按"前缀之后只跟着它自己"来编码。

    这不是省事的写法，而是两个不变量成立的前提：
      - 候选顺序不变性：候选 k 的位置只依赖它自己的长度，与它排第几无关。
        若用自然位置，候选被置换后到前缀的相对偏移会变，RoPE 分数随之改变，
        不变性就只是近似的。
      - 分块不变性：候选 k 无论在哪个 chunk 里位置都相同，所以一次前向与
        分块前向给出同一组 logits。
    不同候选占用重叠的位置区间是安全的：crosstalk=False 时它们互不可见。

    `cand_id` 是必需的（含候选时）：候选块在 seg_id 上首尾相连（<sep> 也是
    SEG_CANDIDATE），只有 cand_id 的跳变能标出块边界。
    """
    bsz, seq_len = seg_id.shape
    dev = seg_id.device
    pos = torch.arange(seq_len, device=dev).expand(bsz, seq_len)

    cand = seg_id == SEG_CANDIDATE
    if not bool(cand.any()):
        return pos

    if cand_id is None:
        raise ValueError("含候选的打包序列必须提供 cand_id：相邻候选在 seg_id 上无法区分")

    if prefix_len is None:
        is_prefix = (seg_id == SEG_STATE) | (seg_id == SEG_QUESTION)
        prefix_len = is_prefix.sum(1, keepdim=True)
    else:
        prefix_len = torch.as_tensor(prefix_len, device=dev, dtype=torch.long).expand(bsz, 1)

    same_as_prev = torch.cat([
        torch.zeros(bsz, 1, dtype=torch.bool, device=dev),
        cand_id[:, :-1] == cand_id[:, 1:],
    ], dim=1)
    starts = cand & ~same_as_prev                          # 每个候选 span 的第一个位置
    block_start = torch.cummax(pos * starts, dim=1).values  # 该位置所属候选 span 的起点
    return torch.where(cand, prefix_len + (pos - block_start), pos)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class DecisionConfig(PretrainedConfig):
    model_type = "minisystemone"

    def __init__(self, hidden_size=512, num_hidden_layers=8, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.vocab_size = kwargs.get("vocab_size", 6400)
        self.num_attention_heads = kwargs.get("num_attention_heads", 8)
        # GQA 8/4 是 26.89M 档的实测配置。GQA 在这里的动机是参数效率而非
        # KV cache 压缩（决策模型不生成）。设为 8 即得 MHA，本尺寸下差别很小。
        self.num_key_value_heads = kwargs.get("num_key_value_heads", 4)
        self.head_dim = kwargs.get("head_dim", hidden_size // self.num_attention_heads)
        self.intermediate_size = kwargs.get("intermediate_size", 1280)
        self.hidden_act = kwargs.get("hidden_act", "silu")
        self.max_position_embeddings = kwargs.get("max_position_embeddings", 8192)
        self.rope_theta = kwargs.get("rope_theta", 10000.0)
        self.rms_norm_eps = kwargs.get("rms_norm_eps", 1e-6)
        self.dropout = kwargs.get("dropout", 0.0)
        self.tie_word_embeddings = kwargs.get("tie_word_embeddings", True)
        self.num_segments = kwargs.get("num_segments", NUM_SEGMENTS)
        self.use_gradient_checkpointing = kwargs.get("use_gradient_checkpointing", False)

        # 决策模型的两个结构开关
        self.candidate_crosstalk = kwargs.get("candidate_crosstalk", False)
        self.prefix_blocked = kwargs.get("prefix_blocked", True)

        self.pad_token_id = kwargs.get("pad_token_id", 0)
        self.sep_token_id = kwargs.get("sep_token_id", 3)
        self.mask_token_id = kwargs.get("mask_token_id", 4)
        self.bos_token_id = None
        self.eos_token_id = None


# ---------------------------------------------------------------------------
# 基础组件（沿用 MiniMind 的实现）
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        return (self.weight * self.norm(x.float())).type_as(x)


def precompute_freqs_cis(dim: int, end: int, rope_base: float = 10000.0):
    """返回 (end, dim) 的 cos/sin 表，按 position_ids 索引使用。"""
    freqs = 1.0 / (rope_base ** (torch.arange(0, dim, 2)[: dim // 2].float() / dim))
    t = torch.arange(end, dtype=torch.float32)
    freqs = torch.outer(t, freqs)
    freqs_cos = torch.cat([torch.cos(freqs), torch.cos(freqs)], dim=-1)
    freqs_sin = torch.cat([torch.sin(freqs), torch.sin(freqs)], dim=-1)
    return freqs_cos, freqs_sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=2):
    """q,k: (B, S, H, D)；cos,sin: (B, S, D)。

    unsqueeze_dim=2（不是 MiniMind 的 1）：这里 cos 带 batch 维（因为按
    position_ids 索引，各样本位置可以不同），所以在 H 维上插入广播维。
    """
    def rotate_half(x):
        return torch.cat((-x[..., x.shape[-1] // 2:], x[..., : x.shape[-1] // 2]), dim=-1)

    q_embed = (q * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(q) * sin.unsqueeze(unsqueeze_dim))
    k_embed = (k * cos.unsqueeze(unsqueeze_dim)) + (rotate_half(k) * sin.unsqueeze(unsqueeze_dim))
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """(B, S, n_kv_heads, D) -> (B, S, n_kv_heads*n_rep, D)。n_rep==1 时原样返回。"""
    if n_rep == 1:
        return x
    bs, slen, n_kv, head_dim = x.shape
    return (x[:, :, :, None, :]
            .expand(bs, slen, n_kv, n_rep, head_dim)
            .reshape(bs, slen, n_kv * n_rep, head_dim))


# ---------------------------------------------------------------------------
# Attention
# ---------------------------------------------------------------------------
class Attention(nn.Module):
    def __init__(self, config: DecisionConfig):
        super().__init__()
        self.n_local_heads = config.num_attention_heads
        self.n_local_kv_heads = config.num_key_value_heads
        self.n_rep = self.n_local_heads // self.n_local_kv_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.n_local_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_local_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_local_heads * self.head_dim, config.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.dropout = config.dropout

    def forward(self, x, position_embeddings, attention_mask=None,
                past_key_value=None, use_cache=False):
        bsz, seq_len, _ = x.shape
        xq = self.q_proj(x).view(bsz, seq_len, self.n_local_heads, self.head_dim)
        xk = self.k_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xv = self.v_proj(x).view(bsz, seq_len, self.n_local_kv_heads, self.head_dim)
        xq, xk = self.q_norm(xq), self.k_norm(xk)
        cos, sin = position_embeddings
        xq, xk = apply_rotary_pos_emb(xq, xk, cos, sin)

        if past_key_value is not None:
            xk = torch.cat([past_key_value[0], xk], dim=1)
            xv = torch.cat([past_key_value[1], xv], dim=1)
        past_kv = (xk, xv) if use_cache else None

        xq = xq.transpose(1, 2)
        xk = repeat_kv(xk, self.n_rep).transpose(1, 2)
        xv = repeat_kv(xv, self.n_rep).transpose(1, 2)

        p = self.dropout if self.training else 0.0
        if attention_mask is not None:
            # float additive mask：0.0 允许，NEG_INF 屏蔽。走 EFFICIENT/CUDNN 融合内核。
            out = F.scaled_dot_product_attention(xq, xk, xv, attn_mask=attention_mask, dropout_p=p)
        else:
            out = F.scaled_dot_product_attention(xq, xk, xv, dropout_p=p, is_causal=False)

        out = out.transpose(1, 2).reshape(bsz, seq_len, -1)
        out = self.resid_dropout(self.o_proj(out))
        return out, past_kv


class FeedForward(nn.Module):
    def __init__(self, config: DecisionConfig, intermediate_size: int = None):
        super().__init__()
        inter = intermediate_size or config.intermediate_size
        self.gate_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, inter, bias=False)
        self.down_proj = nn.Linear(inter, config.hidden_size, bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MiniSystemOneBlock(nn.Module):
    def __init__(self, config: DecisionConfig):
        super().__init__()
        self.self_attn = Attention(config)
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = FeedForward(config)

    def _forward(self, hidden_states, position_embeddings, attention_mask, past_key_value, use_cache):
        residual = hidden_states
        hidden_states, present = self.self_attn(
            self.input_layernorm(hidden_states), position_embeddings,
            attention_mask, past_key_value, use_cache,
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.mlp(self.post_attention_layernorm(hidden_states))
        return hidden_states, present

    def forward(self, hidden_states, position_embeddings, attention_mask=None,
                past_key_value=None, use_cache=False):
        # 梯度检查点在 Encoder 里按层包裹 _forward，这里保持纯前向
        return self._forward(hidden_states, position_embeddings, attention_mask,
                             past_key_value, use_cache)


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, config: DecisionConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.embed_segments = nn.Embedding(config.num_segments, config.hidden_size)
        self.dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList([MiniSystemOneBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        freqs_cos, freqs_sin = precompute_freqs_cis(
            config.head_dim, config.max_position_embeddings, config.rope_theta
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, input_ids, seg_id=None, cand_id=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False):
        bsz, seq_len = input_ids.shape
        past_key_values = past_key_values or [None] * len(self.layers)

        hidden = self.embed_tokens(input_ids)
        if seg_id is not None:
            hidden = hidden + self.embed_segments(seg_id)
        hidden = self.dropout(hidden)

        if position_ids is None:
            if seg_id is not None:
                position_ids = build_position_ids(seg_id, cand_id)
            else:
                start = past_key_values[0][0].shape[1] if past_key_values[0] is not None else 0
                position_ids = torch.arange(start, start + seq_len, device=input_ids.device)[None, :].expand(bsz, -1)
        pos_cos = self.freqs_cos[position_ids]     # (B, S, head_dim)
        pos_sin = self.freqs_sin[position_ids]
        position_embeddings = (pos_cos, pos_sin)

        presents = []
        use_ckpt = self.config.use_gradient_checkpointing and self.training
        for layer, past in zip(self.layers, past_key_values):
            if use_ckpt:
                hidden, present = checkpoint(
                    layer._forward, hidden, position_embeddings, attention_mask, past, use_cache,
                    use_reentrant=False,
                )
            else:
                hidden, present = layer(hidden, position_embeddings, attention_mask, past, use_cache)
            presents.append(present)
        hidden = self.norm(hidden)
        return hidden, presents


# ---------------------------------------------------------------------------
# 决策头
# ---------------------------------------------------------------------------
class AttnPool(nn.Module):
    """在 state ∪ question 的 span 上做 attention pool，得到 z。

    残差到 mean 是刻意的：纯 attention pool 可能塌缩到单个 token，
    那会毁掉校准（分布会变得过度自信）。残差保证 z 始终携带全序列摘要。
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.q = nn.Parameter(torch.zeros(hidden_size))
        self.k = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o = nn.Linear(hidden_size, hidden_size, bias=False)
        self.ln = nn.LayerNorm(hidden_size)

    def forward(self, hidden, mask):
        # 线性层用 model dtype（bf16 下权重也是 bf16），分数与归约升到 fp32：
        # 求和顺序无关紧要，但 softmax 与均值需要 fp32 才不会丢校准所需的分辨率。
        dt = hidden.dtype
        h_f = hidden.float()
        a = (self.k(hidden) @ self.q.to(dt)).float() / math.sqrt(hidden.size(-1))
        a = a.masked_fill(~mask, NEG_INF).softmax(-1)
        z = self.o(torch.einsum("bs,bsh->bh", a, h_f).to(dt))
        m = mask.unsqueeze(-1).to(h_f.dtype)
        mean = (h_f * m).sum(1) / m.sum(1).clamp(min=1)
        # layer_norm 内部按 fp32 累加，所以 bf16 输入不会丢精度
        return self.ln((z.float() + mean).to(dt))


class DecisionHead(nn.Module):
    """唯一的 head。Noul / Choice / Score 的差别只在 schema，不在这里。"""

    def __init__(self, config: DecisionConfig):
        super().__init__()
        h = config.hidden_size
        self.pool = AttnPool(h)
        self.in_ln = nn.LayerNorm(4 * h)
        self.fc1 = nn.Linear(4 * h, h)
        self.act = nn.GELU()
        self.mid_ln = nn.LayerNorm(h)
        self.fc2 = nn.Linear(h, 1)

    @staticmethod
    def pool_candidates(hidden, cand_span, cand_mask):
        """候选 span 上的 masked mean-pool。用 cumsum 实现，避免 (B,K,S,h) 中间张量。"""
        bsz, seq_len, _ = hidden.shape
        pad = torch.zeros(bsz, 1, hidden.size(-1), device=hidden.device, dtype=torch.float32)
        hc = torch.cat([pad, hidden.float().cumsum(1)], dim=1)      # hc[i] = sum(hidden[:i])
        start = cand_span[..., 0].clamp(min=0)
        end = cand_span[..., 1].clamp(min=0, max=seq_len)
        valid = cand_mask & (end > start)
        start = torch.where(valid, start, torch.zeros_like(start))
        end = torch.where(valid, end, torch.zeros_like(end))
        gather_idx = end.unsqueeze(-1).expand(-1, -1, hidden.size(-1))
        total = hc.gather(1, gather_idx) - hc.gather(1, start.unsqueeze(-1).expand(-1, -1, hidden.size(-1)))
        count = (end - start).clamp(min=1).unsqueeze(-1).to(total.dtype)
        return (total / count).to(hidden.dtype)

    def score_from_z(self, z, cand_vecs, cand_mask):
        """给定 z 和候选向量打分。分块推理时 z 由缓存的前缀算出，走这条路径。"""
        zi = z[:, None, :].expand_as(cand_vecs)
        feats = torch.cat([zi, cand_vecs, zi * cand_vecs, (zi - cand_vecs).abs()], dim=-1)
        logits = self.fc2(self.mid_ln(self.act(self.fc1(self.in_ln(feats))))).squeeze(-1)
        return logits.masked_fill(~cand_mask, NEG_INF)

    def forward(self, hidden, prefix_mask, cand_vecs, cand_mask):
        z = self.pool(hidden, prefix_mask)
        return self.score_from_z(z, cand_vecs, cand_mask), z


# ---------------------------------------------------------------------------
# 输出
# ---------------------------------------------------------------------------
@dataclass
class DecisionOutput:
    loss: torch.Tensor = None
    logits: torch.Tensor = None           # (B, K) 原始 logit，温度拟合需要
    z: torch.Tensor = None
    hidden_states: torch.Tensor = None
    past_key_values: list = None
    loss_dict: dict = None


# ---------------------------------------------------------------------------
# 主模型
# ---------------------------------------------------------------------------
class MiniSystemOneForDecision(PreTrainedModel):
    config_class = DecisionConfig

    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.head = DecisionHead(config)
        self.post_init()

    def pool_candidates(self, hidden, cand_span, cand_mask):
        return DecisionHead.pool_candidates(hidden, cand_span, cand_mask)

    def forward(self, input_ids, seg_id=None, cand_id=None, cand_span=None, cand_mask=None,
                prefix_mask=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False,
                target=None, is_ord=None, lambda_brier=0.5, lambda_ord=0.5,
                brier_normalize=False, **kwargs):
        if seg_id is not None:
            # mask 与 prefix_mask 一律从 seg_id / cand_id 推出，不要求调用方传。
            # 除了省一次 (B,1,S,S) 的搬运，更重要的是**漏传时不会静默退化成全连通
            # 注意力** —— 那会同时毁掉候选顺序不变性与分块一致性，且不报错。
            from model.serialize import build_attn_mask, build_prefix_mask

            if attention_mask is None:
                attention_mask = build_attn_mask(
                    seg_id, cand_id, self.config.candidate_crosstalk,
                    self.config.prefix_blocked, dtype=self.dtype,
                )
            if prefix_mask is None:
                prefix_mask = build_prefix_mask(seg_id)

        hidden, presents = self.encoder(
            input_ids, seg_id=seg_id, cand_id=cand_id, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
        )
        cand_vecs = self.pool_candidates(hidden, cand_span, cand_mask)
        logits, z = self.head(hidden, prefix_mask, cand_vecs, cand_mask)

        loss, loss_dict = None, {}
        if target is not None:
            loss, loss_dict = self.compute_loss(logits, target, cand_mask, is_ord,
                                                lambda_brier, lambda_ord,
                                                brier_normalize=brier_normalize)
        return DecisionOutput(
            loss=loss, logits=logits, z=z, hidden_states=hidden,
            past_key_values=presents, loss_dict=loss_dict,
        )

    @staticmethod
    def compute_loss(logits, target, cand_mask, is_ord=None,
                     lambda_brier=0.5, lambda_ord=0.5, brier_normalize=False):
        """L = CE + lambda_b * Brier + is_ord * lambda_o * CDF-MSE。全部在 fp32 上算。"""
        logits = logits.float()
        # 屏蔽槽位置为 NEG_INF，softmax 后为 0，因此它们在各项中贡献 0
        p = torch.softmax(logits.masked_fill(~cand_mask, NEG_INF), dim=-1)
        t = target.float()

        loss_ce = -(t * torch.log(p.clamp_min(1e-12))).sum(-1).mean()

        brier = ((p - t) ** 2).sum(-1)
        if brier_normalize:
            brier = brier / cand_mask.sum(-1).clamp(min=1)
        loss_brier = brier.mean()

        loss = loss_ce + lambda_brier * loss_brier
        loss_dict = {"ce": loss_ce.detach(), "brier": loss_brier.detach()}

        if is_ord is not None and lambda_ord > 0 and is_ord.sum() > 0:
            cdf_p = p.cumsum(-1)[..., :-1]
            cdf_t = t.cumsum(-1)[..., :-1]
            per_item = ((cdf_p - cdf_t) ** 2).mean(-1)
            loss_ord = (per_item * is_ord.float()).sum() / is_ord.float().sum().clamp(min=1)
            loss = loss + lambda_ord * loss_ord
            loss_dict["ord"] = loss_ord.detach()

        loss_dict["loss"] = loss.detach()
        return loss, loss_dict

    @torch.inference_mode()
    def decide(self, input_ids, seg_id=None, cand_id=None, cand_span=None, cand_mask=None,
               prefix_mask=None, attention_mask=None, temperature=1.0):
        """单次前向返回概率分布。"""
        out = self.forward(
            input_ids, seg_id=seg_id, cand_id=cand_id, cand_span=cand_span, cand_mask=cand_mask,
            prefix_mask=prefix_mask, attention_mask=attention_mask,
        )
        logits = out.logits.float()
        p = torch.softmax(logits.masked_fill(~cand_mask, NEG_INF) / temperature, dim=-1)
        return p, logits

    # -- 分块推理 ----------------------------------------------------------
    # 因 prefix_blocked=True 且 candidate_crosstalk=False，打包序列精确可划分：
    # 前缀隐状态与候选集无关，每个候选只依赖前缀和自身 span。所以前缀 K/V
    # 可以缓存并在多个候选分块之间复用。RoPE 使分块位置精确 —— 相对位置下
    # 候选到前缀和自身 span 的注意力偏移与分块起点无关。
    @torch.inference_mode()
    def encode_prefix(self, state_ids, question_ids, sep_id):
        """编码前缀（state+question），返回 (z, past_key_values, prefix_len)。"""
        from model.serialize import pack_example, build_attn_mask, build_prefix_mask

        dev = next(self.parameters()).device
        ids, seg, cid, _ = pack_example(state_ids, question_ids, [], sep_id)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        seg_t = torch.tensor([seg], dtype=torch.long, device=dev)
        cid_t = torch.tensor([cid], dtype=torch.long, device=dev)
        attn = build_attn_mask(seg_t, cid_t, self.config.candidate_crosstalk,
                               self.config.prefix_blocked, dtype=self.dtype)
        hidden, presents = self.encoder(input_ids, seg_id=seg_t, attention_mask=attn, use_cache=True)
        z = self.head.pool(hidden, build_prefix_mask(seg_t))
        return z, presents, len(ids)

    @torch.inference_mode()
    def score_candidates(self, z, past_key_values, prefix_len, candidates_ids, sep_id):
        """对一个候选分块打分，复用已缓存的前缀 K/V。返回 float logits (1, K)。"""
        dev = next(self.parameters()).device
        ids, seg, cid, spans = [], [], [], []
        for k, cs in enumerate(candidates_ids):
            start = len(ids)
            ids.extend(cs if cs else [sep_id])
            seg.extend([SEG_CANDIDATE] * len(cs if cs else [sep_id]))
            cid.extend([k] * len(cs if cs else [sep_id]))
            ids.append(sep_id); seg.append(SEG_CANDIDATE); cid.append(k)
            spans.append((start, len(ids)))

        chunk_len = len(ids)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        seg_t = torch.tensor([seg], dtype=torch.long, device=dev)
        cid_t = torch.tensor([cid], dtype=torch.long, device=dev)

        # 候选可 attend 全部前缀 key + 自身 span 的 key
        same = (cid_t.unsqueeze(-1) == cid_t.unsqueeze(1)) & (cid_t.unsqueeze(-1) >= 0)
        own = torch.where(same, 0.0, NEG_INF).to(self.dtype)                       # (1,S,S)
        pref = torch.zeros(1, chunk_len, prefix_len, dtype=self.dtype, device=dev)
        attn = torch.cat([pref, own], dim=-1).unsqueeze(1)                          # (1,1,S,P+S)

        # 每个候选都从 prefix_len 起算位置，与该候选落在哪个 chunk 无关
        position_ids = build_position_ids(seg_t, cid_t, prefix_len=prefix_len)
        hidden, _ = self.encoder(input_ids, seg_id=seg_t, attention_mask=attn,
                                 position_ids=position_ids, past_key_values=past_key_values)

        k = len(spans)
        cand_span = torch.tensor([[s, t] for s, t in spans], dtype=torch.long, device=dev).unsqueeze(0)
        cand_mask = torch.ones(1, k, dtype=torch.bool, device=dev)
        cand_vecs = self.pool_candidates(hidden, cand_span, cand_mask)
        logits = self.head.score_from_z(z, cand_vecs, cand_mask)
        return logits, cand_mask

    @torch.inference_mode()
    def decide_chunked(self, state_ids, question_ids, candidates_ids, sep_id,
                       chunk=64, temperature=1.0, max_prefix_tokens=7000):
        """任意候选数（含 255）的决策。候选分块，前缀只编码一次。"""
        z, past, prefix_len = self.encode_prefix(state_ids, question_ids, sep_id)
        all_logits, all_mask = [], []
        for j in range(0, len(candidates_ids), chunk):
            lg, mk = self.score_candidates(z, past, prefix_len,
                                           candidates_ids[j:j + chunk], sep_id)
            all_logits.append(lg)
            all_mask.append(mk)
        logits = torch.cat(all_logits, dim=-1)
        mask = torch.cat(all_mask, dim=-1)
        p = torch.softmax(logits.float().masked_fill(~mask, NEG_INF) / temperature, dim=-1)
        return p, logits

    # -- 问题摊销：一份 state 配 N 个问题 ------------------------------------
    # 这是本架构**最强**的效率主张，而它是结构性的而非硬件性的：因为
    # `prefix_blocked=True`，state 的隐状态不依赖任何候选、也不依赖 question
    # （state 不可 attend 到二者），所以同一份 state 配 N 个问题时只需编码一次。
    # 省下的是 N−1 份 state 的注意力与前馈 —— 而 state 通常远长于 question。
    #
    # `decide_chunked` 缓存的是 **state+question**，那是一份前缀只能配一组候选；
    # 跨问题复用必须在 state 之后**切开**，所以需要下面这对接口，不能靠拼前缀绕过。
    @torch.inference_mode()
    def encode_state(self, state_ids, sep_id):
        """只编码 state（含尾随 `<sep>`），返回 (隐状态, K/V, 长度)。

        注意力就是「state 内部全可见」，此时序列里既没有 question 也没有候选，
        所以与完整打包序列里 state 那一段的注意力模式**逐元素相同** —— 这正是
        缓存可以安全复用的前提，而不是一个近似。
        """
        dev = next(self.parameters()).device
        ids = list(state_ids) + [sep_id]
        seg_t = torch.full((1, len(ids)), SEG_STATE, dtype=torch.long, device=dev)
        cid_t = torch.full((1, len(ids)), -1, dtype=torch.long, device=dev)
        input_ids = torch.tensor([ids], dtype=torch.long, device=dev)
        mask = torch.zeros(1, 1, len(ids), len(ids), dtype=self.dtype, device=dev)
        hidden, presents = self.encoder(input_ids, seg_id=seg_t, attention_mask=mask,
                                        use_cache=True)
        return hidden, presents, len(ids)

    @torch.inference_mode()
    def decide_with_state(self, cached, question_ids, candidates_ids, sep_id,
                          chunk=64, temperature=1.0):
        """在已缓存的 state 上回答**一个**问题（任意候选数）。返回 (p, logits)。

        两步：先把 question 接到缓存的 K/V 后面编码出 `z`，再按 chunk 给候选打分。
        question 那一步不需要 mask —— 它合法可见的恰好就是「全部 state + 自己」，
        而过去与自身拼起来就是全部 key。候选那一步直接复用 `score_candidates`，
        它按 `prefix_len` 重建位置与 block-diagonal mask，于是**位置与单次前向
        完全一致**（候选一律从 `state_len + Q` 起算），这就是缓存路径与
        `decide()` 能给出同一组 logits 的原因。
        """
        saved, past, state_len = cached
        dev = next(self.parameters()).device

        q_ids = list(question_ids) + [sep_id]
        Q = len(q_ids)
        seg_t = torch.full((1, Q), SEG_QUESTION, dtype=torch.long, device=dev)
        cid_t = torch.full((1, Q), -1, dtype=torch.long, device=dev)
        input_ids = torch.tensor([q_ids], dtype=torch.long, device=dev)
        pos = torch.arange(state_len, state_len + Q, device=dev)[None, :].expand(1, -1)
        q_hidden, presents = self.encoder(
            input_ids, seg_id=seg_t, attention_mask=None, position_ids=pos,
            past_key_values=past, use_cache=True)

        # z 池化在 state ∪ question 上 —— 两段的隐状态都已算好，拼起来正是完整
        # 前缀。state 段与 question 段在该掩码下全为真，所以掩码是全 1。
        mask = torch.ones(1, state_len + Q, dtype=torch.bool, device=dev)
        z = self.head.pool(torch.cat([saved, q_hidden], dim=1), mask)

        prefix_len = state_len + Q
        all_logits, all_mask = [], []
        for j in range(0, len(candidates_ids), chunk):
            lg, mk = self.score_candidates(z, presents, prefix_len,
                                           candidates_ids[j:j + chunk], sep_id)
            all_logits.append(lg)
            all_mask.append(mk)
        logits = torch.cat(all_logits, dim=-1)
        cmask = torch.cat(all_mask, dim=-1)
        p = torch.softmax(logits.float().masked_fill(~cmask, NEG_INF) / temperature, dim=-1)
        return p, logits


class MiniSystemOneForMaskedLM(PreTrainedModel):
    """Stage 1 用：同一个 Encoder + 一个与词表绑定的 lm_head。

    head 刻意就是**一个绑定权重的线性层**，不加 BERT 那套 dense/GELU/LayerNorm：
    多加的层只属于 MLM 阶段，`load_state_dict` 时会被整个丢掉，等于告诉读者
    "这部分是白练的"。绑定权重意味着 MLM 阶段**不引入任何新参数**，所以
    "编码器从 0 预训练、决策头是新的"这句话在参数账上是精确的。

    绑定后的 head 形状与决策模型读取的 `encoder` 完全一致，两阶段的 checkpoint
    因此能互相加载编码器部分而不需要任何转换代码。
    """

    config_class = DecisionConfig
    _tied_weights_keys = {"lm_head.weight": "encoder.embed_tokens.weight"}

    def __init__(self, config: DecisionConfig):
        super().__init__(config)
        self.encoder = Encoder(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.encoder.embed_tokens.weight = self.lm_head.weight
        self.post_init()

    def forward(self, input_ids, seg_id=None, attention_mask=None, position_ids=None,
                past_key_values=None, use_cache=False, labels=None, **kwargs):
        hidden, presents = self.encoder(
            input_ids, seg_id=seg_id, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values, use_cache=use_cache,
        )
        if labels is None:
            return DecisionOutput(loss=None, logits=self.lm_head(hidden),
                                  hidden_states=hidden, past_key_values=presents)

        # **只对被 mask 的位置算词表 logits。** 逐位置 MLM 的 loss 只在 labels != -100
        # 处非零（15%），但 `lm_head` 是 512→6400 的线性层，在全部 512 个位置上都跑
        # 就是 6.7 倍的浪费。数学上完全等价 —— 被丢弃的位置既不参与 loss、也不影响
        # 反向（loss 是逐位置求和的，没有跨位置的信息流）。
        # 实测收益只有约 6%（B16×S512：168 → 158 ms/step），因为瓶颈在编码器而非
        # 词表头 —— 保留是因为它免费，不是因为它显著。
        flat_labels = labels.view(-1)
        sel = flat_labels != -100
        h = hidden.reshape(-1, hidden.size(-1))[sel]
        logits = self.lm_head(h)
        loss = F.cross_entropy(logits.float(), flat_labels[sel])
        return DecisionOutput(loss=loss, logits=logits, hidden_states=hidden,
                              past_key_values=presents)
