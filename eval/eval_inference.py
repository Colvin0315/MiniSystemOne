"""
"跑一遍模型，把逐样本结果收下来" —— 训练循环里的 val 和离线 harness 共用的唯一路径。

这个文件存在的理由不是复用，而是**保证同一个数字**。方案 R6 说优化目标与报告指标
必须对齐；如果训练循环里那份 val ECE 和 `eval_harness.py` 报告的那份由两份代码各算
一次，它们迟早会漂移（一次忘了同步 mask、一次忘了转 fp64），而漂移的方向是**训练
日志比报告好看**。所以两者都调 `collect()`，再都调 `metrics_by_provenance()`。

`eval/eval_metrics.py` 刻意保持无模型、无 I/O；模型相关的部分只能落在这里。

**只存 `p`，不存 logits。** 温度作用在 `p` 上是精确的，不需要原始 logit：
`softmax(z/T)_k ∝ exp(z_k/T) = p_k^(1/T)`，多出的 `Σexp(z_j)` 因子在归一化时抵消。
所以 `per_sample` 的体积比存 logits 小一个量级，而任何温度都能事后重算。
"""
import json
import math
import os

import numpy as np
import torch
from transformers import AutoTokenizer

from dataset.decision_dataset import collate_decision
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.trainer_utils import ckpt_info, file_sha1, init_model, verify_tokenizer


def load_decision(checkpoint, tokenizer_dir, device="cuda", **overrides):
    if not str(device).startswith("cuda") or not torch.cuda.is_available():
        raise ValueError("需要可用的 CUDA 环境；不会退回 CPU。")
    if not os.path.isfile(checkpoint):
        raise ValueError(f"找不到决策权重：{checkpoint}")
    info = ckpt_info(checkpoint)
    meta = info.get("meta", {})
    if not info.get("config") or meta.get("stage") != "decision":
        raise ValueError("需要带 config 和 decision 阶段元信息的自描述权重。")
    if not meta.get("tokenizer_sha1"):
        raise ValueError("权重缺少 tokenizer_sha1，无法确认配套词表。")
    try:
        verify_tokenizer(checkpoint, tokenizer_dir)
    except SystemExit as error:
        raise ValueError(str(error)) from error
    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    config = DecisionConfig(**info["config"])
    for key, value in overrides.items():
        if value is not None and getattr(config, key) != value:
            raise ValueError(f"{key}={value} 与权重配置 {getattr(config, key)} 不符")
    if config.vocab_size != len(tok):
        raise ValueError("词表大小与 checkpoint 配置不一致")
    for token, expected in (("<pad>", config.pad_token_id), ("<sep>", config.sep_token_id)):
        if tok.convert_tokens_to_ids(token) != expected:
            raise ValueError(f"{token} 的 token id 与 checkpoint 不一致")
    config.use_gradient_checkpointing = False
    model = init_model(MiniSystemOneForDecision, config, checkpoint, device, strict=True)
    model.eval()
    return tok, model, meta


def load_temperature(path, checkpoint, tokenizer_dir):
    with open(path, encoding="utf-8") as f:
        temp = json.load(f)
    if not isinstance(temp, dict) or not isinstance(temp.get("meta"), dict):
        raise ValueError("温度文件缺少 meta 哈希信息")
    for key, artifact in (("ckpt_sha1", checkpoint),
                          ("tokenizer_sha1", os.path.join(tokenizer_dir, "tokenizer.json"))):
        digest = temp["meta"].get(key)
        if not isinstance(digest, str) or len(digest) not in (12, 16):
            raise ValueError(f"温度文件缺少有效的 {key}")
        if file_sha1(artifact, n=len(digest)) != digest:
            raise ValueError(f"温度文件的 {key} 与当前模型/词表不匹配，请重新校准。")
    values = [temp.get("global")]
    for key in ("primitive", "primitive_k"):
        table = temp.get(key, {})
        if not isinstance(table, dict):
            raise ValueError(f"温度 {key} 必须是对象")
        values.extend(table.values())
    if any(isinstance(t, bool) or not isinstance(t, (int, float))
           or not math.isfinite(t) or not 0.05 <= t <= 20 for t in values):
        raise ValueError("所有温度必须在拟合器支持的 [0.05, 20] 范围内，且必须提供 global 温度")
    return temp

# 逐样本元信息：全部是「不进模型、但报告与误差分析要用」的东西。
META_KEYS = ("id", "source", "provenance", "template_id", "entity_pool",
             "K_full", "renormalized", "labels", "counts", "gen_version")


@torch.inference_mode()
def collect(model, dataset, device, batch_size=32, limit=0):
    """全量（或前 N 条）前向，返回逐样本的 numpy 数组与元信息。

    **保留逐样本结果**是刻意的，不是顺手：per-provenance 分桶、可靠性图、温度
    拟合、误差分析都从这一份重算，不需要再跑一次模型。代价是 JSON 大一些，
    换来的是"改一个分桶方案要重新跑 1.5 小时 GPU"这件事不再发生。

    **`batch_size` 在训练循环里必须等于训练用的 batch。** 缓存分配器按块大小复用；
    训练用 16 而 val 用 32 时，val 要的块（32×S）比训练缓存过的任何一块都大，
    于是一律是新 `cudaMalloc`，预留量直接叠在训练池上。实测：训练池约 6.0 GiB，
    val bs=32 再要 3.9 GiB → 7.92 GiB，越过 R5 的 Windows WDDM 悬崖（8188 MiB 卡
    上超过约 7.5 GB 即静默换页），step 500 那次 val 卡了 6 分钟以上且不报 OOM。
    改成 bs=16 后预留 1.91 GiB，能落回训练池里已有的块。
    """
    was_training = model.training
    model.eval()

    n = min(len(dataset), limit) if limit else len(dataset)
    P, T, M, LV, IO = [], [], [], [], []
    meta = {k: [] for k in META_KEYS}
    primitives = []

    for s in range(0, n, batch_size):
        idx = range(s, min(s + batch_size, n))
        batch = collate_decision([dataset[i] for i in idx], device=device)
        out = model(
            batch["input_ids"], seg_id=batch["seg_id"], cand_id=batch["cand_id"],
            cand_span=batch["cand_span"], cand_mask=batch["cand_mask"],
        )
        # 屏蔽槽位先填 -inf 再 softmax：p 在这些位置严格为 0，于是 `p**(1/T)`
        # 仍是 0，温度变换不会把 padding 槽变成非零质量。
        p = torch.softmax(
            out.logits.float().masked_fill(~batch["cand_mask"], float("-inf")), -1)
        P.append(p.cpu().numpy())
        T.append(batch["target"].cpu().numpy())
        M.append(batch["cand_mask"].cpu().numpy())
        LV.append(batch["level_idx"].cpu().numpy())
        IO.append(batch["is_ord"].bool().cpu().numpy())
        primitives.extend(dataset.records[i]["schema"]["primitive"] for i in idx)
        for k in META_KEYS:
            meta[k].extend(batch[k])

    if was_training:
        model.train()
    if not P:
        return None

    result = {
        # 批与批的候选宽度不同（分桶采样让每批 K 一致，但批间不同），所以先补到
        # 全集合的最大 K。填的值必须让所有指标对此无感：p 填 0、target 填 0、
        # mask 填 False、level 填 -1。于是 padded 槽位在
        # `mask_normalize` 后是零质量，在 `t*log p` 里因 t=0 而不贡献，在
        # `(p-t)^2` 里是 0，在 `p*level` 里是 0 —— 每一项都恰好是恒等元。
        "p": _pad_stack(P, 0.0, np.float64),
        "target": _pad_stack(T, 0.0, np.float64),
        "mask": _pad_stack(M, False, bool),
        "levels": _pad_stack(LV, -1, np.float64),
        "is_ord": np.concatenate(IO),
        "primitive": primitives,
        "n": n,
    }
    result.update({k: meta[k] for k in META_KEYS})
    return result


def _pad_stack(arrays, fill, dtype):
    """把变宽的 (n_i, K_i) 数组补到共同宽度后纵向拼接。"""
    k_max = max(a.shape[1] for a in arrays)
    out = np.full((sum(a.shape[0] for a in arrays), k_max), fill, dtype=dtype)
    r = 0
    for a in arrays:
        out[r:r + a.shape[0], :a.shape[1]] = a
        r += a.shape[0]
    return out


def counts_array(collected):
    """Return annotation counts, with zero marking unobserved counts."""
    counts = collected.get("counts")
    if not counts or not any(c is not None and c > 0 for c in counts):
        return None
    return np.asarray([c if c is not None else 0 for c in counts], dtype=np.float64)


def effective_k(mask):
    """每个样本**实际**呈现的候选数（= mask 的逐行和），不是 padding 后的宽度。

    温度按 K 分桶，而分桶必须用真实候选数 —— 用 padded 宽度会把 K=3 和 K=32 的
    样本混进同一个桶，而它们只因为同一 batch 里有个大 K 样本才被 pad 到一起。
    """
    return mask.sum(-1).astype(int) if mask is not None else None


def split_example(ex, sep_id):
    """把打包序列拆回 (state, question, [候选 token])。

    位置一律从 `cand_spans` 与 `<sep>` 的**实际下标**反查，不重算 tokenizer ——
    重算会引入第二次分词，而两次分词只要有一处不同（BPE 归并边界、空白处理），
    这条路径就会以一个看不懂的方式失败。鲁棒性套件④与效率套件的分块路径都走它。
    """
    ids = ex["input_ids"]
    seps = [i for i, t in enumerate(ids) if t == sep_id]
    if len(seps) < 2:
        return None, None, []
    state_ids = ids[:seps[0]]
    q_ids = ids[seps[0] + 1:seps[1]]
    cand_ids = []
    for s, e in ex["cand_spans"]:
        seg = ids[s:e]
        if seg and seg[-1] == sep_id:
            seg = seg[:-1]
        cand_ids.append(seg)
    return state_ids, q_ids, cand_ids
