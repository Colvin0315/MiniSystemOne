"""
公开语料适配器 —— 把真实数据集改写成统一的决策 schema（见 `docs/DATA_SCHEMA.md`）。

为什么必须有它们：合成生成器能给出**已知的** P*，但那是程序的分布。真实人类标注的
分布只有真实语料里有，而"对着真实人类分歧报告校准"是本项目最诚实的头条数字。
ChaosNLI 每条 100 个标注者，是全套里唯一能把逐条 ECE 压到二项噪声底之上的数据。

三条与合成侧一致的约束，由 `finish()` 统一执行：

  1. **候选逐例重排**（连同 `target` 一起置换），位置不携带信息 —— 由
     `scripts/audit_leakage.py` 的位置探针把关。
  2. `state` 必须**恰好等于** `state_sections` 按呈现顺序用 "\\n" 拼接。下游
     `decision_dataset.fit_state` 会按 priority 丢段后重新拼接，两者不一致时裁剪后的
     state 就与训练分布不同，而这种漂移不报错。
  3. `(source, template_id, entity_pool)` 组合落在**唯一一个** split 里。公开集没有
     模板族，所以 pool 取原生 split 名（无原生 split 则取哈希桶名）；`calib` 需要从
     原生 train 里切，因此单独给一个 pool 名 —— 否则 calib 与 train 会是同一个组合，
     而一个组合不能同时属于两个 split。

`meta.approx_tokens` **不在这里算**：它依赖 tokenizer，由 `scripts/build_dataset.py`
统一盖章。各适配器各算一份的话，换 tokenizer 时必然漏掉一两个。
"""
import hashlib
import random

PUBLIC_GEN_VERSION = "1.0.0"

SPLITS = ("train", "val", "calib", "test_known", "test_ood")

# 无原生 split 的数据集（ChaosNLI、GoEmotions raw）按**整桶**哈希切分。取整必须在桶上
# 做而不是在条数上做：逐条决定 split 的话，同一个 (template, pool) 组合会同时出现在
# train 和 test_known 里，留出集当场漏掉，而所有指标都会因此变好看。
N_BUCKETS = 10
BUCKET_SPLIT = ("train", "train", "train", "train", "train",
                "val", "calib", "test_known", "test_known", "test_known")


def _h(key):
    return int(hashlib.sha1(str(key).encode()).hexdigest()[:8], 16)


def bucket(key, n):
    return _h(key) % n


def hash_split(key):
    return BUCKET_SPLIT[bucket(key, N_BUCKETS)]


def carve_calib(key, n=10):
    """从原生 train 里切出 calib。键上加前缀，与原生 split 的哈希正交。"""
    return bucket(key, n) == 0


def sec(seg, text, priority=3):
    return {"seg": seg, "text": text, "priority": priority}


# 意图路由类适配器（CLINC150 / banking77）共用的候选数采样。放在这里而不是各写一份：
# 两边的语义完全相同，分头写只会让"全量候选"的比例在两处慢慢漂开，而这种漂移不报错，
# 只表现为大 K 路径的覆盖悄悄变少。
#
# FULL_K_RATE 那 1/8 不是难度调节，是**路径覆盖预算**：只有它会让真实数据走到
# K≈150 的分块前向上去（见 clinc150.py 的 docstring）。
FULL_K_RATE = 0.125
K_LO, K_HI = 4, 32


def sample_k(rng, n_avail):
    if n_avail <= K_LO:
        return n_avail
    if rng.random() < FULL_K_RATE:
        return n_avail
    return rng.randint(K_LO, min(K_HI, n_avail))


def _audit_of(item):
    """Preserve audit metadata while reconstructing targets."""
    out = dict(item["target"].get("audit") or {})
    out.update(item.get("audit") or {})
    return out


def finish(adapter, index, split, item, template_id, entity_pool):
    """组装一条 record：重排候选、置换目标、渲染 state、钉死契约。"""
    rng = random.Random(f"{adapter}|{index}")
    cands = list(item["candidates"])
    p = list(item["target"]["p"])
    if len(cands) != len(p):
        raise AssertionError(f"{adapter}::{index}: 候选数 {len(cands)} ≠ 目标长度 {len(p)}")

    # **K ≥ 2 是硬约束。** 只有一个候选时 softmax 恒等于 1，CE 与 Brier 都恒为 0、
    # 梯度恒为 0 —— 而它**在评测里会被记成一次正确预测**。于是准确率被一批"免费的
    # 对"抬高，而抬高的幅度恰好等于这类样本的占比。GoEmotions 有 14.7% 是这样的
    # （所有投票者只投了同一个情绪），所以那个数不是小数点后的问题。
    # 放在这里而不是各适配器里：这是"一条决策样本"的定义，不是某个数据集的怪癖。
    if len(cands) < 2:
        raise AssertionError(f"{adapter}::{index}: 只有 {len(cands)} 个候选，不构成决策")

    order = list(range(len(cands)))
    rng.shuffle(order)
    cands = [cands[j] for j in order]
    p = [p[j] for j in order]

    total = sum(p)
    if abs(total - 1.0) > 1e-9:
        raise AssertionError(f"{adapter}::{index}: 目标和为 {total}，应为 1")
    if any(x < 0 for x in p):
        raise AssertionError(f"{adapter}::{index}: 目标含负概率")

    sections = item["sections"]
    state = "\n".join(s["text"] for s in sections)
    target = {"kind": "soft", "p": p, "provenance": item["provenance"],
              "renormalized": False, "audit": _audit_of(item)}
    # Metrics read counts from target, not from human-readable audit metadata.
    if item["target"].get("counts") is not None:
        target["counts"] = int(item["target"]["counts"])
    rec = {
        "id": f"{adapter}::{split}::{index:07d}",
        "source": f"public:{adapter}",
        "gen_version": PUBLIC_GEN_VERSION,
        "split": split,
        "schema": {
            "primitive": item["primitive"],
            "name": item["schema_name"],
            "desc": item["desc"],
            "positive_label": item.get("positive_label"),
        },
        "state": state,
        "state_sections": sections,
        "question": item["question"],
        "question_paraphrases": item["paraphrases"],
        "candidates": cands,
        "target": target,
        "meta": dict(item.get("meta", {}), template_id=template_id,
                     entity_pool=entity_pool, K_full=len(cands),
                     language="en"),
    }
    if item["primitive"] == "score":
        if any(c["meta"]["level"] is None for c in rec["candidates"]):
            raise AssertionError(f"{adapter}::{index}: score 样本有候选缺 level")
    return rec


class PublicAdapter:
    """适配器基类。子类只需实现 `items()`，其余交给 `build()`。"""

    name = ""
    schema_name = ""
    desc = ""
    primitive = "choice"
    provenance = "hard"
    template_id = ""
    positive_label = None
    #: 数据集自带 train/validation/test 时为 True；否则按哈希桶切分
    native_splits = False

    def items(self, seed=0):
        """yield `(key, native_split, item)`。

        `key` 是稳定且唯一的条目标识（用它做哈希切分，必须与 seed 无关，
        否则换 seed 就会让 split 漂移、旧结果 JSON 失去可比性）。
        `item` 必须含 `sections` / `question` / `paraphrases` / `candidates` /
        `primitive` / `schema_name` / `desc` / `provenance` / `target`。
        """
        raise NotImplementedError


def build(adapter, seed=0, limit=None):
    """展开适配器并按 split 分组。返回 {split: [record, ...]}。"""
    out = {s: [] for s in SPLITS}
    for i, (key, native, item) in enumerate(adapter.items(seed)):
        if limit and i >= limit:
            break
        if adapter.native_splits:
            if native == "train":
                calib = carve_calib(f"{adapter.name}|{key}")
                split, pool = ("calib", "calib") if calib else ("train", "train")
            elif native in ("val", "validation", "dev"):
                split, pool = "val", native
            else:
                split, pool = "test_known", native
        else:
            k = f"{adapter.name}|{key}"
            split = hash_split(k)
            pool = f"h{bucket(k, N_BUCKETS)}"
        out[split].append(
            finish(adapter.name, i, split, item, adapter.template_id, pool))
    return out


# ---------------------------------------------------------------------------
# 注册表。`scripts/build_dataset.py --public` 与 `scripts/audit_leakage.py` 都读它。
# ---------------------------------------------------------------------------
def all_adapters():
    from dataset.adapters import amazon_score, banking77, chaosnli, clinc150, goemotions
    return [chaosnli.ChaosNLI(), clinc150.CLINC150(), banking77.Banking77(),
            goemotions.GoEmotions(), amazon_score.AmazonScore()]


def get(name):
    for a in all_adapters():
        if a.name == name:
            return a
    raise KeyError(f"未注册的适配器 {name}；已注册：{[a.name for a in all_adapters()]}")
