"""
Amazon Reviews (en) —— 唯一的**真实语料上的序数评分**任务。

合成侧 `agent_trace_score` 提供了序数信号，但那是程序渲染出来的 state。这里给的是
真人在真评论上的一到五星，所以它是 `ordinal_mae` 与 `expected_score` 两个指标上
"合成规则能不能迁移到真实文本"的直接证据。

**provenance = `hard`**：单次星级不揭示逐条 P*，但支持观测 Brier、NLL 与分桶 ECE。

两处与合成侧刻意对齐的地方：

  1. **候选文本复用 `lexicon.LEVELS`**（`"[3] neutral"`），与 `agent_trace_score`
     完全同形。序数 loss 靠的是 `meta.level` 的整数，但编码器看到的是**文本** ——
     两边的等级词表若不一致，模型就得学两套序数表面形式，而 CDF 项在两边都只有
     一半的样本量。共用一份词表是零成本的改进。
  2. **五个等级恒全在候选里**，不做子采样。序数的 CDF 是定义在**完整**等级集上的，
     挖掉中间某一档，`cumsum` 立刻说谎（`agent_trace_score` 的注释里同样写了这条）。

`label` 是 0–4，等级是 1–5（已核对：label 0 的文本是差评，4 是好评）。

**两个必须在 README 里说清楚的采集口径问题**（数据集的形状完全正常，不会报任何错）：

  - **train split 不能用**：20 万行的 `label` 全是占位的 0。它是给 SetFit 做少样本
    采样用的，标签本来就该由使用者给。拿它当硬标签，训出来的是一个"所有评论都是一星"
    的基准。所以这个适配器只出 val / test，**它是纯评测集，不参与训练**。
  - **val / test 是均衡的**（每档恰好 1000 条），不是自然星级分布（真实评论是 J 形，
    五星占绝大多数）。所以这里的 accuracy 是**均衡准确率**，与在自然分布上量出来的
    不是同一个数。`expected_score` 和 `ordinal_mae` 同理。
"""
from dataset.adapters import PublicAdapter, sec
from dataset.synth.lexicon import LEVELS

QUESTION = "What is the review's rating?"
PARAPHRASES = [
    "How positive is the review?",
    "Rate the review on the five-point scale.",
    "Where does the review fall on the rating scale?",
    "Assign the review an ordinal rating.",
]


class AmazonScore(PublicAdapter):
    name = "amazon_score"
    schema_name = "review_rating"
    desc = ("Predict the ordinal star rating of one product review on a five-point scale. "
            "The target is a hard label, so this set is excluded from calibration "
            "metrics and reported only for accuracy and ordinal error.")
    primitive = "score"
    provenance = "hard"
    template_id = "pub"
    native_splits = True
    hf_id = "SetFit/amazon_reviews_multi_en"

    def items(self, seed=0):
        from datasets import load_dataset

        ds = load_dataset(self.hf_id)

        # **只取 validation / test，train 是废的。** 这个镜像的 train split 有 20 万行，
        # 但 `label` 全是占位的 0（它是给 SetFit 做少样本采样用的，标签由使用者给）。
        # 拿它当硬标签用会得到一个"所有评论都是一星"的基准 —— 而 `label` 字段名、
        # 取值域、以及 0..4 的合法形状全都正常，没有任何东西会报错。
        for native in ("validation", "test"):
            for i, row in enumerate(ds[native]):
                text = (row["text"] or "").strip()
                if not text:
                    continue
                gold = int(row["label"])
                if not 0 <= gold < len(LEVELS):
                    continue
                assert str(row.get("label_text")) == str(gold), (
                    f"AmazonScore: label_text {row['label_text']!r} 与 label {gold} 不符")

                cands = [{"text": f"[{k + 1}] {LEVELS[k][0]}", "label": LEVELS[k][0],
                          "meta": {"level": k + 1}} for k in range(len(LEVELS))]
                p = [1.0 if k == gold else 0.0 for k in range(len(LEVELS))]

                yield f"{native}|{row['id']}", native, {
                    "sections": [sec("review", f"Review: {text}")],
                    "question": QUESTION,
                    "paraphrases": list(PARAPHRASES),
                    "candidates": cands,
                    "primitive": self.primitive,
                    "schema_name": self.schema_name,
                    "desc": self.desc,
                    "provenance": self.provenance,
                    "target": {"kind": "soft", "p": p,
                               "provenance": self.provenance,
                               "renormalized": False,
                               "audit": {"row": i, "stars": gold + 1,
                                         "native_split": native}},
                    "meta": {"variant": self.name},
                }
