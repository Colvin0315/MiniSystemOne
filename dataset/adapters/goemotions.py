"""
GoEmotions —— **收它是为了反衬 ChaosNLI 的样本量，不是为了拿它当校准证据。**

每条约 3–5 个标注者，而且标注者**可以多选**（一条评论同时打上 joy 和 gratitude 是
合法标注）。所以这里的 `p` 有两层构造性失真，README 必须直说：

  1. **N 只有 3–5**，于是 `p` 的取值被钉在很小的分母上 —— 3 个标注者时唯一的可能值
     是 {0, 1/3, 2/3, 1}。逐条 ECE 在这种分辨率下**无法**低于二项噪声底，量到的
     大部分是标注噪声。ChaosNLI 的 N≈100 才让那条底压得下去。
  2. **多选被折成单一分布**：把票数除以**总票数**（而不是标注者数）才得到和为 1 的
     分布。这在"每人恰好选一个"时才等于真实的 P(y|x)，而 GoEmotions 不是。
     `audit` 里同时记 `n_annotators` 与 `n_votes`，就是为了让这个失真当场可见 ——
     `n_votes > n_annotators` 的样本占比越高，这张表距离真正的分布就越远。

  这两条都不是 bug，是**数据集本身的分辨率**。一个只在 N=100 上报告校准、不展示
  N=4 会退化成什么样的项目，是在藏起自己方法的适用边界。

`neutral` 是 28 个情绪里的一个（不是"没情绪"），所以它是普通候选，不做特殊处理。

**没有原生 split**，按整桶哈希切分。
"""
import collections

from dataset.adapters import PublicAdapter, sec

# 28 个情绪列。硬编码而不是从 features 里"减去非情绪列"推导 —— 后者在上游新增一列
# 元数据（比如又加一个质检字段）时会把那一列**静默当成情绪**，而候选文本变成布尔值
# 这种事不会报错，只会让准确率莫名其妙地掉。下面有断言兜底。
EMOTIONS = (
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
)

QUESTION = "Which emotion does the text express?"
PARAPHRASES = [
    "What emotion is the text expressing?",
    "Identify the emotion conveyed by the text.",
    "Which of the listed emotions best matches the text?",
    "Classify the text by the emotion it expresses.",
]

MAX_K = 32


class GoEmotions(PublicAdapter):
    name = "goemotions"
    schema_name = "emotion_distribution"
    desc = ("Predict the distribution over emotions that human annotators assigned to one "
            "short Reddit comment. Only 3 to 5 annotators see each comment and they may "
            "select several emotions, so the target is coarse and noisy by construction.")
    primitive = "choice"
    provenance = "human_annotators"
    template_id = "pub"
    native_splits = False
    hf_id = "google-research-datasets/go_emotions"

    def items(self, seed=0):
        from datasets import load_dataset

        ds = load_dataset(self.hf_id, "raw", split="train")
        missing = [e for e in EMOTIONS if e not in ds.column_names]
        assert not missing, f"GoEmotions: features 里缺情绪列 {missing}"

        # raw 是**逐标注者**的行（211k 行 ≈ 5 万条评论），必须按 id 归组才能得到分布。
        groups = {}
        for row in ds.select_columns(["id", "text", "rater_id",
                                      "example_very_unclear", *EMOTIONS]):
            cid = row["id"]
            g = groups.get(cid)
            if g is None:
                g = groups[cid] = {"text": row["text"] or "", "votes":
                                   collections.Counter(), "n": 0, "unclear": False}
                if not g["text"]:
                    del groups[cid]
                    continue
            g["n"] += 1
            if row["example_very_unclear"]:
                g["unclear"] = True
            for e in EMOTIONS:
                if row[e]:
                    g["votes"][e] += 1

        dropped_k1 = 0
        kept = 0
        for cid, g in groups.items():
            if g["unclear"]:
                continue
            votes = g["votes"]
            total = sum(votes.values())
            if total <= 0:
                continue
            # **只有一个情绪的样本必须丢弃。** 全部投票者都只投了同一个情绪（实测占
            # 14.7%），于是候选集只剩一项 —— softmax 恒等于 1，CE/Brier/梯度全恒为 0，
            # 而评测会把它记成一次正确预测。留着就是把 GoEmotions 的准确率凭空抬高
            # 十几个点，抬高的幅度恰好是这个占比。`finish()` 里有断言兜底。
            #
            # 代价要说清楚：丢掉的正是**最一致**的那批样本，所以剩下的分布比原始
            # GoEmotions 更偏"有分歧"。这对本项目的用途（拿它反衬 ChaosNLI 的 N=100）
            # 反而更贴题，但它确实不再是数据集原本的边缘分布。
            if len(votes) < 2:
                dropped_k1 += 1
                continue
            kept += 1

            # 按票数取前 K —— N≤5 时永远取不满，守卫是给上游放宽标注数留的。
            top = votes.most_common(MAX_K)
            picks = [e for e, _ in top]
            n_votes = sum(v for _, v in top)
            p = [v / n_votes for _, v in top]
            p[-1] = 1.0 - sum(p[:-1])       # 浮点残差塞进最后一项，理由同 chaosnli

            yield cid, "train", {
                "sections": [sec("comment", f"Comment: {g['text']}")],
                "question": QUESTION,
                "paraphrases": list(PARAPHRASES),
                "candidates": [{"text": e, "label": e, "meta": {"level": None}}
                               for e in picks],
                "primitive": self.primitive,
                "schema_name": self.schema_name,
                "desc": self.desc,
                "provenance": self.provenance,
                "target": {"kind": "soft", "p": p, "provenance": self.provenance,
                           "renormalized": False,
                           # 同 chaosnli：`counts` 才是 `binomial_noise_floor` 读的字段。
                           # 这里 N 只有 3–5，地板会大到肉眼可见 —— 那正是收录
                           # GoEmotions 的目的：反衬 ChaosNLI 的 N≈100。
                           "counts": g["n"],
                           "audit": {"n_annotators": g["n"], "n_votes": total}},
                "meta": {"variant": self.name},
            }

        print(f"  goemotions: 归组 {len(groups)} 条 → 保留 {kept} 条"
              f"（丢弃单情绪 {dropped_k1} 条，占 {100 * dropped_k1 / max(len(groups), 1):.1f}%；"
              f"其余为标记 unclear 或无有效投票）")
