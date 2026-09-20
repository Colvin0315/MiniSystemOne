"""ChaosNLI empirical annotation distributions; finite frequencies are not exact P*."""
from dataset.adapters import PublicAdapter, sec

# `label_dist` 的分量顺序是 [e, n, c]。这一条容易搞反 —— "contradiction" 的首字母是
# c，但它排在最后，不是第二。
ORDER = ("e", "n", "c")
TEXT = {"e": "entailment", "n": "neutral", "c": "contradiction"}

QUESTION = "Does the premise entail the hypothesis?"
PARAPHRASES = [
    "Is the hypothesis entailed by the premise?",
    "Given the premise, does the hypothesis follow?",
    "Judge whether the hypothesis is supported by the premise.",
    "What is the logical relationship between the premise and the hypothesis?",
]


def _f(x):
    """numpy 标量转 Python float —— json.dump 不认 numpy 类型，会直接 TypeError。"""
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _normalize(p):
    """归一化并把浮点残差塞进最后一项。

    `finish()` 要求目标和与 1 的偏差小于 1e-9。`[x/s for x in dist]` 在多数整数分布上
    已经够准，但把最后一维改成 `1 - sum(其余)` 能让它是**构造上**精确的，省得某个
    分布恰好卡在容差边缘上，让整次构建以一个看不出原因的 AssertionError 结束。
    """
    s = sum(p)
    out = [x / s for x in p]
    if len(out) > 1:
        out[-1] = 1.0 - sum(out[:-1])
    return out


class ChaosNLI(PublicAdapter):
    name = "chaosnli"
    schema_name = "nli_three_way"
    desc = ("Three-way natural language inference over one premise and one hypothesis. "
            "The target is the distribution of roughly 100 human annotators, so the "
            "irreducible noise is real human disagreement rather than a program rule.")
    primitive = "choice"
    provenance = "human_annotators"
    template_id = "pub"
    native_splits = False
    hf_id = "metaeval/chaos-mnli-ambiguity"

    def items(self, seed=0):
        from datasets import load_dataset

        ds = load_dataset(self.hf_id, split="train")

        for i, row in enumerate(ds):
            premise = (row.get("premise") or "").strip()
            hypothesis = (row.get("hypothesis") or "").strip()
            dist = row.get("label_dist") or []
            counts = row.get("label_count") or []
            counter = row.get("label_counter") or {}
            if not premise or not hypothesis or len(dist) != 3:
                continue

            # `label_dist` 是 `label_count` 归一化来的，两者同序（[e, n, c]），而
            # `label_counter` 是**按名字**索引的 —— 用它对一遍序，是唯一能挡住上游
            # 悄悄重排 `label_dist` 的检查。重排之后目标和仍然是和为 1 的合法分布，
            # 一切照跑，只是 e 与 c 的语义被对调了，而所有指标都会因此变差。
            #
            # `or 0` 不能省：1599 行里有 174 行把"某类零票"编码成了 `None` 而不是 0
            # （Arrow 的缺失值表示）。直接 float() 会 TypeError，`get(c, 0)` 也挡不住 ——
            # 键是存在的，值是 None。
            for j, c in enumerate(ORDER):
                if abs(float(counter.get(c) or 0) - float(counts[j])) > 0.5:
                    raise AssertionError(
                        f"chaosnli: label_dist/label_count 的第 {j} 位（应为 {c}）与 "
                        f"label_counter 不符 —— 上游可能改了分量顺序")

            vals = [float(x) for x in dist]
            if sum(vals) <= 0:
                continue
            p = _normalize(vals)
            # Use the annotation counts from the same source as label_dist.
            n_ann = int(round(sum(float(x) for x in counts))) or int(round(sum(vals)))

            # key 用数据集自带的 `uid`：它稳定、唯一，且与行号无关。用行号的话，
            # 数据集重传或重排会让同一篇 premise 从 train 静默跳到 test_known。
            key = row.get("uid") or f"{premise}||{hypothesis}"

            yield key, "train", {
                "sections": [sec("premise", f"Premise: {premise}"),
                             sec("hypothesis", f"Hypothesis: {hypothesis}")],
                "question": QUESTION,
                "paraphrases": list(PARAPHRASES),
                "candidates": [{"text": TEXT[c], "label": c, "meta": {"level": None}}
                               for c in ORDER],
                "primitive": self.primitive,
                "schema_name": self.schema_name,
                "desc": self.desc,
                "provenance": self.provenance,
                "target": {"kind": "soft", "p": p, "provenance": self.provenance,
                           "renormalized": False,
                           # Counts describe annotation uncertainty, not an ECE lower bound.
                           "counts": n_ann,
                           "audit": {"n_annotators": n_ann,
                                     "entropy": _f(row.get("entropy")),
                                     "gini": _f(row.get("gini")),
                                     "row": i}},
                "meta": {"variant": self.name},
            }
