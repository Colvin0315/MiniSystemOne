"""
CLINC150 —— 全套里最强的准确率信号，也是"候选集是动态的"最干净的测试。

150 个意图 + 一个 `oos`（out-of-scope）。每个样本的候选 = 正确意图 + 若干负例，
所以候选集是**真的**逐例不同，而不是同一个 150 路分类头换了个说法。

两处刻意的设计：

  1. **约 1/8 的样本给全部 150 个候选。** 其余给 K∈[4,32] 的随机子集。这不只是为了
     "混合难度"：K=150 是**真实数据走大 K 路径**的唯一机会，而大 K 路径（分块前向、
     prefix KV 复用）正是本项目最强的效率主张。合成侧只有 `calendar_slot` 会到那个
     量级，真实侧一个都没有的话，那条路径就只在合成分布上被验证过。
     `decision_dataset` 训练时会把它子采样回 ≤32，所以这只影响评测与超集形态。

  2. **`oos` 的答案是一个 `<abstain>` 候选，不是第 151 个类。** 与 `security_gate`
     共用同一个特殊 token —— 弃权是**同一个概念**，两个来源各造一个词表项，模型就得
     学两遍。`ABSTAIN_TEXT` 从那里导入而不是各写一份，就是为了让这条契约不可能漂移。

**provenance = `hard`**：单一观测标签不揭示逐条 P*，但支持观测 Brier、NLL 与分桶 ECE。
"""
import random

from dataset.adapters import PublicAdapter, sample_k, sec
from dataset.synth.security_gate import ABSTAIN_TEXT

QUESTION = "Which intent does the utterance express?"
PARAPHRASES = [
    "What is the intent behind the utterance?",
    "Classify the utterance by intent.",
    "Which of the listed intents matches the utterance?",
    "Identify the user's intent.",
]


class CLINC150(PublicAdapter):
    name = "clinc150"
    schema_name = "intent_routing"
    desc = ("Route one short user utterance to one of 150 in-scope intents, or abstain "
            "when the utterance is out of scope. The candidate set is drawn per example.")
    primitive = "choice"
    provenance = "hard"
    template_id = "pub"
    native_splits = True
    hf_id = "clinc/clinc_oos"

    def items(self, seed=0):
        from datasets import load_dataset

        ds = load_dataset(self.hf_id, "plus")
        # 意图名从 features 里取，不硬编码 150 个字符串：上游改名单时我们会跟着变，
        # 而不是静默地把候选文本换成旧名字。
        names = list(ds["train"].features["intent"].names)
        # **`oos` 不在末位**，它是 index 42。按"最后一个"取会得到一个叫 `change_volume`
        # 的普通意图 —— 那样 oos 样本会被当成普通意图去抽负例，而 `change_volume`
        # 会变成一个永远正确的弃权候选类别。用名字查，不猜位置。
        oos = names.index("oos")
        in_scope = [i for i in range(len(names)) if i != oos]
        n_scope = len(in_scope)

        for native in ("train", "validation", "test"):
            for i, row in enumerate(ds[native]):
                text = (row["text"] or "").strip()
                if not text:
                    continue
                gold = int(row["intent"])
                key = f"{native}|{text}|{gold}"
                rng = random.Random(f"{self.name}|{key}")

                if gold == oos:
                    picks = rng.sample(in_scope, sample_k(rng, n_scope) - 1)
                    cands = [{"text": names[j], "label": names[j],
                              "meta": {"level": None}} for j in picks]
                    cands.append({"text": ABSTAIN_TEXT, "label": "abstain",
                                  "meta": {"level": None}})
                    answer = "abstain"
                else:
                    others = [j for j in in_scope if j != gold]
                    picks = rng.sample(others, sample_k(rng, n_scope) - 1)
                    cands = [{"text": names[gold], "label": names[gold],
                              "meta": {"level": None}}]
                    cands += [{"text": names[j], "label": names[j],
                               "meta": {"level": None}} for j in picks]
                    answer = names[gold]

                p = [1.0 if c["label"] == answer else 0.0 for c in cands]
                yield key, native, {
                    "sections": [sec("utterance", f"Utterance: {text}")],
                    "question": QUESTION,
                    "paraphrases": list(PARAPHRASES),
                    "candidates": cands,
                    "primitive": self.primitive,
                    "schema_name": self.schema_name,
                    "desc": self.desc,
                    "provenance": self.provenance,
                    "target": {"kind": "soft", "p": p,
                               "provenance": self.provenance,
                               "renormalized": False, "audit": {"row": i,
                                                                "native_split": native}},
                    "meta": {"variant": self.name},
                }
