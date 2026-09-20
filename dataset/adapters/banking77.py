"""
Banking77 —— "难负例真的难"的测试集。

77 个银行客服意图，全部是**同一个域内的近邻**（`card_arrival` vs `card_delivery_estimate`、
`pending_card_payment` vs `pending_transfer`）。与 CLINC150 的区别就在这里：CLINC 的
候选来自 150 个互不相干的域，随机采到的负例大多是白送的；banking77 随便采一个都是
真难。两组数字放在一起，才能把"准确率掉了"归因到候选难度而不是模型变差。

没有 `oos`，所以没有弃权候选 —— 弃权监督由 CLINC150 的 `oos` 和 `security_gate`
提供，这里不重复。

数据集**没有原生 split 之外的划分**，只有 train/test，所以 `carve_calib` 会从 train
里切出 calib（见 `dataset/adapters/__init__.py`）。calib 是拟合温度用的，永不训练。

**用 `datasets.load_dataset` 直连，不走 HF datasets-server**：`PolyAI/banking77` 在
server 上没有 parquet 端点，`/splits` 直接 500。这不是可选的实现风格，是唯一能取到
数据的路径。
"""
import random

from dataset.adapters import PublicAdapter, sample_k, sec

QUESTION = "Which banking intent does the customer query express?"
PARAPHRASES = [
    "What is the customer asking about?",
    "Classify the customer query by banking intent.",
    "Which of the listed banking intents matches the query?",
    "Identify the banking intent behind the query.",
]


class Banking77(PublicAdapter):
    name = "banking77"
    schema_name = "banking_intent"
    desc = ("Route one short customer query to one of 77 fine-grained banking intents. "
            "All intents share a domain, so the negatives are genuinely confusable.")
    primitive = "choice"
    provenance = "hard"
    template_id = "pub"
    native_splits = True
    hf_id = "PolyAI/banking77"

    def items(self, seed=0):
        from datasets import load_dataset

        # `trust_remote_code=True` 是**必需**的：PolyAI/banking77 用自定义加载脚本发
        # 数据集，不加这个参数 datasets 会直接拒绝加载。
        ds = load_dataset(self.hf_id, trust_remote_code=True)
        names = list(ds["train"].features["label"].names)
        n = len(names)

        for native in ("train", "test"):
            for i, row in enumerate(ds[native]):
                text = (row["text"] or "").strip()
                if not text:
                    continue
                gold = int(row["label"])
                key = f"{native}|{text}|{gold}"
                rng = random.Random(f"{self.name}|{key}")

                others = [j for j in range(n) if j != gold]
                picks = rng.sample(others, sample_k(rng, n) - 1)
                cands = [{"text": names[gold], "label": names[gold],
                          "meta": {"level": None}}]
                cands += [{"text": names[j], "label": names[j],
                           "meta": {"level": None}} for j in picks]
                p = [1.0 if c["label"] == names[gold] else 0.0 for c in cands]

                yield key, native, {
                    "sections": [sec("query", f"Customer query: {text}")],
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
                               "audit": {"row": i, "native_split": native}},
                    "meta": {"variant": self.name},
                }
