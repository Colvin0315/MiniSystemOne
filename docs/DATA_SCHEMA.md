# MiniSystemOne 数据 schema（冻结）

一行一个 JSON 对象。生成器（`dataset/synth/*`）、公开集适配器（`dataset/adapters/*`）、
数据集（`dataset/decision_dataset.py`）、评测（`eval/*`）全部按本文件对齐。**改动即版本号
变更**，且必须同时更新 `gen_version` 与已发布的结果 JSON。

```json
{
  "id": "banking_balance::train::0000123",
  "source": "synth:banking_balance",
  "gen_version": "1.0.0",
  "split": "train",

  "schema": {"primitive": "noul", "name": "...", "desc": "...", "positive_label": "yes"},

  "state": "…渲染后的自然语言 state…",
  "state_sections": [{"seg": "account", "text": "…", "priority": 3}],

  "question": "…",
  "question_paraphrases": ["…", "…", "…"],

  "candidates": [{"text": "yes", "label": "yes", "meta": {"level": null}}],

  "target": {
    "kind": "soft",
    "p": [0.70, 0.30],
    "provenance": "explicit_rng",
    "renormalized": false,
    "audit": {"margin": -412.5, "q_raw": 0.7011, "tau": 250.0}
  },

  "meta": {"K_full": 2, "approx_tokens": 210, "template_id": "t07", "entity_pool": "p3"}
}
```

## 字段

| 字段 | 约束 |
|---|---|
| `id` | `<generator>::<split>::<序号>`，全局唯一 |
| `source` | `synth:<生成器名>` 或 `public:<适配器名>` |
| `gen_version` | 语义化版本。**测试集只用冻结版本重生成**，版本号写进每个结果 JSON |
| `split` | `train` / `val` / `calib` / `test_known` / `test_ood` |
| `schema.primitive` | `noul` / `choice` / `score`。**三者共用一份 forward 与一份 loss**，差别只在这里与 `is_ord` |
| `schema.positive_label` | 仅 `noul` 用。**正向标签由它指定，绝不按下标** —— 否则候选打乱后 `p_yes` 语义会翻转 |
| `state` | 渲染后的自然语言，模型实际读到的就是它 |
| `state_sections[].seg` | 段名，取自 `lexicon.SECTIONS`。供段落感知截断按 `priority` 丢弃 |
| `state_sections[].priority` | 越大越先被保留；截断时先丢小的 |
| `question` | 训练用的问题文本 |
| `question_paraphrases` | ≥3 条复述。**不参与训练**；本仓库目前没有任何脚本消费它（原本供一套复述扰动评测用，该评测未产出） |
| `candidates[].text` | 候选文本，顺序即呈现顺序。生成器与适配器**必须逐例重排** |
| `candidates[].label` | 候选的稳定标识，用于正确性判定与 target 对齐 |
| `candidates[].meta.level` | 仅 `score` 用，1–5 的序数等级 |
| `target.kind` | `soft`（有完整分布）或 `hard`（one-hot，`p` 退化为 0/1） |
| `target.p` | 长度 = `len(candidates)`，**和为 1**。若是子采样后的目标，须重新归一化 |
| `target.provenance` | `explicit_rng` / `marginalized` / `tie_set` / `human_annotators` / `hard` |
| `target.renormalized` | 候选被 K 子采样后目标重新归一化过则为 `true` |
| `target.counts` | **标注者人数**（标量，逐样本）。仅 `human_annotators` 有；合成集的 `P*` 是精确值，所以没有这个字段、地板本就是 0。它是 `binomial_noise_floor` 的**唯一**输入 —— 不写它，报告里"噪声校正后 ECE"那一行会静默消失 |
| `target.audit` | 生成器内部量（`margin`、`q_raw`、`tau`、隐藏 `h`…）。供 oracle 上界与特征充分性测试，**不进模型** |
| `meta.K_full` | 候选超集大小。评测用全集，训练子采样 |
| `meta.template_id` | 模板标识。**split 按 template_id × entity_pool 划分**，绝不随机逐条划分 |
| `meta.entity_pool` | 实体池标识，取自 `lexicon.POOLS` |
| `meta.approx_tokens` | 预估 token 数，供 `CandidateBucketSampler` 与截断预算使用 |

## provenance 与指标的关系

`target.provenance` 决定该样本**进入哪些指标**。混在一起算 ECE 没有意义，因为各来源的
不可约噪声底不同。

| provenance | 含义 | 来源 | 校准指标 |
|---|---|---|---|
| `explicit_rng` | 目标是已知的随机决策规则 | 合成 | ✅ |
| `marginalized` | 目标是隐藏变量的边缘化 | 合成 | ✅ |
| `tie_set` | 状态不决定唯一答案，目标是有效集上的均匀分布 | 合成 | ✅ |
| `human_annotators` | 真实人类标注分歧分布 | ChaosNLI / GoEmotions | ✅ |
| `hard` | 硬标签 | CLINC150 / banking77 / Amazon | ❌ **排除在所有校准指标之外** |

## 三条硬性约束（由 audit 脚本强制）

1. **split 按 `template_id` × `entity_pool` 划分** —— 这是 `test_known` 成为真正留出集的前提。
   `build_dataset.py` 里有断言，不满足直接失败。
2. **候选逐例重排** —— 防止位置泄漏。`scripts/audit_leakage.py` 的位置探针（特征只有
   `(position_onehot, K)`）在每个 split 上必须只达随机水平。
3. **软目标必须可追溯** —— `target.p` 只能由生成器自身逻辑、从它明确知道"是否渲染进了
   `state`"的量算出，并在 `target.audit` 留下记录。随手写死的软数字**禁止合入**。
   `audit_synthetic.py` 的特征充分性测试会从渲染后的 state 文本里抽取数值字段做逻辑回归，
   无法在容差内恢复 `P*` 的样本即为渲染 bug，丢弃并报告丢弃率。
