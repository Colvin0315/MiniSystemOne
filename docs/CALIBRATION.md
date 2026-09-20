# 校准文档

本项目最核心的主张是**原生校准的概率输出**。本文说明：校准在这个设定下意味着什么、
为什么参数化只能是标量温度、怎么拟合、怎么报告、以及在什么情况下报告的 ECE
**不能**被解释成模型的校准误差。

---

## 1. 为什么 one-hot 标签学不出校准

先讲清楚要解决的问题，否则后面每一步看起来都像无谓的谨慎。

Brier loss 作用在 one-hot 目标上时，退化为一个"把置信度推向极端"的正则项。
它**不会**教会模型"我报 0.8 的时候应该 80% 正确"——因为训练信号里根本没有
"80%" 这个信息。模型能学到的最好东西是"这个样本我确定"，而这恰好是最不校准的行为。

**要演示校准，训练数据里必须有已知真实条件分布 `P*(y|x)` 的样本。**
这不是我们选的技巧，是这条任务线的硬前提。

于是数据来源决定了一切。按 `target.provenance` 分桶：

| provenance | 含义 | 来源 | 进校准指标? |
|---|---|---|---|
| `explicit_rng` | 目标是已知的随机决策规则 | 合成 G1 `banking_balance` | ✓ |
| `marginalized` | 目标是隐藏变量的边缘化 | 合成 G3 `agent_trace_score` / G4 `security_gate` | ✓ |
| `tie_set` | 状态确实不决定唯一答案，目标为有效集上的均匀分布 | 合成 G2 `tool_router` / G4 | ✓ |
| `human_annotators` | 真实人类标注分歧分布 | ChaosNLI (N≈100) | ✓ |
| `hard` | 硬标签，**不可约噪声底未知** | CLINC150 / banking77 / Amazon | **✗** |

### 两条纪律

**纪律 1：`hard` 排除在所有校准指标之外。**
硬标签的 `P*` 是 one-hot，它对"真实的不可约不确定性"一无所知 ——
一条本质上有歧义的 CLINC150 样本，其 `P*` 依然是一个尖峰。
拿它算 ECE，度量的其实是标注噪声而不是模型。

**纪律 2：所有指标按 provenance 分桶报告。**
混在一起算 ECE 没有意义：不同来源的不可约噪声底不同（`tie_set` 的底接近 0，
`human_annotators` 的底由标注人数决定），混样的 ECE 度量的是混样比例。
`eval/eval_harness.py` 的输出 JSON 里 `by_provenance` 是一等字段就是这个原因。

---

## 2. 为什么参数化只能是**标量温度**

这一节要显著地讲，因为它让校准故事自洽而非事后技巧。

**候选集是动态的，所以 per-class 的参数化没有指称对象。**
候选 `k=3` 在样本 A 里是 "transfer_ownership"，在样本 B 里是 `Neutral`，
在样本 C 里是一个从未见过的 intent。一个 per-class 的温度向量（或 per-class
的 bias / prior 修正）在这些 `k=3` 之间没有任何共享语义 —— 它拟合的是
"位置 3"，而位置已被证明与答案无关（`audit_leakage.py` 的门禁）。

**标量温度是唯一能跨候选集迁移、且能用于模型从未见过的 schema 的校准参数化。**

这一点还顺带解释了为什么 `DecisionHead` 里那两层 LayerNorm 是必要的：
它们把 logit 的**尺度**约束住，使其不随 K 与 primitive 漂移。若 logit 尺度在
K=2 与 K=32 之间差一个量级，单一温度就不够用了 —— 而多温度又回到上面那个
没有指称对象的死路。

---

## 3. 拟合方法

`trainer/calibrate_temperature.py`。

### 参数化：拟合 `log T`，不是 `T`

```python
def objective(log_temp):
    return -(t * torch.log_softmax(lp / log_temp.exp(), dim=-1)).sum(-1).mean()
```

拟合 `log T` 而不是 `T`，理由是**梯度尺度均匀**。`T` 的正半轴是 `(0, ∞)`，
直接优化 `T` 会让"T 应该小一点"和"T 应该大一点"的步长尺度完全不对称 ——
LBFGS 的曲率估计在这种参数化下很差。`log T` 上优化是对称的，且 `T > 0`
由构造保证，不需要投影。

### 入参用 `log_p` 而不是 `p`，且**显式**屏蔽 padding 槽

`log_softmax` 吃的是 logit 空间，屏蔽槽只要被推到足够负，就在归一化项里严格为 0。
若改传概率 `p` 再算 `p**(1/T)` 归一化，就需要让 0 在除法里活下来，得多写一层 clip。
数值上更稳的那条路更好走。

**两处必须写对，写错了数字仍然会出来，只是不再是拟合的那套：**

1. **屏蔽槽必须真的被屏蔽，不能靠"上游已经把它变成很小的概率"。**
   上游传进来的是 `log(clip(p, 1e-30)) = -69.08`，不是 `-inf`。它在 `t=0` 处不贡献
   损失，却**会**通过 `log_softmax` 的归一化项影响其余槽位。T≈1 时 `exp(-69)` 可以
   忽略，T 大时不行 —— T=12、128 槽里 126 个是 padding 时，实测拟合出的 T 从真值
   **15.0 掉到 5.99**。padding 最多的 `(33,255)` 桶正是最容易中招的地方。
2. **屏蔽值用有限的大负数（`NEG_LP = -1e4`），不用 `-inf`。**
   `-inf` 的泄漏确实严格为零，但 `log_temp` 的梯度里会出现 `0 · (-lp/T²)` =
   `0 · ∞` = NaN，LBFGS 一步就把 T 变成 NaN —— 失败得比第 1 条更彻底，也更显眼。

### 优化器与钳位

- LBFGS，`line_search_fn="strong_wolfe"`，`max_iter=200`（默认）。
- `float64` 全程。**不是过度谨慎**：温度拟合的目标函数对 `log T` 的曲率在
  接近最优时可能很小，fp32 的梯度噪声会让 LBFGS 在最优附近震荡。
  这个计算只在 6k 样本上做一次，代价可以忽略。
- 钳位 `[0.05, 20]`（`T_MIN` / `T_MAX`）。**钳位是诊断而非保护**：
  一个正常的模型不会拟合出 `T` 贴边界。若它贴了边界，说明数据或 logit 有问题
  （见 §6），这时应该去看而不是让钳位悄悄兜住。

### 拟合集必须是 `calib` split

```
train 180k  |  val 6k  |  calib 6k（只用于拟合温度，永不训练）  |  test_known 18k
```

**`calib` 永不参与训练。** 在训练集上拟合温度、再在同一集合上报告 ECE，
得到的是一个必然偏乐观的数字（温度会过拟合到训练集的具体 logit 分布）。
6k 校准样本是独立留出的，这一点由 `build_dataset.py` 的 split 划分强制。

---

## 4. 三个粒度，全部报告

```python
K_BUCKETS = ((2, 2), (3, 4), (5, 8), (9, 32), (33, 255))
```

| 粒度 | 参数个数 | 说明 |
|---|---|---|
| `global` | 1 | 单一温度，跨全部样本 |
| `primitive` | 3 | noul / choice / score 各一个 |
| `primitive_k` | 15 | primitive × K 桶 |

**三数对比比一句断言更好。** 只报告 global 温度就宣称"标量温度够用"是循环论证 ——
它当然够用，因为那是你唯一拟合的东西。给出三个粒度、报告各自的 NLL 改善，
读者才能自己判断"再加粒度还有多少收益"。

6k 校准样本下，15 个桶每桶仍有数百样本（`primitive_k` 的每格 ≥ 约 130 个，
最窄的 `(2,2)` 桶样本最多）。这个量级足够拟合一个标量。

结果存 `out/calibration/T.json`，**附 tokenizer 与 checkpoint 的哈希**。
哈希是必需的：温度是绑定到具体权重的，换 checkpoint 或换 tokenizer 后
旧的 `T.json` 就失效了。没有哈希，读者无法判断一个 `T.json` 是否适用于手上的模型。

---

## 5. 报告的纪律（R6）

**优化目标与报告指标错配是这个领域最常见的自欺方式。**
典型形态：优化 Brier（proper score，有良好梯度）却报告 ECE（非 proper score，
且可以通过换分桶改善）。

因此：

> **ECE + Brier + NLL + 可靠性图恒一起报告，默认 equal-mass 分桶，
> 分桶方案打进每个输出 JSON。单独变动的指标不予采信。**

### 为什么 ECE 默认用 **equal-mass** 分桶而不是 equal-width

等宽分桶在小模型的**过度自信尾部**会被主导。一个 26M 模型在真实数据上会产生
一批 `p≈0.99` 的样本，等宽分桶会把它们全塞进 `[0.9, 1.0]` 这一格，
于是 ECE 的主要贡献来自那一格 —— 而 `[0.1, 0.2]`、`[0.2, 0.3]` 这些格子里
可能只有几十个样本，统计噪声大且被稀释。结果是 ECE **看起来比实际好**。

equal-mass 分桶让每格样本数相同，噪声均匀，且尾部的一格只占 1/N 的权重。

但分桶方案本身会影响 ECE 的数值，所以**分桶方案必须写进输出 JSON**。
这是"单独变动的指标不予采信"的具体执行方式。

---

## 6. `binomial_noise_floor`：为什么它是一等函数

`eval/eval_metrics.py` 里的 `binomial_noise_floor` **不是脚注，是必需的报告项**。

ChaosNLI 每条样本约 100 个人工标注。当真实 `p = 0.5` 时，
`p̂ = k/100` 的标准误约 `sqrt(0.25/100) = 0.05`。
也就是说，**逐条 ECE 无法测到 0.05 以下** —— 一个完美校准的模型，
对着这些含噪的目标，也会表现出约 0.05 的 ECE。

不报这个数字，读者会把 0.05 解读成模型没校准；
报了这个数字，0.05 才能被正确解读成"这是标注噪声的底，不是模型的误差"。

所以报告格式是：

> 原始 ECE **和** 噪声校正后的 ECE（同标注数下的理想校准模型的期望 ECE）。

展示这个校正，是严谨的校准报告与幼稚报告的分界线。GoEmotions（N=3..5，
使 `p ∈ {0, 0.2, ...}`）之所以被收录，**正是为了反衬 ChaosNLI 的 N=100** ——
README 要大声说明这一点，否则读者会以为 GoEmotions 的坏数字是模型的问题。

---

## 7. 什么情况下报告的 ECE 不能解释成模型的校准误差

这一节是本文最重要的一节。ECE 度量的是"模型 vs 我们给它的目标 `P*`"的差距。
当 `P*` 本身有问题时，ECE 度量的是**生成器 bug**。

三种情形：

### (a) `P*` 依赖未渲染进 state 的信息（R1）

若 `P*` 依赖一个模型看不见的量，**没有模型能学到它**，报告的 ECE 会是一个
看似合理但毫无意义的坏数字。

缓解（三条都便宜，全做，见 `docs/DESIGN.md` §12）：

1. **oracle 上界** —— 用生成器内部特征（`audit.margin`、`q_raw`、隐藏 `h`）
   训一个小 MLP，报其 ECE，应 ≈0。**这是唯一能把"模型未校准"与"目标不可学"
   分开的手段。**
2. **特征充分性测试** —— 从**渲染后的 state 文本**抽取数值字段做逻辑回归，
   验证能否在容差内恢复 `P*`。失败者 = 渲染 bug，丢弃并报丢弃率。
3. **`gen_version` 冻结** —— 测试集只用冻结版本重生成，版本号记入每个结果 JSON。

### (b) `hard` 标签被混进校准指标

见 §1 纪律 1。混合 provenance 的 ECE 不予报告。

### (c) 软目标是被随手写死的数字

**禁止合入。** 合法软目标（见 `docs/DESIGN.md` §11）的共同点是：
目标由生成器自身逻辑、从它**明确知道"是否渲染进了 state"**的量算出，
且 `audit` 字段里留下记录。随手写死的 `[0.7, 0.3]` 要么不可学（模型只能学到均值），
要么让 ECE 度量生成器的想象力。

---

## 8. 温度校准**不**做什么

写清楚否定项，避免读者以为它是万能药。

- **它不改变准确率。** 温度只重新缩放 logits，`argmax` 不变。
  所以 `accuracy` / `accuracy_soft` / `schema 错误率` 在温度拟合前后**完全相同** ——
  这一点在 `eval/eval_harness.py` 的输出里可以直接核（`温度/xxx` 那几行的 `acc`
  必须与未校准行逐位相同）：若拟合后准确率变了，说明实现有 bug。

- **它不修正系统性的先验漂移。** 温度是一个对称的缩放，
  无法表达"模型整体上把 `yes` 想得太多"。要修这个需要 per-class 参数化，
  而没有指称对象（见 §2）。所以本项目**不做** prior correction，
  并且如果一个模型的 ECE 主要来自系统性漂移而非过/欠自信，
  温度会修不动它 —— 这本身是一个应该被报告的结果，不是需要隐藏的失败。

- **它不能外推到比训练时更大的 K。** `K_BUCKETS` 的最大桶是 `(33, 255)`，
  拟合温度时这个桶里的样本 K 分布与推理时若差别很大（例如推理全用 K=255），
  该桶的温度会是次优的。**本项目尚未跑测这一点**（原计划由一套"变候选数"的评测
  暴露），所以它是一个已知的未验证缺口，不是已验证的结论。

- **它不适用于 OOD。** 见 README 的诚实边界：真实 OOD 输入上的校准退化是应当
  被展示的，不是被温度掩盖的。

---

## 9. 旗舰实验（步骤 10）的复现路径

四步，注意中间的依赖方向：**温度必须先拟合，再让 harness 用它产出逐样本结果，
图只读 harness 的输出。** 图不读 checkpoint，也不读 T.json —— 它读的是已经
应用过温度的 `per_sample`。

```bash
# 1. 训练决策模型（产物 out/decision/decision.pth）
python trainer/train_decision.py --encoder out/mlm/mlm.pth

# 2. 在 calib split 上拟合三种粒度的温度 -> out/calibration/T.json
#    （T.json 里含 ckpt / tokenizer 的 sha1，用来确认它与手上的模型配套）
python trainer/calibrate_temperature.py \
    --ckpt out/decision/decision.pth \
    --data dataset/synth \
    --out out/calibration

# 3. 在**留出集**上用拟合好的温度跑 harness，产出逐样本结果
#    test_known 给"已知 P*"下的校准；公开集给真实人类分歧下的校准。
#    两个都要跑 —— 只出 test_known 的漂亮图是本项目最容易被犯的不诚实行为。
python eval/eval_harness.py --ckpt out/decision/decision.pth \
    --data dataset/synth --sets test_known \
    --temperature out/calibration/T.json --out out/eval/decision
python eval/eval_harness.py --ckpt out/decision/decision.pth \
    --data dataset/public --sets test_known \
    --out out/eval/public            # 真实集不套合成温度，见下方注

# 4. 出图（含二项噪声带与每桶 n）
#    `--eval` 要到 checkpoint 那一层：harness 把结果写成
#    `<out>/<ckpt 名>/<数据集>_<split>.json`，文件名带数据集是为了让合成集与
#    公开集的两跑能共存于一个 `--out` 根（否则第二次会静默覆盖第一次）。
python eval/make_reliability_plot.py \
    --eval out/eval/decision/decision --sets test_known \
    --binning equal_mass --out assets

# 5. 真实人类分歧那一张。`--source` 在这里是**必需的**，不是可选项：
#    `human_annotators` 同时盖住 ChaosNLI（N≈100）与 GoEmotions（N=3–5），
#    两者噪声底差一个数量级以上，混成一根柱子度量的是混合比例。
python eval/make_reliability_plot.py \
    --eval out/eval/public/decision --sets test_known \
    --source public:chaosnli --binning equal_mass --out assets
```

> **注：不要把合成集上拟合的温度套到 ChaosNLI 上。** 两个集的 logit 分布不同，
> 温度是从 `test_known` 那类样本的置信度分布里拟合出来的。真实集上的正确做法是
> 单独在它的 dev split 上拟合，或者直接报未校准的原始 ECE。
> 混用会产出一个好看但错误的数字。

**图上有两件事必须出现，否则这张图不成立：**

1. **二项噪声带**（来自 `binomial_noise_floor`），让读者看到"完美模型长什么样"。
2. **每桶样本数 n**，让读者能判断哪些桶的数字可信。

两个评测集并排：

- **`test_known`** —— 已知 `P*`，理想校准可达 ≈0（减去噪声底）。
  **但注意它也是合成数据**，所以它证明的是"模型在被训练的那类分布上校准了"，
  不是"模型在真实世界校准了"。
- **公开集里的 ChaosNLI 子集**（`dataset/public`，`source=public:chaosnli`，
  474 条）—— 真实人类分歧分布，N≈100。**这是唯一真正的真实校准证据**，
  且必须在 README 里配合噪声底一起解读。

**只报 `test_known` 的漂亮校准图、不提 ChaosNLI，是本项目最容易被犯的
不诚实行为。** 两张图恒一起出。

> 它不是一个叫 `chaosnli_dev` 的 split。ChaosNLI 是 `dataset/public/test_known.jsonl`
> 里 `source == "public:chaosnli"` 的那 474 条 —— 与 CLINC150、banking77、
> GoEmotions、Amazon 同住一个文件。所以取这一张图必须用
> `--source public:chaosnli` 过滤，而不是换一个 `--sets`。
