# MiniSystemOne

> **从 0 训练一个概率决策模型 —— 不用 LLM，不做解码，不生成 JSON。**

一个约 2700 万参数的模型：输入 **state + 类型化问题**，输出**带候选概率的类型化决策**，
**单次并行前向**完成。是否校准需要实测，并非结构保证。没有自回归循环，不生成文本，不靠受限解码兜底。

从随机初始化开始训练，风格对齐 MiniMind：单一共享双向编码器、单一决策头、单一 loss。
历史完整实验数字使用一台 8 GB 笔记本 GPU 与原始语料测得，不是 quickstart 结果，
也不保证换硬件或换数据后得到相同数字。

## 从这里开始：三条路线

在仓库根目录执行命令，先安装[复现](#复现)一节的环境。模型训练/推理使用 CUDA，
推荐路线不回退到 CPU。**目前没有公开权重下载 URL**；需要本地已有配套权重和 tokenizer。

1. **已有本地权重，直接体验。** 跑过 quickstart 后可输入无标签请求：
   ```bash
   python inference.py --ckpt out/quickstart/decision/decision.pth --tokenizer out/quickstart/tokenizer --input examples/inference/choice.json
   # 可选：应用配套温度文件
   python inference.py --ckpt out/quickstart/decision/decision.pth --tokenizer out/quickstart/tokenizer --input examples/inference/choice.json --temperature out/quickstart/calibration/T.json
   ```
   另外两种 primitive 用 `noul.json` / `score.json`。不传温度会明确标记未校准。
   换成本地其他 checkpoint 时必须同时配对其 tokenizer 和温度文件。
   CLI 返回 JSON 是程序序列化，不是模型自回归生成 JSON。
2. **离线从零训练玩具模型。** 安装依赖后执行：
   ```bash
   python scripts/quickstart.py
   ```
   双语合成 train 模板、h128/L2、MLM/决策各 20 次更新，依次跑 tokenizer→数据→MLM→
   决策→温度拟合→留出评测→三例推理，产物隔离在 `out/quickstart/`，不静默覆盖已有目录。
   明确跳过正式压缩率门禁，但仍验证特殊 token。它教完整流程，**不代表业务效果，
   不复现下面 26.89M 历史实验表**。
3. **完整实验复现。** 见[复现](#复现)。自行取得自然语料，准备
   `dataset/pretrain_zh.jsonl` 和 `dataset/pretrain_en.jsonl` 后逐阶段运行。
   训练命令不会神奇地下载原始语料；替换来源就是新实验，不能宣称复现历史表。

本次 RTX 4070 Laptop 实测：按[固定配置](configs/quickstart.json)全流程 **90.5 秒**；
MLM 长度 256、决策长度 2048，以容纳大候选集。训练峰值已分配张量显存分别为
0.097/0.210 GB，**不是整卡显存需求**。96 条留出样本准确率 29.2%，只能证明流程跑通，
不能作为业务效果；[实测摘要](results/quickstart.json)可核验。其他显存档位未实测。

下一步：[第一个自定义任务](docs/FIRST_TASK.md)，客服工具路由与人工接管。
新增 schema/候选只是输入契约支持，不等于学会新任务。默认独立候选评分在固定前缀与
温度 T 下满足 `p_i/p_j=exp((s_i-s_j)/T)`，新增其他候选只改变归一化，不改变旧两项
概率比；这限制集合依赖推理。低分桶 ECE 不证明单条请求正确或 OOD 安全。


---

## 什么是 "System One" 模型

2026-09-15，TypeSafe AI 发布了 **Jev** —— 第一个 "System One Model"。它**完全不生成文本**：
你给它一份 **state**（非结构化上下文）和一组**类型化问题**，它**一次并行前向**返回
**带概率的类型化决策**。

以下是它官方文档里公布的特性：

| 特性 | 具体内容 |
|---|---|
| **三个 primitive** | `Noul` —— 是/否问题的 P(true)。`Choice` —— **2–255** 个动态候选上的分布。`Score` —— 2–10 个有序等级上的分布加期望分。 |
| **一次调用回答多个问题** | 一份 14 问的评分表一次请求答完。他们的 `parallel_questions` cookbook 实测"批量一次调用**便宜 12.2 倍、快 10.0 倍，且答案不变**"。 |
| **不做解码** | 没有自回归循环、没有待解析的 JSON，因此不存在 schema 错误要处理。 |
| **可复现** | 同一请求重复 15 次，逐题概率**标准差 σ = 0.0102** —— 而在同一条件下 LLM"**在 temperature 0 也会自相矛盾**"。 |
| **按置信度路由** | "答案告诉你**做什么**，置信度告诉你**该不该动**。" 对概率卡阈值来 通过 / 人工 / 拒绝。 |
| **投机扇出** | 一次问很多问题，包括你未必需要的那些，由代码决定哪些是相关的。 |
| **RLCD 训练** | "Reinforcement Learning for Calibrated Decisions" —— 训练目标是**校准**，不是下一个 token 的似然。 |

Jev 是闭权重、只提供 API 的。

### 本仓库是什么

**MiniSystemOne 是对这个思路的从 0 教学式重写** —— 与 MiniMind 之于 LLaMA/GPT 配方的关系
相同，只是对象换成了决策模型而不是对话模型。

- **从随机初始化开始。** 其他所有 Jev 的开源复刻都是从现成 LLM（Qwen3-0.6B、Gemma…）
  继续训练。这一个从零开始：新建 BPE、MLM 预训练编码器、再接决策头。全链路
  **26.89M 参数、单卡 8GB 笔记本 GPU 训 4.7 小时** —— tokenizer、预训练、决策训练、
  温度校准全部在本仓库内。
- **一个编码器、一个 head、一个 loss。** Noul、Choice、Score 不是三条代码路径，
  而是**同一个**候选集上的 softmax：Noul 是 `{yes, no}`，Score 是 `{1..5}`。
  **弃权是一个候选，不是一个分支。**
- **校准必须实测。** 硬标签交叉熵和 Brier 都是 proper scoring rules，可在期望上
  学习条件概率；软标签不是必要条件。本仓库使用已知合成条件分布，便于直接监督
  和检查完整分布，而不是因为硬标签不能校准。
- **诚实是构造出来的。** 本项目自己预测错的地方（问题摊销、延迟），README 直接写明并划掉；
  模型失败的地方（真实文本、OOD），数字照报不埋。

---

## ⚠️ 这不是 Jev 的复现

本项目是一个**独立的教学实现**。它的灵感来自 TypeSafe AI 的 Jev（2026-09-15）所推广的
"System One Model" 思路，就像 [MiniMind](https://github.com/jingyaogong/minimind) 是
对 LLaMA/GPT 训练方法的**教学式重写**，而不是对某个具体模型的复现。

**本项目不蒸馏 Jev 的输出，也不做任何逆向工程。** TypeSafe 的客户协议明确禁止这些
行为。本仓库的全部训练数据，要么由 `dataset/synth/` 里的程序生成器产出，要么经由
`dataset/adapters/` 从公开数据集转换而来。
**没有任何数据从 Jev 流入训练，也没有任何 Jev 输出被提交进本仓库。**

唯一会碰到 Jev 的地方是一组**对比评测**（`eval/compare_apis.py`，结果见下面的对比小节）：
同一份 state、问题与候选集分别发给 `jev-latest` 和本模型，只报**聚合**指标。benchmark 不是蒸馏 —— 但它在协议下
是**另一个问题**，所以它会待在一个独立的、显式 opt-in、需要 API key 的脚本里，
**不在任何训练或数据构建路径上**。MCA 是否允许，由账号持有人判断，本仓库不替它假设。

我们也**不宣称击败 Jev**，在任何维度上。见[诚实边界](#诚实边界)——
两边的延迟甚至不是同一个量纲。

---

## 一张图看懂

```
state tokens            question tokens      candidate 1   candidate 2   ...
[ ...账户数据... ]      [ 这是欺诈吗？ ]      [fraud]       [legit]       ...
        seg=STATE              seg=QUESTION      seg=CANDIDATE
        └──────────── 一次双向前向 ─────────────┘
                            │
               state ∪ question 上做 span pool → z
                            │
              逐候选打分  f(z, c_i) → logit_i
                            │
              在 K 个候选上做 softmax → p
```

**三个 primitive，一个 head。** Noul（是/否）、Choice（动态候选集，上限 255）、
Score（序数）**是同一个操作**：在一个候选集上做 softmax。Noul 只是 `{yes, no}`，
Score 只是 `{1,2,3,4,5}`。没有单独的 sigmoid head；
**弃权是一个候选，不是一个分支** —— 因为弃权**本身就是**在选项之间做决策。

设计细节见 [`docs/DESIGN.md`](docs/DESIGN.md)，校准方法论见
[`docs/CALIBRATION.md`](docs/CALIBRATION.md)，冻结的数据契约见
[`docs/DATA_SCHEMA.md`](docs/DATA_SCHEMA.md)。

### 为什么校准才是重点

**硬标签可以学习和评估校准。** 观测类别交叉熵和 multiclass Brier 的条件期望
都在真实条件分布处最优；单条 one-hot 不揭示完整分布，但跨样本经验风险可用于估计它。
软标签是便利，不是必要条件。

本仓库用已知随机规则（`explicit_rng`）、隐藏变量条件边缘化（`marginalized`）、
定义的并列集（`tie_set`）构造便于直接检查的目标；ChaosNLI 每条约 100 人的标注
（`human_annotators`）是经验频率，不自动等于真实条件分布。

报告总体与 provenance 子组并说明组成。**分桶 top-label ECE** 比较置信度与桶内
`t[argmax p]`，不证明逐条分布准确或 OOD 理解能力。详见[指标契约](docs/CALIBRATION.md)。
新输出以 `distribution_l2`（逐样本类别平方差**求和**后跨样本平均）替代旧 `brier`，
新增 `expected_brier = distribution_l2 + mean(1-sum(t**2))`，将旧 `calibration`
聚合改为 `soft_targets`，不输出兼容别名。标注诊断改名 `ece_annotation_reference`，
仅是特定假设下的 Monte Carlo 参考量；删除 `ece_corrected`，不从 ECE 相减。
以下历史表只将旧距离列改名 distribution L2，不填入未测量的历史 expected Brier。

---

## 结果

下面全部来自 `out/decision/decision.pth` —— 26.89M 参数，在 67.6M token 上从随机
初始化做 MLM 预训练，再做 33,795 步决策训练。训练峰值显存 **3.85 GB**。
复现命令见[复现](#复现)。

与 `jev-latest` 和一个通用 LLM 的外部对比在本节末尾。此处不含任何估计值。

### 合成留出集上的校准

`test_known`，n=18,000，K 最大 255。`soft_targets` 行（历史键 `calibration`）
剔除 hard 是为了描述软目标子集，不是因为硬标签不能测校准。总体与子组回答不同问题。

| | acc | ECE | distribution L2 | NLL |
|---|---|---|---|---|
| 未校准 | 0.647 | 0.0047 | 0.0249 | 1.0821 |
| 全局温度 | 0.647 | 0.0038 | 0.0249 | 1.0821 |
| per-(primitive × K) 温度 | 0.647 | 0.0046 | 0.0248 | 1.0800 |
| **soft_targets 子集**（剔 hard，n=16,361） | **0.612** | **0.0061** | 0.0273 | — |

![合成集可靠性图](assets/reliability_synth_test_known.png)

**这个历史拟合中温度影响很小。** `T = 0.965`，NLL 从 1.0629 变为 1.0628。
它说明在该分布上收益有限，不证明逐条分布准确，也不证明 one-hot 训练无法校准。

### 该读的是这张表，不是准确率那一列

`accuracy` 是 argmax 一致率（目标并列时按首个下标），不是
`accuracy_soft = mean(t[argmax p])`。`tie_set` 的软正确率上限为 `1/k`，
one-hot 下软正确率就是观测准确率。下表报告软正确率及目标定义的 oracle 上界
`mean(max_k t_k)`。总体指标受组成影响，逐来源报告帮助解释，而不是否定混合总体。

| 来源 | n | 模型 | oracle 上界 | 达成率 |
|---|---|---|---|---|
| `tool_router` | 3,000 | 0.7137 | 0.7140 | **100.0%** |
| `security_gate` | 3,000 | 0.6403 | 0.6426 | **99.6%** |
| `refund_policy` | 3,000 | 0.7561 | 0.7601 | **99.5%** |
| `agent_trace_score`（`marginalized`） | 3,000 | 0.6366 | 0.6450 | **98.7%** |
| `banking_balance`（`explicit_rng`） | 3,000 | 0.4544 | 0.6091 | **74.6%** |
| `calendar_slot` | 3,000 | 0.0920 | 0.1416 | **65.0%** |

六个生成器里四个基本被解掉了。没解掉的两个恰好是该最难的，而且原因不同：

- **`banking_balance`** 是算术那一个 —— `q = σ((B − A − fees)/τ)` —— 也是校准最差的
  一个（**ECE 0.1637**，其余在 0.0035–0.0495）。26M 的编码器做不了精确算术，只能近似，
  而它的置信度跟着近似走、不跟着答案走。
- **`calendar_slot`** 是大 K 那一个（K 到 255）。模型和 oracle 都在 0.1 附近，所以这是
  数据里本来就有的歧义，不是模型失效 —— 但它仍只拿到可达成的 65%。

### BoW 门禁（预登记）

方案要求模型在 `test_known` 上比 bag-of-words 词法探针高 ≥0.25，否则判定生成器
过度模板化、**此时不得报告任何模型指标**。门禁数字在任何 checkpoint 存在之前就已
记录（词法上界 0.315）。

```
真实模型 test_known 准确率 0.647
模型 − 词法上界 = +0.332，门槛 ≥0.25  → 通过
```

门禁**通过**。注意它过的是汇总的 `accuracy`；经得起推敲的读法是上面那张逐来源表。

### 真实人类分歧上的校准（ChaosNLI）

![ChaosNLI 可靠性图](assets/reliability_public_test_known_public-chaosnli.png)

474 条，每条约 100 位标注者。保留历史原始 **ECE 0.0613** 和标注 MC 参考量
**0.0068**；旧减法值 **0.0545** 只记录为撤回的解释，**不再称作校正后 ECE**。
该参考量假设标注独立、置信度就是真实 top-label 概率，不是通用噪声下界。
旧嵌入图片可能仍带已废弃的地板/校正标题；用当前脚本重画才是新口径。
右图未应用温度，因此没有测量合成温度向真实文本迁移的成败。

混合 `human_annotators` 的 **ECE 0.3167** 描述该特定混合总体，不代表每个来源。
同时报告子组：GoEmotions 只有 3–5 位标注者，历史 **ECE 0.3253**、MC 参考量
**0.0044**。混合指标在组成和目的明确时合法，两个 MC 数字都不能从 ECE 扣除。

### 合成 → 真实的差距

**0.612 → 0.424**（合成 soft_targets 子集 → 真实文本上的 ChaosNLI / GoEmotions）。
从程序生成的规则跨到真实自然语言，大约损失 **19 个点**。更难的公开 split
（CLINC150、banking77、Amazon）准确率 0.157，高于随机但离可用很远。

这是本项目最有信息量的单一数字，它作为局限报告，不作为脚注。

### 效率 —— 以及本项目猜错的两个数

在真实请求路径上实测（`model.decide_chunked`），200 条**各不相同的**样本，
RTX 4070 Laptop。

| | 数值 |
|---|---|
| 延迟（B=1，逐样本） | **中位 20.24 ms**，p95 27.09 ms，范围 17.23–33.23 ms |
| 吞吐 | 49.4 样本/s |
| 中位样本 | state 87 token、question 8 token、3 个候选 |
| 峰值显存 | **0.13 GB**，在 B∈{1,8} × K∈{2,32,128,255} 上持平 |
| K=255 | 中位 49.97 ms（走分块路径） |

**延迟受固定开销支配、不受计算量支配 —— 这一点是可测的。** K=2 要 19.88 ms，
K=32 要 18.37 ms：多出 30 个候选**不花任何代价**，因为每次调用的固定开销占了大头。
这也解释了设计中引用的 5.87 ms（在备好的张量上裸前向）为何活不下来 ——
真实路径每次都要重建 mask、位置和打包。

**问题摊销没有兑现。** 设计按前缀 KV 复用预测 N=16 时约 3.3×。实测：

| N 个问题共享一份 state | 朴素 | 缓存 | 加速比 |
|---|---|---|---|
| 1 | 19.07 ms | 25.86 ms | **0.74×** |
| 4 | 78.45 ms | 85.83 ms | 0.91× |
| 16 | 440.04 ms | 423.20 ms | **1.04×** |
| 64 | 1451.66 ms | 1184.07 ms | 1.23× |

拆解：`encode_state` 7.31 ms、`encode_prefix` 8.69 ms、整问 19.85 ms。也就是说缓存后
一个问题要 8.69 ms 却只用 11 个 token —— 省下的量在算术上真实存在，但被同一份固定
开销淹没。**架构性质本身没有疑问**（冒烟测试 `[D]` 显示前缀隐状态与候选**逐比特**无关，
这正是复用**正确**的依据），但在这份任务实际产生的序列长度下（state ≈ 87 token），
几乎没有什么可摊的。摊销要长 state 才有意义，而这份数据没有。

### 与 Jev 和一个通用 LLM 的对比

`eval/compare_apis.py` —— 48 条，按六个生成器分层，候选数 ≤ 8，**三个系统同一批输入、
同一份 `eval_metrics.py` 打分**。

| 系统 | n | 剔除 | 软准确率 | **ECE** | distribution L2 | **ms/条** | 输出 token |
|---|---|---|---|---|---|---|---|
| **ours** | 48 | **0** | 0.5968 | **0.0248** | 0.0235 | **4.3** | **0** |
| `jev-latest` | 48 | 0 | 0.5226 | 0.2164 | 0.2165 | 1464.6 | 2,781 |
| `deepseek-flash` | 37 | **11** | **0.6711** | 0.0538 | 0.0524 | 5650.9 | 139,731 |

**先读让步，再读数字。**

- **我们是在这个分布上训出来的，另两个是零样本。** 这一列对我们有利 —— 而 DeepSeek
  **仍然在准确率上赢了我们**（0.6711 vs 0.5968），同时丢掉了推理预算被吃光的 23% 样本。
  这正是[诚实边界](#诚实边界)那条：**不在准确率上跟语言模型比。**
- **DeepSeek 的 11 次空返回是预算造成的，不是硬失败。** 它的 API 接受 `max_tokens`
  到 65536；我们设了 8192，推理阶段全吃光了。这是**延迟/成本换可靠性**的取舍 ——
  而 4.3 ms、输出 0 个 token 的我们这边不存在这个取舍。
- **Jev 在这张表上的 ECE 不构成对它官方主张的反驳。** 他们公布的是**重复调用之间的
  稳定性**（在它们自己的任务上 σ = 0.0102），不是"与我们的生成器 P\* 是否吻合"。
  这是两个不同的命题，这张表只测了后者。
- **n = 48，每来源 8 条。** 这展示的是形状，不是可引用的数字。

渲染充分性检查报告 `P*` 可从文本 **100% 恢复**，即输入包含预期证据。
在这个小规模历史样本上，Jev 的 **distribution L2（平方差之和）是我们的 8.7 倍**。
这仅是这些输入上的结果，不是系统能力排名或校准证明：我们在该任务族上训练，
这种优势同时影响概率指标和准确率。

复现：`python eval/compare_apis.py`（显式 opt-in；需要 `TYPESAFE_API_KEY` 与
`DEEPSEEK_API_KEY`；**不在任何训练或数据构建路径上**；只写聚合指标，绝不落外部模型的
逐样本输出）。

### 预登记的门禁 —— 在存在任何模型之前就记下

`scripts/audit_synthetic.py` 只跑数据、不碰模型，所以下面这些数字在第一个
checkpoint 训练出来之前就已固定。放在这里，是为了让模型的成绩能对着一个
**不是事后挑的**地板来读。

```
========== 2. schema 契约 ==========
  检查 210000 条，问题 0        （train 180k / val 6k / calib 6k / test_known 18k）

========== 3. split 不相交（数据层复核） ==========
  train 384、val 48、calib 48、test_known 240 个 (source, template, pool) 组合
  相交 0

========== 4. 渲染充分性 ==========
  所有软目标样本、所有生成器、所有 split：100% 可恢复
  （这一项验证的是 P* 只依赖真正渲染进 `state` 的量；
   只要有一条失败，就意味着模型被要求去预测一次抛硬币）
```

**词法门禁（R2）。** 用一个 bag-of-words 逻辑回归 —— 特征为候选自身的词，
加上每个候选词是否也在 `state`/`question` 里出现 —— 在 `train` 里按来源分层的
3 万条样本上训练，在留出集上打分。它是生成器可能被指控"奖励浅层匹配"的
最强非语义对手。

| split | 词法上界 | 随机水平 | 领先 |
|---|---|---|---|
| `val` | 0.336 | 0.284 | +0.052 |
| `calib` | 0.332 | 0.284 | +0.047 |
| `test_known` | **0.315** | 0.280 | +0.036 |

**门禁：真实模型必须比 0.315 高出 ≥0.25，即在 `test_known` 上 ≥0.565。**
达不到就说明生成器过度模板化，**此时不得报告任何模型指标**，必须先多样化。
**对最终 checkpoint 实跑结果：0.647，即 +0.332 —— 通过。** 重跑门禁用：

```bash
python scripts/audit_synthetic.py --model_eval out/eval/decision/decision/synth_test_known.json
```

（文件名里带数据集的 stem —— `eval_harness.py` 写的是
`out/eval/<--out>/<ckpt stem>/<data stem>_<set>.json`。因为旗舰流程会把
`dataset/synth` 和 `dataset/public` 的 `test_known` 写进**同一个目录**，
不带 stem 第二次会静默覆盖第一次。）

词法领先只有 +0.036 —— 门槛低得刻意：它说明这些任务是由**对渲染出来的证据做
算术与规则应用**决定的，而不是由哪些词挨着哪些词决定的。有两行逐生成器的读法
现在就写下来，否则它们在表里看起来像失败：

- **`tool_router` 领先 +0.157**（en +0.211 / 混排 +0.220）。这是设计如此、不是泄漏：
  把意图短语映射到工具名**就是**这个任务，所以词法重叠是预期解法，
  而不是绕过它的捷径。
- **`refund_policy` 得 −0.008**，即随机水平 —— 探针一无所获。另一条答案泄漏探针
  （只读 state + question，**不含候选**）在它上面得 0.812，那不是泄漏而是查找表：
  它的输入空间离散到足以被背下来。这个让步条件跟着任何 `refund_policy` 数字走。

其余看**去数字**那一列：`banking_balance` 0.358（−0.017）、
`agent_trace_score` 0.209（+0.009）、`security_gate` 0.340（+0.007）——
都在随机水平，这正是算术型任务该有的样子。`calendar_slot` 落在 0.003，
**低于**随机：字符串匹配根本够不到它的候选。

### 已经验证的部分 —— 不需要训练

这些是**架构性质**，所以在随机初始化下就成立，现在就能用
`python scripts/smoke_test.py` 复现。它们是关于**设计**的承重主张，
与模型学得好不好无关。

```
[A] 候选独立性        max|Δlogit| = 7.8e-03
[B] 置换不变性        max|Δp|     = 5.4e-04
[C] 分块不变性        max|Δp|     = 6.2e-04
[D] 前缀不受候选影响   max|Δh_prefix| = 0.000e+00   ← 精确为零
[E] mask 健全性       全屏蔽行=0，含 inf=0，非法可见=0
[F] 位置分配          prefix_len=42，K=5，不一致=0
[H] 255 候选单次前向   S=6312 → logits (1, 255)，峰值 1740 MB，
                       与分块路径的 max|Δp| = 2.8e-05
```

注意 **[D] 精确等于 `0.0`**，不只是很小。在 `prefix_blocked=True` 下，
前缀的隐状态与"有哪些候选"**逐比特无关** —— 前缀阻塞把这些注意力行直接屏蔽掉了。
正是这个精确性让前缀 K/V 复用是**正确**的而不是近似的，
也是问题摊销主张最终所依赖的东西。

`[B]`/`[C]` 只是近似，因为 bf16 的加法不满足结合律；容差是 `2e-02`，
而实测值落在容差内部两个数量级的地方。

---

## 诚实边界

这一节是**在结果出来之前**刻意先写定的，为的是让它不能事后被悄悄调整成好看的样子。
它是一组**可被否证的预测**。

### 26M 从 0 训练的编码器**应该能做到**

- 学到**程序可验证的合成决策规则**并达到高准确率 —— 当规则在词法上简单、
  且证据显式存在于 state 中时。
- 产出**真正校准的**概率，对着 `test_known` 的已知 `P*` 与 ChaosNLI 的
  100 人标注分布验证。
- 在 **ECE** 上击败微调 AR LLM 的**verbalized confidence** ——
  用**同一套** ECE 代码路径评分。
- **单次前向**完成决策，显存占用小，且在典型候选长度下支持
  **255 个候选**。
- 在 `candidate_crosstalk=False` 时给出**精确的候选顺序不变性**。
- ~~N=16 个问题共享一份 state 时约 3.3× 的问题摊销。~~ **已被实测否定 ——
  见上面「结果 / 效率」一节。** 实测 N=16 只有 1.04×，N=1 甚至是 0.74×。
  那个预测假设 state 会主导序列长度；而在这份数据的实际长度下（state ≈ 87 token），
  每次调用的固定开销把省下的部分淹没了。它当时被标为"预测而非承诺"，而它没有成立。
- ~~单次决策约 6 ms、批量约 2.8 ms/样本。~~ 那是裸前向的数字。真实请求路径实测
  **中位 20.24 ms / p95 27.09 ms** —— 受固定开销支配，K=2 → K=32 延迟持平即可看出。
  成立的是显存那一半：**0.13 GB**。

### **做不到**

- **匹配 LLM 的开放域 NLU。** ChaosNLI 准确率会远低于 LLM。
  那个数字是作为**校准**演示报告的，README 会这么说明，而不是把它藏起来。
- **在没有任务 schema 的情况下处理任意真实文本。** 它是一个 **schema 绑定的决策模型**，
  不是通用助手。
- **在真实文本上达到自己的合成准确率。** 这个差距会被实测并作为头号诚实数字报告，
  而不是被埋掉。
- **在准确率上赢过语言模型。** 主张是延迟、原生校准分布、
  以及**构造上为零的 schema 错误** —— 不是准确率。
- **保证 OOD 输入上的概率可靠。** 本项目不作此保证。ChaosNLI 历史 ECE 0.0613
  描述被评测集合，不是单条请求保证，也不是 OOD 检测实验。
- **以任何意义替代 LLM。** 它是一个组件。本 README 的框架是
  "一个 26M 的决策原生模型长什么样、代价是多少"，而不是"这是 LLM 的替代品"。

### 明确承诺

**如果 26M 模型最终只能在合成生成器上有效，本 README 会照实说明。**
合成与真实之间的差距本身就是一个真实结果，无论它好不好看都会被报告。

### 已知的度量口径问题

提前记下来，因为其中任何一条若在事后才被发现，都会看起来像找借口：

1. **延迟与 Jev 的 70–500 ms 不可比。** 那个数字是 *LLM 推理*延迟（在文本上跑解码循环），
   我们的是*单次前向*。量纲不同，主张不同。我们不宣称击败它。
2. **这里的 ChaosNLI 只有 MNLI 部分**（我们能拿到的镜像），1599 条 ——
   不是完整的 SNLI+MNLI+ANLI 并集。它的 `test_known` 只有 474 条，偏薄。
   equal-mass 分桶下每桶约 50 条；每张图上都印出每桶条数。
3. **公开集不是自然分布。** `amazon_score` 的 val/test 是类别均衡的（每档 1000 条），
   而不是真实评论的 J 形分布，所以它的 accuracy 是**均衡准确率**。
   且它的 `train` split 不可用（标签是占位符），因此它是纯评测集。
4. **SDPA 后端结论测于 torch 2.5.1+cu121**，而 `requirements.txt` 钉的是 2.6.0+cu124。
   float-vs-bool mask 与 EFFICIENT-vs-MATH 这两条结论需要在钉住的版本上重新验证。
5. **MLM 的墙钟时间比粗读方案时以为的更长。** 默认是 8 个 epoch、约 95M token
   （总共见过约 540M token），即约 **21 token/参数** —— 对 26M 模型而言大致是
   算力最优的，但远多于"1 个 epoch 32 分钟"。

---

## 两个档位

两档的形状刻意对齐 MiniMind，使得 warm-start 消融不需要改任何形状参数。
用 `python scripts/model_stats.py` 复核 —— 它在 CPU 上实例化模型类并打印拆分表，
所以你可以在几秒内证伪这里的任何一个数字。

| 档位 | hidden | layers | heads | ffn | 总参数 | 构成 |
|---|---|---|---|---|---|---|
| **26M**（主目标） | 512 | 8 | 8/4（GQA） | 1280 | **26.89M** | encoder 22.03M + embed 3.28M + head 1.58M |
| **65M** | 768 | 8 | 8/4（GQA） | 2304 | **65.11M** | encoder 56.64M + embed 4.92M + head 3.55M |

Stage 1（`MiniSystemOneForMaskedLM`）是 **25.31M** —— 恰好比决策模型**少** 1.58M，
差的就是 `DecisionHead`；它的 `lm_head` 与 `embed_tokens` 绑定，**新增 0 参数**。
跑一次 `python scripts/model_stats.py` 就能在几秒内核掉这张表里的每个数。

---

## Tokenizer

新建 BPE，**vocab 6400**，11 个特殊 token，`model_max_length=8192`，无 BOS/EOS，
**无 `chat_template`**（决策模型没有对话模板；规范序列化是**代码**
—— `model/serialize.py` —— 而不是 Jinja 字符串）。

```
<pad> <unk> <cls> <sep> <mask> <trunc> <yes> <no> <abstain> <ans> </ans>
 0     1     2     3     4      5       6     7     8        9     10
```

`<yes>`、`<no>`、`<abstain>` 刻意是**单 token** 候选：这使 Noul 两个答案的 pooled
向量最干净；而且 Noul 是一等 primitive，给它的答案原子 token 是真实的建模选择，
不是图方便。

实测压缩率（`python trainer/train_tokenizer.py --tokenizer_path model`）。
参照列是 MiniMind 自己的 tokenizer 在同一组样例上的值：

| 样例集 | 字/token | 门槛 | MiniMind 参照 |
|---|---|---|---|
| 中文 | **1.44** | 1.42 | 1.40 |
| 英文 | **3.16** | 3.15 | 3.39 |
| 混排 | **2.53** | 2.50 | 2.41 |
| **真实生成器文本** | **3.13** | **2.60** | 2.03 |

关于这张表有两句诚实的话：

- **前三行是示意，本身不是有意义门禁。** 英文只比门槛高 0.01 —— 那一行基本是空的。
  它被刻意设在 MiniMind 参照值**之下**，因为英文样例是 LLM 写的说明文，
  而那正是参照 tokenizer 的主场。我们真正要服务的英文是 CLINC150 / banking77
  的短用户话语、GoEmotions 与 Amazon 评论。
- **最后一行才是真正的门禁。** 它量的是真实生成器渲染出来的文本 —— 模型真正会读到的东西。
  这个量直接决定序列长度、截断率，以及 255 个候选能否单次前向装下。

原设计文档写的是 `en ≥ 3.60`。**那个目标不可能达标** ——
MiniMind 自己的 tokenizer 在同一组样例上只有 3.39 —— 也就是说它当初是**拍的而不是测的**。
现已替换为实测值并附上解释。

---

## 仓库结构

```
model/
  model_system_one.py     Config、RMSNorm、RoPE、Attention、Encoder、AttnPool、
                          DecisionHead、ForDecision / ForMaskedLM / ForCausalLM
  serialize.py            唯一的规范序列化：pack_example、build_attn_mask、
                          build_position_ids、collate_packed、头尾截断
  tokenizer.json          新建 BPE，vocab 6400，11 个特殊 token
  tokenizer_config.json
dataset/
  pretrain_corpus.py      语料的事实来源：fetch（一次性、联网）+ blend，
                          以及与 tokenizer / MLM 共用的 iter_corpus
  decision_dataset.py     DecisionDataset（含 K 子采样）、collate_decision、
                          CandidateBucketSampler
  mlm_dataset.py          span masking，80/10/10
  synth/                  六个生成器，注册表 + base + lexicon
  adapters/               chaosnli、clinc150、banking77、goemotions、amazon_score
trainer/
  trainer_utils.py        get_lr、Logger、init_model、save_checkpoint、oom_retry、
                          unbuffer_stdout
  train_tokenizer.py      训练 BPE 并强制压缩率门槛
  train_mlm.py            Stage 1：MLM 预训练
  train_decision.py       Stage 2：CE + λ_b·distribution L2 + λ_o·CDF-MSE
  calibrate_temperature.py  LBFGS 拟合 log T，三种粒度
eval/
  eval_metrics.py         纯函数：accuracy、nll、distribution_l2、expected_brier、ece、reliability_curve、
                          risk_coverage_curve、ordinal_mae、expected_score、
                          ece_annotation_reference、bootstrap_ci
  eval_harness.py         输出 metrics / by_provenance / per_sample
  eval_efficiency.py      延迟、吞吐、显存、问题摊销
  make_reliability_plot.py
scripts/
  build_dataset.py        合成生成器或公开适配器
  audit_synthetic.py      BoW 门禁 —— 全仓库最有价值的脚本
  audit_leakage.py        位置探针 —— 必须停在随机水平
  model_stats.py          参数构成表
  tok_probe.py            tokenizer 压缩率探针（调 BPE 时用）
  smoke_test.py           不变量断言
docs/                     DESIGN.md、DATA_SCHEMA.md、CALIBRATION.md
assets/                  README 里嵌入的两张可靠性图
```

风格沿用 MiniMind：无 `pyproject.toml`、无包安装、脚本直接跑 + `sys.path.append`、
argparse-only CLI、`AdamW`、autocast + `GradScaler`、swanlab 可选、中文 help 文本。

**不入库的：** `out/`（checkpoint、评测 JSON、日志）、`*.pth`、`dataset/**/*.jsonl`、
`dataset/pretrain_en*.jsonl`。全部可由上面的脚本重建 —— 逐条的取舍理由写在
`.gitignore` 里。

### checkpoint 里有什么

`mlm.pth` 与 `decision.pth` 是**自描述**的：文件里带着它被构建时的 config 和训练数据的
出处，所以你不必手抄超参，也不必猜某份词表是不是配它。

```python
from trainer.trainer_utils import ckpt_info
print(ckpt_info("out/decision/decision.pth")["meta"])
# {'stage': 'decision', 'step': 33795, 'n_params': 26889729,
#  'tokenizer_sha1': 'bf7a131a109ea445', 'gen_version': '1.0.0',
#  'encoder_init': 'mlm.pth', 'trained_on': 'synth', 'max_len': 1024, 'epochs': 3}
```

有两道门禁用这份元信息，它们存在的原因都是**失败时不会出声**：

- **`init_model` 拒绝形状不符的配置。** checkpoint 记着 `hidden_size=512` 而你传 768 时
  它直接退出。否则 `strict=False` 会把一部分参数留在随机初始化，评测照跑、表格照出 ——
  一整张看着合理的数字，来自一个半随机的模型。
- **`verify_tokenizer` 拒绝不配套的词表。** 两份词表 vocab 都是 6400，embedding 形状
  永远是对的，载入从不报错，而每个 token id 都指向与训练时不同的词。

这两道门禁在权重首次训出来时都不存在，是准备发布时才补上的。
`scripts/model_stats.py` 仍然是核上面那张参数表的手段。

---

## 复现

```bash
conda create -n minimind python=3.12
conda activate minimind
pip install -r requirements.txt
```

### 0. 语料（一次性，需要联网）

`dataset/pretrain_en*.jsonl` **不入库**，历史英文语料约 61 MB。下列命令联网取得并
混合 Alpaca/Wikitext。中文需另外自行取得：原始默认读取兄弟 MiniMind 仓库的
`dataset/pretrain_t2t_mini.jsonl`。按照该项目说明与使用条款获取对应来源/版本，
放置或转换为 **`dataset/pretrain_zh.jsonl`**，UTF-8，每行 `{"text":"非空正文"}`。
这里不编造该历史本地文件的下载 URL；需自行核验来源，替换语料不等于复现旧表。
Tokenizer/MLM 只消费本地文件，不会自动下载原始语料。

```bash
python dataset/pretrain_corpus.py fetch --out dataset/pretrain_en_alpaca.jsonl \
    --dataset tatsu-lab/alpaca --fields instruction,input,output
python dataset/pretrain_corpus.py fetch --out dataset/pretrain_en_wiki.jsonl \
    --dataset Salesforce/wikitext --config wikitext-103-raw-v1 --fields text \
    --strip_wiki_title --min_chars 600 --n_docs 60000
python dataset/pretrain_corpus.py blend --out dataset/pretrain_en.jsonl
```

> **缺失语料现在明确报错，不静默跳过。** 下方两个本地路径必须有可用数据。
> `--allow_missing_corpus` 只允许显式少一路并报告组成变化，不算完整复现；空文件或
> 损坏记录仍报错。无需外部语料的教学流程请使用 `scripts/quickstart.py`。

### 1. Tokenizer

```bash
python trainer/train_tokenizer.py --pretrain_path dataset/pretrain_zh.jsonl \
    --en_path dataset/pretrain_en.jsonl
```

新建 BPE（vocab 6400），训练语料 = 预训练语料 **∪ 合成决策语料**。
这个并集不显然但重要：若 tokenizer 没见过 `transfer_ownership`、`Neutral`、
`¥1,240.00`，它们会碎成很多 token，候选变长，scorer 拿到的 pooled 向量噪声变大。
成本为零。

我们**不**复用 MiniMind 的 tokenizer。它的 36 个特殊 token 大多是决策模型永不产生的
视觉/音频/工具 token；而且它**缺少** `<mask>`、`<sep>`、`<pad>`、`<trunc>`，
其 `pad_token` 恰恰**就是**它的 eos token。

### 2. 数据

```bash
python scripts/build_dataset.py                        # 合成
python scripts/build_dataset.py --public               # 公开适配器
python scripts/audit_leakage.py --data dataset/synth
python scripts/audit_synthetic.py --data dataset/synth
```

**这两个 audit 是门禁，不是报告。** 若 `audit_synthetic.py` 不通过 ——
即 bag-of-words 逻辑回归在 `test_known` 上追到模型 25 个点以内 ——
生成器就判定为过度模板化，**必须先多样化，才能报告任何模型结果**。

split 按 **`template_id` × `entity_pool` 划分，绝不随机逐条划分。**
这是 `test_known` 成为真正留出集的前提，由 `build_dataset.py` 里的断言强制。

### 3. 训练

```bash
python trainer/train_mlm.py --pretrain_path dataset/pretrain_zh.jsonl \
    --en_path dataset/pretrain_en.jsonl --num_workers 0 --save_optimizer
python trainer/train_decision.py --encoder out/mlm/mlm.pth \
    --num_workers 0 --save_optimizer
```

从最近的完整保存点继续时，保持**相同训练参数**，去掉仅用于初始化的
`--encoder`/`--init_checkpoint` 后加 `--resume`；不要改变数据/词表、batch/accum、epochs 或学习率计划：

```bash
python trainer/train_mlm.py --pretrain_path dataset/pretrain_zh.jsonl \
    --en_path dataset/pretrain_en.jsonl --num_workers 0 --save_optimizer \
    --resume out/mlm/mlm_opt.pth
python trainer/train_decision.py --num_workers 0 --save_optimizer \
    --resume out/decision/decision_opt.pth
```

`--encoder` 是 MLM 编码器初始化；同阶段权重初始化属于**新训练**；
`--resume ..._opt.pth` 才恢复完整状态。需新版完整保存文件，不能用旧 optimizer-only
文件或推理权重冒充恢复。精确恢复目前要求 workers=0；恢复的是最后保存点，不是崩溃
瞬间未保存的计算，也不保证跨硬件/版本逐比特相同。
`--max_steps` 是绝对总更新预算；`--stop_after_steps` 只暂停本次运行，不缩短学习率计划。
恢复时去掉暂停选项，而不是把总预算当作额外训练步数。
训练选项保留名称：`--lambda_brier` 加权 distribution L2，`--brier_normalize`
将每条平方损失除以有效候选数。

### 4. 校准与评测

```bash
python trainer/calibrate_temperature.py \
    --ckpt out/decision/decision.pth --data dataset/synth --out out/calibration

python eval/eval_harness.py --ckpt out/decision/decision.pth \
    --data dataset/synth --sets test_known \
    --temperature out/calibration/T.json --out out/eval/decision

# 历史真实文本路线报告原始概率，没有测试温度迁移。
python eval/eval_harness.py --ckpt out/decision/decision.pth \
    --data dataset/public --sets test_known --out out/eval/public

python eval/make_reliability_plot.py \
    --eval out/eval/decision/decision --sets test_known --binning equal_mass \
    --out assets

# 按 source 选择 ChaosNLI，便于和 GoEmotions 的任务/标注协议分别解释。
python eval/make_reliability_plot.py \
    --eval out/eval/public/decision --sets test_known \
    --source public:chaosnli --binning equal_mass --out assets
```

> **温度迁移必须实测。** 合成 calib 温度可能改善也可能恶化真实文本指标；历史原始
> 概率路线没有测试它。做本域校准应在独立本域 calib/dev 拟合，冻结后测试；明确标记的
> 跨域迁移实验也合法。标量温度是动态候选下简单、置换等变的选择，不是唯一可迁移
> 的参数化，更不保证新 schema 已校准。

`calibrate_temperature.py` 写出的 `T.json` 里含 **checkpoint 与 tokenizer 的 SHA1**，
因为温度是绑定到具体权重的 —— 否则来自另一个模型的 `T.json` 会被静默接受并照常出图。

---

## Windows 上会咬你的两件事

两条都在开发中实测踩过，且都属于"看起来完全是别的问题"的那类失败。

1. **历史 pyarrow/Torch 导入顺序故障。** 早期 Windows 环境出现过反向导入时退出码
   139、无 traceback 的问题。当前固定环境已经跑通 quickstart，不能据此推广成普遍的
   导入顺序规则；当前 `train_mlm.py` 也没有占位的 `import datasets`。

2. **7.5–8 GB 附近有一个显存悬崖，而且它不崩。** Windows WDDM 在显存不足时把页面
   换到共享内存，而不是抛 OOM：实测 32×1024 的配置**不会** OOM，只是**慢 10 倍**
   （297 ms → 5936 ms/step）。所以这里的预算是 **峰值 < 6.5 GB**，而不是"塞进 8 GB"；
   越过这条线时 `peak_vram_warn()` 会喊出来 —— 因为如果没有这条告警，
   "运行莫名变慢"的唯一解释就变成了"我的代码写得慢"。

**不要**把 MiniMind 的默认 batch（32/8）照搬到 8 GB 卡上。你会得到一个莫名很慢的运行，
而不是一个报错。

---

## 环境

实测环境：Python 3.12，torch 2.6.0+cu124，RTX 4070 Laptop（8188 MiB），bf16。

吞吐强依赖你的硬件。`train_mlm.py` 每 `--log_interval` 打印 tok/s，
就是为了让你能在自己机器上重算墙钟时间，而不是信本文件里的任何数字。

---

## License

Apache-2.0，见 [LICENSE](LICENSE)。
