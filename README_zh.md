<div align="center">

# 🚀 MiniSystemOne

### 在消费级显卡上，从零搭建一个输出决策概率的小模型

**中文** | [English](README.md)

⚡ [快速开始](#quick-start) · 🚀 [完成第一个任务](#first-task) · 📚 [数据集](#datasets) · 🛠️ [逐步训练](#training) · 📉 [训练损失](#loss) · 📊 [模型评测](#evaluation)

</div>

![MiniSystemOne：消费级 GPU、共享编码器与六步学习路线](assets/readme/overview.png)

---

## 🌱 项目介绍

给模型一段上下文、一个问题和一组候选，让它直接返回每个选项的概率——这是 MiniSystemOne 要教你搭建的模型。

例如，用户说“我想查询账户余额”，程序需要在“查询余额”“执行转账”“交给人工”之间选择。我们希望学会的不只是调用一个接口，还包括：文本怎样变成 token，编码器怎样读取证据，候选怎样被打分，概率怎样通过数据学出来，以及怎样判断这些概率是否值得信任。

本项目受 [MiniMind](https://github.com/jingyaogong/minimind) 的教学方式启发，面向想亲手训练小模型的初学者。你将从新建 BPE 词表和随机初始化的编码器出发，走完 **数据准备 → MLM 预训练 → 决策训练 → 温度校准 → 评测 → 任务接入**。默认决策模型约 **26.89M 参数**，核心训练循环直接使用 PyTorch。

模型形式受到 Jev / System One 思路启发；这是独立的教学实现，不是 Jev 内部架构或 RLCD 训练方法的复现。当前训练采用监督学习，不使用 Jev 输出作为训练数据。

### 🎓 你会学到什么

- 从头训练 tokenizer，理解词表、特殊 token 和序列打包。
- 搭建共享编码器，把不同类型的决策统一为候选打分。
- 构造有可追溯目标分布的数据，区分硬标签、软目标和人类分歧。
- 阅读并修改 MLM、交叉熵、Brier 和序数损失的实现。
- 区分训练集、验证集、校准集、留出测试集和跨任务测试集。
- 用自己的数据训练一个小任务，并按概率决定自动处理还是人工接管。

### 🧩 三种问题，一个模型

| 类型 | 适合的问题 | 输出 |
|---|---|---|
| **Noul** | 是否满足条件？是否通过检查？ | `P(yes)` 和 `P(no)` |
| **Choice** | 应该选择哪个工具、意图或动作？ | 2–255 个动态候选上的分布 |
| **Score** | 应该打几分？执行质量在哪个等级？ | 2–10 个等级上的分布及期望分 |

三者共享一个编码器和一个候选打分头。模型不逐 token 生成答案；普通打包路径并行计算候选，大候选集可以分块计算。最终 JSON 由 Python 组装，不是模型生成的文本。

> **当前状态：**最小训练链路、推理接口及回归测试已跑通。旧版权重和历史图表使用过修复前的序数损失，完整模型的修复后成绩需要重新训练、校准和评测；本页不把旧数字当成当前效果。仓库不包含 `.pth` 权重，新用户可以先走下面的最小训练路线。

<a id="quick-start"></a>
## ⚡ Ⅰ · 快速开始：先跑通一次完整流程

### 1. 准备环境

训练使用 NVIDIA CUDA GPU；推理和基础测试也支持 CPU。项目已有单张 RTX 4070 Laptop 8GB 的历史训练经验，但具体显存和速度取决于 batch、序列长度及候选数。

```shell
git clone https://github.com/Colvin0315/MiniSystemOne.git
cd MiniSystemOne
conda create -n minisystemone python=3.12 -y
conda activate minisystemone
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
```

下文命令都在仓库根目录执行，使用单行写法，兼容 PowerShell 和 Bash。训练前确认 CUDA 为 `True`。完整训练的默认尺寸是 `hidden_size=512`、`num_hidden_layers=8`；快速路线使用 `128 × 2` 的小模型。

### 2. 从零跑一个小模型

这条路线只用程序生成的数据，不下载外部语料，也不需要另一个 MiniMind 仓库。所有产物放到 `out/quickstart/`，使用新目录可以避免覆盖已有训练结果。

**这里的目标是理解并验证流程。每阶段只训练几步，不承诺模型已经学会业务任务。**

**🔤 ① 训练自己的 tokenizer**

```shell
python trainer/train_tokenizer.py --pretrain_path= --en_path= --n_docs 0 --n_synth 200 --out_dir out/quickstart/tokenizer --skip_eval
```

产物是 `tokenizer.json` 和 `tokenizer_config.json`。`--pretrain_path=`、`--en_path=` 表示显式关闭这两路外部语料；`--skip_eval` 仅在这个小规模演示中跳过压缩率检查。

**🧩 ② 生成决策数据**

```shell
python scripts/build_dataset.py --tokenizer out/quickstart/tokenizer --out out/quickstart/data --per_gen_train 20 --per_gen_val 4 --per_gen_calib 4 --per_gen_test_known 4 --per_gen_test_ood 4
```

六个生成器合计产生 120 条训练样本，以及各 24 条验证、校准和已知任务测试样本。默认没有留出整个生成器，所以 `test_ood` 为空；设置 `--per_gen_test_ood` 本身不会创建 OOD 任务。

**🧠 ③ MLM 预训练：先学习文本表示**

```shell
python trainer/train_mlm.py --tokenizer out/quickstart/tokenizer --pretrain_path= --en_path= --n_docs 0 --n_synth 40 --hidden_size 128 --num_hidden_layers 2 --max_len 256 --batch_size 2 --epochs 1 --max_steps 5 --save_optimizer --no_swanlab --out out/quickstart/mlm --log_dir out/quickstart/mlm/logs
```

**🎯 ④ 决策训练：学习给候选分配概率**

```shell
python trainer/train_decision.py --tokenizer out/quickstart/tokenizer --data out/quickstart/data --encoder out/quickstart/mlm/mlm.pth --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --epochs 1 --max_steps 5 --val_limit 12 --val_every 0 --save_optimizer --no_swanlab --out out/quickstart/decision --log_dir out/quickstart/decision/logs
```

**📊 ⑤ 校准与测试：用没有参与训练的数据检查概率**

```shell
python trainer/calibrate_temperature.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --data out/quickstart/data --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --lbfgs_steps 10 --out out/quickstart/calibration
python eval/eval_harness.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --data out/quickstart/data --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --sets test_known --out out/quickstart/eval
```

第二条命令报告未校准结果；若要并列比较温度校准结果，添加 `--temperature out/quickstart/calibration/T.json`。这个小校准集只用于演示。

**🚀 ⑥ 输入自己的问题**

```shell
python scripts/decide.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --input examples/choice.json --device cpu
```

到这里，你已经走完从词表、随机权重到决策输出的完整路径。检查以下产物，能把每一步与代码对应起来：

| 路径 | 内容 |
|---|---|
| `out/quickstart/tokenizer/` | 新训练的词表 |
| `out/quickstart/data/` | 数据 split 和构建 manifest |
| `out/quickstart/mlm/mlm.pth` | 预训练编码器 |
| `out/quickstart/decision/decision.pth` | 决策模型 |
| `out/quickstart/decision/decision_opt.pth` | 完整续训状态 |
| `out/quickstart/calibration/T.json` | 温度与配套权重、词表哈希 |
| `out/quickstart/eval/decision/data_test_known.json` | 指标和逐样本结果 |

<a id="first-task"></a>
## 🚀 Ⅱ · 用模型完成第一个任务

我们用“把客服请求分配给合适的工具”贯穿这个例子。模型负责选择，应用代码负责后续流程。

### 1. 描述状态、问题和候选

[examples/choice.json](examples/choice.json) 可以直接修改：

```json
{
  "state": "The user asks to check their account balance. Available tools: balance_lookup retrieves the balance; transfer_funds moves money.",
  "question": "Which tool matches the user's request?",
  "primitive": "choice",
  "candidates": ["balance_lookup", "transfer_funds", "abstain"]
}
```

`state` 放模型作判断需要的证据，`question` 说明要做什么判断，`candidates` 定义允许返回的选项。候选可以随请求变化，但增加一个名称不会自动教会模型新的业务规则。

### 2. 调用并消费结果

把下面代码保存为仓库根目录的 `route_demo.py`，运行 `python route_demo.py`。这里使用快速路线产物；正式使用时替换成你在该任务上训练和验证过的权重。

```python
import json
from model.inference import DecisionPredictor

predictor = DecisionPredictor(
    checkpoint="out/quickstart/decision/decision.pth",
    tokenizer="out/quickstart/tokenizer",
    device="cpu",
)
with open("examples/choice.json", encoding="utf-8") as f:
    request = json.load(f)

result = predictor.predict(request)
print(json.dumps(result, ensure_ascii=False, indent=2))

# 0.8 只是演示阈值；上线前应在自己的验证集上选择。
choice = result["choice"]
if result["confidence"] < 0.8 or choice == "abstain":
    destination = "人工处理"
else:
    destination = {
        "balance_lookup": "余额查询流程",
        "transfer_funds": "转账申请流程",
    }[choice]
print("请求分配到：", destination)
```

输出包含所有候选的 `probabilities`、最高概率候选 `choice`、对应的 `confidence`，以及 `state_truncated`、`temperature_applied`。这段程序完成的是分流决策，不会真的调用银行接口。

### 3. 同一个接口回答另外两类问题

![三种接口的真实输出与人工接管演示](assets/readme/task-demo.png)

上图来自修复后快速路线的小模型，对应仓库中的三个示例输入。只训练五步时，概率仍接近均匀；Choice 请求因置信度低于示例阈值 0.8 进入人工处理。这展示了接口与兜底流程，不代表模型已经学会任务。

```shell
python scripts/decide.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --input examples/noul.json --device cpu
python scripts/decide.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --input examples/score.json --device cpu
```

- **Noul：**只需 `state`、`question`、`primitive: "noul"`，候选固定为 `yes/no`，额外返回 `p_true`。
- **Score：**传候选文本和一一对应的 `levels`，例如 `[3, 1, 2]`；额外返回 `expected_score = Σ p_i × level_i`。

推理入口默认保留最多 512 个 state token，问题和单个候选最多 128 token；state 截断会在输出中标记。`--max_state_tokens` 和 `--chunk` 可调整上下文预算和候选分块大小。`--calibration T.json` 可加载配套温度，但只能把相应校准域上的效果视为已验证，不能直接推广到任意任务。

<a id="datasets"></a>
## 📚 Ⅲ · 数据集：模型究竟在学什么

本项目把数据分成三层：**语言预训练语料、合成决策数据、公开真实文本数据**。它们解决不同问题，不应混为同一种训练材料。

### 1. 语言预训练语料

| 数据 | 用途 | 本地文件 |
|---|---|---|
| MiniMind `pretrain_t2t_mini.jsonl` 或自己的中文文本 | 中文表示与词汇 | 本教程使用 `dataset/pretrain_zh.jsonl` |
| `tatsu-lab/alpaca` | 英文指令、会话表达；拼接 instruction/input/output 当作普通文本 | `dataset/pretrain_en_alpaca.jsonl` |
| `Salesforce/wikitext`，`wikitext-103-raw-v1` | 补充英文文本量 | `dataset/pretrain_en_wiki.jsonl` |
| 本仓库合成任务的渲染文本 | 补充工具名、等级、金额和业务术语 | 构建语料时程序生成 |

预训练输入是一行一篇文本，例如 `{"text": "账户已经通过身份核验。"}`。这一阶段没有“选哪个候选”的标签；tokenizer 学切词，MLM 学恢复被遮盖的 token。使用 Alpaca 的文本不等于在做对话 SFT。

### 2. 六类合成决策任务

合成数据由 [dataset/synth/](dataset/synth/) 的规则程序生成，包含中文、英文和混合文本。目标概率来自明确的生成规则，不是手工填一个看起来合理的置信度。

| 生成器 | 学习任务 | 类型 | 目标来源 |
|---|---|---|---|
| `banking_balance` | 读取余额、金额、费用，判断批准或路由 | Noul / Choice | 已知随机规则 |
| `tool_router` | 根据请求意图选工具 | Choice | 唯一答案或并列有效答案 |
| `agent_trace_score` | 根据执行轨迹评估等级 | Score | 对未完全观测的量边缘化 |
| `security_gate` | 根据风险证据选择放行、人工、拒绝 | Choice | 边缘化或并列答案 |
| `refund_policy` | 组合先验与核验证据判断退款 | Noul | 已知概率规则或边缘化 |
| `calendar_slot` | 根据可用时间选择候选时段 | Choice | 已知随机选择规则 |

默认全量构建的 split 如下，可在 [manifest](dataset/synth/manifest.json) 中核对：

| Split | 样本数 | 什么时候用 |
|---|---:|---|
| `train` | 180,000 | 更新模型权重 |
| `val` | 6,000 | 观察训练、选择设置 |
| `calib` | 6,000 | 固定模型后拟合温度 |
| `test_known` | 18,000 | 同一任务族中留出的模板或实体池 |
| `test_ood` | 默认 0 | 显式留出整个生成器后才有样本 |

划分以 `template_id × entity_pool` 组合为单位。`test_known` 留出模板或实体池；`val/calib` 使用与训练不重叠的组合。这样可以减少仅靠记住相似句子获得高分的情况。

### 3. 一条训练样本长什么样

训练 JSONL 的 `candidates` 是带元数据的对象，推理 JSON 的 `candidates` 则是字符串列表。二者用途不同。

```json
{
  "id": "my_router::train::000001",
  "source": "custom:router",
  "gen_version": "1.0.0",
  "split": "train",
  "state": "用户希望查询账户余额。",
  "question": "应该调用哪个工具？",
  "schema": {"primitive": "choice", "name": "customer_router"},
  "candidates": [
    {"text": "balance_lookup", "label": "balance"},
    {"text": "transfer_funds", "label": "transfer"},
    {"text": "abstain", "label": "abstain"}
  ],
  "target": {"kind": "hard", "p": [1.0, 0.0, 0.0], "provenance": "hard"}
}
```

`target.p` 与候选顺序一一对应，和为 1。重排候选时必须同步重排目标。Score 还需在每个候选的 `meta.level` 中提供实际等级；排序和距离计算使用它，而不是候选的位置。

硬标签是合法的概率学习信号；已知软分布则更适合直接研究预测分布与目标分布的差异。不要把单条标签随意改成 0.8 来“制造校准”。完整约定见 [数据格式](docs/DATA_SCHEMA.md)。

### 4. 公开真实文本数据与评测

公开数据由 [dataset/adapters/](dataset/adapters/) 转成相同格式，使用项目自己的候选构造与 split。因此这里的分数不能直接当作原始任务排行榜成绩。

| 数据源 | 转换后的任务 | 在本项目中观察什么 |
|---|---|---|
| `metaeval/chaos-mnli-ambiguity` | 三分类自然语言推断；多位标注者形成目标分布 | 真实人类分歧上的概率预测 |
| `clinc/clinc_oos`（plus） | 意图选择，含范围外意图处理 | 真实用户请求及较大候选集 |
| `PolyAI/banking77` | 银行业务意图选择 | 接近业务场景的分类迁移 |
| `google-research-datasets/go_emotions`（raw） | 情绪投票分布 | 少量标注者下的分歧 |
| `SetFit/amazon_reviews_multi_en` | 一到五星的序数判断 | 真实文本上的等级预测 |

有三个重要口径：GoEmotions 的多选投票被归一化为分布，仅保留至少两个获票情绪，因此不等于原始 28 类评测；Amazon 适配器只读取 validation/test；默认决策训练只读取 `dataset/synth`，**构建公开数据并不会自动把它加入训练**。

公开数据优先沿用原始划分，并按适配器规则留出校准集；没有适用原始划分的来源使用稳定哈希桶。实际数量以本次生成的 `manifest.json` 为准。

<a id="training"></a>
## 🛠️ Ⅳ · 逐步搭建与完整训练

这一节使用默认的 26.89M 配置。为了与仓库自带 tokenizer、旧权重区分，新词表、数据和权重统一存放在 `out/full/`。

### 📚 Step 1 · 准备语言语料

从 [MiniMind 数据集](https://huggingface.co/datasets/jingyaogong/minimind_dataset/tree/main) 下载 `pretrain_t2t_mini.jsonl`，保存为 `dataset/pretrain_zh.jsonl`；也可以换成自己的 `{"text": ...}` JSONL。再准备英文数据：

```shell
python dataset/pretrain_corpus.py fetch --out dataset/pretrain_en_alpaca.jsonl --dataset tatsu-lab/alpaca --fields instruction,input,output
python dataset/pretrain_corpus.py fetch --out dataset/pretrain_en_wiki.jsonl --dataset Salesforce/wikitext --config wikitext-103-raw-v1 --fields text --strip_wiki_title --min_chars 600 --n_docs 60000
python dataset/pretrain_corpus.py blend --out dataset/pretrain_en.jsonl
```

下载阶段需要联网，之后训练读取本地文件。外部数据遵循各自的数据集许可。这里显式传中文路径，避免依赖脚本原来的 `../minimind/...` 默认位置。

### 🔤 Step 2 · 训练 BPE 词表

```shell
python trainer/train_tokenizer.py --pretrain_path dataset/pretrain_zh.jsonl --en_path dataset/pretrain_en.jsonl --out_dir out/full/tokenizer
```

目标词表大小为 6,400，包含 `<pad>`、`<sep>`、`<mask>`、`<trunc>` 等特殊 token。工具名和数字格式也参与词表训练，便于后续候选编码。脚本会检查压缩率；tokenizer 变化后，需要重新构建数据并训练配套模型。

### 🔎 Step 3 · 生成并检查决策数据

```shell
python scripts/build_dataset.py --tokenizer out/full/tokenizer --out out/full/synth
python scripts/audit_leakage.py --data out/full/synth
python scripts/audit_synthetic.py --data out/full/synth
```

审计检查位置捷径、split 分离、规则目标能否由渲染文本恢复，并运行词法基线。首次运行还没有模型结果，词法对照只报告基线；模型训练完成后可通过 `--model_eval` 进行比较。

### 🧠 Step 4 · 搭建共享编码器

核心代码在 [model/model_system_one.py](model/model_system_one.py)，序列化在 [model/serialize.py](model/serialize.py)。建议先沿着这一条数据路径阅读：

```text
state + question + candidates
             │
        BPE tokenizer
             │
[state] [SEP] [question] [SEP] [candidate 1] [SEP] ...
             │
    共享 Transformer 编码器
     ├─ 汇聚 state + question → z
     └─ 汇聚每个候选           → c_i
             │
      同一个打分函数 f(z, c_i)
             │
       softmax(logits) → p
```

编码器采用 RMSNorm、RoPE、SwiGLU 和分组查询注意力。注意力不是全局无约束的双向连接：state 只看 state；question 看 state 和自身；候选看前缀及自身，不看其他候选。这个结构允许缓存前缀，并让候选重排不改变对应语义的打分。

默认参数量可以直接核对：

```shell
python scripts/model_stats.py --tier 26m
```

Noul 是两个候选，Choice 是动态候选集，Score 是带数值等级的候选集，因此无需为它们建立三个独立分类器。

### 🔥 Step 5 · MLM 预训练编码器

```shell
python trainer/train_mlm.py --tokenizer out/full/tokenizer --pretrain_path dataset/pretrain_zh.jsonl --en_path dataset/pretrain_en.jsonl --out out/full/mlm --log_dir out/full/mlm/logs --save_optimizer --no_swanlab
```

这一阶段从随机权重开始。MLM 临时输出头与 token embedding 共享权重，训练目标是恢复被遮盖的文本位置。产物 `mlm.pth` 用于初始化下一阶段的编码器。

### 🎯 Step 6 · 训练决策头和编码器

```shell
python trainer/train_decision.py --tokenizer out/full/tokenizer --data out/full/synth --encoder out/full/mlm/mlm.pth --out out/full/decision --log_dir out/full/decision/logs --save_optimizer --no_swanlab
```

加载预训练编码器，新建决策头，**两者一起更新**。默认训练候选采样上限为 32；Score 保留完整等级集，评测使用完整候选集。训练按候选数量和长度分桶以减少 padding。

| 参数 | MLM 默认 | 决策默认 | 如何理解 |
|---|---:|---:|---|
| `hidden_size` / `num_hidden_layers` | 512 / 8 | 512 / 8 | 两阶段必须匹配 |
| `batch_size` | 16 | 16 | 显存不足先降低 |
| `max_len` | 512 | 1024 | 决策输入还要容纳问题与候选 |
| `epochs` | 8 | 3 | 数据完整遍历次数 |
| `learning_rate` | 0.001 | 0.0005 | 另配 warmup 和余弦衰减 |
| `accum` | 1 | 1 | 梯度累积的 microbatch 数 |

`--use_checkpoint` 可以用计算换激活显存。减小 batch 后可增加 `--accum`，但不能据此假设所有效果完全不变。其他配置见各脚本的 `--help`。

### 💾 Step 7 · 保存与续训

普通 `.pth` 保存推理权重、模型配置和元数据；加 `--save_optimizer` 后另存 `*_opt.pth`，包含优化器、scaler、epoch、下一批位置及随机状态。

恢复上面的默认决策训练：

```shell
python trainer/train_decision.py --tokenizer out/full/tokenizer --data out/full/synth --resume out/full/decision/decision_opt.pth --out out/full/decision --log_dir out/full/decision/logs --save_optimizer --no_swanlab
```

保持原始数据、词表、训练设置与总 `epochs` 不变。`--max_steps N` 可在总 step 达到 N 时保存退出；恢复时删除这个限制即可继续原计划。MLM 同样支持 `--resume`。旧文件若缺少完整续训状态会被拒绝，不会伪装成精确恢复。

<a id="loss"></a>
## 📉 Ⅴ · 训练的 loss 是什么

### 先看实际训练曲线

![历史 MLM、决策训练损失及验证集 NLL](assets/readme/training.png)

这张图直接读取历史训练日志：MLM 记录到 66,000 step，决策训练记录到 33,750 step。浅线是记录的 batch loss，深线是最近 21 个记录点的均值；右侧是验证集 NLL。日志中的重启已分开，只保留最后一次运行。不同任务与候选数会带来 loss 波动，因此不能只凭一段下降曲线判断泛化效果。

**图中决策实验早于序数损失修复，作为训练过程记录保留，不作为当前代码的成绩。** [原始数值快照与绘图说明](assets/readme/README.md)随仓库提供，下面解释的是当前实现。

### 1. MLM：恢复被遮盖的 token

MLM 对约 15% 的位置做 span masking。被选中的位置中，约 80% 替换为 `<mask>`、10% 替换为随机 token、10% 保留原 token；特殊 token 不参与遮盖。

$$
\mathcal L_{\mathrm{MLM}}=-\frac{1}{|M|}\sum_{j\in M}\log P_\theta(x_j\mid\widetilde{x})
$$

这里 $M$ 是被选中的位置，$\widetilde{x}$ 是经过扰动的输入。只在这些位置计算交叉熵。它教编码器利用上下文，不负责决定业务候选。

### 2. 决策训练：学习整个候选分布

设模型预测为 $p$，目标为 $t$，当前样本有 $K$ 个候选：

$$
\mathcal L_{\mathrm{CE}}=-\sum_{i=1}^{K}t_i\log p_i,\qquad
\mathcal L_{\mathrm{Brier}}=\sum_{i=1}^{K}(p_i-t_i)^2
$$

- **CE**：目标支持的候选，应得到相应概率质量。
- **Brier**：直接惩罚预测分布与目标分布的平方差。默认不除以 K，`--brier_normalize` 可用于对照。
- **序数项**：只用于 Score，利用等级之间的距离，而不把“差一档”和“差四档”视为同样的错误。

Score 先按真实等级 $l_1<\cdots<l_K$ 排序，计算累积分布 $F_p(j)=\sum_{i\le j}p_i$：

$$
\mathcal L_{\mathrm{ord}}=
\frac{\sum_{j=1}^{K-1}(l_{j+1}-l_j)\,[F_p(j)-F_t(j)]^2}
{l_K-l_1}
$$

候选呈现顺序可以打乱，计算序数项时仍按 `meta.level` 排序，padding 不参与计算。一个 batch 的总损失为：

$$
\mathcal L=\operatorname{mean}_{B}(\mathcal L_{\mathrm{CE}})
+0.5\operatorname{mean}_{B}(\mathcal L_{\mathrm{Brier}})
+0.5\operatorname{mean}_{B_{\mathrm{Score}}}(\mathcal L_{\mathrm{ord}})
$$

没有 Score 样本时省略最后一项；权重由 `--lambda_brier`、`--lambda_ord` 控制。

**一个容易混淆的点：**硬标签交叉熵和 Brier 也能学习条件概率。软目标让本项目更容易直接检查分布误差，但不保证模型天然校准。对于软目标，本项目的 `brier` 字段是平方分布距离；标准观测标签 Brier 的期望还要加上 $1-\sum_i t_i^2$。更多解释见 [校准文档](docs/CALIBRATION.md)。

### 3. 温度校准：训练后调整概率的尖锐程度

冻结模型，在独立 `calib` 集上最小化 NLL，拟合正温度 $T$：

$$
p_i^{(T)}=\operatorname{softmax}(z/T)_i
$$

$T>1$ 通常让分布更平缓，$T<1$ 让它更尖锐。脚本比较全局、按 primitive、按 primitive×候选数分组的温度。它不会修复错误的候选排序，也不能代替模型学习任务。

<a id="evaluation"></a>
## 📊 Ⅵ · 评测：模型答得对，还是只是很自信

### 已有实验：先观察差距

![历史模型在合成与公开数据上的差距，以及温度校准前后 ECE](assets/readme/evaluation.png)

历史模型在合成留出集上表现较好，但迁移到公开真实文本时，NLL 和分布误差明显增加。两边的任务、候选数和目标分布不同，这不是严格控制变量的对比；它提醒我们合成数据上的成绩不能直接代表真实任务能力。右图在同一个合成测试集上比较校准前后的 ECE，较低 ECE 也不等于更高准确率。

<details>
<summary><b>展开：与 Jev / DeepSeek API 的历史小样本对比</b></summary>

![48 条合成样本的历史 API 探索性对比](assets/readme/api-comparison.png)

这次实验每个合成任务取 8 条，共 48 条，候选数不超过 8。MiniSystemOne 与 Jev 各有 48 条有效输出；DeepSeek 仅 37 条有效输出，11 条解析失败被排除在质量指标之外。因此不能把柱状图当作同样本公平排名，也不支持“超过某模型”的普遍结论。摘要没有记录 API 模型版本，图中也没有把本地推理与网络调用延迟直接比较。

</details>

以上均为修复前的历史实验。图表保留真实数值与局限，修复后的完整模型仍需重新训练评测。所有实验图均由记录程序绘制；首页主图是 AI 生成的教学示意图。可运行 `python scripts/make_readme_figures.py` 从仓库中的数值快照重绘，详见[图表来源与生成提示词](assets/readme/README.md)。

### 1. 校准与合成留出测试

```shell
python trainer/calibrate_temperature.py --tokenizer out/full/tokenizer --ckpt out/full/decision/decision.pth --data out/full/synth --out out/full/calibration
python eval/eval_harness.py --tokenizer out/full/tokenizer --ckpt out/full/decision/decision.pth --data out/full/synth --sets test_known --temperature out/full/calibration/T.json --out out/full/eval
python eval/make_reliability_plot.py --eval out/full/eval/decision --sets test_known --out out/full/plots
```

若 GPU 显存紧张，可给前两条命令增加 `--batch_size 2`。评测路径的文件名包含数据目录名，例如 `out/full/eval/decision/synth_test_known.json`。

| 指标 | 回答的问题 | 阅读方式 |
|---|---|---|
| `accuracy` | 是否命中目标分布的 argmax？ | 软目标存在并列时会受 tie-breaking 影响 |
| `accuracy_soft` | 被选候选的目标概率是多少？ | 硬标签下等于准确率，软标签下是期望正确率 |
| `nll` | 是否给目标支持的答案足够概率？ | 越小越好 |
| `brier` | 预测分布离目标分布多远？ | 当前实现为平方分布距离 |
| `ece` | 置信度桶内的预测与正确率是否一致？ | 越小越好，但单独看它不够 |
| `ordinal_mae` | 预测等级分布移动多远才能到目标？ | 等级单位的 Wasserstein-1 距离 |
| `expected_score_mae` | 期望分误差有多大？ | Score 专用 |

按 `source` 和 `provenance` 分组读结果，再看汇总。历史字段 `calibration` 表示排除 `hard` 的分布目标子集，并不表示硬标签无法评测校准。

### 2. 到真实文本上再测一次

```shell
python scripts/build_dataset.py --public --tokenizer out/full/tokenizer --out out/full/public
python eval/eval_harness.py --tokenizer out/full/tokenizer --ckpt out/full/decision/decision.pth --data out/full/public --sets test_known --batch_size 2 --out out/full/eval_public
python eval/make_reliability_plot.py --eval out/full/eval_public/decision --sets test_known --source public:chaosnli --out out/full/plots_public
```

这里先报告原始概率，不自动套用合成数据拟合的温度。公开集虽也叫 `test_known`，对只训练过合成任务的模型而言仍然是跨数据域测试。若用公开数据继续训练，应保留其独立校准集和测试集，并明确说明训练来源已改变。

### 3. 测试未见过的任务族

```shell
python scripts/build_dataset.py --tokenizer out/full/tokenizer --out out/full/synth_ood --ood_generators calendar_slot security_gate
```

这会把指定生成器全部留到 `test_ood`。要做严格的任务族泛化实验，必须用 `out/full/synth_ood/train.jsonl` **重新训练一个决策模型**，再测其 `test_ood`；不能拿已经在这两类任务上训练过的全量模型来宣称 OOD 能力。若声称整个链路都未见过这些任务，还需在 tokenizer/MLM 语料中排除相应生成器；默认合成预训练语料并未做这种隔离。

### 4. 速度、显存与代码正确性

```shell
python eval/eval_efficiency.py --tokenizer out/full/tokenizer --ckpt out/full/decision/decision.pth --data out/full/synth --out out/full/efficiency
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cpu --skip_big
```

效率报告与任务效果分开阅读。候选数、state 长度、batch、分词和打包开销都会改变端到端延迟。架构测试覆盖候选重排、分块和前缀隔离；回归测试覆盖序数、截断、弃权、推理和续训。GPU 续训一致性测试的开启方式见 [详细入门指南](docs/QUICKSTART.md)。

<a id="custom-task"></a>
## 🎯 Ⅶ · 换成你自己的任务

以客服工具路由为例：

1. **定义选择。** 明确每个候选的含义，以及什么情况下应交给人工。可同时成立的多个判断，分别建 Noul 问题。
2. **准备样本。** 按前面的 JSONL 格式记录真实请求、候选和标签；覆盖不同表达、缺失信息与容易混淆的请求。
3. **分开数据。** 在 `dataset/my_task/` 准备 `train.jsonl`、`val.jsonl`、`calib.jsonl`、`test_known.jsonl`。按会话、用户或模板隔离，避免改写泄漏。
4. **训练并评测。** 先用自己的验证集观察错误，再在校准集拟合温度，最后只在测试集报告结果。
5. **接入程序。** 用同一候选命名调用 `DecisionPredictor`，在验证集上选择路由阈值，记录覆盖率和错误率。

使用前面完整路线的编码器与词表：

```shell
python trainer/train_decision.py --tokenizer out/full/tokenizer --data dataset/my_task --encoder out/full/mlm/mlm.pth --out out/my_task --log_dir out/my_task/logs --save_optimizer --no_swanlab
python trainer/calibrate_temperature.py --tokenizer out/full/tokenizer --data dataset/my_task --ckpt out/my_task/decision.pth --out out/my_task/calibration
python eval/eval_harness.py --tokenizer out/full/tokenizer --data dataset/my_task --ckpt out/my_task/decision.pth --sets test_known --temperature out/my_task/calibration/T.json --out out/my_task/eval
```

然后将任务示例里的 checkpoint 换为 `out/my_task/decision.pth`，tokenizer 换为 `out/full/tokenizer`。如需应用相应温度，构造 `DecisionPredictor` 时传 `calibration="out/my_task/calibration/T.json"`。

<a id="code-map"></a>
## 🧭 Ⅷ · 代码阅读顺序

| 顺序 | 文件 | 先理解什么 |
|---:|---|---|
| 1 | [train_tokenizer.py](trainer/train_tokenizer.py) | 文本怎样变成 token |
| 2 | [serialize.py](model/serialize.py) | 输入布局、segment、mask、候选 span |
| 3 | [model_system_one.py](model/model_system_one.py) | 编码器、pooling、候选打分和 loss |
| 4 | [decision_dataset.py](dataset/decision_dataset.py) | 截断、候选采样和 batch |
| 5 | [train_mlm.py](trainer/train_mlm.py) / [train_decision.py](trainer/train_decision.py) | 优化器与训练循环 |
| 6 | [eval_metrics.py](eval/eval_metrics.py) / [calibrate_temperature.py](trainer/calibrate_temperature.py) | 如何测概率、如何校准 |
| 7 | [inference.py](model/inference.py) / [decide.py](scripts/decide.py) | 如何把模型接入一个任务 |

进一步阅读：[架构设计](docs/DESIGN.md) · [数据格式](docs/DATA_SCHEMA.md) · [校准说明](docs/CALIBRATION.md) · [详细入门指南](docs/QUICKSTART.md)。

## 💡 常见问题

**它能像聊天模型一样回答任意问题吗？** 不能。当前目标是给定上下文和候选的任务决策，能力取决于训练分布。概率合法、输出格式合法都不等于答案正确。

**为什么还需要 MLM？** 它先提供文本表示。可以用 `--encoder=` 从随机权重直接训练决策，作为“预训练是否有用”的消融，而不是默认假设它一定更好。

**CPU 可以训练吗？** 当前两个训练入口要求 CUDA；CPU 支持推理、参数统计和基础测试。先用小规模路线检查自己的环境。

**更换候选有什么限制？** 默认各候选独立打分，适合“这个选项是否符合证据”。对于“选出当前候选集合的中位数”等依赖候选之间关系的任务，这个结构有表达限制。

**怎样避免显存不足？** 先减小 `--batch_size`，再考虑序列长度和 `--use_checkpoint`。尤其在 Windows 上，显存压力可能先表现为严重变慢，而不立即报 OOM。

**我只想尽快试用。** 从快速路线生成自己的小权重，再运行 `scripts/decide.py`；若已有配套的正式权重和 tokenizer，也可以直接调用。不要混用不同训练 run 的词表、权重和温度。

## 🙏 致谢与许可

感谢 [MiniMind](https://github.com/jingyaogong/minimind) 对从零训练与教学实践的启发，感谢公开数据集作者。决策模型的产品形式参考 [TypeSafe 的 Jev 介绍](https://typesafe.ai/blog/introducing-system-one-models-and-jev)；概率评分理论可参考 [Gneiting 与 Raftery](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf)。

代码采用 [Apache-2.0](LICENSE)。外部语料的许可与使用条件请查阅对应数据源。
