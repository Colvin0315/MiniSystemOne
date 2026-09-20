# 从第一次推理到自己的决策任务

本教程先演示使用，再用小规模合成数据走通从零训练。命令均在仓库根目录执行，使用单行写法以兼容 PowerShell 与 Bash。模型输出合法候选上的概率，不保证答案正确或在新业务上校准。

## 1. 环境与验证

训练需要支持 CUDA 的 NVIDIA GPU；推理和回归测试也支持 CPU。现有正式实验来自 RTX 4070 Laptop，其他显卡的速度和显存需要自行测量。

```shell
conda create -n minisystemone python=3.12 -y
conda activate minisystemone
python -m pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
python scripts/smoke_test.py --device cpu --skip_big
```

普通 unittest 会跳过需要 CUDA 的训练进程测试。PowerShell 中可另运行：

```powershell
$env:MINISYSTEMONE_GPU_TESTS='1'
python -m unittest discover -s tests -p test_training_integration.py -v
```

该测试在临时目录训练很小的模型，比较连续训练与中途恢复的完整精度权重，包括不足一个梯度累积窗口的 epoch 尾部。它不是模型效果评测，也不证明跨设备逐比特可复现。

## 2. 用已有权重推理

若已有 `out/decision/decision.pth` 及配套 `model/` tokenizer：

```shell
python scripts/decide.py --ckpt out/decision/decision.pth --input examples/noul.json --device cpu
python scripts/decide.py --ckpt out/decision/decision.pth --input examples/choice.json --device cpu
python scripts/decide.py --ckpt out/decision/decision.pth --input examples/score.json --device cpu
```

没有权重时，先完成下一节。仓库不包含 `.pth` 文件，也不假设已有可下载的新版本权重。修复前权重仍能推理，但其序数训练存在已知问题，不能代表修复后的效果。

输入契约：

| primitive | 额外输入 | 额外输出 |
|---|---|---|
| `noul` | 候选固定为 `yes`、`no`，可省略 candidates | `p_true` |
| `choice` | `candidates`：2–255 个不同的非空字符串 | 所有候选的概率 |
| `score` | 2–10 个候选及同序的不同非负 `levels` | `expected_score` |

公共字段是 `state`、`question`、`primitive`。输出 `choice`、`confidence` 和 `probabilities`；Score 的等级与候选顺序对应，不要求按大小排列。三个示例仅展示接口，未保证在它们上面的预测质量。

默认最多保留 512 个 state token，头尾截断会令 `state_truncated=true`。问题及单个候选最多 128 token；超出时报错。可用 `--max_state_tokens` 调整，但丢失证据会影响结果。默认每块 16 个候选，可用 `--chunk` 调整。

Python 调用：

```python
import json
from model.inference import DecisionPredictor

predictor = DecisionPredictor("out/decision/decision.pth", "model", device="cpu")
with open("examples/choice.json", encoding="utf-8") as f:
    result = predictor.predict(json.load(f))
print(result)
```

若要应用温度，传 `--calibration out/calibration/T.json`。入口会检查权重及词表哈希；哈希匹配仅保证文件配套，不保证校准分布适合新业务。未传温度时输出 `temperature_applied=false`，这不等于已经验证模型校准。

## 3. 最小从零训练流程

下面使用独立的 `out/quickstart/`，新训 BPE 和随机初始化的小编码器，只用程序生成的语料，不依赖旁边的 MiniMind 仓库，也不下载外部数据。需要 CUDA；请使用尚未存放重要产物的输出目录。

这是一条流程验证路线，**不是 README 历史效果的复现配方**。为了快速走通，跳过 tokenizer 压缩率门禁、缩小模型和数据，并只训练少量步；这些产物不能用作质量结论。

```shell
python trainer/train_tokenizer.py --pretrain_path= --en_path= --n_docs 0 --n_synth 200 --out_dir out/quickstart/tokenizer --skip_eval
python scripts/build_dataset.py --tokenizer out/quickstart/tokenizer --out out/quickstart/data --per_gen_train 20 --per_gen_val 4 --per_gen_calib 4 --per_gen_test_known 4 --per_gen_test_ood 4
python trainer/train_mlm.py --tokenizer out/quickstart/tokenizer --pretrain_path= --en_path= --n_docs 0 --n_synth 40 --hidden_size 128 --num_hidden_layers 2 --max_len 256 --batch_size 2 --epochs 1 --max_steps 5 --save_optimizer --no_swanlab --out out/quickstart/mlm --log_dir out/quickstart/mlm/logs
python trainer/train_decision.py --tokenizer out/quickstart/tokenizer --data out/quickstart/data --encoder out/quickstart/mlm/mlm.pth --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --epochs 1 --max_steps 5 --val_limit 12 --val_every 0 --save_optimizer --no_swanlab --out out/quickstart/decision --log_dir out/quickstart/decision/logs
python trainer/calibrate_temperature.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --data out/quickstart/data --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --lbfgs_steps 10 --out out/quickstart/calibration
python eval/eval_harness.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --data out/quickstart/data --hidden_size 128 --num_hidden_layers 2 --batch_size 2 --sets test_known --out out/quickstart/eval
python scripts/decide.py --tokenizer out/quickstart/tokenizer --ckpt out/quickstart/decision/decision.pth --input examples/choice.json --device cpu
```

每阶段的意义：tokenizer 把文本变成整数；MLM 学习文本表示；决策训练学习条件分布；`calib` 只拟合温度；`test_known` 只报告独立评测。小校准集只是演示，不足以证明概率可靠。

正式训练时应补入语言语料，取消 `--skip_eval`、取消步数限制并运行数据审计。MLM 对非空但不存在的语料路径会报错；仅显式 `--pretrain_path=` 或 `--en_path=` 才表示不要该来源。不要把小规模教程结果与历史 26.89M 模型比较。

## 4. 中断后继续

训练时加 `--save_optimizer` 才会生成完整的 `*_opt.pth`；普通 `.pth` 仅用于推理或初始化。按原训练命令保留数据、词表、模型尺寸、batch、accum、seed、学习率和**总 epochs**，添加：

```shell
python trainer/train_decision.py --resume out/decision/decision_opt.pth --save_optimizer --no_swanlab
```

这条简写只适用于原本就是默认参数的训练。小规模教程则复制原命令，删除 `--max_steps 5`，添加对应的 `--resume ..._opt.pth`。

`--max_steps N` 是达到总 optimizer step N 后保存退出，方便验证续训；它不改变计划的学习率总步数。续训从记录的下一批开始，恢复模型、优化器、scaler、Python/NumPy/Torch/CUDA 随机状态。要求保持软件、硬件环境一致；旧优化器文件缺少完整位置及随机状态时会拒绝续训，不假装精确恢复。`--encoder` 仍可用于权重初始化，但会新建优化器和训练计划。

## 5. 训练自己的第一个任务

建议从三个客服工具开始：余额查询、转账、人工处理。若几个决策可同时为真，分别建 Noul 问题；如果只能选一个工具，用 Choice。增加候选名称并不会自动让模型掌握新任务，需要训练数据及独立评测。

在 `dataset/my_task/` 准备 `train.jsonl`、`val.jsonl`、`calib.jsonl`、`test_known.jsonl`。每行是一个完整 JSON 对象，例如下面这一行（只是格式示例，不能复制成训练与测试的共同样本）：

```json
{"id":"balance-001","state":"用户希望查询账户余额。","question":"应该调用哪个工具？","schema":{"primitive":"choice"},"candidates":[{"text":"balance_lookup","label":"balance"},{"text":"transfer_funds","label":"transfer"},{"text":"human_review","label":"human"}],"target":{"p":[1,0,0],"provenance":"hard"},"source":"my_customer_service","gen_version":"my-task-v1"}
```

更多字段及约束见 [DATA_SCHEMA.md](DATA_SCHEMA.md)。目标顺序必须与候选一致且和为 1；硬标签是合法训练信号。不要凭感觉把 1 改成 0.8 当作“校准标签”。软标签可以来自已知随机规则或多位独立标注者，并应记录来源。

按客户/会话/模板等业务实体隔离 split，避免同一请求改几个字出现在多个集合。覆盖拒答、近义候选、不同顺序、缺失证据和常见失败样本。准备好后：

```shell
python trainer/train_decision.py --data dataset/my_task --encoder out/mlm/mlm.pth --out out/my_task --log_dir out/my_task/logs --save_optimizer --no_swanlab
python trainer/calibrate_temperature.py --data dataset/my_task --ckpt out/my_task/decision.pth --out out/my_task/calibration
python eval/eval_harness.py --data dataset/my_task --ckpt out/my_task/decision.pth --sets test_known --temperature out/my_task/calibration/T.json --out out/my_task/eval
```

以上假设使用默认 512×8 模型及 `model/` 词表；其他尺寸须显式传同样参数。业务阈值在验证集选择，测试集报告覆盖率与错误率；不要在测试集上反复挑阈值。`eval.eval_metrics.risk_coverage_curve` 支持软目标与弃权候选，但目前没有自动生成阈值路由报告的命令。

## 6. 候选打分的边界

默认架构中，每个候选只看 state、问题和自己。这支持分块及顺序一致性，但在前缀不变时，新增一个候选原则上不改变原有两个候选的概率比。依赖整个候选集合的比较任务（例如选出当前集合的中位数）不能默认适用。还应测试重复/近义候选与没有正确选项的输入；合法输出格式不等于可靠业务决策。
