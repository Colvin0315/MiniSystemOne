# 第一个自定义任务：客服请求 → 工具或人工

这是一个 **independent educational implementation** 的小教程，不是可上线客服系统。
我们从本地合成语料学习流程，再训练自己的决策任务；不使用 Jev 或任何外部模型输出蒸馏。
生成器离线运行，不加入全局合成任务注册表，不访问网络。所有模型训练、校准和推理使用 CUDA。

## 1. 先跑通 quickstart

在仓库根目录、已安装依赖且 CUDA 可用的 Python 环境中运行。此机器可用 `conda activate minimind`；
也可以使用 `C:/Coding/Anaconda3/envs/minimind/python.exe` 代替下文的 `python`。
以下命令每条独立执行，不依赖 Bash 的续行语法。

```bash
python scripts/quickstart.py
```

它在全新的 `out/quickstart/` 完成最小流程。已有产物时不要覆盖，选择新的输出目录并同步修改下文路径。
本教程需要它产出的配套 tokenizer `out/quickstart/tokenizer/` 和 MLM 编码器
`out/quickstart/mlm/mlm.pth`，不是把 quickstart 的通用决策头当作已经学会客服任务。
快速配置为 hidden size 128、2 层；短训和模板预训练只证明流程能运行，不保证业务准确率。
没有 CUDA 或配套权重就先解决环境问题，不回退 CPU、不随机初始化冒充已有模型。

## 2. 定义一个小而明确的任务

每次互斥选择一个候选：

| label | 固定含义 |
|---|---|
| `order` | 查询订单物流状态 |
| `refund` | 登记订单退款申请（不直接退款） |
| `account` | 查询账户登录问题 |
| `human` | 转人工客服，进一步确认需求 |

`human` 是参与训练的显式候选，不是第五个隐含类别。预测它时总是交给人工；
预测工具但置信度不足时，也交给人工。这里的人工标签来自“需求不明确、需要人工沟通”的可见文本，
**不意味着模型已经见过所有未知业务**。

- **Choice**：选项互斥、一次选择一个，概率在候选集合上归一化。
- **多个独立 Noul**：如果“查订单”和“查账户”可以同时成立，应为每个条件分别询问 yes/no，
  并指定 `schema.positive_label`，而不是用一个 Choice 强迫它们竞争。
  独立 Noul 的正类概率之间不必和为 1，不能直接当成同一 Choice 分布。
- **Score**：适用于有序等级，例如满意度 1–5，候选的 `meta.level` 表示数值等级，输出可以取期望分数。

修改 schema 名称、候选文本或增加工具不会自动学会新业务；应重新定义标注规则、准备数据、训练、校准并选策略。
模型独立打分候选也有表达限制：固定请求下两个候选的概率比由各自得分决定，新增第三项不自动学会它们之间的业务关系。

### 手写一条完整训练样本

下面是一整条样本的展开形式；存为 JSONL 时，每条对象必须占一行。它属于合法的硬标签训练：
`p` 是 one-hot，顺序与 `candidates` 一一对应。硬标签同样支持交叉熵、Brier、NLL 和 ECE，不要求编造软概率。

```json
{
  "id": "customer_tool_routing::train::manual0001",
  "source": "synth:customer_tool_routing",
  "gen_version": "1.0.0",
  "split": "train",
  "schema": {
    "primitive": "choice",
    "name": "customer_tool_routing",
    "desc": "选择一个客服工具；信息不足或超出支持范围时转人工。"
  },
  "state": "请查询订单 C1000 的物流状态。",
  "state_sections": [
    {"seg": "notes", "text": "请查询订单 C1000 的物流状态。", "priority": 3}
  ],
  "question": "这个请求应交给哪个工具或人工客服？",
  "question_paraphrases": ["应该选择哪个客服工具？", "请选一个处理渠道。", "此请求应如何分派？"],
  "candidates": [
    {"text": "转人工客服，进一步确认需求", "label": "human", "meta": {"level": null}},
    {"text": "查询订单物流状态", "label": "order", "meta": {"level": null}},
    {"text": "登记订单退款申请（不直接退款）", "label": "refund", "meta": {"level": null}},
    {"text": "查询账户登录问题", "label": "account", "meta": {"level": null}}
  ],
  "target": {
    "kind": "hard",
    "p": [0, 1, 0, 0],
    "provenance": "hard",
    "renormalized": false,
    "audit": {"correct_label": "order"}
  },
  "meta": {
    "K_full": 4,
    "approx_tokens": 256,
    "template_id": "order-0",
    "entity_pool": "train",
    "entity_id": "C1000"
  }
}
```

`target.audit`、`id`、`split` 和数据元信息不会作为模型输入；不能把答案偷偷写进 `state`。
`state_sections` 是训练时的截断结构，本例的 `notes` 与 `state` 相同。
Choice 的 level 为 null；`approx_tokens` 只是训练分桶估计，不是推理长度保证。
推理只传 `schema/state/question/candidates`，不需要 target。

## 3. 独立生成并检查四个 split

```bash
python examples/customer_tool_routing.py build --out out/customer/data --seed 0
```

产生 train 128、val 32、calib 32、test_known 32 条 JSONL，四类均衡。
先分配每类的四种不同措辞模板和四个不相交的实体池，再生成样本：
训练实体 C1000–C1031，val C1100–C1107，calib C1200–C1207，test C1300–C1307。
模板和实体在 split 间均不重叠；代码检查 ID、state、模板、实体、实体池不泄漏。
共享问题和候选文本是任务定义，不是复制训练样本。候选逐条按 seed 确定性重排，one-hot 在重排后按 label 对齐。
已有非空 `--out` 会报错，包括上次未完成的生成目录，绝不默默覆盖。

若换成自己的数据，仍按上面 schema 写四个文件，并在生成之前划分客户/订单/模板或时间组，
不要把同一客户、同一请求的改写随机分到训练与测试。保留本脚本的四个候选语义；
若更换业务定义，也要同步修改脚本的 allowlist 和数据验证，不只是改 JSON 名称。
这里的留出集只衡量这个小任务的保留措辞与实体，不能作为一般 OOD 成绩。

## 4. 用 MLM 编码器训练自己的决策头

```bash
python trainer/train_decision.py --data out/customer/data --tokenizer out/quickstart/tokenizer --encoder out/quickstart/mlm/mlm.pth --hidden_size 128 --num_hidden_layers 2 --max_len 512 --batch_size 4 --k_min 4 --k_max 4 --epochs 10 --max_steps 100 --warmup_steps 2 --num_workers 0 --device cuda --save_optimizer --no_swanlab --out out/customer/decision --log_dir out/customer/logs --save_interval 50 --log_interval 10
```

数据目录与训练输出分开，避免生成器的防覆盖检查碰到日志/权重。
`--encoder` 是 MLM → decision 初始化，决策头新训练；不是恢复上次优化器。
`--epochs 10` 提供足够批次，`--max_steps 100` 把本次绝对训练预算限制在 100 次更新。
固定 K=4 让这份教学样本始终看到人工和三个工具，不在本例增加候选子采样难度。
`--save_optimizer` 为需要恢复训练时保留完整恢复状态，但本教程不展开 resume。
得到 `out/customer/decision/decision.pth` 后冻结这个权重，不根据 test 结果继续挑版本。

## 5. 先在 calib 拟合温度，再只用 val 选策略

```bash
python trainer/calibrate_temperature.py --ckpt out/customer/decision/decision.pth --tokenizer out/quickstart/tokenizer --data out/customer/data --split calib --batch_size 4 --out out/customer/calibration
python examples/customer_tool_routing.py select --data out/customer/data --ckpt out/customer/decision/decision.pth --tokenizer out/quickstart/tokenizer --temperature out/customer/calibration/T.json --policy out/customer/policy.json --max_risk 0.1
```

脚本固定使用 **global 温度**，复用根目录 `inference.py` 的 `Predictor` 和公共指标库。
Predictor 验证温度对应的 checkpoint/tokenizer 哈希；教程另外要求温度元信息 `split=calib`。
温度是概率缩放，不会创造缺失的业务知识。calib 拟合完成后才能运行 select。
当前温度元信息提供 split 和模型/词表绑定；它不是数据来源的可信证明，应自己确保传入的 calib 就是上述独立文件。

策略只读取 val 标签选阈值，复用 `eval_metrics.risk_coverage_curve`：
完整置信度并列组一起接受或拒绝，人工候选的位置逐行按 label 找，绝不假定人工在最后。
在经验 accepted risk ≤ `--max_risk` 的所有可行点中选最大 coverage；风险曲线不要求单调。
如果没有任何正覆盖的可行阈值，保存 `"threshold": null` 表示全部人工，不写非法 JSON 的 Infinity。

`policy.json` 保存阈值、任务/候选定义、温度粒度、val 指标以及 checkpoint、tokenizer 文件、温度文件和四个数据文件的 SHA-256。
只计算 test 文件哈希，不用 test 标签调参。文件已存在就拒绝重写。
换权重、词表、温度或数据必须创建新的实验策略，而不是继续使用旧阈值。
这是防误用的内容绑定，不是签名或防恶意篡改的安全边界。

## 6. 独立动作：只评测一次冻结策略

```bash
python examples/customer_tool_routing.py evaluate --data out/customer/data --ckpt out/customer/decision/decision.pth --tokenizer out/quickstart/tokenizer --temperature out/customer/calibration/T.json --policy out/customer/policy.json --out out/customer/test_report.json
```

也可省略 data/ckpt/tokenizer/temperature，从策略记录的绝对路径读取。显式路径可迁移产物，但文件内容哈希必须一致。
`evaluate` 仅读取 `test_known` 预测与标签，无阈值拟合选项，不接受 `--max_risk`；不覆盖已有报告。
不要根据这个 test 报告再调参数、重新挑阈值后仍称它是独立测试。

报告包含：

- `coverage = accepted / n`：自动工具覆盖率，预测人工从不计入自动覆盖。
- `accepted_risk = accepted_errors / accepted`：接受样本的观测错误率；accepted=0 时为 **null 而不是 0**。
- `handoff_fraction = human_handoff / n`：人工比例，包含模型选人工及阈值拒绝。
- `n/accepted/accepted_errors/model_human/threshold_handoff/human_handoff`：可核对的计数。
- `metrics_all`：公共 `eval_metrics.compute_metrics` 在全部样本上计算的
  `distribution_l2`、`expected_brier`、`nll`、`ece` 等，不仅是接受子集。
  ECE 使用 15 桶 equal-mass top-label 口径。小样本 ECE 很不稳定。

本例全是 one-hot，所以 distribution L2 与 expected Brier 相等；软目标下两者相差
`mean(1 - sum(t**2))`。硬标签 proper scores 合法，ECE 也不能证明逐条概率正确。
**val 上 ≤10% 是样本内经验约束，不是测试集、未来流量或未知业务的 10% 风险保证**。
全部转人工是合法结果，不应为演示效果放宽阈值或编造成功数字。

## 7. 实际推理和安全 stub 分发

```bash
python examples/customer_tool_routing.py dispatch --policy out/customer/policy.json --state "请查询订单 C9000 的物流状态。"
python examples/customer_tool_routing.py dispatch --policy out/customer/policy.json --state "请为我们的企业开通跨境税务申报和报关服务。" --unseen
```

第二条是没有训练的业务请求。请阅读本机实际输出的 `prediction.candidates`、
`selected_label`、`confidence`、`calibration`、`input_tokens` 和最终 `dispatch`，不要预填预测数字。
`not_trained_example: true` 仅由人为提供的 `--unseen` 标注产生，**不是模型检测到了 OOD**，也不会强迫分发到人工。
如果它高置信地选错工具，保留这个真实输出；即使碰巧转人工，也不能声称解决了未知业务拒答。

所有 handler 都是本地 allowlist stub：输出 `dry_run: true`、`side_effects: false`，
不会联网、退款、发邮件或修改账户。模型返回的字符串绝不作为函数名任意调用，更不会执行 shell。
人工接管本身也只是打印结果，没有创建真实工单。

`dispatch --input request.json` 可替代 `--state`。JSON 使用上文的
`schema/state/question/candidates` 四字段，候选可以重排，但 label 与文本含义必须保持。
每次 dispatch 都检查冻结产物哈希，因此这份教学脚本仍需要策略记录的数据文件；它不是部署打包工具。
Predictor 默认 CUDA，严格检查序列长度，超预算报错而不是静默截断。
生产调用还需要身份权限、参数验证、真实人工渠道和业务安全检查，不属于此 stub。

## 8. 本次 GPU 实测与失败边界

2026-09-20，在 RTX 4070 Laptop 上按上述 100 次决策更新命令完成整条流程。
[聚合结果](../results/customer_tool_routing.json)记录：val 准确率 50%，test_known 准确率 65.625%。
在 val 的 `max_risk=0.1` 约束下，没有满足条件的正覆盖阈值，策略保存 `threshold=null`；
test 自动覆盖率为 0、人工比例 100%、接受错误率为 null。没有为提高演示覆盖率而重新选阈值。

订单查询例子选中 order，校准置信度约 0.690，但按冻结策略仍转人工。
未训练的报关/税务请求选中 human，置信度约 0.256；单个成功转人工不证明未知任务检测能力。
这次实测展示流程和局限，不是业务部署验收。

## 9. 无模型单元测试与 GPU 验证边界

```bash
python -m unittest discover -s tests -p test_routing.py -v
```

这些测试只使用标准库和 NumPy，覆盖候选/标签对齐、split 隔离、防覆盖、并列置信度、
逐行人工位置、全人工、零覆盖 null、产物修改拒绝及 test 不调阈值。它们不加载模型，也不是 CPU 推理性能测试。
真实 GPU 验收应另行完整运行 build → train → calib → select → evaluate → dispatch，
以生成的报告和实际推理输出为准；单元测试通过不能代替业务质量验证。
