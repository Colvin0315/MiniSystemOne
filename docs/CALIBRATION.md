# 校准与指标契约

本项目输出候选概率；是否校准需要在明确的评测总体上验证，不能由模型结构、训练目标或单个低 ECE 数字保证。

## 1. 硬标签与软标签都能训练和评估校准

设真实条件分布为 q(y|x)。对观测硬标签 Y 最小化交叉熵或 multiclass Brier，
其条件期望都在 p=q 处最优（proper scoring rules）。单条 one-hot 不是整个条件分布，
但跨样本的经验风险可以学习条件概率。有限数据、模型容量、优化和分布偏移会影响实际结果。
Brier 不是 label smoothing，也不会自动封顶置信度。软标签不是校准的必要条件。

已知软目标便于直接监督和检查完整分布：`explicit_rng` 是已知随机规则，
`marginalized` 是隐藏变量的条件边缘化，`tie_set` 是定义的有效集均匀目标，
`human_annotators` 是有限标注者的经验频率。最后一种不自动等于真实条件分布。
`hard` 是观测类别，也可以报告 ECE、NLL 和观测 Brier。

所有来源报告 `all` 与各 provenance。混合评测合法，但必须说明组成、权重和评测目的，
并检查子组，以免正负校准误差在同一桶抵消。`soft_targets` 只是上述四种软目标来源的
聚合，不表示只有它们有资格评测校准；当它与 `all` 重合时不重复输出。

## 2. distribution L2 与 expected Brier

对每条样本在有效候选上分别归一化 p 和 t，屏蔽槽不贡献任何项：

```
distribution_l2 = mean_i sum_k (p_ik - t_ik)^2
expected_brier = distribution_l2 + mean_i (1 - sum_k t_ik^2)
               = mean_i E_{Y~t_i} sum_k (p_ik - 1[Y=k])^2
```

这是类别平方差**求和后跨样本平均**，不是每类别 MSE。one-hot 时附加项为零，
两者都等于观测 multiclass Brier。软目标下 expected Brier 是相对于**给定 t** 的期望，
不是声称已知真实 q。给定固定目标，两者梯度相同，数值不同。

训练选项 `--lambda_brier`、`--brier_normalize` 和内部 loss logger 的 `brier`
保留原名；所训练的平方项是 distribution L2。默认不除以 K；显式归一化时每条除以
有效候选数（无 mask 时为 K），再跨样本平均。模型 forward 显式传递该开关。

### 输出键迁移（没有兼容别名）

| 旧输出/函数 | 新契约 |
|---|---|
| `brier` | `distribution_l2`；新增独立的 `expected_brier` |
| `calibration` 聚合 / `CALIBRATION_PROVENANCE` | `soft_targets` / `SOFT_TARGET_PROVENANCE` |
| `ece_noise_floor` / `binomial_noise_floor` | `ece_annotation_reference`（语义也改变，不能只改键名） |
| `ece_corrected` | 删除，不从 ECE 中扣参考量 |

旧 README 表中的 Brier 数字实际是 distribution L2，只改列名，不重算或捏造历史
expected Brier。历史文件和旧图片可能仍带旧名字；需重跑生成新格式，不把旧图片中的
“地板/校正后”标题当作有效解释。

## 3. ECE 测什么

top-label ECE 先以 `argmax(p)` 选类，以 `max(p)` 分桶，再比较桶平均置信度与
桶平均 `t[argmax(p)]`。硬标签下后者自然是 0/1 观测正确率。
ECE 不是 proper score；它依赖分桶、样本量和评测总体，不能证明逐条分布准确、
条件校准、排序最优或未知任务/OOD 理解能力。很低的总体 ECE 也可能掩盖子组失准。

同时报告 ECE、distribution L2、expected Brier、NLL、可靠性图、桶样本数和来源组成。
默认 equal-mass；并列置信度不强拆，实际桶数可能较少。equal-width 也有效，两种
方案没有普遍优劣，比较时须固定设置并记录。图上的点不是置信区间。

## 4. 标注 Monte Carlo 参考量，不是噪声下界

`ece_annotation_reference` 固定模型 top-label 置信度 c_i，并假设它就是真实正确概率，
模拟独立 `Binomial(N_i, c_i)/N_i`，重复计算分桶 ECE 后取均值。
这只是特定假设、样本量、标注数和分桶下的诊断量，**不是通用噪声下界、不是置信区间，
也不是可从观测 ECE 中相减的偏差估计**。

只在有限且正的 counts 子集上建桶和模拟，报告 `ece_annotation_reference_n`。
混合 counts 中缺失/零计数的行被排除，不伪造零正确率。没有可用计数时返回 null。
单条 N=100、q=0.5 的频率标准误约 0.05，不等于分桶 ECE 的下界；桶内平均可能显著
减小波动，标注相关性也可能破坏独立假设。

历史 ChaosNLI 原始 ECE **0.0613**、模拟量 **0.0068** 保留；旧减法值 **0.0545**
仅是已撤回的算术解释，不再作为校正后 ECE。GoEmotions 原始 ECE **0.3253**、历史
模拟量 **0.0044**，混合 `human_annotators` ECE **0.3167**。这些不是重新测量结果；
来源任务和标注协议不同，应分组说明，而不是宣称混合指标一概无效。

## 5. 为什么本项目选标量温度

动态候选的槽位没有固定类别语义，按槽位拟合温度或偏置会依赖候选顺序。
标量温度 `softmax(logits/T)` 是简单、正值、保持排序且候选置换等变的选择，
**不是唯一**能处理动态候选的校准方法；共享/等变映射也可能做到，本项目不实现它们。
LayerNorm 是建模选择，不保证不同 K 或任务的 logit 尺度相同，更不保证校准。

`trainer/calibrate_temperature.py` 在独立 `calib` 上拟合 `log T`，不在测试集挑温度。
目标是 NLL，LBFGS 默认 `max_iter=200`、`strong_wolfe`、float64，T 范围 `[0.05,20]`。
达到边界需诊断，可能表示极端置信度或样本不足，不自动证明数据有 bug。

显式屏蔽 padding：历史诊断中 T=15.0、128 槽中 126 个 padding 时，漏屏蔽使拟合
降至 5.99。有限负屏蔽值避免 `0 * inf` 梯度 NaN；概率路径也必须保持 mask 槽为零。

三个粒度为 `global`、`primitive` 和 `primitive_k`，K 桶为
`(2,2),(3,4),(5,8),(9,32),(33,255)`。按粒度报告；小样本桶不足以支撑强结论。
`T.json` 的 checkpoint/tokenizer 哈希必须匹配，加载的温度须在 `[0.05,20]` 内，
避免任意极小温度导致幂运算下溢。未知分组按实现回退到全局温度，
这只是运行规则，不是已校准的保证。

温度不改变 argmax，也不保证修正系统性先验偏移、未知 schema 或更大 K。
**跨分布迁移可能成功也可能失败，必须实测**，不能声称标量天然可迁移或必然不能迁移。
合成温度应用到真实文本可以作为明确标记的迁移实验；若要本域校准，应在本域独立
calib/dev 拟合，冻结后在本域测试。历史公开集图未应用温度，因此不能据此判断迁移结果。

## 6. 目标构造与模型限制

若训练目标依赖未暴露的随机变量，应对其条件边缘化；把隐藏变量的一次实现当作已知
条件分布会让分布误差解释失效。渲染充分性、生成器版本冻结和 oracle 检查有助于诊断，
但单独一个 oracle 或 ECE 不是证明任务理解的充分条件。软目标须可追溯，不任意编造。

默认独立候选评分下，固定 state/question 和温度时 `p_i/p_j=exp((s_i-s_j)/T)`，
新增其他候选不改变这两个候选的概率比，只改变归一化；这也限制候选间比较/集合依赖任务。
换 schema 或候选文本只是输入契约支持，不代表模型学会了新任务。

`risk_coverage_curve` 按完整置信度并列组采用 `confidence >= threshold`。
预测人工/弃权候选不算自动接受；coverage 分母是全部请求，risk 分母是接受请求，
正确率用 t[pred]。初始点为零覆盖、阈值 +inf、risk=NaN；JSON 中应写 null 而非 0。
在 val 选策略、冻结后仅在 test 报告，详见 [FIRST_TASK.md](FIRST_TASK.md)。

## 7. 复现可靠性图

以下为完整路线，前置本地模型、配套 tokenizer 和数据；快速教学路线见 README。

```bash
python trainer/calibrate_temperature.py \
    --ckpt out/decision/decision.pth --data dataset/synth --out out/calibration
python eval/eval_harness.py --ckpt out/decision/decision.pth \
    --data dataset/synth --sets test_known \
    --temperature out/calibration/T.json --out out/eval/decision
# 历史公开集路线报告原始概率；没有测试温度迁移。
python eval/eval_harness.py --ckpt out/decision/decision.pth \
    --data dataset/public --sets test_known --out out/eval/public
python eval/make_reliability_plot.py \
    --eval out/eval/decision/decision --sets test_known --binning equal_mass --out assets
python eval/make_reliability_plot.py \
    --eval out/eval/public/decision --sets test_known \
    --source public:chaosnli --binning equal_mass --out assets
```

图只读取 harness 的 `per_sample`，不重新调用模型。默认展示 soft_targets，
`--include_hard` 可加入硬标签；`--source` 是分析选择而非校准合法性门槛。
公开 ChaosNLI 是 `dataset/public/test_known.jsonl` 内的 `source=public:chaosnli`
（历史 474 条），不是单独的 split。图记录每桶 n；没有通用二项噪声带或扣除后的 ECE。
