# 设计文档

> **修复与历史结果：** 旧 checkpoint 在候选打乱后的数组位置上计算序数 CDF。当前损失按实际等级排序、忽略 padding，并用等级间距加权 CDF 平方差；`ordinal_mae` 是 `sum(abs(F_p-F_t)*grade_gap)`，无 0.5 系数。既有表格和图片是历史资产，需重新训练、拟合温度并评测。


本文记录**为什么这么设计**，以及**代码里刻意不做什么**。README 讲的是"这是什么、
跑出来多少"，本文讲的是"为什么不那样做"。凡是与 MiniMind 参考实现不同的地方，
这里都要有一条对应的理由 —— 否则读者会以为是疏漏。

---

## 0. 一句话

> state + typed questions → 带校准概率的类型化决策，**单次并行前向**，不生成文本。

三个 primitive：**Noul**（yes/no）、**Choice**（动态候选集，上限 255）、
**Score**（序数等级）。它们**不是三个 head**，见 §2。

---

## 1. 与 MiniMind 的全部差异（以及理由）

| # | 差异 | 理由 |
|---|---|---|
| 1 | 双向编码器（非 causal） | 决策不需要自回归生成。没有"下一个 token"这回事，整条序列一次编码完。 |
| 2 | RoPE 按 `position_ids` 索引，不是连续切片 | 打包格式需要**任意位置**：候选 span 要从前缀末尾重新起算，分块时 chunk 起点不同。MiniMind 的 `freqs[start:start+len]` 切片假设位置连续，做不到。见 §4。 |
| 3 | float additive mask，不是 bool | 实测更快更省显存，且不会掉进 MATH 后端。**注意这组数测于 torch 2.5.1+cu121**，换版本需重测（`python scripts/smoke_test.py --bench_mask`）。 |
| 4 | 不用 `enable_gqa=True` | 融合内核要求 q/k/v head 数相同，只有 CUDNN 接受 `enable_gqa`。沿用显式 `repeat_kv` expand（`n_rep==1` 时它直接返回原张量，无开销）。 |
| 5 | `rope_theta = 1e4`（MiniMind 用 1e6） | MiniMind 面向 32k 上下文；我们的预算是 ≤8192，1e4 在短程上的分辨率明显更好，而决策所需的全部关系都是局部的。 |
| 6 | 不用 `[CLS]`；`<cls>` 进词表但 v0 不使用 | `z` 由 state ∪ question span 上的 attention pool 得到。池化比一个必须"学会代表整句"的 `[CLS]` 位置更可控，且天然带 mask（见 §3 的 `AttnPool`）。`<cls>` 保留是留给消融。 |
| 7 | **不复用 MiniMind 的 `tokenizer.json`** | 见 §7。 |
| 8 | 单 head，无 sigmoid/BCE 的"Noul head" | 见 §2。 |
| 9 | 没有 `chat_template` | 决策模型没有对话模板，规范序列化是**代码**（`model/serialize.py`）而非 Jinja 字符串。 |
| 10 | 新增 7 个特殊 token | `<sep> <mask> <trunc> <yes> <no> <abstain> <ans> </ans>`。前四个是打包/预训练的结构刚需；`<yes>/<no>/<abstain>` 见 §2；`<ans>/</ans>` 目前无使用者（原为 AR baseline 的 loss mask 保留）。 |

`.gitignore` 里 `model_tok_legacy/`、`model_tok_v*/` 是调 tokenizer 时的中间版本，
最终词表只有 `model/`。

---

## 2. 只有一个 head：Noul / Choice / Score 是同一个东西

**Noul = 候选集固定为 `{yes, no}` 的 Choice。Score = 候选集是序数等级的 Choice。**
只有一份 forward、一份 loss。三者的差别仅在 schema 对象和 `is_ord` 标志。

因此**明确不实现** sigmoid + BCE 的"Noul head"。那会是第二个 head，
破坏"单 head 统一"这个核心主张，而且它买不到任何东西 —— K=2 时 softmax + CE
就是 sigmoid + BCE。

### 由此派生的三条强制约定

1. **Noul 的正向标签由 `schema.positive_label` 指定，绝不按下标。**
   候选会被打乱（这是反泄漏的必要措施，见 §8），按下标取会把 `p_yes` 的语义翻转。
   这条如果写错，模型仍然能训、准确率仍然合理，只有校准图会悄悄错 —— 属于最难发现的一类 bug。

2. **Score 的等级序号同时渲染进候选文本**（`"[3] neutral"`），让编码器**能看见**序关系。
   光靠 `level_idx` 张量是看不见的 —— 它不进前向。

3. **弃权是一个候选**（G4 的 `abstain`，CLINC150 的 `oos`），不是一个分支。
   零成本、零新 head，且概念正确：弃权**就是**在选项里做决策。
   `eval_metrics.risk_coverage_curve` 提供了按置信度弃权的曲线函数，
   **但本仓库没有脚本产出它** —— 函数在，调用者不在。

`<yes>/<no>/<abstain>` 作为**单 token** 候选（而不是让 "yes" 走正常 BPE）是刻意的：
pooled 向量最干净，是仅有的三个"语义"特殊 token，且理由正当 ——
Noul 是一等 primitive，给它的答案原子 token 是真实的建模选择。

---

## 3. 注意力契约 —— 整个设计精确性的来源

两个开关：

| 开关 | 默认 | 作用 |
|---|---|---|
| `prefix_blocked` | `True` | 前缀（state / question）**不可** attend 到候选。反方向不阻塞：候选可自由 attend 全部前缀。 |
| `candidate_crosstalk` | `False` | 候选之间不可互相 attend。 |

许可表（`allowed[query][key]`）：

| query ↓ \ key → | STATE | QUESTION | CANDIDATE |
|---|---|---|---|
| STATE | full | ✗ | ✗ |
| QUESTION | full | full | ✗ |
| CANDIDATE | full | full | `crosstalk` ? full : **仅自身 span** |

外加：任何 token 都不 attend 到 PAD。

### `crosstalk=False` 的语义

> **每个候选的 logit 独立计算，只有 softmax 归一化把它们耦合起来。**

这正是"加入一个无关候选不应改变其他候选得分"的**正确归纳偏置**。所以它是默认值，
开着它是消融实验（步骤 15）。改一个开关就能把这条主张变成实测数字，也是 §5、§6
两个不变量的前提。

### 三个实现要点，每一个都对应一次踩坑

1. **mask 用 float additive，不是 bool。** 见 §1 第 3 条。
2. **绝不能出现整行全被 mask 的 query 行** —— softmax 会产生 NaN，NaN 经 LayerNorm
   污染**全序列**（不只是那一行）。所以：
   - PAD query 行**放行全部真实 key**。输出会被丢弃，且 attention 是按 query 行独立的，
     不会污染真实 token。
   - `NEG_INF = -1e9` 而不是 `-inf`。即使某行意外全被 mask，softmax 也退化为均匀分布
     而不是 NaN。这是一道**故意留的缓冲**：它的代价是让"全 mask"这个 bug 变成静默的
     数值退化而非崩溃。因此不能靠 NaN 来发现这类 bug，只能靠 `scripts/smoke_test.py`。
3. **`build_attn_mask` 在没有结构时返回 `None`**（纯文本或全局放行）。调用方必须接受
   `None` 并走无 mask 快路径 —— 传一个全 0 的 mask 会强行把 attention 从无 mask
   快路径踢出去，白白变慢。

---

## 4. RoPE 位置规则（`build_position_ids`）

> 前缀（state ∪ question，含各自的尾随 `<sep>`）用**自然位置** `0..P-1`；
> **每个候选 span（含其尾随 `<sep>`）的位置从 `P` 重新开始。**

也就是：每个候选都按"前缀之后只跟着它自己"来编码。

这不是省事的写法，而是两个不变量成立的**前提**：

- **候选顺序不变性**：候选 k 的位置只依赖它自己的长度，与它排第几无关。
  若用自然位置，候选被置换后到前缀的相对偏移会变，RoPE 分数随之改变，
  不变性就只是近似的。
- **分块不变性**：候选 k 无论落在哪个 chunk 里位置都相同，所以一次前向与
  分块前向给出**逐元素相同**的 logits。

用绝对位置编码这两条**都不成立**。这是选 RoPE 的第二个独立论据（第一个是
MiniMind 的实现可复用）。

`<sep>` **计入它终止的那个 span 的 pooling**，并继承该 span 的 seg_id
（BERT 的 `[SEP]`-as-summary 惯例）。所以 `state` 的池化范围是
`state tokens + 尾随 <sep>`。

---

## 5. 决策头

```python
class AttnPool(nn.Module):        # z 的来源：state ∪ question 上的 attention pool
    def forward(self, H, mask):
        a = (self.k(H) @ self.q) / sqrt(h)
        a = a.masked_fill(~mask, NEG_INF).softmax(-1)
        z = einsum('bs,bsh->bh', a, H)
        mean = (H * mask[...,None]).sum(1) / mask.sum(1,keepdim=True).clamp(min=1)
        return self.ln(z + mean)   # 残差到 mean
```

`z` 是一个查询向量，落在 state ∪ question 的 **token 集合**上（不是候选上）。

**`+ mean` 这一步的残差不是为了梯度好走，是为了防止 pool 塌缩到单个 token。**
池化权重 `a` 是全序列 softmax，在训练早期很容易锁死在某个高范数 token 上；
一旦塌缩，`z` 就退化成"某一个 token 的表示"，问题条件信息全丢，校准直接崩。
残差到 masked mean 保证 `z` 永远至少含有"整段前缀的平均"这个成分。

**`in_ln` + `mid_ln` 用于稳定表示尺度。** 它们不保证校准，也不保证 logit 尺度不随 K 和
primitive 漂移。没有它们，单一温度不够用 —— 大 K 样本的 logit 尺度天然更大，
温度校准会在 K 之间来回拉扯。

```python
class DecisionHead(nn.Module):
    def forward(self, H, prefix_mask, cand_vecs, cand_mask):
        z = self.pool(H, prefix_mask)
        zi = z[:,None,:].expand_as(cand_vecs)
        f = torch.cat([zi, cand_vecs, zi*cand_vecs, (zi-cand_vecs).abs()], -1)  # 4h
        logits = self.fc2(self.mid_ln(self.act(self.fc1(self.in_ln(f))))).squeeze(-1)
```

四个拼接项各有职责：

| 项 | 职责 |
|---|---|
| `z` | 单独作为线性项会被 softmax 吸收（加常数不影响分布）而无用；但进**非线性** MLP 后让每个候选的得分以问题为条件 —— 这是 per-question 校准的来源。 |
| `z ⊙ c_i` | 双线性交互，主要判别信号。 |
| `abs(z − c_i)` | 序数信号（Score 必需）与 off-schema 检测。 |
| `c_i` | 让候选自身的先验可以被学（如某类标签天然少见）。 |

候选侧用 **masked mean-pool（0 参数）**，不用 attention pool：候选很短且同质，
在这个尺度上再加 2h² 参数只会过拟合。

**不另做 cross-attention 模块。** 编码器是双向的，question token 已经在同一次前向里
attend 过 state，`H[question_positions]` 里已经含有"问题条件下的 state 读法"。
额外的 cross-attention 是冗余的 2h² 参数。

参数核对用 `scripts/model_stats.py`，它会显式对照 `AttnPool` 的实测参数与解析式
`2h²+3h`。

---

## 6. Loss

```
L = CE(p*, softmax(logits)) + λ_b · Brier(p*, p) + is_ord · λ_o · CDF-MSE(p*, p)
```

- **λ_b = 0.5，作用于全部样本。** 这里的 `Brier(p*,p)` 是平方距离 `sum((p-p*)^2)`。对软目标它是超额期望 Brier，标准采样标签的期望 Brier 还需加 `1-sum(p*^2)`。CE 与平方距离在理想条件下具有相同最优分布，但梯度并不只是逐元素缩放；平方距离的 logit 梯度为 `2p_j[(p_j-t_j)-sum_k p_k(p_k-t_k)]`。硬标签上的 Brier 不是 label smoothing，也不自动封顶置信度。默认不按 K 归一化；`--brier_normalize` 用于对照实验。

- **λ_o = 0.5，仅 Score 样本。** 先按真实数值等级升序排列有效候选与目标，再比较相邻等级前的 CDF。以等级间距加权平方差，并按有效等级跨度归一化；padding 不贡献间距或误差。由逐样本 `is_ord` 门控，允许与非 Score 样本同 batch。评测 `ordinal_mae` 则用绝对 CDF 差乘等级间距求和，即一维 Wasserstein-1 距离，不使用 0.5 系数。

- **计算一律在 fp32**。bf16 下 Brier 与 CDF-MSE 的差值会被显著截断。

---

## 7. Tokenizer：为什么新建而不是复用 MiniMind 的

**新建 BPE，vocab 6400，11 个特殊 token。不复用 `../minimind/model/tokenizer.json`。**

复用毫无收益：MiniMind 的 tokenizer 是为了社区共享**权重**，而我们的架构、head、
词表用途都不同，无权重共享可能。而且它有两个具体问题：

- 它的 36 个特殊 token 里有 21 个是视觉/音频/工具 token（`<|image_pad|>`、
  `<tts_pad|>`、`<tool_call>`…），决策模型永不产生；留着会**诱导读者以为这模型能对话**。
- 它**缺少**我们需要的（`<mask>` / `<sep>` / `<pad>` / `<trunc>`），
  且其 `pad_token` 就是 `<|endoftext|>`，与 eos 语义冲突。

### 词表

```python
SPECIALS = ["<pad>","<unk>","<cls>","<sep>","<mask>","<trunc>",
            "<yes>","<no>","<abstain>","<ans>","</ans>"]
```

`tokenizer_config.json` 写 `model_max_length=8192`、`pad_token="<pad>"`、
`bos/eos_token=None`、**无 `chat_template`** —— 理由见 §1 第 9 条。

### 训练语料 = 预训练语料 ∪ 合成决策语料

这条不显然但重要：决定决策质量的是**工具名、部门标签、等级标签、金额、日期、状态词**。
若 tokenizer 没见过 `transfer_ownership` / `Neutral` / `¥1,240.00`，它们会碎成很多 token，
候选变长，scorer 拿到的 pooled 向量噪声变大。成本为零，直接改善被测量的东西。

### 验证门槛（`trainer/train_tokenizer.py`）

- 往返解码一致；
- 压缩率 **中文 ≥1.40 字/token、英文 ≥3.60 字/token、混排 ≥2.20**；
- 所有特殊 token 往返一致；
- `len(tokenizer) == 6400`。

压缩率表打进 README。`--tokenizer_path ../minimind/model` 保留为省 10 分钟训练的退路，
但**所有 README 数字均基于新建 tokenizer**，这一点必须在 README 里注明。

---

## 8. 打包格式与批处理

```
pos:  [ state tokens ][<sep>][ question tokens ][<sep>][ c1 ][<sep>][ c2 ][<sep>] ...
seg:   2 2 2 2 2 2 2   2       3 3 3 3 3 3      3      4 4    4    4 4    4
```

`cand_id` 在 collate 里**一次向量化 scatter 生成**，使 block-diagonal mask
只需一次张量比较（`cand_id[:,:,None] == cand_id[:,None,:]`）而不是逐样本循环。

`collate_decision` 返回：`input_ids (B,S)` / `seg_id` / `cand_id` / `cand_span (B,K,2)` /
`cand_mask (B,K)` / `level_idx (B,K)` / `target (B,K)` / `is_ord (B,)` / `primitive` / `meta`。
`S` 向上取整到 8 的倍数。

### K 分桶采样，不用固定 K padding

若生成器 K 在 `{2..32}` 上大致均匀，固定 `K_max=32` 会浪费约 50% 的候选槽位。
`CandidateBucketSampler` 按 `(K, total_len//64)` 分组，保证每 batch K 一致、
长度差 <64，padding 浪费 <8%。推理时另用固定 K + mask（那里 K 确实任意，
且必须服务单个样本）。

**训练上限：K ≤ 32, S ≤ 1024。** 这是实测的甜点（4.35 GB / 55k tok/s），
距显存悬崖还有 3 GB。见 §9。

### 候选子采样

`candidates` 存的是**超集**。`__getitem__` 时按 `K ~ Uniform{2..32}` 子采样，
但 **`p*_k ≥ 0.05` 的候选恒被保留**；其余无放回采样，目标重新归一化并记
`renormalized: true`。一举两得：免费的 K 增广 + 变 K 评测所需的超集。

---

## 9. 显存预算：为什么是 6.5 GB 而不是 8 GB

**实测：32×1024 的配置峰值 8.3 GB，不 OOM，但慢 10 倍**（297 ms → 5936 ms/step）。
Windows WDDM 在显存见底时**静默换页到共享内存**，代价是速度而非报错。

所以：

- 硬性预算 **训练峰值 < 6.5 GB**，比"塞进 8GB"严格得多。这是本方案用保守 batch +
  梯度累积而非极限打包的原因。
- `peak_vram_warn()` 在超预算时**只告警不中止** —— 但必须响，因为**不会 OOM**。
  没有这条告警，"运行莫名变慢 10 倍"就没有解释来源。
- `oom_retry` 对半砍 batch 重试。不是防御性编程：候选数 K 与 state 长度都是变的，
  峰值显存在样本间波动；没有它，一次超标的 batch 毁掉一整晚训练。
- `--use_checkpoint` 逐层梯度检查点，激活降约 2.5×，速度降约 30%。

**不要照抄 MiniMind 的默认值**（batch 32 / accum 8）。在本机上那会得到一个
莫名很慢的运行，而不是一个报错。

---

## 10. 数据完整性的三道门禁

三道门禁都是**硬门禁**（不通过就不该报告任何模型结果）。它们存在的理由都是
"数字看起来合理但其实是假象"这一类失败 —— 这类失败比崩溃危险得多。

### 门禁 1：`audit_synthetic.py` —— 反模板化

在训练集上训一个 **bag-of-words 逻辑回归**，在 `test_known` 上评测。
**门槛：unigram-BoW 准确率必须比模型低至少 25 个百分点。**

反模板化措施（强制）：每问 ≥20 个复述模板、≥5 个不相交实体池、state 段落顺序随机、
数字/日期/币种格式随机、中英混排约 35/35/30、表面干扰项（无关账户、作废条款、
过期历史）、所有实体 ID 用 `sha1(seed||counter)[:8]`。

### 门禁 2：`audit_leakage.py` —— 位置探针

用**纯位置探针**（特征只有 `(position_onehot, K)`）证明重排有效。
它在每个 split 上必须**只达随机水平**。

这里的坑：并列目标（`tie_set`）上，硬 argmax 会把位置 0 记成正确，于是并列越多的
来源数字越好看。所以位置探针必须用**软口径**评分，否则它会在并列样本上
"通过"而实际什么都没证明。

### 门禁 3：按 `template_id × entity_pool` 划分 split

**绝不随机逐条划分。** 这是 `test_known` 成为真正留出集的前提，
由 `build_dataset.py` 里的断言强制。

### 候选信息泄漏的三条通道（R3）

| 通道 | 检查 |
|---|---|
| 位置相关（gold 恒排第一） | 门禁 2 的位置探针 |
| state 泄漏（state 文本直接给出答案） | 每生成器做答案泄漏 audit：在 `state + question`（**不含候选**）上跑 BoW 探针，要求其准确率显著低于完整模型 |
| crosstalk 泄漏 | 候选置换不变性定量测量（`scripts/smoke_test.py` 的 [B]） |

---

## 11. 校准的学习目标与可追溯数据

交叉熵与 Brier 是适当评分规则，one-hot 观测也能在期望上学习真实条件概率。已知软分布有助于受控实验和减少抽样噪声，但不是校准的必要条件，也不保证训练后的模型校准。详见 [校准方法](CALIBRATION.md)。

本项目区分 `explicit_rng`、`marginalized`、`tie_set`、`human_annotators` 与 `hard`。历史 `metrics.calibration` 子集按约定排除 `hard`；硬标签依然可用于校准评测。按 provenance 和来源分层可揭示子群问题，指定混合分布的汇总 ECE 也有意义，但不能替代分层结果。

### 软目标的合法构造（三者共同点：目标由生成器自身逻辑、从它明确知道"是否渲染进了
state"的量算出，`audit` 里留下记录）

随手写死的软数字**禁止合入** —— 那种目标要么不可学（模型只能学到均值），
要么让 ECE 度量生成器 bug 而非模型。

- **(a) `explicit_rng`** — 环境里有一个 state 未暴露的随机化。例：银行以
  `q = σ((B − A − fees)/τ)` 批准交易，`B/A/fees/τ` 全部印在 state 里，
  但银行的 RNG 没有。则 `P(approve|x) = q` **精确成立**。模型在训练集里看到同一
  `(B,A)` 两次不同结果，必须回归到频率。这是最干净的校准设定，且诚实 ——
  不确定性在世界里，不在 state 里。
- **(b) `marginalized`** — 生成器维护隐藏变量 `h`（风险分、用户严格度、残差连续信号），
  只在 state 里**粗粒度**暴露（5 档标签、1 位小数读数）。目标为
  `Σ_h P(y|h,x_visible)·P(h|exposed_label)`。这产生教学上最理想的行为：
  **同一问题在 state 具体时分布尖锐、在 state 粗糙时分布宽泛。**
  这个对比是"模型在做推断而非模式匹配"的最好证据。
- **(c) `tie_set`** — 程序检测出有效答案集，目标为均匀分布。

R1 风险（P* 可能依赖未渲染进 state 的信息）的缓解见 §12。

---

## 12. R1 缓解：怎么证明"目标可学"

若 `P*` 依赖未渲染进 state 的信息，模型**无法**学到，而报告的 ECE 度量的是
生成器 bug 而非模型 —— 这会在产生一个看似合理的坏数字的同时让旗舰实验失效。

三条缓解，都便宜，全做：

1. **oracle 上界** —— 用生成器**内部**特征（`audit.margin`、`q_raw`、隐藏 `h`）
   训一个小 MLP 并报告其 ECE。它应该 ≈0。这把"模型未校准"与"目标不可学"
   分离开：oracle 好 + 模型差 = 模型的锅；oracle 也差 = 目标本身有问题。
2. **特征充分性测试**（在 `audit_synthetic.py` 里）—— 对每个软样本，验证从
   **渲染后的 state 文本中抽取**的数值字段做逻辑回归能否在容差内恢复 `P*`。
   失败者即为渲染 bug，**丢弃并报告丢弃率**。
3. **`gen_version` 冻结** —— 测试集只用冻结版本重新生成，版本号记入每个结果 JSON。

---

## 13. 外部对比实验

**自训一个 AR baseline 的方案已放弃。** 原计划是同 backbone 训一个 AR 变体做四路
对比（自由生成 / 受限解码 / scored / verbalized confidence），代码写过也冒烟过，
但已连同 `MiniSystemOneForCausalLM` 一起删除 —— 理由：**它比不过外部对比。**
一个自己训 20 分钟的 26M AR 变体，证明不了"决策原生比自回归好"，只证明"我们训的
这个 AR 变体弱"。真正有说服力的对照是别人已经训好的模型。

对比对象改为**外部 API**：

| 对比对象 | 是什么 | 怎么比 |
|---|---|---|
| **TypeSafe `jev-latest`** | 同一个概念的生产实现 | 同一份 state + question + 候选集，走它们的 choice/noul/score primitive |
| **Qwen API** | 通用 LLM 的三种用法 | ① 自由生成标签 ② 受限解码（候选 trie）③ **verbalized confidence**（输出 `{"choice":..., "confidence": 0.62}`） |

**口径纪律（这部分继承自被删掉的四路对比，仍然成立）：**

1. **同一套指标代码。** 所有对象的 ECE / Brier / NLL 都由 `eval/eval_metrics.py`
   算，不另外实现。用同一指标检验各系统的概率输出，
   避免预设谁的校准更好 —— 而两份实现各算一次必然漂移（方案 R6）。
2. **报告 schema 错误率。** 自由生成的 LLM 会输出候选集以外的字符串，这是它的
   固有失败模式；我们的单次前向构造上为 0。这一列不报，对比就是不完整的。
3. **延迟与成本一起报，并写清口径。** `前向次数` 单独看会骗人：一次 LLM 解码调用的
   串行前向次数与我们的 1 次前向不是一回事，token 数也不同。所有计时包
   `torch.cuda.synchronize()`，外部 API 用服务端报告的延迟并注明网络开销包含与否。
4. **不落任何外部模型的逐样本输出到仓库。** 只报聚合指标。这不只是卫生问题 ——
   外部模型的输出**绝不进训练路径**，本项目从 0 训练，与它们没有任何数据依赖。

> ⚠️ **法律边界。** 用 TypeSafe API 做对比评测属于"benchmark 使用"，与
> **蒸馏/逆向工程**是两回事，后者是本项目从立项起就排除的。但 MCA 里是否有关于
> 竞争性使用或 benchmark 的独立条款，需要账号持有人自行确认。代码侧的设计约束是
> 硬的：对比脚本必须显式 opt-in、需要 API key、**不在任何训练/数据构建路径上**。

---

## 14. 分块推理与前缀复用

因 `prefix_blocked=True` 且 `crosstalk=False`，打包序列**精确可划分**：
前缀隐状态与候选无关，每个候选只依赖前缀和自身 span。于是：

- 前缀 K/V 可缓存复用（沿用 MiniMind 的 `past_key_value` 拼接机制，无需新抽象）；
- **RoPE 使分块位置精确** —— 相对位置下偏移量一致。绝对位置会彻底失效，
  这是选 RoPE 的第二个独立论据（见 §4）。

**26M 实测：`S_pref=700` 时，255 个典型长度候选（21 tok）总计 6055 tok，
单次前向即可容纳。** 分块路径因此是安全网而非主路径。

`encode_state` / `decide_with_state` 提供跨问题的 state 复用：state 在注意力上被
禁止看到 question 与候选，所以它的隐状态与"问什么、有哪些候选"**无关**。
N=16 个问题共享一份 state 时，成本 ≈ `L_state + N·(L_q + L_cand)` 而非
`N·(L_state + L_q + L_cand)`。

**这是全项目最强的效率主张，因为它是架构属性而非硬件属性** ——
换一个 `prefix_blocked=False` 的模型，这个数就退化到 1×。
`eval/eval_efficiency.py` 的 `suite_amortization` 负责把它测出来，
且**加速比不写死**（方案里猜的是 N=16 时 ~3.3×）：它取决于 state 在总 token 里的占比，
而那随数据而变，所以同时报出 token 构成让读者自己核。

---

## 15. 长 state 截断

按序三种（`serialize.truncate_head_tail` 是第 ② 种）：

1. **段落感知丢弃** —— `state_sections` 带 `priority`，溢出时先丢低优先级段再重渲染。
   廉价，且是个教学点：结构化 state 可以语义截断而非盲目切。
2. **头尾切片** —— 保留 0.3 预算的头 + 0.7 预算的尾，中间一个 `<trunc>`。
   近期偏置对 agent trace 是正确先验，因为当前处境在末尾。
3. 硬左截断 + 前置 `<trunc>` 标记。

原计划由一套"截断敏感性"评测测出实际曲线；**本项目尚未跑，所以它仍是猜测，
不是报告结果。**

---

## 16. 代码风格约定（沿用 MiniMind）

- 无 `pyproject.toml`、无包安装、脚本直接跑 + `sys.path.append`。
- argparse-only CLI，中文 help 文本。
- `AdamW` + `torch.autocast` + `GradScaler`。
- `========== N. 阶段名 ==========` 分节横幅。
- swanlab 可选；**非交互式（重定向 / 后台 / CI）时强制关闭**，
  因为它首次运行会在 stdin 上弹三选一问卷并**阻塞等待** ——
  后台跑起来像卡死，实际是在等一个永远不会到来的按键。
- `Logger.log` 用 `print(..., flush=True)`：重定向到文件时 stdout 变成块缓冲，
  日志会落后真实进度上千步，看着像训练卡死。
- Windows 上 `datasets`(pyarrow) **必须先于 torch** import，否则进程静默消失
  （退出码 139，零输出）。`train_mlm.py` 保留 `import datasets  # noqa: F401` 就是这个原因。

---

## 17. 明确不做（v0 边界）

- **不做 RL。** RLCD 是原方案训练方法的一部分，但本项目 v0 的论点是"单次前向 +
  直接输出概率并评估校准"，这个论点在纯监督下就能检验。RL 留作后续。
- **不做第二个 head**（sigmoid Noul head）。见 §2。
- **不做通用助手。** 它是 **schema-bound** 决策模型：没有任务 schema 时，
  它不处理任意真实文本。README 必须说清楚，否则读者会拿 ChatGPT 的预期来用它。
- **不宣称击败 Jev。** 延迟口径不同（见 README 的诚实边界一节）。
- **不使用 TypeSafe 的 API、不蒸馏、不逆向。** 数据全部来自程序生成 + 公开数据集。
