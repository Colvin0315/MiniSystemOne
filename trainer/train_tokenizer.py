"""
MiniSystemOne tokenizer —— 全新 BPE，**不复用** MiniMind 的 tokenizer.json。

为什么不复用（详见 docs/DESIGN.md）：
  1. 无权重共享可能。MiniMind 的 tokenizer 存在是为了社区共享**权重**；我们的
     架构、head、词表用途都不同，复用零收益。
  2. MiniMind 的 36 个特殊 token 里有 21 个是视觉/音频/工具 token
     （<|image_pad|>、<tts_pad|>、<tool_call>…），决策模型永不产生，而且会
     诱导读者以为这个模型能对话。
  3. 它**缺少**我们需要的 <mask> / <sep> / <pad> / <trunc>，且它的 pad_token
     就是 <|endoftext|>，与 eos 语义冲突。

训练语料 = 预训练语料 ∪ 合成决策语料。这条不显然但重要：决定决策质量的是
**工具名、部门标签、等级标签、金额、日期、状态词**。若 tokenizer 没见过
`transfer_ownership` / `neutral` / `¥1,240.00`，它们会碎成很多 token，候选变长，
scorer 拿到的 pooled 向量噪声变大。成本为零，直接改善被测量的东西。

用法：
    python trainer/train_tokenizer.py                       # 默认 150k 预训练 + 20k 合成
    python trainer/train_tokenizer.py --n_docs 20000        # 快速冒烟
    python trainer/train_tokenizer.py --synthetic_only --n_synth 2000 --skip_eval --out_dir out/tutorial_tokenizer
"""
import argparse
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from tokenizers import decoders, models, pre_tokenizers, trainers, Tokenizer

from dataset.pretrain_corpus import iter_corpus
from dataset.synth import lexicon
from dataset.synth.lexicon import TOOLS, LEVELS

# 顺序即 id，**不要重排**：model/model_system_one.py 的 DecisionConfig 默认
# pad=0 / sep=3 / mask=4 依赖这个顺序，脚本末尾有断言校验。
SPECIALS = [
    "<pad>",      # 0  padding 与 "无候选" 槽位
    "<unk>",      # 1
    "<cls>",      # 2  预留，v0 不用（z 由 AttnPool 得到），留给消融
    "<sep>",      # 3  终止 span，并计入该 span 的 pooling
    "<mask>",     # 4  MLM 的损坏 token，决策训练中永不出现
    "<trunc>",    # 5  头尾切片时插在中间
    "<yes>",      # 6  Noul 的单 token 候选
    "<no>",       # 7
    "<abstain>",  # 8  弃权是一等候选，不是第二个 head
    "<ans>",      # 9  仅 AR baseline 的 loss 区间标记
    "</ans>",     # 10
]

VOCAB_SIZE = 6400
MAX_LEN = 8192

# 压缩率门槛（字/token）。参考值：**MiniMind 自己的 tokenizer** 在同一组样例上
# 得 zh 1.40 / en 3.39 / mix 2.41。
#
# en 的门槛**低于**参考值，这是有意的，不是放水：SAMPLE_TEXTS["en"] 是 LLM 生成的
# 说明文，正是参考 tokenizer 的主场。我们真正要服务的英文是 CLINC150 / banking77 的
# 短用户话语、GoEmotions 与 Amazon 评论（实测会话域 3.47 vs 参考 3.24），以及本地
# 英文预训练语料（维基 3.04 vs 2.79）。拿说明文上 5% 的打包损失换决策域覆盖，划得来。
# 原方案写的 en ≥ 3.60 是**不可能达标**的 —— 参考 tokenizer 自己只有 3.39，
# 说明那个数字当初是拍的而不是测的。
COMPRESSION_GATES = {"zh": 1.42, "en": 3.15, "mix": 2.50}

# **本 tokenizer 真正的那道门禁。** 上面三行量的是人造样例，这一行量的是
# `dataset/synth/*.py` 真实渲染出来的 state+question+candidates —— 也就是模型实际
# 会读到的东西。它才是序列长度、截断率、以及 255 候选能否单次前向的直接决定量。
#
# 实测：参考 tokenizer（用 lexicon 表面语料训的）**2.03**，当前 tokenizer 3.13。
# 门槛 2.60 坐在两者之间 —— 远低于已达成的值，且高于它要挡掉的那个。
REAL_TEXT_GATE = 2.60

# 决策域探针的**参照值**，不是门槛。见 `evaluate()` 里对为什么不能当门槛的说明。
# 参考 tokenizer 实测：工具名 max 4 / 等级标签 max 2 / 金额 max 7 / 日期 max 6。
PROBE_REFERENCE = {"工具名": 4, "等级标签": 2, "金额": 7, "日期": 6}

SAMPLE_TEXTS = {
    "zh": [
        "人工智能是计算机科学的一个分支，它企图了解智能的实质，并生产出一种新的能以人类智能相似的方式做出反应的智能机器，该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。人工智能从诞生以来，理论和技术日益成熟，应用领域也不断扩大，可以设想，未来人工智能带来的科技产品，将会是人类智慧的“容器”。",
        "星际航行是指在星系内甚至星系间的空间中进行的航行。由于宇宙空间极其广阔，传统的化学火箭动力在恒星间航行时显得力不从心。科学家们提出了多种方案，包括离子推进器、核热火箭、甚至是利用反物质作为能源的设想。此外，曲率驱动和虫洞旅行等科幻概念也在理论物理研究中被反复探讨。",
        "工作区 tenant 的账户状态为已暂停，账单部门在 2026年9月15日 提交了一笔 ¥1,240.00 的退款申请，订单编号为 8f3a91c2。限额条款规定单月累计退款不得超过 50,000 元，且该组织存在一条两天前产生的争议记录。",
    ],
    "en": [
        "Large language models (LLMs) are a type of artificial intelligence (AI) trained on vast amounts of text data to understand and generate human-like language. These models use deep learning techniques, specifically transformers, to process and predict the next word in a sequence. LLMs like GPT-4, Llama, and Claude have demonstrated remarkable capabilities in coding, translation, and creative writing. However, they also face challenges such as hallucinations, where the model generates factually incorrect information.",
        "The development of sustainable energy is crucial for the future of our planet. As climate change continues to impact global weather patterns, transitioning from fossil fuels to renewable sources like solar, wind, and hydroelectric power has become an urgent priority. Innovations in battery storage technology and smart grid management are essential to ensure a reliable energy supply.",
        "The workspace 8f3a91c2 has status suspended. The billing department filed a refund of USD 1,240.00 on 2026-09-15 against order 4c1de077, which exceeds the 12.5 percent monthly refund limit but remains under the absolute cap of $50,000.",
    ],
    "mix": [
        "Python 是一种高级编程语言，以其简洁的语法和强大的生态系统而闻名。It is widely used in data science, machine learning, and web development. 开发者可以利用 NumPy, Pandas, and PyTorch 等库快速构建复杂的应用。学习 Python 的过程非常愉快，因为它的代码读起来就像英语一样。Whether you are a beginner or an expert, Python offers something for everyone.",
        "应该执行 transfer_ownership 还是 transfer_funds？Workspace 状态为 active，但组织在 2026-09-15 存在一条 disputed 记录。建议先调用 get_audit_log 查看 recent activity，再决定是否 allow 或 abstain。",
    ],
}


def train(args):
    if args.tokenizer_path:
        # 省 10 分钟训练的退路。注意：README 里的所有数字都基于**新建** tokenizer。
        print(f"复用已有 tokenizer：{args.tokenizer_path}（所有 README 数字应基于 --tokenizer_path '' 的结果）")
        return args.tokenizer_path

    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=SPECIALS,
    )
    # Omit the progress length: requested counts are upper bounds for local corpora.
    tokenizer.train_from_iterator(iter_corpus(args), trainer=trainer)
    tokenizer.decoder = decoders.ByteLevel()

    os.makedirs(args.out_dir, exist_ok=True)
    tokenizer.save(os.path.join(args.out_dir, "tokenizer.json"))

    # added_tokens 里只有 SPECIALS 是 special，其余（BPE 学出的）必须显式置 False，
    # 否则会被当成特殊 token 而在解码时被跳过。
    tok_path = os.path.join(args.out_dir, "tokenizer.json")
    with open(tok_path, "r", encoding="utf-8") as f:
        tok_data = json.load(f)
    for info in tok_data.get("added_tokens", []):
        info["special"] = info["content"] in SPECIALS
    with open(tok_path, "w", encoding="utf-8") as f:
        json.dump(tok_data, f, ensure_ascii=False, indent=2)

    added_tokens_decoder = {}
    for token in SPECIALS:
        added_tokens_decoder[str(tokenizer.token_to_id(token))] = {
            "content": token, "lstrip": False, "normalized": False,
            "rstrip": False, "single_word": False, "special": True,
        }

    config = {
        "add_bos_token": False,
        "add_eos_token": False,
        "add_prefix_space": False,
        "added_tokens_decoder": added_tokens_decoder,
        "additional_special_tokens": [],
        "bos_token": None,
        "eos_token": None,
        "clean_up_tokenization_spaces": False,
        "legacy": True,
        "model_max_length": MAX_LEN,
        "pad_token": "<pad>",
        "unk_token": "<unk>",
        # 决策模型没有对话模板：规范序列化是代码（model/serialize.py），不是 Jinja 字符串。
        # 这个偏离 MiniMind 的地方在 docs/DESIGN.md 里明确记录。
        "tokenizer_class": "PreTrainedTokenizerFast",
    }
    with open(os.path.join(args.out_dir, "tokenizer_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)
    print(f"tokenizer 已写入 {args.out_dir}")
    return args.out_dir


# ---------------------------------------------------------------------------
def probe_groups():
    """决策域探针：四类 tokenizer 必须切得动的词。

    金额与日期用 lexicon 的格式化器现造，而不是手写几个特例 —— 特例会被
    针对性优化，格式化器造出的正是训练时真实出现的表面形式。
    """
    rng = random.Random(0)
    return [
        ("工具名", list(TOOLS[:12])),
        ("等级标签", [en for en, _ in LEVELS]),
        ("金额", [lexicon.fmt_amount(rng, rng.randrange(0, 500000), "zh") for _ in range(6)]),
        ("日期", [lexicon.fmt_date(rng, rng.randrange(0, 700), "zh") for _ in range(6)]),
    ]


def real_text_ratios(tok, n_per_gen=40):
    """在每个生成器的真实输出上量 (字符数, token 数)。

    与 `SAMPLE_TEXTS` 的区别是根本性的：那三组样例是**我手写的**，量的是"这个
    tokenizer 在我挑的句子上好不好"；这里量的是**模型实际会读到的东西**。
    参考 tokenizer 在样例上 2.55、在真实文本上 2.03 —— 差值就是它没见过的
    句子骨架、百分比、时刻与规则术语。

    只用 `train` split：评测集的文本形状与它同源，没有额外信息。
    """
    from dataset.synth import build_all

    out = []
    for g in build_all(seed=7):
        chars = ntoks = 0
        for rec in g.generate("train", n_per_gen):
            text = rec["state"] + "\n" + rec["question"]
            text += "".join(c["text"] for c in rec["candidates"])
            chars += len(text)
            ntoks += len(tok.encode(text, add_special_tokens=False))
        out.append((g.name, chars, ntoks))
    return out


def evaluate(tokenizer_dir):
    from transformers import AutoTokenizer
    from model.model_system_one import DecisionConfig

    tok = AutoTokenizer.from_pretrained(tokenizer_dir)
    cfg = DecisionConfig()
    ok = True

    print("-" * 72)
    print(f"词表长度：{len(tok)}（目标 {VOCAB_SIZE}）")
    ok &= len(tok) == VOCAB_SIZE

    # 特殊 token 的 id 必须与模型 config 默认值一致，否则 mask / sep 全错位
    checks = [("<pad>", cfg.pad_token_id), ("<sep>", cfg.sep_token_id), ("<mask>", cfg.mask_token_id)]
    for name, want in checks:
        got = tok.convert_tokens_to_ids(name)
        good = got == want
        ok &= good
        print(f"  {name:10s} id={got:5d}  期望 {want}  {'OK' if good else 'FAIL'}")

    # 全部特殊 token 往返一致
    bad = [s for s in SPECIALS if tok.decode([tok.convert_tokens_to_ids(s)]) != s]
    ok &= not bad
    print(f"特殊 token 往返：{'OK' if not bad else f'FAIL {bad}'}")

    # 往返解码一致（普通文本）
    for lang, texts in SAMPLE_TEXTS.items():
        for t in texts:
            if tok.decode(tok.encode(t)) != t:
                print(f"往返不一致 [{lang}]：{t[:40]}...")
                ok = False
    print(f"普通文本往返：{'OK' if ok else 'FAIL'}")

    # 压缩率门槛
    print("-" * 72)
    print("压缩率（字/token）：")
    for lang, texts in SAMPLE_TEXTS.items():
        ratios = [len(t) / len(tok.encode(t)) for t in texts]
        avg = sum(ratios) / len(ratios)
        gate = COMPRESSION_GATES[lang]
        good = avg >= gate
        ok &= good
        detail = " ".join(f"{r:.2f}" for r in ratios)
        print(f"  {lang:4s} 平均 {avg:.2f}  门槛 {gate:.2f}  [{detail}]  {'OK' if good else 'FAIL'}")

    # 真实生成器文本的压缩率 —— 本文件里唯一与任务分布同源的那道门禁。
    print("-" * 72)
    print("真实生成器文本（字/token）—— **这道才是门禁**：")
    total_c = total_t = 0
    for name, chars, ntoks in real_text_ratios(tok):
        total_c += chars
        total_t += ntoks
        print(f"  {name:20s} {chars/ntoks:5.2f}  ({chars} 字 / {ntoks} tok)")
    real_avg = total_c / total_t
    good = real_avg >= REAL_TEXT_GATE
    ok &= good
    print(f"  合计 {real_avg:.2f}  门槛 {REAL_TEXT_GATE:.2f}  "
          f"{'OK' if good else 'FAIL'}")

    # 决策域探针是**诊断**，不是门槛。为什么不能当门槛：它的阈值当初是从
    # `lexicon.corpus_doc`（生成器写好之前的表面语料）上标定的，而那份语料**每篇**
    # 都塞了 2–8 个工具名的候选列表 —— 工具名的密度是真实决策数据的好几倍。
    # 于是在它上面标出的"工具名 ≤4"是那个语料的产物，不是决策任务的性质；照它判，
    # 会把一个在真实文本上压缩率高 50% 的 tokenizer 判为不合格。词表预算固定
    # （6400）时，这份表与上面那道压缩率门禁**此消彼长**，所以它只能报数。
    print("-" * 72)
    print("决策域探针（诊断；括号内为参照 tokenizer 的值）：")
    for name, items in probe_groups():
        counts = [len(tok.encode(p, add_special_tokens=False)) for p in items]
        worst = max(counts)
        print(f"  {name:8s} max={worst}（参照 {PROBE_REFERENCE[name]}）  {counts}")

    print("-" * 72)
    print("全部通过" if ok else "存在未通过项")
    return ok


def validate_tokenizer(tokenizer_dir):
    """Always check the loading/ID contract, even when quality gates are skipped."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(tokenizer_dir, local_files_only=True)
    for expected, token in enumerate(SPECIALS):
        actual = tok.convert_tokens_to_ids(token)
        if actual != expected or tok.decode([actual]) != token:
            raise ValueError(f"Tokenizer contract failed for {token}: "
                             f"expected id {expected}, got {actual}.")
    if tok.pad_token_id != 0 or tok.unk_token_id != 1:
        raise ValueError("Tokenizer pad/unk IDs do not match the model contract.")
    print(f"tokenizer 本地加载与特殊 token 契约通过（词表 {len(tok)}）")


def main():
    p = argparse.ArgumentParser(description="MiniSystemOne tokenizer 训练与门槛校验")
    p.add_argument("--pretrain_path", default="dataset/pretrain_zh.jsonl",
                   help="中文预训练语料 jsonl（每行 {\"text\": ...}）")
    p.add_argument("--en_path", default="dataset/pretrain_en.jsonl",
                   help="英文预训练语料 jsonl；缺失必须显式允许回退")
    p.add_argument("--synthetic_only", action="store_true",
                   help="仅用双语 train 模板语料演示离线流程，不替代自然语言预训练")
    p.add_argument("--allow_missing_corpus", action="store_true",
                   help="显式允许跳过缺失自然语料；空或损坏的文件仍报错")
    p.add_argument("--en_share", type=float, default=0.5, help="英文应占的字符比例")
    p.add_argument("--out_dir", default="model", help="tokenizer.json 输出目录")
    p.add_argument("--n_docs", type=int, default=150000, help="中英混合预训练文档数")
    p.add_argument("--n_synth", type=int, default=20000, help="合成决策语料篇数")
    p.add_argument("--tokenizer_path", default="", help="非空则跳过训练，直接复用该目录的 tokenizer")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip_eval", action="store_true")
    args = p.parse_args()

    out = train(args)
    validate_tokenizer(out)
    if args.skip_eval:
        print("已显式跳过正式压缩率质量门禁；此 tokenizer 不代表正式训练质量。")
    else:
        sys.exit(0 if evaluate(out) else 1)


if __name__ == "__main__":
    main()
