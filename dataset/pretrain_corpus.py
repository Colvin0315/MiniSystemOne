"""
预训练语料的事实来源 —— tokenizer 训练与 MLM 预训练共用同一份混合逻辑。

两个来源：
  1. MiniMind 的 pretrain jsonl（中文为主，**实测 77% 汉字、仅约 6% 拉丁字母**）。
  2. 公开英文语料（用 `fetch` 子命令一次性抓到本地）。

为什么必须补英文：MiniMind 的语料英文覆盖几乎为零。只在这上面训 tokenizer，
词表里英文词片只有 457 个（MiniMind 自己的 tokenizer 是 1051），MLM 阶段没有
英文则编码器的英文表示会很弱 —— 而 ChaosNLI / CLINC150 / banking77 / GoEmotions
/ Amazon 这些**英文公开集正是项目最诚实的头条数字**。

英文为什么是 alpaca + wikitext 两份而不是一份（实测，见 train_tokenizer.py 的门槛注释）：

  - **alpaca（指令式英文）供域**。它是会话/指令域，与 CLINC150、banking77 的短
    用户话语、GoEmotions 与 Amazon 评论同域。实测短会话话语压缩率：alpaca 参与
    的混合 3.47 > 纯 alpaca 3.39 > MiniMind 参考 3.24。维基类散文是对照组里最差
    的域 —— 而模型**永远不会读维基**。
  - **wikitext 供量**。alpaca 只有 15.8M 字符 / 4.78M token，单靠它英文在 MLM 里
    占比不到 5%，撑不起 26M 编码器的英文表示。wikitext 提供 44M 字符把英文总量
    抬到 ~60M 字符 / ~16.6M token。
  - 代价是维基散文之外的一点点说明文打包损失（MiniMind 自己的样例文本 3.21 vs
    参考 3.39），换来的是决策域覆盖翻倍：工具名 9→4 token、等级标签 4→2。

训练过程不联网：英文语料在 `fetch`/`blend` 时一次性落成 jsonl，之后只读本地文件。

用法（三步重建 dataset/pretrain_en.jsonl，总计约 20 分钟）：
    python dataset/pretrain_corpus.py fetch --out dataset/pretrain_en_alpaca.jsonl \\
        --dataset tatsu-lab/alpaca --fields instruction,input,output
    python dataset/pretrain_corpus.py fetch --out dataset/pretrain_en_wiki.jsonl \\
        --dataset Salesforce/wikitext --config wikitext-103-raw-v1 --fields text \\
        --strip_wiki_title --min_chars 600 --n_docs 60000
    python dataset/pretrain_corpus.py blend --out dataset/pretrain_en.jsonl
"""
import argparse
import json
import os
import random
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

MIN_CHARS = 200          # 过短的文档（wikitext 的残段/列表）对语言建模没有价值

# 合成语料里各生成器的采样权重（理由见 `iter_synth_docs`）。工具名 60 个全在
# `tool_router` 里，且它就是准确率主力，所以给到 3 倍份额。
GEN_VOCAB_WEIGHTS = {"tool_router": 3.0, "agent_trace_score": 1.5}


def _read_jsonl(path, text_key):
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            text = obj.get(text_key)
            if text:
                yield text


def iter_mixed(zh_path=None, en_path=None, max_docs=200000, en_share=0.5,
               zh_key="text", en_key="text"):
    """按**字符预算**混合中英语料，而非按文档数 —— 两种语言的字节密度差 3 倍。

    en_share 是英文应占的字符比例。每次取当前占比低于目标的那一路，
    所以是自校正的：任一路提前耗尽也不会让比例崩掉。
    """
    zh = _read_jsonl(zh_path, zh_key) if zh_path and os.path.exists(zh_path) else iter(())
    en = _read_jsonl(en_path, en_key) if en_path and os.path.exists(en_path) else iter(())
    zh_done = en_done = False
    zh_chars = en_chars = 0
    n = 0

    while n < max_docs and not (zh_done and en_done):
        total = zh_chars + en_chars
        take_en = (en_chars / total < en_share) if total else (en_share > 0)
        if en_done:
            take_en = False
        elif zh_done:
            take_en = True

        try:
            text = next(en) if take_en else next(zh)
        except StopIteration:
            if take_en:
                en_done = True
            else:
                zh_done = True
            continue

        if take_en:
            en_chars += len(text)
        else:
            zh_chars += len(text)
        n += 1
        yield text

    total = max(zh_chars + en_chars, 1)
    print(f"语料混合：{n} 篇，中文 {zh_chars/1e6:.1f}M 字 ({100*zh_chars/total:.0f}%) "
          f"+ 英文 {en_chars/1e6:.1f}M 字符 ({100*en_chars/total:.0f}%)")


def render_doc(rec):
    """把一条决策样本摊平成一篇文本文档，供 tokenizer / MLM 使用。

    带上 `question_paraphrases`：词表若不覆盖它们，任何按复述做的扰动评测测到的
    都会是分词而不是语义。（本仓库目前**没有**这样的评测；保留该字段是为了让
    词表不必为此重训。）
    """
    parts = [rec["state"], rec["question"]]
    parts += list(rec.get("question_paraphrases") or ())
    parts += [c["text"] for c in rec["candidates"]]
    return "\n".join(p for p in parts if p)


def iter_synth_docs(n_docs, seed=0):
    """从**真实生成器**采样决策文档。

    `lexicon.iter_corpus_docs` 是生成器写好之前的表面替身：它覆盖了工具名、等级
    标签、金额、日期这些**词类型**，但覆盖不到生成器真正的句子骨架 —— 中文动作词
    与对象词（`转移所有权`）、百分比（`62.5%`）、时刻与日名（`08:00`/`周一`）、
    规则句里的术语（`检出率`/`先验胜算`/`排期规则`）。承重词恰好全部落在这几类里，
    而它们正是候选文本本身 —— 碎成 4–8 个 token 就直接变成 scorer 的噪声。

    **按词表密度而非按文档数均分。** 词表的预算是固定的（6400），一个词能否拿到
    合并取决于它在语料里的出现次数。工具名（60 个）全部集中在 `tool_router` 一个
    生成器里，均分就等于只给它 1/6 的合成预算 —— 实测那样工具名会碎到 4–8 个
    token，而等级标签是 5 个词散在四个生成器里、反而不到 4 个。多给承载大词表的
    生成器一点份额，是**为类型覆盖服务**的采样，不是为好看的指标服务的：真实
    state 的压缩率（下面 `tok_probe.py` 量的那个）是唯一会被它损害的指标，所以
    加权重之后必须复测它。
    """
    from dataset.synth import build_all

    gens = build_all(seed=seed)
    rng = random.Random(seed)
    weights = [GEN_VOCAB_WEIGHTS.get(g.name, 1.0) for g in gens]
    total_w = sum(weights)
    pool = []
    for g, w in zip(gens, weights):
        pool.extend(g.generate("train", max(1, round(n_docs * w / total_w))))
    rng.shuffle(pool)
    for rec in pool[:n_docs]:
        yield render_doc(rec)


def iter_corpus(args):
    """中英混合预训练语料为主，合成决策语料按目标比例均匀插入。

    合成语料的作用是**补词汇**（工具名/标签/金额/日期/时刻/术语），不是主导分布 ——
    它模板重复度极高，占比过高会挤掉自然语言统计。目标配比约
    n_synth / (n_docs + n_synth)。

    放在这里而不是 `train_tokenizer.py` 里，是因为 **MLM 阶段必须看到和 tokenizer
    完全相同的混合分布**。各写一份，"均匀插入"的间隔算法迟早会漂移，而漂移的表现
    只是 MLM 的英文占比悄悄变了 —— 不报错，只是模型变弱。

    两者共用这一个函数、但**可以传不同的 `n_synth`**，这是有意的：tokenizer 要的是
    **类型覆盖**（一个词出现够多次就能拿到合并），MLM 要的是**自然分布**（合成语料
    模板重复度极高，占比过高会挤掉自然语言统计）。所以 tokenizer 用 60k、MLM 用
    20k。共享的是插入算法，不是那个数字。
    """
    rng = random.Random(args.seed)
    synth = list(iter_synth_docs(args.n_synth, seed=args.seed))
    rng.shuffle(synth)

    every = max(1, args.n_docs // max(args.n_synth, 1))
    n_pre = 0
    for doc in iter_mixed(args.pretrain_path, args.en_path, max_docs=args.n_docs,
                          en_share=args.en_share):
        yield doc
        n_pre += 1
        if n_pre % every == 0 and synth:
            yield synth.pop()
    yield from synth                      # 预训练语料不足时把合成语料补完
    print(f"语料构成：中英混合 {n_pre} 篇 + 合成 {args.n_synth - len(synth)} 篇")


# ---------------------------------------------------------------------------
def fetch(args):
    """把公开英文语料抓成本地 jsonl。只在建语料时联网，训练阶段不联网。

    同一份代码抓两种语料：alpaca（`--fields instruction,input,output`）供域、
    wikitext（`--fields text --strip_wiki_title`）供量。两者由 `blend` 合成
    最终语料，理由见模块 docstring。
    """
    from datasets import load_dataset

    fields = [s for s in args.fields.split(",") if s]
    print(f"流式下载 {args.dataset}/{args.config} ... (字段 {fields})")
    ds = load_dataset(args.dataset, args.config, split="train", streaming=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    n = written = 0
    chars = 0
    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for row in ds:
            parts = [str(row.get(k) or "").strip() for k in fields]
            text = "\n".join(p for p in parts if p)
            if args.strip_wiki_title:
                # wikitext 每篇以 " = Title = " 开头，标题行对语言建模是噪声
                text = text.split("\n", 1)[-1].strip() if text.startswith("=") else text
            n += 1
            if len(text) < args.min_chars:
                continue
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            written += 1
            chars += len(text)
            if written % 20000 == 0:
                print(f"  已写 {written} 篇 / {chars/1e6:.1f}M 字符（扫描 {n} 篇）")
            if written >= args.n_docs:
                break
    os.replace(tmp, args.out)
    print(f"完成：{written} 篇 / {chars/1e6:.1f}M 字符 -> {args.out}"
          f"（共扫描 {n} 篇，丢弃 {n-written} 篇过短文档）")


def blend(args):
    """把 alpaca 与 wikitext 按**字符预算**合成最终英文语料。

    直接复用 iter_mixed —— 它本来就是"按字符比例自校正地取两路"，这里只是
    把两路都当成英文。wiki_share 是 wikitext 应占的字符比例；取 0.67 意味着
    alpaca 全程耗尽后 wikitext 补满，即"能给多少 alpaca 就给多少"。
    """
    tmp = args.out + ".tmp"
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for text in iter_mixed(args.alpaca, args.wiki, max_docs=args.max_docs,
                               en_share=args.wiki_share, zh_key="text", en_key="text"):
            f.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
            n += 1
    os.replace(tmp, args.out)
    print(f"完成：{n} 篇 -> {args.out}")


def main():
    p = argparse.ArgumentParser(description="MiniSystemOne 预训练语料准备")
    sub = p.add_subparsers(dest="cmd", required=True)

    f = sub.add_parser("fetch", help="抓取公开英文语料到本地 jsonl")
    f.add_argument("--out", default="dataset/pretrain_en.jsonl")
    f.add_argument("--n_docs", type=int, default=150000)
    f.add_argument("--dataset", default="tatsu-lab/alpaca")
    f.add_argument("--config", default="default")
    f.add_argument("--fields", default="instruction,input,output",
                   help="用逗号分隔的字段名，按序拼接成一篇文档")
    f.add_argument("--min_chars", type=int, default=MIN_CHARS)
    f.add_argument("--strip_wiki_title", action="store_true")

    b = sub.add_parser("blend", help="按字符预算合成 alpaca + wikitext")
    b.add_argument("--alpaca", default="dataset/pretrain_en_alpaca.jsonl")
    b.add_argument("--wiki", default="dataset/pretrain_en_wiki.jsonl")
    b.add_argument("--out", default="dataset/pretrain_en.jsonl")
    b.add_argument("--wiki_share", type=float, default=0.67)
    b.add_argument("--max_docs", type=int, default=200000)

    args = p.parse_args()
    if args.cmd == "fetch":
        fetch(args)
    elif args.cmd == "blend":
        blend(args)


if __name__ == "__main__":
    main()
