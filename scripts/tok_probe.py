"""量一下 tokenizer 在**真实生成器输出**上的表现 —— 判断是否需要重训。

`dataset/synth/lexicon.py::corpus_doc` 是为"生成器还没写出来"准备的表面语料，
它覆盖了工具名/标签/金额/日期这些**词类型**，但覆盖不到生成器真正的句子骨架
（中文动作词与对象词、百分比、时刻、日名、规则句里的术语）。

重训 tokenizer 意味着废弃一个正在跑的 MLM。所以先量：这些缺口到底有多大。
只看两个数：真实 state 的压缩率，以及承重词的碎词数。
"""
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from transformers import AutoTokenizer

from dataset.synth import build_all
from dataset.synth.lexicon import LEVELS, STATUSES, TOOLS

TOK_DIR = "model"

# 这些是"候选/标签/金额被渲染成几个 token"的直接读数。碎词数越高，
# scorer 拿到的 pooled 向量噪声越大 —— 这是 plan 里点名要防的那件事。
#
# 工具名/等级/状态**从 lexicon 现取**而不是手写：手写的那份曾经写着
# `Standard`/`Platinum`，而 `LEVELS` 里根本没有这两个词（实际是
# `very poor`…`excellent`，全小写）。探针测了一些数据里不存在的字符串，
# 读数看着没问题，结论却是空的。
PROBE = ([*TOOLS[:8], *[en for en, _ in LEVELS], *[en for en, _ in STATUSES[:4]]]
         + [
             "¥1,240.00", "1,240.00 元", "62.5%", "百分之 62.5",
             "08:00", "17:45", "周一", "Wed", "morning", "上午", "晚间",
             "转移所有权", "取消订阅", "删除回调", "查询用量报告",
             "检出率", "误报率", "先验胜算", "似然比", "后验胜算",
             "排期规则", "权重相同", "等可能", "弃权",
         ])


def main():
    dirs = sys.argv[1:] or [TOK_DIR]
    toks = [(d, AutoTokenizer.from_pretrained(d)) for d in dirs]
    print("vocab = " + ", ".join(f"{d}:{len(t)}" for d, t in toks))

    print("\n=== 逐词对照（各 tokenizer 的碎词数）===")
    hdr = "  " + f"{'probe':28}" + "".join(f"{d:>16}" for d, _ in toks)
    print(hdr)
    for s in PROBE:
        ns = [len(t(s, add_special_tokens=False)["input_ids"]) for _, t in toks]
        print(f"  {s!r:28}" + "".join(f"{n:>16}" for n in ns))

    tok = toks[-1][1]
    print(f"\n=== 承重词碎词数（{dirs[-1]}）===")

    print("\n=== 承重词碎词数 ===")
    bad = 0
    for s in PROBE:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        pieces = tok.convert_ids_to_tokens(tok(s, add_special_tokens=False)["input_ids"])
        flag = "" if n <= 2 else "  <-- 碎"
        if n > 2:
            bad += 1
        print(f"  {n}  {s!r:28} {pieces}{flag}")
    print(f"碎词(>2)比例：{bad}/{len(PROBE)}")

    print("\n=== 真实 state 压缩率 ===")
    for g in build_all(seed=7):
        recs = g.generate("train", 40)
        chars = toks = 0
        for r in recs:
            text = r["state"] + "\n" + r["question"]
            text += "".join(c["text"] for c in r["candidates"])
            chars += len(text)
            toks += len(tok(text, add_special_tokens=False)["input_ids"])
        print(f"  {g.name:20} {chars/toks:5.2f} 字/token   "
              f"({chars} 字 / {toks} tok)")


if __name__ == "__main__":
    main()
