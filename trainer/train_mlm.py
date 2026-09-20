"""
Stage 1：MLM 预训练编码器。

这一阶段存在的理由只有一个 —— **决策头需要一个有语言表示的编码器**。从随机初始化
直接上决策训练，编码器要在学语言的同时学决策规则，而决策样本只有 6 万条；MLM 用
上千万 token 的自然语言先把表示垫起来，是这个小模型唯一的免费午餐。

产物是**编码器**，不是分类器。所以：

  - 保存的 checkpoint 会被 `train_decision.py` 用 `strict=False` 载入，
    带过去的只有 `encoder.*`；`lm_head` 被丢掉（它与 `embed_tokens` 绑定，
    本来就与编码器共享权重，丢了不损失任何已学信息）。
  - MLM 的最终 loss 本身**不是**要报告的指标。它只用来确认训练没崩。

与决策格式的关系见 `dataset/mlm_dataset.py` 的模块注释（结构 token 冲突、
长度分布、两个学习率）—— 结论是**分开阶段、分开数据集、不混**。

用法：
    python trainer/train_mlm.py --smoke                       # 2 分钟冒烟
    python trainer/train_mlm.py                               # 默认 400k 篇
    python trainer/train_mlm.py --bench_seconds 30            # 只测 tok/s
"""
import argparse
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.mlm_dataset import MLMDataset, collate_mlm
from dataset.pretrain_corpus import iter_corpus
from model.model_system_one import DecisionConfig, MiniSystemOneForMaskedLM
from trainer.trainer_utils import (
    Logger, data_manifest, file_sha1, get_lr, init_model, peak_vram_warn,
    save_checkpoint, unbuffer_stdout,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: MLM 预训练")
    p.add_argument("--tokenizer", default="model", help="tokenizer 目录")
    p.add_argument("--pretrain_path", default="../minimind/dataset/pretrain_t2t_mini.jsonl",
                   help="中文预训练语料 jsonl（缺省则只用英文 + 合成）")
    p.add_argument("--en_path", default="dataset/pretrain_en.jsonl")
    p.add_argument("--n_docs", type=int, default=400000, help="中英混合语料篇数")
    p.add_argument("--n_synth", type=int, default=20000, help="合成决策语料篇数")
    p.add_argument("--en_share", type=float, default=0.5, help="英文目标字符占比")
    # 这两项的默认值取决于 --smoke，所以留 None 在 main 里定；写死成 512/16 会让
    # `--smoke --max_len 512` 这类组合被静默覆盖（冒烟规模本该只是**默认值**，
    # 不是强制值，否则没法用它去测真实配置的吞吐）。
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--accum", type=int, default=1, help="梯度累积步数")
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-3,
                   help="峰值学习率；余弦表实际峰值是它的 0.55 倍")
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--out", default="out/mlm")
    p.add_argument("--log_dir", default="out/mlm/logs")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true", help="逐层梯度检查点")
    p.add_argument("--save_optimizer", action="store_true",
                   help="另存优化器状态（约 3 倍体积，续训才需要）")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--resume", default=None, help="续训用的 *_opt.pth")
    p.add_argument("--bench_seconds", type=int, default=0,
                   help="只跑这么久的吞吐测定，然后退出并报预计 epoch 时间")
    p.add_argument("--smoke", action="store_true", help="极小规模端到端冒烟")
    return p.parse_args()


def main():
    unbuffer_stdout()
    args = parse_args()
    if args.smoke:
        args.n_docs, args.n_synth = 2000, 500
        args.epochs, args.log_interval, args.save_interval = 1, 10, 200
        args.out, args.log_dir = "out/mlm_smoke", "out/mlm_smoke/logs"
        # 冒烟按定义就是非交互的，不该把时间花在等 swanlab 问卷上
        args.no_swanlab = True
    args.max_len = args.max_len or (256 if args.smoke else 512)
    args.batch_size = args.batch_size or (8 if args.smoke else 16)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本项目的所有实测数字都基于 GPU；--device cpu 不在支持范围内。")

    print("========== 1. 语料 ==========")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    ds = MLMDataset(tok, iter_corpus(args), max_len=args.max_len, seed=args.seed)
    if len(ds) == 0:
        raise SystemExit("语料为空 —— 检查 --pretrain_path / --en_path 是否存在")

    print("\n========== 2. 模型 ==========")
    config = DecisionConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tok), use_gradient_checkpointing=args.use_checkpoint,
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        sep_token_id=tok.convert_tokens_to_ids("<sep>"),
        mask_token_id=tok.convert_tokens_to_ids("<mask>"),
    )
    model = init_model(MiniSystemOneForMaskedLM, config, args.resume, device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数 {n_param/1e6:.2f}M  max_len {args.max_len}  batch {args.batch_size}"
          f"  accum {args.accum}")

    # 写进 checkpoint 的自描述信息，见 `save_checkpoint`。没有它，下载权重的人只能回
    # README 手抄 h/L，也无从核对权重与词表是否配套。
    #
    # MLM 的语料是"中英混合 + 合成"，不是某一个 `gen_version` 的全集，所以这个键叫
    # `synth_gen_version`（只指其中合成那部分），不冒充整体版本。
    ckpt_meta = {
        "stage": "mlm",
        "tokenizer_sha1": file_sha1(os.path.join(args.tokenizer, "tokenizer.json")),
        "n_params": n_param,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "corpus": {"pretrain": args.pretrain_path, "en": args.en_path,
                   "n_docs": args.n_docs, "n_synth": args.n_synth},
        "synth_gen_version": data_manifest("dataset/synth").get("gen_version"),
    }

    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=collate_mlm, num_workers=args.num_workers,
                        drop_last=True)
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        opt.load_state_dict(ck["optimizer"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        start_step = ck.get("step", 0)
        print(f"  续训自 step {start_step}")

    logger = Logger(args.log_dir, name="mlm", use_swanlab=not args.no_swanlab)
    tokens_per_step = args.batch_size * args.max_len * args.accum

    print("\n========== 3. 训练 ==========")
    print(f"  每 epoch {steps_per_epoch} step，共 {total_steps} step，"
          f"每 step {tokens_per_step} token")
    step, t0, tok_seen = start_step, time.time(), 0
    bench_t0, bench_tok = time.time(), 0
    model.train()
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        ds.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            cur_lr = get_lr(step, total_steps, args.learning_rate)
            cur_lr *= min(1.0, (step + 1) / max(args.warmup_steps, 1))   # 线性 warmup
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / args.accum
            scaler.scale(loss).backward()

            if (micro + 1) % args.accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)

            step += 1
            tok_seen += tokens_per_step
            bench_tok += tokens_per_step
            # 报 `out.loss`（未除 accum）而不是反传用的那个：除过的那个会被
            # accum 的取值改变数值，跨配置不可比。
            if step % args.log_interval == 0 or step == start_step + 1:
                dt = time.time() - t0
                logger.log({"step": step, "epoch": epoch, "loss": out.loss.item(),
                            "lr": cur_lr, "tok_s": tok_seen / max(dt, 1e-9),
                            "elapsed_min": dt / 60})
                t0, tok_seen = time.time(), 0

            if args.save_interval and step % args.save_interval == 0:
                save_checkpoint(model, os.path.join(args.out, "mlm.pth"),
                                opt if args.save_optimizer else None, scaler, step,
                                config=config, meta=ckpt_meta)
                print(f"    ckpt -> {args.out}/mlm.pth")

            if args.bench_seconds and time.time() - bench_t0 > args.bench_seconds:
                stop = True
                break

    if args.bench_seconds:
        dt = time.time() - bench_t0
        tok_s = bench_tok / max(dt, 1e-9)
        if step == start_step:
            print("\n  测定期间一步都没跑完 —— 加长 --bench_seconds 再试")
        else:
            print(f"\n  吞吐 {tok_s/1000:.1f}k tok/s（{bench_tok} token / {dt:.1f}s）")
            print(f"  预计每 epoch {(len(ds)*args.max_len/tok_s)/60:.1f} 分钟，"
                  f"{args.epochs} 个 epoch 共 "
                  f"{(steps_per_epoch*args.epochs*tokens_per_step/tok_s)/60:.0f} 分钟")
        print("  （这个数字来自**你的**硬件。README 里的吞吐是在 RTX 4070 Laptop "
              "上测的，不能直接套用。）")
        peak_vram_warn()
        logger.finish()
        return

    save_checkpoint(model, os.path.join(args.out, "mlm.pth"),
                    opt if args.save_optimizer else None, scaler, step,
                    config=config, meta=ckpt_meta)
    peak = peak_vram_warn()
    print(f"\n  训练结束：step {step}，峰值显存 {peak:.2f} GB")
    print(f"  编码器 checkpoint -> {args.out}/mlm.pth")
    print(f"  下一步：python trainer/train_decision.py --encoder {args.out}/mlm.pth")
    logger.finish()


if __name__ == "__main__":
    main()
