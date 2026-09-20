"""Stage 1: MLM encoder training with optimizer-boundary resume."""
import argparse
import hashlib
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
    Logger, data_manifest, file_sha1, get_lr, init_model, load_resume,
    optimizer_update, peak_vram_warn, restore_rng, restore_training, resume_state,
    save_checkpoint, seed_training, tokenizer_fingerprint, unbuffer_stdout,
    validate_train_args, verify_tokenizer, write_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 1: MLM 预训练")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--pretrain_path", default="dataset/pretrain_zh.jsonl")
    p.add_argument("--en_path", default="dataset/pretrain_en.jsonl")
    p.add_argument("--synthetic_only", action="store_true")
    p.add_argument("--allow_missing_corpus", action="store_true")
    p.add_argument("--n_docs", type=int, default=400000)
    p.add_argument("--n_synth", type=int, default=20000)
    p.add_argument("--en_share", type=float, default=0.5)
    p.add_argument("--max_len", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--out", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--save_optimizer", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--resume", default=None, help="精确续训：新版 *_opt.pth；保持原训练参数")
    p.add_argument("--init_checkpoint", default=None, help="同阶段权重初始化；重置优化器和训练计划")
    p.add_argument("--max_steps", type=int, default=0, help="绝对更新预算，0=全部 epochs")
    p.add_argument("--stop_after_steps", type=int, default=0,
                   help="本次运行更新这么多步后暂停，不改变 LR horizon")
    p.add_argument("--bench_seconds", type=int, default=0)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main():
    unbuffer_stdout()
    args = parse_args()
    if args.smoke:
        args.n_docs, args.n_synth = 2000, 500
        args.epochs, args.log_interval, args.save_interval = 1, 10, 200
        args.no_swanlab = True
    args.out = args.out or ("out/mlm_smoke" if args.smoke else "out/mlm")
    args.log_dir = args.log_dir or os.path.join(args.out, "logs")
    args.max_len = args.max_len if args.max_len is not None else (256 if args.smoke else 512)
    args.batch_size = args.batch_size if args.batch_size is not None else (8 if args.smoke else 16)
    validate_train_args(args)
    started = time.perf_counter()
    device = seed_training(args.seed, args.device)
    os.makedirs(args.out, exist_ok=True)

    print("========== 1. 语料 ==========")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    ds = MLMDataset(tok, iter_corpus(args), max_len=args.max_len, seed=args.seed)
    if len(ds.flat) < args.max_len:
        raise ValueError("语料为空或不足一个完整 MLM chunk")
    epoch_batches = [(len(ds) + args.batch_size - 1) // args.batch_size] * args.epochs
    fingerprints = {
        "tokenizer": tokenizer_fingerprint(args.tokenizer),
        "tokens": hashlib.sha1(memoryview(ds.flat)).hexdigest(),
        "corpus": {} if args.synthetic_only else {
            "zh": file_sha1(args.pretrain_path, n=40),
            "en": file_sha1(args.en_path, n=40)},
    }
    training = resume_state(args, fingerprints, epoch_batches)
    total_steps = training["total_steps"]

    print("\n========== 2. 模型 ==========")
    config = DecisionConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tok), use_gradient_checkpointing=args.use_checkpoint,
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        sep_token_id=tok.convert_tokens_to_ids("<sep>"),
        mask_token_id=tok.convert_tokens_to_ids("<mask>"),
    )
    if args.init_checkpoint:
        verify_tokenizer(args.init_checkpoint, args.tokenizer)
    model = init_model(MiniSystemOneForMaskedLM, config, args.init_checkpoint,
                       device, strict=True)
    saved = load_resume(args.resume, config, "mlm", training) if args.resume else None
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数 {n_param/1e6:.2f}M  max_len {args.max_len}  batch {args.batch_size}")
    ckpt_meta = {
        "stage": "mlm", "tokenizer_sha1": file_sha1(os.path.join(args.tokenizer, "tokenizer.json")),
        "n_params": n_param, "max_len": args.max_len, "epochs": args.epochs,
        "corpus": {"pretrain": args.pretrain_path, "en": args.en_path,
                   "n_docs": args.n_docs, "n_synth": args.n_synth,
                   "synthetic_only": args.synthetic_only},
        "synth_gen_version": data_manifest("dataset/synth").get("gen_version"),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    logger = Logger(args.log_dir, name="mlm", use_swanlab=not args.no_swanlab)
    step, start_epoch, next_batch = 0, 0, 0
    pending_rng = None
    if saved:
        pending_rng = restore_training(saved, model, opt, scaler)
        step, start_epoch, next_batch = saved["step"], saved["epoch"], saved["next_batch"]
        ckpt_meta = saved["meta"]
        training.update(epoch=start_epoch, next_batch=next_batch)
        del saved
        print(f"  续训 step={step}, epoch={start_epoch}, next_batch={next_batch}")
    start_step = step
    stop_at = min(total_steps, step + args.stop_after_steps) if args.stop_after_steps else total_steps

    def save():
        save_checkpoint(model, os.path.join(args.out, "mlm.pth"),
                        opt if args.save_optimizer else None, scaler, step,
                        config=config, meta=ckpt_meta, training=training)

    print(f"\n========== 3. 训练：每 epoch batches={epoch_batches}，LR horizon={total_steps} ==========")
    t0, tok_seen, bench_tok = time.perf_counter(), 0, 0
    bench_t0 = t0
    model.train()
    opt.zero_grad(set_to_none=True)
    stop = step >= stop_at
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        ds.set_epoch(epoch)
        order = torch.randperm(len(ds), generator=torch.Generator().manual_seed(args.seed + epoch)).tolist()
        batches = [order[i:i + args.batch_size] for i in range(0, len(order), args.batch_size)]
        cursor = next_batch if epoch == start_epoch else 0
        loader = DataLoader(ds, batch_sampler=batches[cursor:], collate_fn=collate_mlm,
                            num_workers=args.num_workers,
                            generator=torch.Generator().manual_seed(args.seed + epoch))
        iterator = iter(loader)
        if pending_rng is not None:
            restore_rng(pending_rng)
            pending_rng = None
        for micro, batch in enumerate(iterator, start=cursor):
            window = min(args.accum, len(batches) - (micro // args.accum) * args.accum)
            cur_lr = get_lr(step, total_steps, args.learning_rate)
            cur_lr *= min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for group in opt.param_groups:
                group["lr"] = cur_lr
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / window
            scaler.scale(loss).backward()
            tok_seen += batch["input_ids"].numel()
            bench_tok += batch["input_ids"].numel()
            if (micro + 1) % args.accum and micro + 1 != len(batches):
                continue
            optimizer_update(model, opt, scaler)
            step += 1
            training.update(epoch=epoch + (micro + 1 == len(batches)),
                            next_batch=0 if micro + 1 == len(batches) else micro + 1)
            if step % args.log_interval == 0 or step == start_step + 1:
                dt = time.perf_counter() - t0
                logger.log({"step": step, "epoch": epoch, "loss": out.loss.item(),
                            "lr": cur_lr, "tok_s": tok_seen / max(dt, 1e-9),
                            "elapsed_min": dt / 60})
                t0, tok_seen = time.perf_counter(), 0
            if args.save_interval and step % args.save_interval == 0:
                save()
            stop = step >= stop_at or (args.bench_seconds and time.perf_counter() - bench_t0 >= args.bench_seconds)
            if stop:
                break
    if pending_rng is not None:
        restore_rng(pending_rng)
    save()
    if args.bench_seconds:
        dt = time.perf_counter() - bench_t0
        print(f"  吞吐 {bench_tok / max(dt, 1e-9) / 1000:.1f}k tok/s（本机实测）")
    peak = peak_vram_warn()
    write_summary(args.out, step, started, peak)
    print(f"\n  训练结束/暂停：step {step}/{total_steps}，峰值显存 {peak:.2f} GB")
    print(f"  编码器 checkpoint -> {args.out}/mlm.pth")
    logger.finish()


if __name__ == "__main__":
    main()
