"""Stage 2: CE + distribution L2 + ordinal CDF loss.

Validation reports all examples and soft_targets separately. Hard-label ECE is
valid too; top-label ECE does not establish per-example distribution accuracy.
"""
import argparse
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from dataset.decision_dataset import CandidateBucketSampler, DecisionDataset, collate_decision
from eval.eval_inference import collect, counts_array
from eval.eval_metrics import metrics_by_provenance
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.trainer_utils import (
    Logger, capture_rng, ckpt_info, data_manifest, file_sha1, get_lr, init_model,
    load_resume, optimizer_update, peak_vram_warn, restore_rng, restore_training,
    resume_state, save_checkpoint, seed_training, tokenizer_fingerprint,
    unbuffer_stdout, validate_train_args, verify_tokenizer, write_summary,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: 决策训练")
    p.add_argument("--data", default="dataset/synth", help="含 train/val jsonl 的目录")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--encoder", default=None,
                   help="MLM 编码器初始化（非严格跨阶段）；空字符串=随机初始化")
    p.add_argument("--init_checkpoint", default=None, help="决策权重初始化（严格同阶段），重新训练")
    p.add_argument("--resume", default=None, help="精确续训：新版 *_opt.pth；保持原训练参数")
    p.add_argument("--out", default=None)
    p.add_argument("--log_dir", default=None)
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=32)
    p.add_argument("--keep_p_min", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lambda_brier", type=float, default=0.5)
    p.add_argument("--lambda_ord", type=float, default=0.5)
    p.add_argument("--brier_normalize", action="store_true")
    p.add_argument("--crosstalk", action="store_true")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--val_every", type=int, default=500)
    p.add_argument("--train_limit", type=int, default=0)
    p.add_argument("--val_limit", type=int, default=0)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--save_optimizer", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--max_steps", type=int, default=0, help="绝对更新预算，0=全部 epochs")
    p.add_argument("--stop_after_steps", type=int, default=0,
                   help="本次运行更新这么多步后暂停，不改变 LR horizon")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def evaluate(model, dataset, device, limit=0, batch_size=16):
    rng, was_training = capture_rng(), model.training
    try:
        col = collect(model, dataset, device, batch_size=batch_size, limit=limit)
        if col is None:
            return None
        col["metrics"] = metrics_by_provenance(
            col["p"], col["target"], col["mask"], provenance=col["provenance"],
            is_ord=col["is_ord"], levels=np.where(col["levels"] >= 0, col["levels"], 0),
            counts=counts_array(col),
        )
        return col
    finally:
        # Evaluation must not change the subsequent training stream.
        restore_rng(rng)
        model.train(was_training)


def main():
    unbuffer_stdout()
    args = parse_args()
    if args.smoke:
        args.epochs, args.batch_size = 1, 8
        args.val_every, args.log_interval, args.save_interval = 50, 10, 100
        args.val_limit = args.val_limit or 200
        args.no_swanlab = True
    args.out = args.out or ("out/decision_smoke" if args.smoke else "out/decision")
    args.log_dir = args.log_dir or os.path.join(args.out, "logs")
    validate_train_args(args)
    if args.init_checkpoint and args.encoder:
        raise ValueError("--init_checkpoint 与 --encoder 不能同时使用")
    if args.encoder is None and not (args.resume or args.init_checkpoint):
        args.encoder = "out/mlm_smoke/mlm.pth" if args.smoke else "out/mlm/mlm.pth"
    started = time.perf_counter()
    device = seed_training(args.seed, args.device)
    os.makedirs(args.out, exist_ok=True)

    print("========== 1. 数据 ==========")
    tok = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    train_path = os.path.join(args.data, "train.jsonl")
    val_path = os.path.join(args.data, "val.jsonl")
    ds_train = DecisionDataset(train_path, tok, max_len=args.max_len, k_min=args.k_min,
                               k_max=args.k_max, keep_p_min=args.keep_p_min,
                               augment_k=True, seed=args.seed, limit=args.train_limit)
    ds_val = DecisionDataset(val_path, tok, max_len=args.max_len, augment_k=False, seed=args.seed)
    for dataset, split in ((ds_train, "train"), (ds_val, "val")):
        if any(rec.get("split") != split for rec in dataset.records):
            raise ValueError(f"{split} 文件包含其他 split 的样本")
    print(f"  train {len(ds_train)} 条   val {len(ds_val)} 条   K∈[{args.k_min},{args.k_max}]")
    sampler = CandidateBucketSampler(ds_train, args.batch_size, shuffle=True, seed=args.seed)
    epoch_batches = []
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        epoch_batches.append(len(sampler))
    fingerprints = {
        "tokenizer": tokenizer_fingerprint(args.tokenizer),
        "train": file_sha1(train_path, n=40), "val": file_sha1(val_path, n=40),
        "manifest": file_sha1(os.path.join(args.data, "manifest.json"), n=40),
    }
    training = resume_state(args, fingerprints, epoch_batches)
    total_steps = training["total_steps"]

    print("\n========== 2. 模型 ==========")
    config = DecisionConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tok), use_gradient_checkpointing=args.use_checkpoint,
        candidate_crosstalk=args.crosstalk,
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        sep_token_id=tok.convert_tokens_to_ids("<sep>"),
    )
    ckpt = None if args.resume else (args.init_checkpoint or args.encoder or None)
    if ckpt:
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"初始化权重不存在：{ckpt}；随机训练请显式 --encoder ''")
        stage = ckpt_info(ckpt).get("meta", {}).get("stage")
        if stage is None:
            weights = torch.load(ckpt, map_location="cpu", weights_only=True)
            weights = weights.get("model", weights)
            stage = "decision" if any(k.startswith("head.") for k in weights) else "mlm"
            del weights
        if args.init_checkpoint and stage != "decision":
            raise ValueError("--init_checkpoint 需要 decision 权重；MLM 请用 --encoder")
        if stage not in ("mlm", "decision"):
            raise ValueError(f"不支持的初始化 stage：{stage}")
        verify_tokenizer(ckpt, args.tokenizer)
    strict = bool(args.init_checkpoint) or (bool(ckpt) and stage == "decision")
    model = init_model(MiniSystemOneForDecision, config, ckpt, device, strict=strict)
    saved = load_resume(args.resume, config, "decision", training) if args.resume else None
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数 {n_param/1e6:.2f}M  crosstalk={args.crosstalk}")
    ckpt_meta = {
        "stage": "decision", "tokenizer_sha1": file_sha1(os.path.join(args.tokenizer, "tokenizer.json")),
        "gen_version": data_manifest(args.data).get("gen_version"),
        "n_params": n_param, "max_len": args.max_len, "epochs": args.epochs,
        "encoder_init": os.path.basename(ckpt) if ckpt else "random",
        "trained_on": os.path.basename(os.path.normpath(args.data)),
    }
    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    logger = Logger(args.log_dir, name="decision", use_swanlab=not args.no_swanlab)
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
        save_checkpoint(model, os.path.join(args.out, "decision.pth"),
                        opt if args.save_optimizer else None, scaler, step,
                        config=config, meta=ckpt_meta, training=training)

    def validate():
        res = evaluate(model, ds_val, device, args.val_limit, args.batch_size)
        if res:
            m, soft = res["metrics"]["all"], res["metrics"].get("soft_targets", {})
            logger.log({"step": step, "val_acc": m["accuracy"], "val_ece": m["ece"],
                        "val_distribution_l2": m["distribution_l2"], "val_nll": m["nll"],
                        "val_ece_soft_targets": soft.get("ece", float("nan")),
                        "val_acc_soft_targets": soft.get("accuracy", float("nan"))})
            with open(os.path.join(args.out, "val_last.json"), "w", encoding="utf-8") as f:
                json.dump({"step": step, "metrics": res["metrics"]}, f, ensure_ascii=False, indent=2)

    print(f"\n========== 3. 训练：每 epoch batches={epoch_batches}，LR horizon={total_steps} ==========")
    model.train()
    opt.zero_grad(set_to_none=True)
    t0 = time.perf_counter()
    stop = step >= stop_at
    for epoch in range(start_epoch, args.epochs):
        if stop:
            break
        sampler.set_epoch(epoch)
        batches = list(sampler)
        if len(batches) != epoch_batches[epoch]:
            raise RuntimeError("epoch batch 数量与 LR 计划不一致")
        cursor = next_batch if epoch == start_epoch else 0
        loader = DataLoader(ds_train, batch_sampler=batches[cursor:], collate_fn=collate_decision,
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
            batch = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    batch["input_ids"], seg_id=batch["seg_id"], cand_id=batch["cand_id"],
                    cand_span=batch["cand_span"], cand_mask=batch["cand_mask"],
                    target=batch["target"], is_ord=batch["is_ord"],
                    lambda_brier=args.lambda_brier, lambda_ord=args.lambda_ord,
                    brier_normalize=args.brier_normalize,
                )
                loss = out.loss / window
            scaler.scale(loss).backward()
            if (micro + 1) % args.accum and micro + 1 != len(batches):
                continue
            optimizer_update(model, opt, scaler)
            step += 1
            training.update(epoch=epoch + (micro + 1 == len(batches)),
                            next_batch=0 if micro + 1 == len(batches) else micro + 1)
            if step % args.log_interval == 0 or step == start_step + 1:
                d = {"step": step, "epoch": epoch, "lr": cur_lr,
                     "elapsed_s": time.perf_counter() - t0}
                d.update({"distribution_l2" if k == "brier" else k: float(v)
                          for k, v in out.loss_dict.items()})
                logger.log(d)
                t0 = time.perf_counter()
            if args.val_every and step % args.val_every == 0:
                validate()
            if args.save_interval and step % args.save_interval == 0:
                save()
            if step >= stop_at:
                stop = True
                break
    if pending_rng is not None:
        restore_rng(pending_rng)
    save()
    validate()
    peak = peak_vram_warn()
    write_summary(args.out, step, started, peak)
    print(f"\n  训练结束/暂停：step {step}/{total_steps}，峰值显存 {peak:.2f} GB")
    print(f"  -> {args.out}/decision.pth")
    logger.finish()


if __name__ == "__main__":
    main()
