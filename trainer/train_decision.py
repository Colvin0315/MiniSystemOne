"""
Stage 2：决策训练。`L = CE + λ_b·Brier + is_ord·λ_o·CDF-MSE`。

这一阶段是项目的主体。它和普通分类训练的区别只有一条，但那条是全部：

    **目标是分布，不是标签。**

`provenance` 决定这份分布从哪来（`explicit_rng` 的银行 RNG、`tie_set` 的并列集、
`marginalized` 的边缘化），而 `hard` 的目标是 one-hot —— 它在 DATA_SCHEMA 里被
排除在**所有校准指标**之外，因为它里面没有任何关于"世界有多不确定"的信息，
拿它算 ECE 只会把整体数字稀释得好看。所以训练用的 val 上同时报两套：
全部样本、以及剔除 hard 的 `calibration` 子集。

另外三条不显然但重要的约定：

1. **`calib` split 只在温度校准里用，永不训练。** 在校准集上训过的模型，其温度
   拟合必然过拟合，可靠性图会假性变好。这里断言 sample 出的 batch 只来自 train。
2. **权重不初始化**（`--encoder ""` 时）等于从随机初始化直接训决策 —— 这是方案 R4
   的消融之一，用来量化"MLM 预训练到底值多少"。
3. **Brier 不做 K 归一化**（默认），理由见 `model_system_one.compute_loss`；
   `--brier_normalize` 提供出来是为了让读者自己看到那个差别。

用法：
    python trainer/train_decision.py --smoke                      # 100 步冒烟
    python trainer/train_decision.py --encoder out/mlm/mlm.pth
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

from dataset.decision_dataset import (
    CandidateBucketSampler, DecisionDataset, collate_decision,
)
from eval.eval_inference import collect, counts_array
from eval.eval_metrics import metrics_by_provenance
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.trainer_utils import (
    Logger, data_manifest, file_sha1, get_lr, init_model, peak_vram_warn,
    save_checkpoint, unbuffer_stdout,
)


def parse_args():
    p = argparse.ArgumentParser(description="Stage 2: 决策训练")
    p.add_argument("--data", default="dataset/synth", help="含 train/val jsonl 的目录")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--encoder", default=None,
                   help="Stage 1 的编码器。缺省 out/mlm/mlm.pth（--smoke 下退到 "
                        "out/mlm_smoke/mlm.pth）；**空字符串 = 从随机初始化**（R4 消融）")
    p.add_argument("--out", default="out/decision")
    p.add_argument("--log_dir", default="out/decision/logs")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--k_min", type=int, default=2)
    p.add_argument("--k_max", type=int, default=32, help="训练硬上限，见方案 R5")
    p.add_argument("--keep_p_min", type=float, default=0.05)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--accum", type=int, default=1)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=5e-4)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--lambda_brier", type=float, default=0.5)
    p.add_argument("--lambda_ord", type=float, default=0.5)
    p.add_argument("--brier_normalize", action="store_true")
    p.add_argument("--crosstalk", action="store_true",
                   help="打开候选互看（这是消融，默认关；见方案套件①）")
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--val_every", type=int, default=500, help="每多少步跑一次 val")
    p.add_argument("--train_limit", type=int, default=0,
                   help="train 只用前 N 条（0=全部）；冒烟用")
    p.add_argument("--val_limit", type=int, default=0, help="val 只用前 N 条（0=全部）")
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--save_interval", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--use_checkpoint", action="store_true")
    p.add_argument("--save_optimizer", action="store_true")
    p.add_argument("--no_swanlab", action="store_true")
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


# ---------------------------------------------------------------------------
def evaluate(model, dataset, device, limit=0, batch_size=16):
    """训练循环内的 val。**转发到 `eval_inference.collect`，不自己再写一遍前向。**

    这不是为了复用代码，是为了保证"训练日志里的 ECE"和"`eval_harness.py` 报告的
    ECE"是同一个数。两份各算一次的实现迟早会漂移（一次忘了同步 mask、一次忘了
    升 fp64），而漂移的方向通常**是训练日志比报告好看** —— 那正好是最不该发生的
    一种错误（方案 R6）。

    **`batch_size` 由调用方传 `args.batch_size`，不另取默认值。** 缓存分配器按块
    大小复用：训练用 16 而 val 用 32 时，val 要的块（32×S）比训练缓存过的任何一块
    都大，一律走新的 `cudaMalloc`，预留量整个叠在训练池上 —— 实测 6.0 GiB + 3.9 GiB
    = 7.92 GiB，越过 R5 的 Windows WDDM 悬崖（8188 MiB 卡上超过约 7.5 GB 即静默
    换页、不报 OOM，表现为 step 500 那次 val 卡住 6 分钟以上）。对齐到 16 之后
    实测预留 1.91 GiB，能落回训练池已缓存的块里。
    """
    col = collect(model, dataset, device, batch_size=batch_size, limit=limit)
    if col is None:
        return None
    col["metrics"] = metrics_by_provenance(
        col["p"], col["target"], col["mask"], provenance=col["provenance"],
        is_ord=col["is_ord"], levels=np.where(col["levels"] >= 0, col["levels"], 0),
        counts=counts_array(col),
    )
    return col


def main():
    unbuffer_stdout()
    args = parse_args()
    if args.smoke:
        args.epochs, args.batch_size = 1, 8
        args.val_every, args.log_interval, args.save_interval = 50, 10, 100
        args.out, args.log_dir = "out/decision_smoke", "out/decision_smoke/logs"
        args.val_limit = args.val_limit or 200
        # 冒烟按定义就是非交互的，不该把时间花在等 swanlab 问卷上
        args.no_swanlab = True
    if args.encoder is None:
        # 冒烟接 `train_mlm.py --smoke` 的产物，让"两步连跑"开箱可用；
        # 生产路径接 `train_mlm.py` 的产物。显式传 `--encoder ''` 不受影响。
        args.encoder = ("out/mlm_smoke/mlm.pth" if args.smoke else "out/mlm/mlm.pth")

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本项目的所有实测数字都基于 GPU；--device cpu 不在支持范围内。")

    print("========== 1. 数据 ==========")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    train_path = os.path.join(args.data, "train.jsonl")
    val_path = os.path.join(args.data, "val.jsonl")
    for path in (train_path, val_path):
        if not os.path.exists(path):
            raise SystemExit(f"缺少 {path} —— 先跑 scripts/build_dataset.py")
    ds_train = DecisionDataset(train_path, tok, max_len=args.max_len, k_min=args.k_min,
                               k_max=args.k_max, keep_p_min=args.keep_p_min,
                               augment_k=True, seed=args.seed, limit=args.train_limit)
    # val 不做 K 增广：评测要可复现，而且要覆盖**全部**候选（K_full），
    # 否则"K 变化时是否还准"这件事在训练循环里根本看不到。
    ds_val = DecisionDataset(val_path, tok, max_len=args.max_len, augment_k=False,
                             seed=args.seed)
    print(f"  train {len(ds_train)} 条   val {len(ds_val)} 条   K∈[{args.k_min},{args.k_max}]")
    print(f"  calib split **不在此读取** —— 它只用于温度校准（见 calibrate_temperature.py）")

    sampler = CandidateBucketSampler(ds_train, args.batch_size, shuffle=True, seed=args.seed)
    loader = DataLoader(ds_train, batch_sampler=sampler, collate_fn=collate_decision,
                        num_workers=args.num_workers)
    steps_per_epoch = max(1, len(loader) // args.accum)
    total_steps = steps_per_epoch * args.epochs

    print("\n========== 2. 模型 ==========")
    config = DecisionConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tok), use_gradient_checkpointing=args.use_checkpoint,
        candidate_crosstalk=args.crosstalk,
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        sep_token_id=tok.convert_tokens_to_ids("<sep>"),
    )
    ckpt = args.encoder or None
    if ckpt and not os.path.exists(ckpt):
        raise SystemExit(f"--encoder {ckpt} 不存在；要从随机初始化训练请显式传 --encoder ''")
    model = init_model(MiniSystemOneForDecision, config, ckpt, device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"  参数 {n_param/1e6:.2f}M  "
          f"crosstalk={args.crosstalk}  prefix_blocked={config.prefix_blocked}")

    # 自描述信息，见 `save_checkpoint`。`gen_version` 取自**训练数据的 manifest**
    # 而不是命令行参数 —— 这样「这个权重是在哪版数据上训的」是数据自己说的，
    # 不是人复述的。
    dm = data_manifest(args.data)
    ckpt_meta = {
        "stage": "decision",
        "tokenizer_sha1": file_sha1(os.path.join(args.tokenizer, "tokenizer.json")),
        "gen_version": dm.get("gen_version"),
        "n_params": n_param,
        "max_len": args.max_len,
        "epochs": args.epochs,
        "encoder_init": os.path.basename(ckpt) if ckpt else "random",
        "trained_on": os.path.basename(os.path.normpath(args.data)),
    }

    opt = torch.optim.AdamW(model.parameters(), lr=args.learning_rate,
                            betas=(0.9, 0.95), weight_decay=0.01)
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=True)
    except (AttributeError, TypeError):
        scaler = torch.cuda.amp.GradScaler(enabled=True)
    logger = Logger(args.log_dir, name="decision", use_swanlab=not args.no_swanlab)

    print("\n========== 3. 训练 ==========")
    print(f"  每 epoch {steps_per_epoch} step，共 {total_steps} step")
    print(f"  loss = CE + {args.lambda_brier}·Brier"
          f"{f' / K' if args.brier_normalize else ''}"
          f" + is_ord·{args.lambda_ord}·CDF-MSE")
    best = None
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        for micro, batch in enumerate(loader):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                     for k, v in batch.items()}
            cur_lr = get_lr(step, total_steps, args.learning_rate)
            cur_lr *= min(1.0, (step + 1) / max(args.warmup_steps, 1))
            for g in opt.param_groups:
                g["lr"] = cur_lr

            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(
                    batch["input_ids"], seg_id=batch["seg_id"], cand_id=batch["cand_id"],
                    cand_span=batch["cand_span"], cand_mask=batch["cand_mask"],
                    target=batch["target"], is_ord=batch["is_ord"],
                    lambda_brier=args.lambda_brier, lambda_ord=args.lambda_ord,
                )
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

            if step % args.log_interval == 0:
                d = {"step": step, "epoch": epoch, "lr": cur_lr,
                     "step_s": (time.time() - t0) / args.log_interval}
                d.update({k: float(v) for k, v in out.loss_dict.items()})
                logger.log(d)
                t0 = time.time()

            if args.val_every and step % args.val_every == 0:
                res = evaluate(model, ds_val, device, args.val_limit, args.batch_size)
                if res:
                    m, mc = res["metrics"]["all"], res["metrics"].get("calibration", {})
                    # **val 上必须同时看 ECE 与 Brier。** ECE 不是 proper score，
                    # 单独盯它会被分桶方案带偏（方案 R6）。
                    logger.log({"step": step, "val_acc": m["accuracy"],
                                "val_ece": m["ece"], "val_brier": m["brier"],
                                "val_nll": m["nll"],
                                "val_ece_calib": mc.get("ece", float("nan")),
                                "val_acc_calib": mc.get("accuracy", float("nan"))})
                    with open(os.path.join(args.out, "val_last.json"), "w",
                              encoding="utf-8") as f:
                        json.dump({"step": step, "metrics": res["metrics"]}, f,
                                  ensure_ascii=False, indent=2)

            if args.save_interval and step % args.save_interval == 0:
                save_checkpoint(model, os.path.join(args.out, "decision.pth"),
                                opt if args.save_optimizer else None, scaler, step,
                                config=config, meta=ckpt_meta)

    save_checkpoint(model, os.path.join(args.out, "decision.pth"),
                    opt if args.save_optimizer else None, scaler, step,
                    config=config, meta=ckpt_meta)
    res = evaluate(model, ds_val, device, args.val_limit, args.batch_size)
    if res:
        with open(os.path.join(args.out, "val_last.json"), "w", encoding="utf-8") as f:
            json.dump({"step": step, "metrics": res["metrics"]}, f,
                      ensure_ascii=False, indent=2)
        m = res["metrics"]["all"]
        print(f"\n  val 全部样本    acc {m['accuracy']:.3f}  ECE {m['ece']:.4f}  "
              f"Brier {m['brier']:.4f}  NLL {m['nll']:.4f}")
        for prov, mm in res["metrics"].items():
            if prov in ("all", "calibration"):
                continue
            print(f"    {prov:16s} n={mm['n']:6d}  acc {mm['accuracy']:.3f}  "
                  f"ECE {mm['ece']:.4f}")
    peak = peak_vram_warn()
    print(f"\n  训练结束：step {step}，峰值显存 {peak:.2f} GB")
    print(f"  -> {args.out}/decision.pth")
    print(f"  下一步：python eval/eval_harness.py --ckpt {args.out}/decision.pth")
    logger.finish()


if __name__ == "__main__":
    main()
