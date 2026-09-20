"""
温度校准：在 `calib` split 上拟合标量温度，写 `out/calibration/T.json`。

**为什么只有一个标量温度，而不是 per-class 的偏置。** 候选集是动态的：候选 k=3
在每条样本里都是不同的实体，"第 3 类的温度"没有指称对象。标量温度是唯一能
(a) 跨候选集迁移、(b) 用在模型**从未见过的 schema** 上的参数化。这一点是整个
校准故事自洽的关键，不是事后找补。

**拟合 `log T` 而不是 `T`。** `T` 在 (0, ∞) 上，梯度尺度两端差几个数量级；`log T`
在 ℝ 上均匀，LBFGS 的收敛判断才正常，而且不需要把 `T` 的参数化硬塞进 `softplus`
（那会引入一个额外的尺度先验）。钳位在 `[0.05, 20]` 是因为更极端的温度在校准集
（2k 条）上已经是过拟合噪声了。

**三种粒度都拟合、都报告：** 全局 1 个 / per-primitive 3 个 /
per-(primitive × K 桶) 最多 15 个。K 桶取 `eval_metrics.K_BUCKETS`，与 ECE 分桶
共用同一套边界。三数对比比一句"per-K 温度更细"更有信息量：如果全局一个标量就能
追平 15 个桶，那么多出来的自由度就是白算的。

**绝不训练、只在 calib 上拟合。** 在校准集上训过的模型，它的温度必然过拟合，
可靠性图会假性变好。`train_decision.py` 因此根本不读 calib。

用法：
    python trainer/calibrate_temperature.py --ckpt out/decision/decision.pth
"""
import argparse
import hashlib
import json
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
from transformers import AutoTokenizer

from dataset.decision_dataset import DecisionDataset
from eval.eval_inference import collect, counts_array, effective_k
from eval.eval_metrics import (
    K_BUCKETS, apply_temperature, metrics_by_provenance, temp_key,
)
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.trainer_utils import init_model, unbuffer_stdout, verify_tokenizer


def file_sha1(path, chunk=1 << 20):
    """按块流式哈希，不把整份权重读进内存。文件不存在时返回 None ——
    缺失哈希本身是可报告的状态，不该让校准跑不起来。"""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]

T_MIN, T_MAX = 0.05, 20.0

# 屏蔽槽在 logit 空间里的填充值。见 `fit_temperature`：必须是**有限**的大负数，
# 不能用 `-inf`（会让 log_temp 的梯度出现 0·∞ 而变 NaN）。
NEG_LP = -1e4


def parse_args():
    p = argparse.ArgumentParser(description="温度校准（只在 calib split 上拟合）")
    p.add_argument("--ckpt", default="out/decision/decision.pth")
    p.add_argument("--data", default="dataset/synth")
    p.add_argument("--tokenizer", default="model")
    p.add_argument("--out", default="out/calibration")
    p.add_argument("--split", default="calib")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--hidden_size", type=int, default=512)
    p.add_argument("--num_hidden_layers", type=int, default=8)
    p.add_argument("--crosstalk", action="store_true")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--lbfgs_steps", type=int, default=200)
    return p.parse_args()


def fit_temperature(log_p, target, mask, group, max_iter=200):
    """在指定分组内用 LBFGS 最小化 NLL，返回 (T, nll_before, nll_after)。

    参数化 `q = softmax(log_p / T)`，优化变量是 `log T`（见模块注释）。

    **用 `log_p` 而不是 `p` 入参**：屏蔽槽位的 `log_p` 必须是 `-inf`，`log_softmax`
    才把那些位置压到严格 0；若用 `p**(1/T)` 再归一化，0 要在除法里活下来，得多写
    一层 clip。数值上更稳的那条路更好走。

    **`mask` 必须真的用上，不能只是签名里的装饰。** 上游传进来的 `log_p` 是
    `log(clip(p, 1e-30))`，屏蔽槽因此是 `-69.08` 而不是 `-inf`；它虽然在 `t=0`
    处不贡献损失，却**会**通过 `log_softmax` 的归一化项影响其余槽位。T≈1 时
    `exp(-69)` 可以忽略，但 T 大时不是：T=20 时 `exp(-69/20)=0.032`，几个屏蔽槽
    就足以把归一化项抬高，把拟合往小 T 的方向拽。

    屏蔽值用 `NEG_LP = -1e4` 而**不是 `-inf`**：`-inf` 的泄漏确实零，但 `log_temp`
    的梯度里会出现 `0 · (-lp/T²)` = `0 · ∞` = NaN，LBFGS 一步就把 T 变成 NaN。
    `-1e4` 在 float64 下 `exp(-1e4/T)` 对任何 T≥0.05 都下溢到严格的 0，泄漏同样是
    零，而梯度保持有限。
    """
    lp = torch.tensor(log_p, dtype=torch.float64)
    t = torch.tensor(target, dtype=torch.float64)
    sel = torch.tensor(group, dtype=torch.bool)
    lp = lp.masked_fill(~torch.tensor(np.asarray(mask), dtype=torch.bool), NEG_LP)
    lp, t = lp[sel], t[sel]
    if lp.numel() == 0:
        return None

    def objective(log_temp):
        ls = torch.log_softmax(lp / log_temp.exp(), dim=-1)
        # 屏蔽槽上 `t=0` 且 `ls=-inf`，直接相乘得到 NaN。先把 `t=0` 处换成 0 再乘。
        return -(t * torch.where(t > 0, ls, torch.zeros_like(ls))).sum(-1).mean()

    before = float(objective(torch.zeros((), dtype=torch.float64)))
    log_temp = torch.zeros((), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_temp], lr=1.0, max_iter=max_iter,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(log_temp)
        loss.backward()
        return loss

    opt.step(closure)
    temp = float(torch.exp(log_temp).clamp(T_MIN, T_MAX))
    after = float(objective(torch.tensor(np.log(temp), dtype=torch.float64)))
    return temp, before, after


def main():
    unbuffer_stdout()
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        raise SystemExit("本项目的所有实测数字都基于 GPU；--device cpu 不在支持范围内。")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    # 同 eval_harness：词表不配套时温度会拟合到一份错的 p 上，且不会报错。
    verify_tokenizer(args.ckpt, args.tokenizer)
    config = DecisionConfig(
        hidden_size=args.hidden_size, num_hidden_layers=args.num_hidden_layers,
        vocab_size=len(tok), candidate_crosstalk=args.crosstalk,
        pad_token_id=tok.convert_tokens_to_ids("<pad>"),
        sep_token_id=tok.convert_tokens_to_ids("<sep>"),
    )
    model = init_model(MiniSystemOneForDecision, config, args.ckpt, device)

    path = os.path.join(args.data, f"{args.split}.jsonl")
    if not os.path.exists(path):
        raise SystemExit(f"缺少 {path} —— 校准必须在一份**没有参与训练**的 split 上做")
    ds = DecisionDataset(path, tok, max_len=args.max_len, augment_k=False)

    print("\n========== 收 calib 上的逐样本分布 ==========")
    t0 = time.time()
    col = collect(model, ds, device, args.batch_size, args.limit)
    k = effective_k(col["mask"])
    n = col["n"]
    print(f"  {n} 条，K_used {k.min()}–{k.max()}，{time.time()-t0:.0f} 秒")

    # log p 而不是 p：见 fit_temperature 的注释。
    log_p = np.log(np.clip(col["p"], 1e-30, None))
    target, mask = col["target"], col["mask"]
    prim = np.asarray(col["primitive"])

    print("\n========== 拟合 ==========")
    temp = {"global": None}
    rows = []

    groups = [("global", np.ones(n, dtype=bool))]
    for pr in sorted(set(prim.tolist())):
        groups.append((f"primitive:{pr}", prim == pr))
    kk = [temp_key(p, x) for p, x in zip(prim, k)]
    kk = np.asarray(kk)
    for key in sorted(set(kk.tolist())):
        groups.append((f"primitive_k:{key}", kk == key))

    fitted = {}
    for name, sel in groups:
        res = fit_temperature(log_p, target, mask, sel, args.lbfgs_steps)
        if res is None:
            continue
        T, before, after = res
        fitted[name] = T
        rows.append({"group": name, "n": int(sel.sum()), "T": T,
                     "nll_before": before, "nll_after": after})
        print(f"  {name:26s} n={int(sel.sum()):5d}  T={T:6.3f}  "
              f"NLL {before:.4f} -> {after:.4f}  (Δ {after-before:+.4f})")

    temp["global"] = fitted["global"]
    temp["primitive"] = {nm.split(":", 1)[1]: T for nm, T in fitted.items()
                         if nm.startswith("primitive:")}
    temp["primitive_k"] = {nm.split(":", 1)[1]: T for nm, T in fitted.items()
                           if nm.startswith("primitive_k:")}

    # 拟合完立刻在校准集本身上报 ECE —— **并且明确标注这是在拟合集上**。
    # 这个数字必然偏乐观，写进 JSON 是为了让读者能看到"过拟合有多大"，
    # 而不是为了让谁引用它。
    print("\n========== 校准集上的自评（拟合集，必然偏乐观）==========")
    from eval.eval_metrics import apply_temperature
    self_report = {}
    for gran in ("global", "primitive", "primitive_k"):
        if not temp[gran if gran != "primitive_k" else "primitive_k"]:
            continue
        pg = apply_temperature(col["p"], temp, prim, k, mask, gran)
        m = metrics_by_provenance(pg, target, mask, provenance=col["provenance"],
                                  is_ord=col["is_ord"],
                                  levels=np.where(col["levels"] >= 0, col["levels"], 0),
                                  counts=counts_array(col))
        self_report[gran] = m
        print(f"  {gran:12s} ECE {m['all']['ece']:.4f}  Brier {m['all']['brier']:.4f}  "
              f"NLL {m['all']['nll']:.4f}")
    raw = metrics_by_provenance(col["p"], target, mask, provenance=col["provenance"],
                                is_ord=col["is_ord"],
                                levels=np.where(col["levels"] >= 0, col["levels"], 0),
                                counts=counts_array(col))
    print(f"  {'未校准':12s} ECE {raw['all']['ece']:.4f}  Brier {raw['all']['brier']:.4f}  "
          f"NLL {raw['all']['nll']:.4f}")

    os.makedirs(args.out, exist_ok=True)
    out = dict(temp)
    out["meta"] = {
        "ckpt": os.path.abspath(args.ckpt),
        # **哈希是必需的，不是装饰。** 温度绑定到具体权重与具体词表：换一版
        # checkpoint 或重训 tokenizer 后，旧的 T.json 就静默失效了，而它会照常
        # 被 `eval_harness.py --temperature` 吃进去并产出一张看起来正常的可靠性图。
        # 把两个哈希写进来，读者才能判断一个 T.json 是否适用于手上的模型。
        "ckpt_sha1": file_sha1(args.ckpt),
        "tokenizer_sha1": file_sha1(os.path.join(args.tokenizer, "tokenizer.json")),
        "tokenizer_path": os.path.abspath(args.tokenizer),
        "split": args.split, "n": n,
        "k_buckets": [list(b) for b in K_BUCKETS],
        "fit_objective": "NLL on calib split (LBFGS on log T)",
        "clamp": [T_MIN, T_MAX],
        "fits": rows,
        "self_eval_on_fit_split": {
            "note": "在**拟合集**上算的，必然偏乐观；正式数字用 eval_harness.py",
            "uncalibrated": raw["all"], **{g: m["all"] for g, m in self_report.items()},
        },
    }
    out_path = os.path.join(args.out, "T.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\n  -> {out_path}")
    print("  下一步：python eval/eval_harness.py --ckpt "
          f"{args.ckpt} --temperature {out_path}")


if __name__ == "__main__":
    main()
