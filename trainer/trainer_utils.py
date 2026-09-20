"""
训练公共件：学习率、日志、checkpoint、模型初始化、OOM 重试。

沿用 MiniMind 的形状（单文件、argparse、AdamW、autocast + GradScaler、swanlab），
但有三处是**本项目特有且必须保留**的：

1. **`oom_retry`** —— 见下方注释。8GB 卡上这不是"锦上添花"，而是"跑不跑得完"。
2. **`peak_vram_warn`** —— Windows WDDM 在显存见底时**静默换页**而不 OOM：
   32×1024 的配置不会报错，只会慢 10 倍。没有这个告警，用户会以为是自己代码慢。
3. **`lm_checkpoint` 存双份**（推理用的半精度 + 续训用的完整优化器状态），
   后者用 `--save_optimizer` 控制，因为它单个就有 ~3 倍模型大小。
"""
import hashlib
import json
import os
import sys
import time

import torch


# ---------------------------------------------------------------------------
def get_lr(current_step, total_steps, lr):
    """余弦衰减：step=0 为 lr，半程为 0.55*lr，末端为 0.1*lr。

    调用方另乘线性 warmup，因此实际峰值还取决于 warmup 时长。
    """
    import math
    return lr * (0.1 + 0.45 * (1 + math.cos(math.pi * current_step / max(total_steps, 1))))


class Logger:
    """极简日志：可选 swanlab。没装 swanlab 也不能让训练挂掉。"""

    def __init__(self, log_dir=None, project="MiniSystemOne", name=None, use_swanlab=True):
        self.swanlab = None
        self.log_file = None
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = open(os.path.join(log_dir, "train.log"), "a", encoding="utf-8")
        # **非交互式（重定向 / 后台 / CI）时强制关掉 swanlab。** swanlab 首次运行会在
        # stdin 上弹一个三选一的问卷；后台跑时没有 stdin，于是它**阻塞等待**，训练一行
        # 日志都不出、GPU 也不转 —— 看起来像卡死，实际是在等一个永远不会到来的按键。
        # 实测踩过一次：一个后台冒烟跑到天荒地老，最后被 SIGSEGV 收尸。日志与
        # checkpoint 都由本类自己写，所以静默降级不丢任何训练产物。
        if use_swanlab and not sys.stdin.isatty():
            print("  （stdin 不是终端，跳过 swanlab —— 它会弹交互问卷并阻塞后台任务）")
            use_swanlab = False
        if use_swanlab:
            try:
                import swanlab
                swanlab.init(project=project, name=name, logdir=log_dir)
                self.swanlab = swanlab
            except Exception as e:                     # noqa: BLE001
                print(f"  （swanlab 不可用，仅写本地日志：{e}）")

    def log(self, data):
        line = "  ".join(f"{k}={v:.6g}" if isinstance(v, float) else f"{k}={v}"
                         for k, v in data.items())
        # `flush=True` 是给后台运行用的：重定向到文件时 stdout 变成块缓冲，日志会
        # 落后真实进度上千步。实测过一次——看着像训练卡死，实际只是 4 KB 还没攒满，
        # 而当时正在排一个与训练无关的 bug。
        print(line, flush=True)
        if self.log_file:
            self.log_file.write(line + "\n")
            self.log_file.flush()
        if self.swanlab is not None:
            self.swanlab.log(data)

    def finish(self):
        if self.log_file:
            self.log_file.close()
        if self.swanlab is not None:
            self.swanlab.finish()


def unbuffer_stdout():
    """把 stdout 切成行缓冲。长跑脚本的 `main()` 第一行就该调它。

    README 的复现流程把输出重定向到日志（`python ... > out/x.log`），而**重定向时
    stdout 是块缓冲**：一个要跑四十分钟的脚本在 8 KB 攒满之前一个字都不吐，进程活着、
    CPU 在转、输出文件 0 字节 —— 看起来与卡死完全一样。实测在 audit 上踩过一次。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)


# checkpoint 自描述格式的版本号。将来改结构就 +1；读取端据此判断能否理解。
CKPT_FORMAT = 1

# 决定**参数形状**的字段。载入时若 checkpoint 里记的值与本次 config 不符，
# 必须立刻报错 —— 理由见 `init_model`。
SHAPE_KEYS = ("hidden_size", "num_hidden_layers", "num_attention_heads",
              "num_key_value_heads", "head_dim", "intermediate_size",
              "vocab_size", "num_segments")


def _self_desc(blob):
    """从 checkpoint 原始对象里摘出自描述三段。老格式（扁平 state_dict）返回 {}。"""
    if not isinstance(blob, dict):
        return {}
    return {k: blob[k] for k in ("format", "config", "meta") if k in blob}


def ckpt_info(path):
    """读 checkpoint 的自描述信息 → `{"format":…, "config":…, "meta":…}`。

    老格式（`CKPT_FORMAT` 之前的扁平 state_dict）没有这些键，于是返回空 dict ——
    调用方据此**降级**，而不是把它当成错误。权重文件读不动（损坏、被截断）时
    同样返回空 dict，让后续的 `load_state_dict` 去报真正的错。
    """
    if not path or not os.path.exists(path):
        return {}
    try:
        return _self_desc(torch.load(path, map_location="cpu", weights_only=True))
    except Exception:
        return {}


def data_manifest(data_dir):
    """读数据集 manifest（`gen_version` / `tokenizer_sha1` / 各 split 条数）。缺失返回 {}。

    走 JSON 而不是 import `dataset.synth`：后者的导入链牵扯 `datasets`，而本仓库在
    Windows 上有一条硬规矩 —— `datasets`(pyarrow) 必须在 `torch` 之前 import，
    否则进程静默退出且零输出。只为了读一个版本号去冒那个险不值得。
    """
    path = os.path.join(data_dir or "", "manifest.json")
    if not data_dir or not os.path.exists(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def file_sha1(path, n=16):
    """按块流式哈希，不把整份文件读进内存。缺失文件返回 None。"""
    if not path or not os.path.exists(path):
        return None
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def verify_tokenizer(checkpoint, tokenizer_dir):
    """核对 checkpoint 记的词表哈希与即将使用的词表是否一致。

    **权重与词表不配套时不会自己报错。** 两者 vocab 都是 6400，embedding 形状完全
    一样，载入成功、评测照跑、指标照出 —— 只是每个 token id 指向另一个词，所有数字
    都是噪声。这是本仓库里最隐蔽的一类失败，比形状不符还难发现（形状不符至少会
    触发 `init_model` 的过半缺失检查）。所以在这里硬拦。

    老格式 checkpoint 没记哈希 → 打印一行"无法核对"就放过，不假装校验过。
    """
    meta = ckpt_info(checkpoint).get("meta") or {}
    want = meta.get("tokenizer_sha1")
    if not want:
        print(f"  （{os.path.basename(checkpoint)} 未记录词表哈希 —— 老格式，无法核对）")
        return True
    got = file_sha1(os.path.join(tokenizer_dir, "tokenizer.json"))
    if got != want:
        raise SystemExit(
            f"词表与权重不配套：checkpoint 记的是 {want}，"
            f"{tokenizer_dir}/tokenizer.json 是 {got}。\n"
            f"  这是**静默**失败 —— vocab 都是 6400，embedding 形状一致，载入不会报错，"
            f"但每个 token id 指向另一个词，全部指标都是噪声。\n"
            f"  换成与权重配套的那份 tokenizer，或改用对应的 checkpoint。")
    print(f"  词表哈希核对通过 {got}")
    return True


def init_model(model_cls, config, checkpoint=None, device="cuda", strict=False):
    """新建模型；给了 checkpoint 就加载。

    `strict=False` 是刻意的默认：Stage 1 的 `MiniSystemOneForMaskedLM` 比决策模型
    多一个 `lm_head`、少一个 `head`，两者互载时必然有缺失/多余键。这正是设计意图
    （"编码器从 0 预训练、决策头是新的"），所以不能因为 strict 报错就以为出事了。

    **自描述校验**：checkpoint 里记着训练时的 config（`save_checkpoint(config=…)`），
    这里把决定参数形状的那几个字段与本次 config 逐项比对，不符就直接退出。这一层是
    对下面"过半缺失"检查的补强 —— 那份检查只在**大部分**参数没载入时触发，而
    `head_dim` 或 `intermediate_size` 写错时，缺失比例可能不到一半，于是模型带着
    一部分随机初始化安静地跑完评测。
    """
    model = model_cls(config)
    if checkpoint and os.path.exists(checkpoint):
        blob = torch.load(checkpoint, map_location="cpu", weights_only=True)
        info = _self_desc(blob)
        sd = blob.get("model", blob) if isinstance(blob, dict) else blob

        ck_cfg = info.get("config") or {}
        if ck_cfg:
            bad = [(k, ck_cfg.get(k), getattr(config, k, None)) for k in SHAPE_KEYS
                   if k in ck_cfg and ck_cfg[k] != getattr(config, k, None)]
            if bad:
                detail = "；".join(f"{k}: checkpoint {a} ≠ 本次 {b}" for k, a, b in bad)
                raise SystemExit(
                    f"{checkpoint} 记录的模型形状与本次 config 不符：{detail}\n"
                    f"  这就是 h512 的权重配 h768 config 的情况 —— 继续跑不会报错，"
                    f"只会得到一个部分随机初始化的模型，评测数字全是噪声。\n"
                    f"  检查 --hidden_size / --num_hidden_layers 是否与训练时一致。")

        missing, unexpected = model.load_state_dict(sd, strict=strict)
        if not strict:
            print(f"  载入 {checkpoint}")
            m = info.get("meta") or {}
            if m:
                bits = [f"{k}={m[k]}" for k in
                        ("stage", "step", "n_params", "tokenizer_sha1", "gen_version")
                        if k in m]
                print(f"    " + "  ".join(str(b) for b in bits))
            if missing:
                print(f"    未载入（新参数）：{len(missing)} 项，如 {list(missing)[:3]}")
            if unexpected:
                print(f"    丢弃（属于别的阶段）：{len(unexpected)} 项，"
                      f"如 {list(unexpected)[:3]}")
            # 跨阶段互载本来就会有缺失/多余键（见上方注释），所以默认不报错。
            # 但**过半缺失**不是跨阶段，是把 checkpoint 载进了形状不对的模型 ——
            # 通常是 h512 的权重配了 h768 的 config（或反过来）。这时 strict=False
            # 会安静地留下一个随机初始化的模型，评测照跑、数字照出，只是全是噪声。
            total = len(model.state_dict())
            if len(missing) > 0.5 * total:
                raise SystemExit(
                    f"{checkpoint} 有 {len(missing)}/{total} 个参数没载入 —— 这不是"
                    f"跨阶段的正常缺失，而是模型形状与 checkpoint 不符。"
                    f"检查 --hidden_size / --num_hidden_layers 是否与训练时一致。"
                )
    else:
        print("  随机初始化（从 0 训练）")
    return model.to(device)


def save_checkpoint(model, path, optimizer=None, scaler=None, step=0,
                    config=None, meta=None, training_state=None):
    """两件事分开写在两个文件里，因为它们用途不同。

    只存半精度权重（`{name}.pth`）给推理/评测用；完整状态（含优化器动量，约 3 倍
    大小）只在 `save_optimizer` 时另存 `{name}_opt.pth`。混在一起会让每个推理用的
    checkpoint 都白背 3 倍体积。

    **`config` / `meta` 与权重写在同一个文件里，让 checkpoint 自描述。** 没有它们，
    下载权重的人只能回 README 手抄 `h=512 / L=8`，而且没有任何办法核对这份权重和
    `model/tokenizer.json` 是不是配套 —— 见 `verify_tokenizer` 为什么那件事很危险。
    两者都只占几百字节。

    `meta` 里应放（都是可 JSON 化的基本类型，`weights_only=True` 才读得回来）：
    `stage` / `tokenizer_sha1` / `gen_version` / `n_params` / `max_len` / `trained_on`。
    step 会自动并入 `meta`。
    """
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    blob = {"format": CKPT_FORMAT,
            "model": {k: v.half() for k, v in model.state_dict().items()}}
    if config is not None:
        blob["config"] = config.to_dict()
    if meta is not None:
        blob["meta"] = dict(meta, step=step)
    tmp = path + ".tmp"
    torch.save(blob, tmp)
    os.replace(tmp, path)
    if optimizer is not None:
        from trainer.training_state import capture_rng
        opt_path = path.replace(".pth", "_opt.pth")
        tmp = opt_path + ".tmp"
        opt_blob = dict(blob, model=model.state_dict(), optimizer=optimizer.state_dict(),
                        scaler=scaler.state_dict() if scaler is not None else None, step=step)
        if training_state is not None:
            opt_blob['training_state'] = dict(training_state, rng=capture_rng())
        torch.save(opt_blob, tmp)
        os.replace(tmp, opt_path)


class oom_retry:
    """前向/反向 OOM 时对半砍 batch 并重试，最多 `retries` 次。

    这不是"防御性编程"，是 8GB 卡上的必需品：候选数 K 与 state 长度都是变的，
    峰值显存因此在样本之间波动。没有它，一次超标的 batch 会毁掉一整晚的训练；
    有了它，那次 batch 变小一点，训练继续。

    显存没有全部释放时 `torch.cuda.empty_cache()` 是必须的 —— 否则重试会立刻
    再次 OOM（缓存里的碎片不会自动还给驱动）。
    """

    def __init__(self, retries=3):
        self.retries = retries

    def __call__(self, fn, batch, split_fn):
        cur = batch
        for attempt in range(self.retries + 1):
            try:
                return fn(cur)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                if attempt == self.retries:
                    raise
                if len(cur) <= 1:
                    raise
                half = max(1, len(cur) // 2)
                print(f"    OOM：batch {len(cur)} → {half}，重试")
                cur = split_fn(cur, half)
        return None


def peak_vram_warn(limit_gb=6.5, tag=""):
    """训练峰值显存硬预算。超了只告警不中止 —— 但必须响，因为**不会 OOM**。

    Windows WDDM 在显存不足时把页面换到共享内存，代价是慢 10 倍且不报错。
    实测：26M 模型 32×1024 峰值 8.3 GB，不 OOM，但 step 时间从 297 ms 变成 5936 ms。
    因此这条告警是"运行莫名变慢"的唯一解释来源。
    """
    if not torch.cuda.is_available():
        return 0.0
    peak = torch.cuda.max_memory_allocated() / 1e9
    if peak > limit_gb:
        print(f"    **显存告警**{tag}：峰值 {peak:.2f} GB > 预算 {limit_gb} GB。"
              f"不会 OOM，但可能已被换页到共享内存，速度会大幅下降。"
              f"减小 batch / 用 --use_checkpoint / 降 max_len。")
    return peak
