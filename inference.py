import argparse
from contextlib import redirect_stdout
import json
import math
import sys

import numpy as np
import torch

from eval.eval_inference import load_decision, load_temperature
from eval.eval_metrics import apply_temperature, expected_score
from model.serialize import collate_packed, serialize_decision


def validate_request(request):
    if not isinstance(request, dict):
        raise ValueError("请求必须是 JSON 对象")
    schema = request.get("schema")
    if not isinstance(schema, dict) or schema.get("primitive") not in ("noul", "choice", "score"):
        raise ValueError("schema.primitive 必须是 noul、choice 或 score")
    for field in ("state", "question"):
        if not isinstance(request.get(field), str):
            raise ValueError(f"{field} 必须是字符串")
    if not request["question"].strip():
        raise ValueError("question 不能为空")
    candidates = request.get("candidates")
    if not isinstance(candidates, list) or not 2 <= len(candidates) <= 255:
        raise ValueError("candidates 必须包含 2–255 项")
    labels = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("每个候选必须是含 text 和 label 的对象")
        for field in ("text", "label"):
            if not isinstance(candidate.get(field), str) or not candidate[field].strip():
                raise ValueError(f"候选的 {field} 必须是非空字符串")
        labels.append(candidate["label"])
    if len(set(labels)) != len(labels):
        raise ValueError("候选 label 必须唯一")
    if schema["primitive"] == "noul":
        if len(candidates) != 2 or schema.get("positive_label") not in labels:
            raise ValueError("Noul 需要两个候选及匹配 label 的 positive_label")
    if schema["primitive"] == "score":
        if len(candidates) > 10:
            raise ValueError("Score 需要 2–10 个等级候选")
        levels = []
        for candidate in candidates:
            meta = candidate.get("meta")
            level = meta.get("level") if isinstance(meta, dict) else None
            if (isinstance(level, bool) or not isinstance(level, (int, float))
                    or not math.isfinite(level)):
                raise ValueError("Score 每个候选需要有限数值 meta.level")
            levels.append(level)
        if len(set(levels)) != len(levels):
            raise ValueError("Score 等级不能重复")
    return schema["primitive"]


class Predictor:
    def __init__(self, checkpoint, tokenizer="model", temperature=None,
                 device="cuda", max_len=None, granularity="global"):
        if granularity not in ("global", "primitive", "primitive_k"):
            raise ValueError("未知的温度粒度")
        with redirect_stdout(sys.stderr):
            self.tokenizer, self.model, meta = load_decision(checkpoint, tokenizer, device)
        self.device = device
        self.max_len = max_len if max_len is not None else meta.get("max_len", 1024)
        if not isinstance(self.max_len, int) or self.max_len <= 0:
            raise ValueError("max_len 必须是正整数")
        if self.max_len > self.model.config.max_position_embeddings:
            raise ValueError("max_len 超过模型位置预算")
        self.temperature = load_temperature(temperature, checkpoint, tokenizer) if temperature else None
        self.granularity = granularity
        if self.temperature is not None and granularity != "global" and not self.temperature.get(granularity):
            self.granularity = "global"

    @torch.inference_mode()
    def predict(self, request):
        primitive = validate_request(request)
        candidates = request["candidates"]
        sep_id = self.tokenizer.convert_tokens_to_ids("<sep>")
        ids, seg, cid, spans = serialize_decision(
            self.tokenizer, request["state"], request["question"],
            [c["text"] for c in candidates], sep_id,
        )
        if len(ids) > self.max_len:
            raise ValueError(f"输入共 {len(ids)} tokens，超过预算 {self.max_len}；请缩短 state/question/候选。")
        batch = collate_packed(
            [{"input_ids": ids, "seg_id": seg, "cand_id": cid, "cand_spans": spans}],
            self.tokenizer.pad_token_id, sep_id, device=self.device,
        )
        p, _ = self.model.decide(**{key: batch[key] for key in (
            "input_ids", "seg_id", "cand_id", "cand_span", "cand_mask", "prefix_mask")})
        probabilities = p.cpu().numpy().astype(np.float64)
        if self.temperature is not None:
            probabilities = apply_temperature(
                probabilities, self.temperature, [primitive], [len(candidates)],
                granularity=self.granularity,
            )
        probabilities = probabilities[0]
        selected = int(probabilities.argmax())
        result = {
            "primitive": primitive,
            "candidates": [dict(label=c["label"], text=c["text"], probability=float(value))
                           for c, value in zip(candidates, probabilities)],
            "selected_label": candidates[selected]["label"],
            "confidence": float(probabilities[selected]),
            "calibration": self.granularity if self.temperature is not None else "uncalibrated",
            "input_tokens": len(ids),
        }
        if primitive == "noul":
            positive = next(i for i, c in enumerate(candidates)
                            if c["label"] == request["schema"]["positive_label"])
            result["positive_probability"] = float(probabilities[positive])
        elif primitive == "score":
            levels = np.array([[c["meta"]["level"] for c in candidates]])
            result["expected_score"] = float(expected_score(probabilities[None, :], levels)[0])
        return result


def main():
    parser = argparse.ArgumentParser(description="MiniSystemOne 本地决策推理（CUDA）")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--tokenizer", default="model")
    parser.add_argument("--input", default="-", help="JSON 文件；- 从 stdin 读取")
    parser.add_argument("--temperature")
    parser.add_argument("--granularity", choices=("global", "primitive", "primitive_k"), default="global")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_len", type=int)
    args = parser.parse_args()
    try:
        if args.input == "-":
            request = json.load(sys.stdin)
        else:
            with open(args.input, encoding="utf-8") as f:
                request = json.load(f)
        validate_request(request)
        predictor = Predictor(args.ckpt, args.tokenizer, args.temperature,
                              args.device, args.max_len, args.granularity)
        print(json.dumps(predictor.predict(request), ensure_ascii=False, allow_nan=False))
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(2, f"推理失败：{error}\n")


if __name__ == "__main__":
    main()
