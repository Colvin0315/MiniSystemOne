"""Small public inference API using the same tokenizer and candidate scorer as training."""
import contextlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from eval.eval_metrics import apply_temperature
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from model.serialize import encode_text, truncate_head_tail
from trainer.trainer_utils import ckpt_info, file_sha1, init_model, verify_tokenizer


class DecisionPredictor:
    def __init__(self, checkpoint, tokenizer='model', device=None, calibration=None):
        if not Path(checkpoint).is_file():
            raise ValueError(f'Checkpoint not found: {checkpoint}')
        info = ckpt_info(checkpoint)
        if not info.get('config'):
            raise ValueError('Inference requires a self-describing checkpoint with config')
        if info.get('meta', {}).get('stage') != 'decision':
            raise ValueError('Expected a decision checkpoint, not MLM weights')
        if not info['meta'].get('tokenizer_sha1'):
            raise ValueError('Checkpoint must record tokenizer_sha1 for verified inference')
        config = DecisionConfig(**info['config'])
        if config.candidate_crosstalk or not config.prefix_blocked:
            raise ValueError('Chunked inference requires candidate_crosstalk=False and prefix_blocked=True')
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        with contextlib.redirect_stdout(sys.stderr):
            verify_tokenizer(checkpoint, tokenizer)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, local_files_only=True)
            self.model = init_model(MiniSystemOneForDecision, config, checkpoint, self.device, strict=True).eval()
        self.sep = self.tokenizer.convert_tokens_to_ids('<sep>')
        self.trunc = self.tokenizer.convert_tokens_to_ids('<trunc>')
        self.calibration = None
        if calibration:
            with open(calibration, encoding='utf-8-sig') as f:
                temp = json.load(f)
            meta = temp.get('meta', {})
            for key, path in [('ckpt_sha1', checkpoint), ('tokenizer_sha1', str(Path(tokenizer) / 'tokenizer.json'))]:
                digest = meta.get(key)
                if not isinstance(digest, str) or len(digest) not in (12, 16, 40) or file_sha1(path, len(digest)) != digest:
                    raise ValueError(f'Calibration {key} does not match this model/tokenizer')
            values = [temp.get('global'), *temp.get('primitive', {}).values(), *temp.get('primitive_k', {}).values()]
            if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in values):
                raise ValueError('Calibration temperatures must be positive finite numbers')
            self.calibration = temp

    @torch.inference_mode()
    def predict(self, request, max_state_tokens=512, chunk=16):
        if not isinstance(request, dict):
            raise ValueError('Request must be a JSON object')
        for field in ('state', 'question'):
            if not isinstance(request.get(field), str) or not request[field].strip():
                raise ValueError(f'{field} must be a nonempty string')
        primitive = request.get('primitive')
        if primitive not in ('noul', 'choice', 'score'):
            raise ValueError('primitive must be noul, choice or score')
        candidates = request.get('candidates', ['yes', 'no'] if primitive == 'noul' else None)
        if (not isinstance(candidates, list) or not 2 <= len(candidates) <= 255
                or any(not isinstance(c, str) or not c.strip() for c in candidates)):
            raise ValueError('candidates must contain 2–255 nonempty strings')
        if len(set(c.strip() for c in candidates)) != len(candidates):
            raise ValueError('candidates must be unique')
        if primitive == 'noul' and candidates != ['yes', 'no']:
            raise ValueError('Noul uses candidates ["yes", "no"] in that order')
        levels = request.get('levels')
        if primitive == 'score':
            if (len(candidates) > 10 or not isinstance(levels, list) or len(levels) != len(candidates)
                    or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in levels)
                    or len(set(levels)) != len(levels)):
                raise ValueError('Score needs 2–10 candidates with matching distinct finite nonnegative levels')
        if max_state_tokens < 2 or chunk < 1:
            raise ValueError('max_state_tokens must be >= 2 and chunk >= 1')
        state = encode_text(self.tokenizer, request['state'])
        was_truncated = len(state) > max_state_tokens
        state = truncate_head_tail(state, max_state_tokens, self.trunc)
        question = encode_text(self.tokenizer, request['question'])
        encoded = [encode_text(self.tokenizer, c) or [self.sep] for c in candidates]
        if len(question) > 128 or any(len(c) > 128 for c in encoded):
            raise ValueError('Question and each candidate are limited to 128 tokens in this entry point')
        if len(state) + len(question) + max(map(len, encoded)) + 3 > self.model.config.max_position_embeddings:
            raise ValueError('Request exceeds model position budget; reduce max_state_tokens')
        probs, _ = self.model.decide_chunked(state, question, encoded, self.sep, chunk=chunk)
        p = probs.float().cpu().numpy()
        if self.calibration is not None:
            p = apply_temperature(p, self.calibration, [primitive], [len(candidates)])
        p = p[0]
        best = int(np.argmax(p))
        result = dict(primitive=primitive, choice=candidates[best], confidence=float(p[best]),
                      probabilities=[dict(candidate=c, probability=float(v)) for c, v in zip(candidates, p)],
                      state_truncated=was_truncated, temperature_applied=self.calibration is not None)
        if primitive == 'noul':
            result['p_true'] = float(p[0])
        if primitive == 'score':
            result['expected_score'] = float(np.dot(p, levels))
        return result
