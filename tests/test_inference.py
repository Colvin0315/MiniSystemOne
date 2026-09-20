import json
from pathlib import Path
import tempfile
import unittest

import datasets
import torch
from transformers import AutoTokenizer

from model.inference import DecisionPredictor
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.trainer_utils import save_checkpoint, file_sha1

ROOT = Path(__file__).resolve().parents[1]


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.ckpt = str(Path(cls.temp.name) / 'decision.pth')
        config = DecisionConfig(hidden_size=32, num_hidden_layers=1, vocab_size=6400)
        model = MiniSystemOneForDecision(config)
        save_checkpoint(model, cls.ckpt, config=config, meta=dict(stage='decision',
            tokenizer_sha1=file_sha1(str(ROOT / 'model/tokenizer.json'))))
        cls.predictor = DecisionPredictor(cls.ckpt, str(ROOT / 'model'), device='cpu')

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_all_three_primitives_and_score_order(self):
        base = dict(state='All checks passed.', question='Is it successful?')
        noul = self.predictor.predict(dict(base, primitive='noul'))
        self.assertEqual(noul['p_true'], noul['probabilities'][0]['probability'])
        choice = self.predictor.predict(dict(base, primitive='choice', candidates=['pass', 'fail']))
        self.assertAlmostEqual(sum(v['probability'] for v in choice['probabilities']), 1., places=6)
        a = self.predictor.predict(dict(base, primitive='score', candidates=['high', 'low', 'medium'], levels=[3, 1, 2]))
        b = self.predictor.predict(dict(base, primitive='score', candidates=['low', 'medium', 'high'], levels=[1, 2, 3]))
        self.assertAlmostEqual(a['expected_score'], b['expected_score'], places=5)
        json.dumps(a, allow_nan=False)

    def test_invalid_requests(self):
        base = dict(state='state', question='question', primitive='choice')
        for candidates in ([], ['a'], ['a', 'a'], [''], [1, 2]):
            with self.assertRaises(ValueError):
                self.predictor.predict(dict(base, candidates=candidates))
        with self.assertRaises(ValueError):
            self.predictor.predict(dict(base, primitive='score', candidates=['a', 'b'], levels=[1, 1]))
        with self.assertRaises(ValueError):
            self.predictor.predict(dict(base, primitive='noul', candidates=['false', 'true']))

    def test_truncation_is_reported(self):
        result = self.predictor.predict(dict(state='Evidence ' * 100, question='Pass?', primitive='noul'), max_state_tokens=8)
        self.assertTrue(result['state_truncated'])

    def test_calibration_must_match_checkpoint(self):
        path = Path(self.temp.name) / 'T.json'
        path.write_text(json.dumps({'global': 1., 'meta': {'ckpt_sha1': 'wrong', 'tokenizer_sha1': 'wrong'}}))
        with self.assertRaises(ValueError):
            DecisionPredictor(self.ckpt, str(ROOT / 'model'), device='cpu', calibration=str(path))

    def test_checkpoint_without_tokenizer_fingerprint_is_rejected(self):
        blob = torch.load(self.ckpt, weights_only=True)
        del blob['meta']['tokenizer_sha1']
        path = Path(self.temp.name) / 'no_hash.pth'
        torch.save(blob, path)
        with self.assertRaisesRegex(ValueError, 'tokenizer_sha1'):
            DecisionPredictor(str(path), str(ROOT / 'model'), device='cpu')
