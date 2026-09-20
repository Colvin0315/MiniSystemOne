"""Behavior regressions; no trained weights or GPU required."""
import unittest
import json
import tempfile
from pathlib import Path

import datasets  # Windows DLL initialization order
import numpy as np
import torch

from dataset.decision_dataset import fit_state, DecisionDataset, collate_decision
from eval.eval_inference import collect
from eval.eval_metrics import ordinal_mae, risk_coverage_curve, compute_metrics
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from model.serialize import pack_example, collate_packed


class CharacterTokenizer:
    def convert_tokens_to_ids(self, token):
        return {'<pad>': 0, '<sep>': 3, '<trunc>': 5}[token]

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


class CorrectnessTests(unittest.TestCase):
    def test_fractional_levels_survive_data_collation_and_collection(self):
        rec = dict(state='A', question='B', schema={'primitive': 'score'},
                   candidates=[{'text': 'C', 'meta': {'level': .1}}, {'text': 'D', 'meta': {'level': .9}}],
                   target={'p': [.3, .7], 'provenance': 'marginalized'})
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'test.jsonl'
            path.write_text(json.dumps(rec), encoding='utf-8')
            ds = DecisionDataset(path, CharacterTokenizer(), augment_k=False)
            batch = collate_decision([ds[0]])
            np.testing.assert_allclose(batch['level_idx'].numpy(), [[.1, .9]])
            model = MiniSystemOneForDecision(DecisionConfig(hidden_size=32, num_hidden_layers=1, vocab_size=128))
            col = collect(model, ds, 'cpu', batch_size=1)
            np.testing.assert_allclose(col['levels'], [[.1, .9]])

    def test_truncate_discards_low_priority_section_first(self):
        sections = [{"priority": 0, "text": "aaaa"}, {"priority": 1, "text": "BBBBBB"}]
        self.assertEqual(fit_state(sections, 7, CharacterTokenizer(), 999, [10]), [66] * 6)

    def test_truncate_keeps_everything_when_it_fits(self):
        sections = [{"priority": 0, "text": "aa"}, {"priority": 1, "text": "BB"}]
        self.assertEqual(fit_state(sections, 5, CharacterTokenizer(), 999, [10]), [97, 97, 10, 66, 66])

    def test_ordinal_loss_uses_levels_and_ignores_padding(self):
        p = torch.tensor([[.7, .2, .1, 0.]])
        t = torch.tensor([[0., 0., 1., 0.]])
        levels = torch.tensor([[1, 2, 3, -1]])
        mask = levels > 0
        for order in ([0, 1, 2, 3], [3, 0, 2, 1]):
            ix = torch.tensor(order)
            _, d = MiniSystemOneForDecision.compute_loss(
                p[:, ix].clamp_min(1e-12).log(), t[:, ix], mask[:, ix], torch.ones(1),
                level_idx=levels[:, ix])
            self.assertAlmostEqual(d['ord'].item(), .65, places=6)

    def test_ordinal_metric_is_wasserstein_in_level_units(self):
        p = np.array([[.7, .2, .1, 0.]])
        t = np.array([[0., 0., 1., 0.]])
        levels = np.array([[1, 2, 3, -1]])
        for order in ([0, 1, 2, 3], [3, 0, 2, 1]):
            self.assertAlmostEqual(ordinal_mae(p[:, order], t[:, order], levels[:, order], levels[:, order] > 0), 1.6)
        # Uneven levels: mass moved from 1 to 5 costs 4, not one candidate slot.
        self.assertAlmostEqual(ordinal_mae([[1., 0.]], [[0., 1.]], [[1, 5]]), 4.)

    def test_mixed_primitives_slice_levels_with_score_rows(self):
        result = compute_metrics(np.array([[1., 0.], [1., 0.]]),
                                 np.array([[1., 0.], [0., 1.]]),
                                 np.ones((2, 2), dtype=bool), is_ord=np.array([False, True]),
                                 levels=np.array([[-1, -1], [1, 5]]))
        self.assertAlmostEqual(result['ordinal_mae'], 4.)

    def test_abstention_is_not_covered_and_soft_targets_are_used(self):
        p = np.array([[.99, .01], [.2, .8]])
        t = np.array([[1., 0.], [.25, .75]])
        coverage, risk, threshold = risk_coverage_curve(p, t, abstain_idx=0)
        np.testing.assert_allclose(coverage, [.5])
        np.testing.assert_allclose(risk, [.25])
        np.testing.assert_allclose(threshold, [.8])
        self.assertEqual(len(risk_coverage_curve(p[:1], t[:1], abstain_idx=0)[0]), 0)

    def test_equal_confidence_thresholds_group_ties(self):
        cov, risk, thr = risk_coverage_curve(np.array([[.8, .2], [.2, .8]]),
                                           np.array([[1., 0.], [1., 0.]]))
        np.testing.assert_allclose(cov, [1.])
        np.testing.assert_allclose(risk, [.5])

    def test_forward_honors_brier_normalize(self):
        model = MiniSystemOneForDecision(DecisionConfig(hidden_size=32, num_hidden_layers=1,
            num_attention_heads=4, num_key_value_heads=2, vocab_size=32)).eval()
        ids, seg, cid, spans = pack_example([1], [2], [[3], [4]], 5)
        batch = collate_packed([dict(input_ids=ids, seg_id=seg, cand_id=cid,
                                    cand_spans=spans, target=[1., 0.])], 0, 5)
        with torch.no_grad():
            a = model(**batch).loss_dict['brier'].item()
            b = model(**batch, brier_normalize=True).loss_dict['brier'].item()
        self.assertAlmostEqual(b * 2, a, places=6)


if __name__ == '__main__':
    unittest.main()
