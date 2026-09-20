"""NumPy metrics run by default; opt in to the CUDA-only forward test with --cuda."""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from eval.eval_metrics import (
    accuracy, compute_metrics, distribution_l2, ece, ece_annotation_reference,
    expected_brier, mask_normalize, metrics_by_provenance, nll,
    risk_coverage_curve, soft_accuracy,
)

RUN_CUDA = "--cuda" in sys.argv
if RUN_CUDA:
    sys.argv.remove("--cuda")


class MetricsTests(unittest.TestCase):
    def test_analytic_expected_brier(self):
        p = np.array([[0.8, 0.2], [0.1, 0.9]])
        t = np.array([[0.6, 0.4], [0.3, 0.7]])
        self.assertAlmostEqual(distribution_l2(p, t), 0.08)
        self.assertAlmostEqual(expected_brier(p, t), 0.53)
        eye = np.eye(2)
        direct = np.mean([sum(t[i, y] * np.sum((p[i] - eye[y]) ** 2)
                              for y in range(2)) for i in range(2)])
        self.assertAlmostEqual(expected_brier(p, t), direct)
        self.assertAlmostEqual(expected_brier(p, t),
                               distribution_l2(p, t) + np.mean(1 - (t*t).sum(-1)))

    def test_hard_labels_can_be_calibrated(self):
        p = np.tile([0.8, 0.2], (10, 1))
        t = np.eye(2)[[0] * 8 + [1] * 2]
        self.assertAlmostEqual(ece(p, t), 0.0)
        self.assertAlmostEqual(accuracy(p, t), 0.8)
        self.assertAlmostEqual(expected_brier(p, t), distribution_l2(p, t))
        self.assertAlmostEqual(expected_brier(p, t), 0.32)
        self.assertGreater(ece(np.tile([0.99, 0.01], (10, 1)), t), 0.18)
        self.assertLess(nll(p, t), nll(np.tile([0.99, 0.01], (10, 1)), t))

    def test_mask_both_distributions_and_normalize(self):
        p = np.array([[8, 2, np.nan], [1, 1, 2]])
        t = np.array([[3, 2, 999], [0, 2, 2]])
        mask = np.array([[1, 1, 0], [1, 1, 1]], bool)
        pp = np.array([[0.8, 0.2, 0], [0.25, 0.25, 0.5]])
        tt = np.array([[0.6, 0.4, 0], [0, 0.5, 0.5]])
        np.testing.assert_allclose(mask_normalize(p, mask), pp)
        d = ((pp-tt)**2).sum(-1)
        self.assertAlmostEqual(distribution_l2(p, t, mask), d.mean())
        self.assertAlmostEqual(distribution_l2(p, t, mask, normalize=True),
                               np.mean(d / [2, 3]))
        self.assertAlmostEqual(expected_brier(p, t, mask, normalize=True),
                               np.mean((d + 1 - (tt*tt).sum(-1)) / [2, 3]))
        self.assertAlmostEqual(distribution_l2(pp, tt, normalize=True), d.mean()/3)
        self.assertAlmostEqual(expected_brier(pp, tt, normalize=True),
                               expected_brier(pp, tt)/3)
        self.assertAlmostEqual(nll(p, t, mask), nll(pp, tt))
        self.assertAlmostEqual(accuracy(p, t, mask), accuracy(pp, tt))
        np.testing.assert_allclose(soft_accuracy(p, t, mask), [0.6, 0.5])
        with self.assertRaises(ValueError):
            mask_normalize([[0, 0]])
        with self.assertRaises(ValueError):
            mask_normalize([[0.5, 0.5]], [[False, False]])

    def test_provenance_and_new_keys(self):
        p = np.array([[0.8, 0.2], [0.6, 0.4], [0.5, 0.5]])
        t = np.array([[1, 0], [0.6, 0.4], [0.5, 0.5]])
        res = metrics_by_provenance(p, t, provenance=["hard", "explicit_rng", "tie_set"],
                                    counts=np.array([0, 0, 10]))
        self.assertEqual(res["soft_targets"]["n"], 2)
        self.assertIn("ece", res["hard"])
        self.assertNotIn("calibration", res)
        for group in res.values():
            self.assertIn("distribution_l2", group)
            self.assertIn("expected_brier", group)
            self.assertTrue({"brier", "ece_corrected", "ece_noise_floor"}.isdisjoint(group))
        self.assertEqual(res["all"]["ece_annotation_reference_n"], 1)
        self.assertIsNone(res["hard"]["ece_annotation_reference"])

    def test_annotation_reference_uses_only_positive_counts(self):
        p = np.array([[0.6, 0.4], [0.99, 0.01], [0.8, 0.2], [0.55, 0.45]])
        counts = [100, 0, 12, None]
        actual = ece_annotation_reference(p, counts, seed=42)
        subset = ece_annotation_reference(p[[0, 2]], [100, 12], seed=42)
        self.assertEqual(actual, subset)
        self.assertIsNone(ece_annotation_reference(p, [0, None, 0, np.nan]))
        self.assertIsNone(ece_annotation_reference(p, None))
        with self.assertRaises(ValueError):
            ece_annotation_reference(p, [1.5, 0, 0, 0])
        out = compute_metrics(p, p, counts=counts)
        self.assertEqual(out["ece_annotation_reference_n"], 2)
        self.assertAlmostEqual(out["ece"], 0.0)
        self.assertGreater(out["ece_annotation_reference"], 0.0)

    def test_risk_coverage_ties_and_abstain(self):
        p = np.array([[0.1, 0.9], [0.8, 0.2], [0.8, 0.2], [0.6, 0.4]])
        t = np.array([[0, 1], [1, 0], [0.5, 0.5], [0, 1]])
        cov, risk, thr = risk_coverage_curve(p, t, abstain_idx=1)
        np.testing.assert_allclose(cov, [0, 0, 0.5, 0.75])
        np.testing.assert_allclose(risk, [np.nan, np.nan, 0.25, 0.5], equal_nan=True)
        np.testing.assert_allclose(thr, [np.inf, 0.9, 0.8, 0.6])
        perm = [2, 0, 3, 1]
        other = risk_coverage_curve(p[perm], t[perm], abstain_idx=np.array([1]*4))
        for a, b in zip((cov, risk, thr), other):
            np.testing.assert_allclose(a, b, equal_nan=True)
        # Per-row indices differ after candidate permutations; -1 disables abstention.
        c, r, _ = risk_coverage_curve(p, t, abstain_idx=[1, 0, -1, 1])
        self.assertAlmostEqual(c[-1], 0.5)
        self.assertAlmostEqual(r[-1], 0.75)

    def test_masked_risk_and_empty_curve(self):
        p = np.array([[8, 2, 99], [8, 2, 99]])
        t = np.array([[1, 0, 50], [0, 1, 50]])
        mask = np.array([[True, True, False]] * 2)
        cov, risk, thresholds = risk_coverage_curve(p, t, mask=mask, abstain_idx=2)
        np.testing.assert_allclose(cov, [0, 1])
        self.assertAlmostEqual(risk[-1], 0.5)
        np.testing.assert_allclose(thresholds, [np.inf, 0.8])
        cov, risk, thresholds = risk_coverage_curve(np.empty((0, 2)), np.empty((0, 2)))
        np.testing.assert_allclose(cov, [0])
        self.assertTrue(np.isnan(risk[0]))
        self.assertTrue(np.isinf(thresholds[0]))

    def test_all_abstain_and_hard_risk(self):
        p = np.array([[0.8, 0.2], [0.8, 0.2]])
        t = np.array([[1, 0], [0, 1]])
        c, r, _ = risk_coverage_curve(p, t, abstain_idx=0)
        self.assertTrue(np.all(c == 0))
        self.assertTrue(np.isnan(r).all())
        c, r, th = risk_coverage_curve(p, t)
        np.testing.assert_allclose(c, [0, 1])
        self.assertAlmostEqual(r[-1], 0.5)
        self.assertEqual(len(th), 2)


@unittest.skipUnless(RUN_CUDA, "CUDA forward test requires explicit --cuda")
class ForwardCudaTests(unittest.TestCase):
    def test_forward_passes_brier_normalize(self):
        import torch
        from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
        self.assertTrue(torch.cuda.is_available(), "CUDA required; no CPU fallback")
        torch.manual_seed(7)
        model = MiniSystemOneForDecision(DecisionConfig(
            hidden_size=32, num_hidden_layers=1, vocab_size=32,
            num_attention_heads=4, num_key_value_heads=2,
            intermediate_size=64, max_position_embeddings=32)).cuda().eval()
        batch = {
            "input_ids": torch.tensor([[1, 2, 3, 4, 5, 6]], device="cuda"),
            "seg_id": torch.tensor([[2, 2, 3, 3, 4, 4]], device="cuda"),
            "cand_id": torch.tensor([[-1, -1, -1, -1, 0, 1]], device="cuda"),
            "cand_span": torch.tensor([[[4, 5], [5, 6], [0, 0]]], device="cuda"),
            "cand_mask": torch.tensor([[True, True, False]], device="cuda"),
            "target": torch.tensor([[1., 0., 0.]], device="cuda"),
        }
        with torch.inference_mode():
            raw = model(**batch, lambda_brier=0.7, lambda_ord=0, brier_normalize=False)
            norm = model(**batch, lambda_brier=0.7, lambda_ord=0, brier_normalize=True)
        torch.testing.assert_close(raw.logits, norm.logits)
        torch.testing.assert_close(raw.loss_dict["brier"] / 2, norm.loss_dict["brier"])
        torch.testing.assert_close(raw.loss - norm.loss, 0.35 * raw.loss_dict["brier"])
        self.assertGreater(float(raw.loss_dict["brier"]), 0)


if __name__ == "__main__":
    unittest.main()
