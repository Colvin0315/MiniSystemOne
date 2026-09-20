import argparse
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from examples import customer_tool_routing as routing


class ThresholdTests(unittest.TestCase):
    def test_ties_are_not_split(self):
        p = np.array([[.9, .05, .05], [.9, .05, .05]])
        t = np.array([[1, 0, 0], [0, 1, 0]])
        self.assertIsNone(routing.choose_threshold(p, t, 2, .1))
        self.assertEqual(routing.choose_threshold(p, t, 2, .5), .9)

    def test_maximum_feasible_coverage_not_first_crossing(self):
        p = np.array([[.9, .05, .05], [.8, .1, .1], [.6, .2, .2]])
        t = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 0]])
        self.assertAlmostEqual(routing.choose_threshold(p, t, 2, .34), .6)

    def test_per_row_human_positions(self):
        p = np.array([[.9, .05, .05], [.1, .8, .1], [.1, .2, .7]])
        t = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 1]])
        human = np.array([0, 2, 1])
        threshold = routing.choose_threshold(p, t, human, 0)
        report = routing.policy_report(p, t, human, threshold)
        self.assertEqual(report["accepted"], 2)
        self.assertEqual(report["model_human"], 1)
        self.assertEqual(report["accepted_risk"], 0)
        self.assertAlmostEqual(report["coverage"], 2 / 3)

    def test_all_human_and_abstain_all_use_null_risk(self):
        p = np.array([[.1, .9], [.2, .8]])
        t = np.array([[0, 1], [1, 0]])
        self.assertIsNone(routing.choose_threshold(p, t, 1, 1))
        report = routing.policy_report(p, t, 1, None)
        self.assertIsNone(report["accepted_risk"])
        self.assertEqual(report["coverage"], 0)
        self.assertEqual(report["handoff_fraction"], 1)
        self.assertEqual(report["threshold_handoff"], 0)
        json.dumps(report, allow_nan=False)

    def test_no_feasible_threshold_and_empty_curve(self):
        self.assertIsNone(routing.choose_threshold(np.array([[.8, .2]]), np.array([[0, 1]]), 1, 0))
        self.assertIsNone(routing.choose_threshold(np.empty((0, 2)), np.empty((0, 2)), 1, .1))
        for bad in (-.1, 1.1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                routing.choose_threshold(np.array([[.8, .2]]), np.array([[1, 0]]), 1, bad)

    def test_human_tied_with_automatic_rows(self):
        p = np.array([[.8, .1, .1], [.1, .1, .8], [.8, .1, .1]])
        t = np.array([[1, 0, 0], [0, 0, 1], [0, 1, 0]])
        self.assertIsNone(routing.choose_threshold(p, t, 2, .49))
        threshold = routing.choose_threshold(p, t, 2, .5)
        report = routing.policy_report(p, t, 2, threshold)
        self.assertEqual(report["accepted_risk"], .5)
        self.assertEqual(report["accepted"], 2)
        self.assertAlmostEqual(report["coverage"], 2 / 3)


class BuilderTests(unittest.TestCase):
    def test_disjoint_deterministic_aligned_hard_data(self):
        splits = routing.generate(7)
        self.assertEqual(splits, routing.generate(7))
        self.assertNotEqual(splits, routing.generate(8))
        routing.assert_disjoint(splits)
        positions = set()
        for split, rows in splits.items():
            self.assertEqual(len(rows), 128 if split == "train" else 32)
            self.assertEqual({r["target"]["audit"]["correct_label"] for r in rows}, set(routing.TOOLS))
            for row in rows:
                routing.validate_routing_request(row)
                self.assertEqual(row["target"]["provenance"], "hard")
                self.assertEqual(row["target"]["kind"], "hard")
                self.assertEqual(sum(row["target"]["p"]), 1)
                self.assertEqual(row["state_sections"][0]["text"], row["state"])
                self.assertEqual(row["split"], split)
                for candidate, mass in zip(row["candidates"], row["target"]["p"]):
                    self.assertEqual(mass, int(candidate["label"] == row["target"]["audit"]["correct_label"]))
                    self.assertIsNone(candidate["meta"]["level"])
                positions.add(row["target"]["p"].index(1))
        self.assertEqual(positions, {0, 1, 2, 3})

    def test_builder_refuses_nonempty_and_reads_splits(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "data"
            routing.build(out)
            before = (out / "train.jsonl").read_bytes()
            with self.assertRaises(ValueError):
                routing.build(out)
            self.assertEqual((out / "train.jsonl").read_bytes(), before)
            for split in routing.SPLITS:
                self.assertEqual(len(routing.read_split(out, split)), 128 if split == "train" else 32)
            (out / "val.jsonl").write_bytes(before)
            with self.assertRaisesRegex(ValueError, "another split"):
                routing.read_split(out, "val")

    def test_leakage_detection(self):
        splits = routing.generate()
        splits["val"][0]["meta"]["entity_id"] = splits["train"][0]["meta"]["entity_id"]
        with self.assertRaisesRegex(ValueError, "entity_id"):
            routing.assert_disjoint(splits)

    def test_dispatch_contract_rejects_changed_candidate_meaning(self):
        request = routing.request_for("请查询订单 C1 的物流状态。")
        request["candidates"][0]["text"] = "立即转账"
        with self.assertRaises(ValueError):
            routing.validate_routing_request(request)
        with self.assertRaises(ValueError):
            routing.dispatch_stub("execute_shell")
        for label in routing.TOOLS:
            self.assertFalse(routing.dispatch_stub(label)["side_effects"])


class FrozenPolicyTests(unittest.TestCase):
    def test_dispatch_uses_same_normalization_and_keeps_unseen_annotation_only(self):
        probabilities = [.59999999, .1, .1, .19999999]
        threshold = float(routing.top1([probabilities])[1][0])
        args = argparse.Namespace(policy="unused", input=None, state="未训练业务", unseen=True)
        result = {"candidates": [{"label": label, "probability": p}
                                 for label, p in zip(routing.TOOLS, probabilities)],
                  "selected_label": "order", "confidence": probabilities[0]}
        with patch.object(routing, "load_policy", return_value=({"threshold": threshold}, {})), \
             patch.object(routing, "predictor_for") as predictor:
            predictor.return_value.predict.return_value = result
            output = routing.dispatch(args)
            self.assertTrue(output["accepted"])
            self.assertTrue(output["not_trained_example"])
            self.assertEqual(output["dispatch"]["handler"], "order")
            self.assertFalse(output["dispatch"]["side_effects"])

    def test_test_evaluation_never_selects_threshold(self):
        rows = routing.generate()["test_known"]
        p = np.asarray([r["target"]["p"] for r in rows])
        human = np.array([next(i for i, c in enumerate(r["candidates"]) if c["label"] == "human") for r in rows])
        with tempfile.TemporaryDirectory() as temp:
            policy_path = Path(temp) / "policy.json"
            policy_path.write_text("{}", encoding="utf-8")
            args = argparse.Namespace(policy=str(policy_path), ckpt=None, tokenizer=None, temperature=None,
                                      data=None, out=str(Path(temp) / "report.json"))
            policy = {"threshold": None, "artifacts": {}}
            with patch.object(routing, "load_policy", return_value=(policy, {"data": temp})), \
                 patch.object(routing, "read_split", return_value=rows) as read, \
                 patch.object(routing, "predictor_for"), \
                 patch.object(routing, "predict_rows", return_value=(p, p, human)), \
                 patch.object(routing, "choose_threshold", side_effect=AssertionError("test must not tune")):
                report = routing.evaluate(args)
                read.assert_called_once_with(temp, "test_known")
                self.assertEqual(report["coverage"], 0)
                self.assertIsNone(report["accepted_risk"])
                self.assertEqual(set(report["metrics_all"]) & {"distribution_l2", "expected_brier", "nll", "ece"},
                                 {"distribution_l2", "expected_brier", "nll", "ece"})
                with self.assertRaises(ValueError):
                    routing.evaluate(args)

    def test_hash_binding_rejects_changes_to_every_artifact(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            routing.build(root / "data")
            (root / "tokenizer").mkdir()
            (root / "tokenizer" / "tokenizer.json").write_text("{}", encoding="utf-8")
            (root / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
            (root / "decision.pth").write_bytes(b"mock checkpoint; never loaded")
            (root / "T.json").write_text("{}", encoding="utf-8")
            paths = {"ckpt": str(root / "decision.pth"), "tokenizer": str(root / "tokenizer"),
                     "temperature": str(root / "T.json"), "data": str(root / "data")}
            policy = {"version": 1, "schema": routing.SCHEMA, "tools": routing.TOOLS,
                      "granularity": "global", "selection_split": "val", "temperature_split": "calib",
                      "threshold": None, "paths": paths, "artifacts": routing.fingerprint(**paths)}
            policy_path = root / "policy.json"
            routing.write_json(policy_path, policy)
            routing.load_policy(policy_path)
            for artifact in policy["artifacts"].values():
                path = Path(artifact["path"])
                original = path.read_bytes()
                path.write_bytes(original + b" ")
                with self.assertRaisesRegex(ValueError, "artifact mismatch"):
                    routing.load_policy(policy_path)
                path.write_bytes(original)
            policy["threshold"] = float("inf")
            policy_path.write_text(json.dumps(policy), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "threshold"):
                routing.load_policy(policy_path)


if __name__ == "__main__":
    unittest.main()
