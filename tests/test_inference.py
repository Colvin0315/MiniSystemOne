import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.eval_inference import load_temperature
from inference import Predictor, validate_request
from model.model_system_one import DecisionConfig, MiniSystemOneForDecision
from trainer.calibrate_temperature import fit_temperature
from trainer.trainer_utils import file_sha1, save_checkpoint


class RequestTests(unittest.TestCase):
    def setUp(self):
        self.request = json.loads((ROOT / "examples/inference/choice.json").read_text(encoding="utf-8"))

    def test_example_contracts(self):
        for primitive in ("noul", "choice", "score"):
            request = json.loads((ROOT / f"examples/inference/{primitive}.json").read_text(encoding="utf-8"))
            self.assertEqual(validate_request(request), primitive)

    def test_bad_requests(self):
        for bad in (None, [], {}, {"schema": {"primitive": "unknown"}}):
            with self.assertRaises(ValueError):
                validate_request(bad)
        duplicate = copy.deepcopy(self.request)
        duplicate["candidates"][1]["label"] = duplicate["candidates"][0]["label"]
        with self.assertRaisesRegex(ValueError, "唯一"):
            validate_request(duplicate)
        empty = copy.deepcopy(self.request)
        empty["candidates"] = []
        with self.assertRaises(ValueError):
            validate_request(empty)


class InferenceCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not torch.cuda.is_available():
            raise RuntimeError("Inference integration tests require CUDA")
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.tokenizer = str(ROOT / "model")
        tok = AutoTokenizer.from_pretrained(cls.tokenizer, local_files_only=True)
        config = DecisionConfig(hidden_size=32, num_hidden_layers=1, intermediate_size=64,
                                vocab_size=len(tok))
        torch.manual_seed(31)
        model = MiniSystemOneForDecision(config).cuda()
        cls.ckpt = str(cls.path / "decision.pth")
        save_checkpoint(model, cls.ckpt, config=config, meta={
            "stage": "decision", "max_len": 512,
            "tokenizer_sha1": file_sha1(str(ROOT / "model/tokenizer.json")),
        })
        cls.predictor = Predictor(cls.ckpt, cls.tokenizer)
        del model

    @classmethod
    def tearDownClass(cls):
        del cls.predictor
        cls.temp.cleanup()

    def test_outputs_and_score_mapping(self):
        for primitive in ("noul", "choice", "score"):
            request = json.loads((ROOT / f"examples/inference/{primitive}.json").read_text(encoding="utf-8"))
            result = self.predictor.predict(request)
            p = [c["probability"] for c in result["candidates"]]
            self.assertAlmostEqual(sum(p), 1.0, places=6)
            self.assertTrue(all(v >= 0 for v in p))
            if primitive == "noul":
                self.assertEqual(result["positive_probability"], p[1])
            if primitive == "score":
                self.assertAlmostEqual(result["expected_score"], sum((i + 1) * v for i, v in enumerate(p)))
                request["candidates"].reverse()
                reversed_result = self.predictor.predict(request)
                self.assertAlmostEqual(reversed_result["expected_score"], result["expected_score"], places=5)

    def test_missing_and_overlength(self):
        with self.assertRaises(ValueError):
            Predictor(str(self.path / "missing.pth"), self.tokenizer)
        request = json.loads((ROOT / "examples/inference/choice.json").read_text(encoding="utf-8"))
        request["state"] = "request " * 1000
        with self.assertRaisesRegex(ValueError, "超过预算"):
            self.predictor.predict(request)

    def test_wrong_tokenizer_is_library_error(self):
        wrong = self.path / "wrong_tokenizer"
        wrong.mkdir()
        (wrong / "tokenizer.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "词表与权重不配套"):
            Predictor(self.ckpt, str(wrong))

    def test_temperature_identity_checks(self):
        path = self.path / "T.json"
        temp = {"global": 2.0, "meta": {
            "ckpt_sha1": file_sha1(self.ckpt, n=12),
            "tokenizer_sha1": file_sha1(str(ROOT / "model/tokenizer.json"), n=12),
        }}
        path.write_text(json.dumps(temp), encoding="utf-8")
        self.assertEqual(load_temperature(str(path), self.ckpt, self.tokenizer)["global"], 2.0)
        predictor = Predictor(self.ckpt, self.tokenizer, str(path))
        request = json.loads((ROOT / "examples/inference/choice.json").read_text(encoding="utf-8"))
        raw = self.predictor.predict(request)
        actual = predictor.predict(request)
        expected = np.sqrt([c["probability"] for c in raw["candidates"]])
        expected /= expected.sum()
        np.testing.assert_allclose([c["probability"] for c in actual["candidates"]], expected, atol=1e-6)
        global_only = Predictor(self.ckpt, self.tokenizer, str(path), granularity="primitive_k")
        self.assertEqual(global_only.granularity, "global")
        np.testing.assert_allclose(
            [c["probability"] for c in global_only.predict(request)["candidates"]], expected, atol=1e-6)
        for change in ({"global": 0}, {"global": 0.001}, {"global": 21},
                       {"global": float("nan")}, {"meta": {"ckpt_sha1": "0" * 12}}):
            path.write_text(json.dumps(dict(temp, **change)), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_temperature(str(path), self.ckpt, self.tokenizer)

    def test_cli_stdout_is_json(self):
        run = subprocess.run([sys.executable, str(ROOT / "inference.py"), "--ckpt", self.ckpt,
                              "--tokenizer", self.tokenizer, "--input",
                              str(ROOT / "examples/inference/noul.json")],
                             cwd=ROOT, capture_output=True, text=True, encoding="utf-8", check=True)
        self.assertEqual(json.loads(run.stdout)["primitive"], "noul")

    def test_temperature_fit_cuda_with_padding(self):
        p = np.array([[.95, .05, 0], [.9, .1, 0], [.85, .15, 0]])
        t = np.array([[.6, .4, 0], [.55, .45, 0], [.5, .5, 0]])
        temperature, before, after = fit_temperature(np.log(np.clip(p, 1e-30, None)), t,
                                                     p > 0, np.ones(3, dtype=bool), max_iter=20)
        self.assertTrue(np.isfinite(temperature))
        self.assertLessEqual(after, before + 1e-8)


if __name__ == "__main__":
    unittest.main()
