"""Sequential CUDA subprocess integration tests; run with minimind Python."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def worker(stage, trace_path, argv):
    import random
    import runpy
    import numpy as np
    import torch
    import model.model_system_one as models
    import trainer.trainer_utils as utils

    torch.use_deterministic_algorithms(True, warn_only=True)
    traces, updates, scales = [], [], []
    original_config = models.DecisionConfig

    def config(*args, **kwargs):
        kwargs["dropout"] = 0.1
        return original_config(*args, **kwargs)

    models.DecisionConfig = config
    cls = models.MiniSystemOneForMaskedLM if stage == "mlm" else models.MiniSystemOneForDecision
    original_forward = cls.forward

    def forward(self, *args, **kwargs):
        if self.training:
            tensors = {k: v.detach().cpu().clone() for k, v in kwargs.items() if torch.is_tensor(v)}
            if args:
                tensors["input_ids"] = args[0].detach().cpu().clone()
            traces.append({"batch": tensors,
                           "draws": [random.random(), float(np.random.random()),
                                     float(torch.rand(())), float(torch.rand((), device="cuda"))]})
        output = original_forward(self, *args, **kwargs)
        if self.training:
            traces[-1]["loss"] = float(output.loss.detach())
        return output

    cls.forward = forward
    original_scale = torch.amp.GradScaler.scale

    def scale(self, outputs):
        scales.append(float(outputs.detach()))
        return original_scale(self, outputs)

    torch.amp.GradScaler.scale = scale
    original_update = utils.optimizer_update

    def update(model, optimizer, scaler):
        updates.append({"lr": optimizer.param_groups[0]["lr"], "micro_end": len(traces)})
        original_update(model, optimizer, scaler)

    utils.optimizer_update = update
    sys.argv = [f"train_{stage}.py", *argv]
    runpy.run_path(str(ROOT / "trainer" / f"train_{stage}.py"), run_name="__main__")
    torch.save({"batches": traces, "updates": updates, "scales": scales}, trace_path)


class ResumeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA required; no CPU training fallback")
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory(prefix="minisystemone_resume_")
        cls.root = Path(cls.tmp.name)
        cls.tokenizer = cls.root / "tokenizer"
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        special = ["<pad>", "<unk>", "<bos>", "<sep>", "<mask>", "<trunc>",
                   "<yes>", "<no>", "<eos>", "<choice>", "<score>"]
        words = [f"word{i}" for i in range(40)]
        backend = Tokenizer(models.WordLevel({w: i for i, w in enumerate(special + words)},
                                            unk_token="<unk>"))
        backend.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tok = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="<unk>",
                                     pad_token="<pad>", mask_token="<mask>",
                                     additional_special_tokens=special[2:])
        tok.save_pretrained(cls.tokenizer)
        cls.corpus = cls.root / "corpus.jsonl"
        docs = [{"text": " ".join(words[(i + j) % 40] for j in range(15))} for i in range(5)]
        cls.corpus.write_text("".join(json.dumps(x) + "\n" for x in docs), encoding="utf-8")
        cls.data = cls.root / "data"
        cls.data.mkdir()
        records = []
        for i in range(9):
            k = 2 + i % 5
            records.append({"id": f"toy-{i}", "source": "toy", "split": "train",
                            "schema": {"primitive": "choice"}, "state": f"word{i} word1 word2",
                            "question": "word3 word4", "meta": {"approx_tokens": 32},
                            "candidates": [{"label": str(j), "text": f"word{j + 5}"} for j in range(k)],
                            "target": {"p": [1 / k] * k, "provenance": "explicit_rng"}})
        for split in ("train", "val"):
            (cls.data / f"{split}.jsonl").write_text(
                "".join(json.dumps(dict(r, split=split)) + "\n" for r in records), encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def args(self, stage):
        args = ["--tokenizer", str(self.tokenizer), "--device", "cuda", "--hidden_size", "32",
                "--num_hidden_layers", "1", "--epochs", "2", "--batch_size", "2",
                "--accum", "2", "--warmup_steps", "1", "--num_workers", "0",
                "--seed", "17", "--save_optimizer", "--no_swanlab", "--log_interval", "1",
                "--save_interval", "1"]
        if stage == "mlm":
            args += ["--max_len", "16", "--n_docs", "5", "--n_synth", "0",
                     "--pretrain_path", str(self.corpus), "--en_path", str(self.corpus), "--en_share", "0"]
        else:
            args += ["--max_len", "64", "--data", str(self.data), "--encoder", "",
                     "--k_max", "6", "--keep_p_min", "0.9", "--val_every", "1",
                     "--val_limit", "3", "--brier_normalize"]
        return args

    def launch(self, stage, name, extra=(), error=None):
        out = self.root / name
        trace = self.root / (name + "_trace.pth")
        args = [sys.executable, str(Path(__file__).resolve()), "--worker", stage, str(trace),
                *self.args(stage), "--out", str(out), *map(str, extra)]
        env = dict(os.environ, PYTHONHASHSEED="0", PYTHONIOENCODING="utf-8",
                   CUBLAS_WORKSPACE_CONFIG=":4096:8", HF_HUB_OFFLINE="1", TOKENIZERS_PARALLELISM="false")
        result = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True,
                                encoding="utf-8", errors="replace", timeout=180)
        log = result.stdout + result.stderr
        if error is not None:
            self.assertNotEqual(result.returncode, 0, log)
            self.assertIn(error, log)
            return
        self.assertEqual(result.returncode, 0, log)
        checkpoint = self.torch.load(out / f"{stage}_opt.pth", map_location="cpu", weights_only=True)
        trace_data = self.torch.load(trace, map_location="cpu", weights_only=True)
        self.assertEqual(json.loads((out / "summary.json").read_text())["step"], checkpoint["step"])
        return checkpoint, trace_data, out

    def same(self, left, right, path="state"):
        if self.torch.is_tensor(left):
            self.assertTrue(self.torch.is_tensor(right), path)
            self.torch.testing.assert_close(left, right, rtol=2e-6, atol=2e-7, msg=path)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys(), path)
            for key in left:
                self.same(left[key], right[key], f"{path}.{key}")
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right), path)
            for i, (a, b) in enumerate(zip(left, right)):
                self.same(a, b, f"{path}[{i}]")
        elif isinstance(left, float):
            self.assertAlmostEqual(left, right, delta=2e-7 + 2e-6 * abs(left), msg=path)
        else:
            self.assertEqual(left, right, path)

    def compare_training(self, stage, full, resumed):
        for key in ("model", "optimizer", "scaler", "step", "epoch", "next_batch",
                    "total_steps", "epoch_batches", "train_args", "fingerprints", "rng"):
            self.same(full[key], resumed[key], f"{stage}.{key}")

    def assert_window_scaling(self, trace, batches, accum=2):
        index = 0
        for count in batches:
            for micro in range(count):
                window = min(accum, count - (micro // accum) * accum)
                self.assertAlmostEqual(trace["scales"][index], trace["batches"][index]["loss"] / window,
                                       delta=1e-6)
                index += 1
        self.assertEqual(index, len(trace["batches"]))

    def test_uninterrupted_and_resume_mid_epoch_boundary_and_tail(self):
        for stage in ("mlm", "decision"):
            with self.subTest(stage=stage):
                full, trace, _ = self.launch(stage, f"{stage}_full")
                self.assert_window_scaling(trace, full["epoch_batches"])
                expected = sum((n + 1) // 2 for n in full["epoch_batches"])
                self.assertEqual(full["step"], expected)
                self.assertEqual((full["epoch"], full["next_batch"]), (2, 0))
                first_epoch_steps = (full["epoch_batches"][0] + 1) // 2
                for pause in sorted({1, first_epoch_steps}):
                    name = f"{stage}_pause_{pause}"
                    partial, before, out = self.launch(stage, name, ["--stop_after_steps", pause])
                    self.assertEqual(partial["total_steps"], full["total_steps"])
                    self.assertEqual(partial["step"], pause)
                    if pause == first_epoch_steps:
                        self.assertEqual((partial["epoch"], partial["next_batch"]), (1, 0))
                    checkpoint = out / f"{stage}_opt.pth"
                    (out / f"{stage}.pth").unlink()
                    resumed, after, _ = self.launch(stage, name + "_resumed", ["--resume", checkpoint])
                    self.compare_training(stage, full, resumed)
                    self.same(trace["batches"], before["batches"] + after["batches"], "data/masks/augmentation/RNG")
                    self.same(trace["scales"], before["scales"] + after["scales"], "accumulation")
                    self.same([x["lr"] for x in trace["updates"]],
                              [x["lr"] for x in before["updates"] + after["updates"]], "LR")
                done, empty, _ = self.launch(stage, f"{stage}_done", ["--resume", self.root / f"{stage}_full" / f"{stage}_opt.pth"])
                self.compare_training(stage, full, done)
                self.assertEqual(empty["batches"], [])

    def test_absolute_budget_and_fixed_horizon(self):
        for stage in ("mlm", "decision"):
            full, trace, _ = self.launch(stage, f"{stage}_budget", ["--max_steps", 3])
            part, first, out = self.launch(stage, f"{stage}_budget_pause",
                                           ["--max_steps", 3, "--stop_after_steps", 1])
            resumed, last, _ = self.launch(stage, f"{stage}_budget_resume",
                                            ["--max_steps", 3, "--resume", out / f"{stage}_opt.pth"])
            self.assertEqual((full["step"], full["total_steps"], part["total_steps"]), (3, 3, 3))
            self.compare_training(stage, full, resumed)
            self.same(trace["batches"], first["batches"] + last["batches"])

    def test_incomplete_adam_state_rejected(self):
        import copy
        from model.model_system_one import DecisionConfig, MiniSystemOneForDecision, MiniSystemOneForMaskedLM
        from trainer.trainer_utils import restore_training
        for stage, cls in (("decision", MiniSystemOneForDecision), ("mlm", MiniSystemOneForMaskedLM)):
            model = cls(DecisionConfig(hidden_size=32, num_hidden_layers=1,
                                       intermediate_size=64, vocab_size=64)).cuda()
            optimizer = self.torch.optim.AdamW(model.parameters(), lr=1e-3)
            for name, parameter in model.named_parameters():
                if stage == "mlm" and name == "encoder.embed_segments.weight":
                    continue
                parameter.grad = self.torch.ones_like(parameter)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scaler = self.torch.amp.GradScaler("cuda")
            original = {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scaler": scaler.state_dict(), "step": 1, "meta": {"stage": stage}, "rng": {}}
            restore_training(original, model, optimizer, scaler)
            for mutation in ("all", "one", "step", "shape"):
                blob = copy.deepcopy(original)
                state = blob["optimizer"]["state"]
                key = next(iter(state))
                if mutation == "all":
                    state.clear()
                elif mutation == "one":
                    del state[key]
                elif mutation == "step":
                    state[key]["step"] = self.torch.tensor(0.)
                else:
                    state[key]["exp_avg"] = self.torch.zeros(1)
                with self.assertRaisesRegex(ValueError, "resume Adam"):
                    restore_training(blob, model, optimizer, scaler)

    def test_reject_incompatible_or_missing_resume(self):
        for stage in ("mlm", "decision"):
            original, _, out = self.launch(stage, f"{stage}_errors", ["--stop_after_steps", 1])
            good = out / f"{stage}_opt.pth"
            self.launch(stage, f"{stage}_missing", ["--resume", self.root / "missing.pth"], "resume")
            self.launch(stage, f"{stage}_workers", ["--resume", good, "--num_workers", 1], "num_workers 0")
            for flag, value, message in (("--epochs", 3, "train_args"), ("--accum", 3, "train_args"),
                                         ("--max_steps", 2, "train_args"), ("--hidden_size", 64, "config")):
                self.launch(stage, f"{stage}_{flag[2:]}", ["--resume", good, flag, value], message)
            for change, message in (("old", "resume"), ("format", "resume"), ("stage", "stage"), ("cursor", "cursor"),
                                    ("fingerprints", "fingerprints"), ("rng", "resume")):
                damaged = self.torch.load(good, map_location="cpu", weights_only=True)
                if change == "old":
                    damaged = {k: damaged[k] for k in ("model", "optimizer", "step")}
                elif change == "format":
                    damaged["format"] = 2
                elif change == "stage":
                    damaged["meta"]["stage"] = "other"
                elif change == "cursor":
                    damaged["next_batch"] = 1
                elif change == "fingerprints":
                    damaged["fingerprints"]["tokenizer"]["tokenizer.json"] = "changed"
                else:
                    del damaged["rng"]
                path = self.root / f"{stage}_bad_{change}.pth"
                self.torch.save(damaged, path)
                self.launch(stage, f"{stage}_reject_{change}", ["--resume", path], message)
            source = self.corpus if stage == "mlm" else self.data / "train.jsonl"
            old = source.read_bytes()
            try:
                source.write_bytes(old.replace(b"word1", b"word9", 1))
                self.launch(stage, f"{stage}_data_changed", ["--resume", good], "fingerprints")
            finally:
                source.write_bytes(old)
            if stage == "decision":
                train_path = self.data / "train.jsonl"
                old_train = train_path.read_text(encoding="utf-8")
                try:
                    rows = [json.loads(line) for line in old_train.splitlines()]
                    rows[0]["split"] = "calib"
                    train_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
                    self.launch(stage, "decision_reject_calib_in_train", error="split")
                finally:
                    train_path.write_text(old_train, encoding="utf-8")
            tok_file = self.tokenizer / "tokenizer_config.json"
            old = tok_file.read_bytes()
            try:
                tok_file.write_bytes(old + b"\n")
                self.launch(stage, f"{stage}_tokenizer_changed", ["--resume", good], "fingerprints")
            finally:
                tok_file.write_bytes(old)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        worker(sys.argv[2], sys.argv[3], sys.argv[4:])
    else:
        unittest.main()
