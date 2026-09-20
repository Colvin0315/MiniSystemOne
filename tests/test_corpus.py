"""Pure corpus/quickstart contract tests: no CUDA, downloads, or training."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dataset.pretrain_corpus import _read_jsonl, iter_corpus, iter_mixed, iter_synth_docs
from scripts.quickstart import (build_stages, check_data_lengths, check_destination,
                                collect_stage_result, load_config, resolve_path)


class CorpusTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def corpus(self, name, texts):
        path = self.root / name
        path.write_text("".join(json.dumps({"text": text}) + "\n" for text in texts),
                        encoding="utf-8")
        return str(path)

    def args(self, **overrides):
        values = dict(seed=0, n_docs=10, n_synth=3, en_share=0.5,
                      pretrain_path=str(self.root / "missing_zh.jsonl"),
                      en_path=str(self.root / "missing_en.jsonl"),
                      synthetic_only=False, allow_missing_corpus=False)
        values.update(overrides)
        return SimpleNamespace(**values)

    def test_missing_fails_before_synthetic_generation(self):
        for kwargs in ({}, {"pretrain_path": "", "en_path": ""}):
            with self.subTest(kwargs=kwargs), patch("dataset.pretrain_corpus.iter_synth_docs") as synth:
                with self.assertRaisesRegex(FileNotFoundError, "--allow_missing_corpus"):
                    list(iter_corpus(self.args(**kwargs)))
                synth.assert_not_called()

    def test_second_source_missing_fails_before_synth(self):
        path = self.corpus("zh.jsonl", ["自然文本"])
        with patch("dataset.pretrain_corpus.iter_synth_docs") as synth:
            with self.assertRaisesRegex(FileNotFoundError, "Missing en"):
                list(iter_corpus(self.args(pretrain_path=path)))
            synth.assert_not_called()

    def test_allow_missing_reports_real_counts_including_tail(self):
        path = self.corpus("zh.jsonl", ["自然文本"])
        output = io.StringIO()
        with patch("dataset.pretrain_corpus.iter_synth_docs", return_value=iter(["a", "b"])), \
                contextlib.redirect_stdout(output):
            docs = list(iter_corpus(self.args(pretrain_path=path, allow_missing_corpus=True)))
        self.assertCountEqual(docs, ["自然文本", "a", "b"])
        self.assertIn("中英混合 1 篇 + 合成 2 篇", output.getvalue())
        self.assertIn("中文 1 篇", output.getvalue())
        self.assertIn("英文 0 篇", output.getvalue())

    def test_explicit_missing_all_can_fallback(self):
        with patch("dataset.pretrain_corpus.iter_synth_docs", return_value=iter(["synthetic"])):
            self.assertEqual(list(iter_corpus(self.args(allow_missing_corpus=True))), ["synthetic"])
        with self.assertRaisesRegex(ValueError, "No usable corpus"):
            list(iter_corpus(self.args(allow_missing_corpus=True, n_synth=0)))

    def test_synthetic_only_does_not_read_natural_paths(self):
        output = io.StringIO()
        with patch("dataset.pretrain_corpus._validate_sources", side_effect=AssertionError), \
                patch("dataset.pretrain_corpus.iter_synth_docs", return_value=iter(["a", "b"])) as synth, \
                contextlib.redirect_stdout(output):
            docs = list(iter_corpus(self.args(synthetic_only=True, n_docs=0, n_synth=2)))
        self.assertCountEqual(docs, ["a", "b"])
        synth.assert_called_once_with(2, seed=0)
        self.assertIn("中英混合 0 篇 + 合成 2 篇", output.getvalue())

    def test_empty_or_corrupt_files_fail_even_with_fallback(self):
        valid = self.corpus("en.jsonl", ["English text"])
        for contents in ("", "\n  \n", "not json\n", "{}\n", "[]\n",
                         '{"text": 3}\n', '{"text": " "}\n',
                         '{"text": "valid"}\nnot json\n'):
            with self.subTest(contents=contents):
                path = self.root / "bad.jsonl"
                path.write_text(contents, encoding="utf-8")
                with patch("dataset.pretrain_corpus.iter_synth_docs") as synth:
                    with self.assertRaisesRegex(ValueError, "bad.jsonl"):
                        list(iter_corpus(self.args(pretrain_path=str(path), en_path=valid,
                                                  allow_missing_corpus=True, n_docs=1)))
                    synth.assert_not_called()

    def test_invalid_utf8_fails(self):
        path = self.root / "invalid.jsonl"
        path.write_bytes(b'{"text": "\xff"}\n')
        with self.assertRaisesRegex(ValueError, "UTF-8"):
            list(_read_jsonl(path, "text"))

    def test_mixed_exhaustion_keeps_all_available_docs(self):
        zh = self.corpus("zh.jsonl", ["中文", "另一篇"])
        en = self.corpus("en.jsonl", ["English"])
        self.assertCountEqual(list(iter_mixed(zh, en)), ["中文", "另一篇", "English"])

    def test_invalid_budgets_fail_before_synth(self):
        for kwargs in ({"n_docs": -1}, {"n_synth": -1}, {"en_share": 1.1},
                       {"synthetic_only": True, "n_synth": 0}, {"n_docs": 0}):
            with self.subTest(kwargs=kwargs), patch("dataset.pretrain_corpus.iter_synth_docs") as synth:
                with self.assertRaises(ValueError):
                    list(iter_corpus(self.args(**kwargs)))
                synth.assert_not_called()

    def test_synthetic_zero_does_no_work(self):
        with patch("dataset.synth.build_all") as build:
            self.assertEqual(list(iter_synth_docs(0)), [])
            build.assert_not_called()

    def test_real_synthetic_counts_and_determinism(self):
        for n in (1, 6, 13):
            with self.subTest(n=n):
                docs = list(iter_synth_docs(n, seed=4))
                self.assertEqual(len(docs), n)
                self.assertTrue(all(isinstance(doc, str) and doc.strip() for doc in docs))
                self.assertEqual(docs, list(iter_synth_docs(n, seed=4)))


class QuickstartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name)
        self.config = load_config(ROOT / "configs" / "quickstart.json")

    def test_commands_use_exact_flags_and_isolated_outputs(self):
        stages = build_stages(self.config, self.out)
        self.assertEqual([s[0] for s in stages], [
            "tokenizer", "data", "mlm", "decision", "calibration", "eval",
            "inference_noul", "inference_choice", "inference_score"])
        for name, command in stages:
            self.assertEqual(command[0], sys.executable)
            self.assertTrue(Path(command[1]).is_absolute())
            self.assertNotIn("--smoke", command)
            self.assertNotIn("--force", command)
            for flag in ("--out", "--out_dir", "--log_dir", "--tokenizer", "--encoder",
                         "--data", "--temperature", "--ckpt"):
                if flag in command:
                    self.assertTrue(Path(command[command.index(flag) + 1]).is_relative_to(self.out))
        by_name = dict(stages)
        for name in ("tokenizer", "mlm"):
            self.assertIn("--synthetic_only", by_name[name])
        for split in ("train", "val", "calib", "test_known", "test_ood"):
            self.assertIn(f"--per_gen_{split}", by_name["data"])
        for name in ("mlm", "decision"):
            command = by_name[name]
            self.assertEqual(command[command.index("--max_steps") + 1], "20")
            self.assertIn("--save_optimizer", command)
        self.assertEqual(resolve_path("configs/quickstart.json"), ROOT / "configs" / "quickstart.json")

    def test_refuse_nonempty_destination(self):
        check_destination(self.out)
        (self.out / "existing").write_text("do not overwrite", encoding="utf-8")
        with self.assertRaisesRegex(FileExistsError, "Refusing nonempty"):
            check_destination(self.out)

    def test_fixed_candidate_length_overflow_is_actionable(self):
        (self.out / "data").mkdir()
        (self.out / "data" / "train.jsonl").write_text(json.dumps({
            "id": "too_long", "meta": {"approx_tokens": 520}}) + "\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Fixed question/candidates"):
            check_data_lengths(self.out, 512)

    def test_trainer_summary_must_confirm_budget(self):
        (self.out / "mlm").mkdir()
        summary_path = self.out / "mlm" / "summary.json"
        result = {"step": 20, "elapsed_s": 1.5, "peak_vram_gb": 0.1}
        summary_path.write_text(json.dumps(result), encoding="utf-8")
        self.assertEqual(collect_stage_result("mlm", self.out, self.config)["training"], result)
        result["step"] = 19
        summary_path.write_text(json.dumps(result), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "expected 20"):
            collect_stage_result("mlm", self.out, self.config)


if __name__ == "__main__":
    unittest.main()
