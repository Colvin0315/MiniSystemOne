"""Opt-in GPU integration: MINISYSTEMONE_GPU_TESTS=1 python -m unittest discover -s tests."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import datasets
import torch

ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get('MINISYSTEMONE_GPU_TESTS') == '1' and torch.cuda.is_available(),
                     'set MINISYSTEMONE_GPU_TESTS=1 with CUDA to test trainer subprocesses')
class TrainerIntegrationTests(unittest.TestCase):
    def test_both_trainers_resume_to_same_weights(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            corpus = root / 'corpus.jsonl'
            corpus.write_text('\n'.join(json.dumps({'text': 'An account has enough balance to approve the transfer. ' * 4})
                                        for _ in range(10)), encoding='utf-8')
            records = []
            for i in range(10):
                records.append(dict(id=str(i), state='The run completed all checks.', question='Rate the run.',
                    schema={'primitive': 'score'}, candidates=[{'text': str(v), 'meta': {'level': v}} for v in [3, 1, 2]],
                    target={'p': [.7, .1, .2], 'provenance': 'marginalized'}, source='integration'))
            for split in ('train', 'val'):
                (root / (split + '.jsonl')).write_text('\n'.join(map(json.dumps, records)), encoding='utf-8')
            common = ['--tokenizer', str(ROOT / 'model'), '--hidden_size', '32', '--num_hidden_layers', '1',
                      '--batch_size', '2', '--max_len', '64', '--epochs', '2', '--accum', '3',
                      '--save_optimizer', '--save_interval', '0', '--no_swanlab', '--seed', '12']
            for stage, extra in [('mlm', ['--pretrain_path', '', '--en_path', str(corpus), '--n_docs', '10', '--n_synth', '0']),
                                 ('decision', ['--data', str(root), '--encoder', '', '--val_every', '0', '--val_limit', '2'])]:
                def run(out, rest):
                    cmd = [sys.executable, str(ROOT / 'trainer' / f'train_{stage}.py'), *common, *extra,
                           '--out', str(out), '--log_dir', str(out / 'logs'), *rest]
                    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding='utf-8',
                                          env=dict(os.environ, PYTHONIOENCODING='utf-8'), timeout=180)
                    self.assertEqual(proc.returncode, 0, proc.stdout + proc.stderr)
                full, part = root / (stage + '_full'), root / (stage + '_part')
                run(full, [])
                run(part, ['--max_steps', '1'])
                run(part, ['--resume', str(part / f'{stage}_opt.pth')])
                a = torch.load(full / f'{stage}_opt.pth', map_location='cpu', weights_only=True)
                b = torch.load(part / f'{stage}_opt.pth', map_location='cpu', weights_only=True)
                self.assertEqual(a['step'], b['step'])
                self.assertEqual(a['training_state']['epoch'], 2)
                if stage == 'decision':
                    self.assertEqual(a['step'], 4)  # 5 batches/epoch, accum=3, tail flushed
                for name in a['model']:
                    torch.testing.assert_close(a['model'][name], b['model'][name], rtol=0, atol=0,
                                               msg=lambda msg: f'{stage}: {name}: {msg}')
