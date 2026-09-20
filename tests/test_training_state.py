import random
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import datasets
import numpy as np
import torch

from model.model_system_one import DecisionConfig
from trainer import trainer_utils as utils


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class TrainingStateTests(unittest.TestCase):
    def test_resume_matches_next_update_including_randomness(self):
        utils.seed_training(7, 'cuda')
        config = DecisionConfig(hidden_size=32, num_hidden_layers=1)
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.3)).cuda()
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        scaler = torch.amp.GradScaler('cuda')

        def update(m, o, s):
            x = torch.randn(2, 3, device='cuda') + random.random() + np.random.random()
            s.scale(m(x).square().mean()).backward()
            utils.optimizer_update(m, o, s)

        for _ in range(3):
            update(model, opt, scaler)
        args = SimpleNamespace(accum=2, max_steps=0, seed=7)
        progress = utils.resume_state(args, {'data': 'test'}, [4, 4], epoch=1, next_batch=2)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'model.pth')
            utils.save_checkpoint(model, path, opt, scaler, step=3, config=config,
                                  meta={'stage': 'decision'}, training=progress)
            update(model, opt, scaler)
            other = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.3)).cuda()
            other_opt = torch.optim.AdamW(other.parameters(), lr=.01)
            other_scaler = torch.amp.GradScaler('cuda')
            resume_path = str(Path(folder) / 'model_opt.pth')
            resumed = utils.load_resume(resume_path, config, 'decision', progress)
            self.assertEqual((resumed['epoch'], resumed['next_batch'], resumed['step']), (1, 2, 3))
            utils.restore_rng(utils.restore_training(resumed, other, other_opt, other_scaler))
            update(other, other_opt, other_scaler)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            changed = utils.resume_state(SimpleNamespace(accum=2, max_steps=0, seed=8),
                                         {'data': 'test'}, [4, 4])
            with self.assertRaisesRegex(ValueError, 'train_args'):
                utils.load_resume(resume_path, config, 'decision', changed)
            resumed['format'] = 2
            torch.save(resumed, resume_path)
            with self.assertRaisesRegex(ValueError, 'resume'):
                utils.load_resume(resume_path, config, 'decision', progress)

    def test_old_optimizer_checkpoint_is_not_silent_exact_resume(self):
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'old.pth')
            torch.save(dict(model={}, optimizer={}, step=1), path)
            with self.assertRaisesRegex(ValueError, 'resume'):
                utils.load_resume(path, DecisionConfig(), 'decision', {})

    def test_partial_accumulation_window_horizon(self):
        state = utils.resume_state(SimpleNamespace(accum=3, max_steps=0), {}, [5, 5])
        self.assertEqual(state['total_steps'], 4)


if __name__ == '__main__':
    unittest.main()
