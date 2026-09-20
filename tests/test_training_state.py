import random
import tempfile
import unittest
from pathlib import Path

import datasets
import numpy as np
import torch

from trainer import trainer_utils
from trainer import training_state


class TrainingStateTests(unittest.TestCase):
    def test_resume_matches_next_update_including_randomness(self):
        torch.manual_seed(7)
        random.seed(7)
        np.random.seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.3))
        opt = torch.optim.AdamW(model.parameters(), lr=.01)
        def update(m, o):
            x = torch.randn(2, 3) + random.random() + np.random.random()
            o.zero_grad()
            m(x).square().mean().backward()
            o.step()
        update(model, opt)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'model.pth')
            progress = dict(epoch=1, next_batch=2, signature={'seed': 7})
            trainer_utils.save_checkpoint(model, path, opt, step=3, training_state=progress)
            update(model, opt)
            other = torch.nn.Sequential(torch.nn.Linear(3, 3), torch.nn.Dropout(.3))
            other_opt = torch.optim.AdamW(other.parameters(), lr=.01)
            resumed = training_state.restore_training(str(Path(folder) / 'model_opt.pth'),
                                                      other, other_opt, None, {'seed': 7})
            self.assertEqual((resumed['epoch'], resumed['next_batch'], resumed['step']), (1, 2, 3))
            update(other, other_opt)
            for a, b in zip(model.parameters(), other.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, 'signature'):
                training_state.restore_training(str(Path(folder) / 'model_opt.pth'),
                                                other, other_opt, None, {'seed': 8})

    def test_old_optimizer_checkpoint_is_not_silent_exact_resume(self):
        model = torch.nn.Linear(1, 1)
        opt = torch.optim.AdamW(model.parameters())
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'old.pth')
            torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict(), step=1), path)
            with self.assertRaisesRegex(ValueError, 'training_state'):
                training_state.restore_training(path, model, opt, None, {})

    def test_partial_accumulation_window(self):
        self.assertEqual([training_state.accumulation_size(i, 5, 3) for i in range(5)], [3, 3, 3, 2, 2])


if __name__ == '__main__':
    unittest.main()
