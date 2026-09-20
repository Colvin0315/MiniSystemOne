"""Small shared resume contract. Checkpoints are saved at optimizer boundaries."""
import random

import numpy as np
import torch


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def capture_rng():
    ns = np.random.get_state()
    return dict(python=random.getstate(), numpy=[ns[0], ns[1].tolist(), ns[2], ns[3], ns[4]],
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    ns = state['numpy']
    np.random.set_state((ns[0], np.asarray(ns[1], dtype=np.uint32), ns[2], ns[3], ns[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda']:
        if not torch.cuda.is_available() or len(state['cuda']) != torch.cuda.device_count():
            raise ValueError('Resume requires the same CUDA device count')
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def restore_training(path, model, optimizer, scaler, signature):
    ck = torch.load(path, map_location='cpu', weights_only=True)
    state = ck.get('training_state')
    if not state or 'rng' not in state:
        raise ValueError('Checkpoint has no complete training_state; use it only as initialization, not exact resume')
    if state.get('signature') != signature:
        raise ValueError('Resume signature differs: keep training data, tokenizer and training options unchanged')
    model.load_state_dict(ck['model'], strict=True)
    optimizer.load_state_dict(ck['optimizer'])
    if scaler is not None and ck.get('scaler') is not None:
        scaler.load_state_dict(ck['scaler'])
    restore_rng(state['rng'])
    return dict(state, step=ck['step'])


def accumulation_size(micro, batches, accum):
    """Average the final incomplete window by its actual microbatch count."""
    return min(accum, batches - (micro // accum) * accum)


def resume_signature(args, data_hashes):
    # Output locations and stop points can change; schedule/data/model cannot.
    ignored = {'resume', 'encoder', 'out', 'log_dir', 'log_interval', 'save_interval',
               'save_optimizer', 'no_swanlab', 'max_steps', 'bench_seconds', 'val_every'}
    return dict(options={k: v for k, v in vars(args).items() if k not in ignored},
                data=data_hashes)
