"""Rebuild README plots from the committed numeric snapshot; --refresh reads local runs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'assets/readme'
COLORS = ['#138b99', '#e6a23a', '#6a78b8']


def refresh():
    sources = {}

    def read(path):
        raw = (ROOT / path).read_bytes()
        sources[path] = hashlib.sha256(raw).hexdigest()
        return raw.decode('utf-8-sig')

    def log(path):
        train, val = [], []
        for line in read(path).splitlines():
            row = {k: float(v) for k, v in re.findall(r'(\w+)=([-+\d.eE]+)', line)}
            if 'loss' in row:
                if train and row['step'] < train[-1]['step']:
                    train, val = [], []  # Keep only the final run after a restart.
                train.append(row)
            elif 'val_nll' in row:
                val.append(row)
        return {'train': train, 'val': val}

    data = {'status': 'Historical full runs predate correctness repairs. Demo uses the repaired 5-step quickstart.',
            'mlm': log('out/mlm/logs/train.log'),
            'decision': log('out/decision/logs/train.log')}
    data['api'] = json.loads(read('out/compare_apis.json'))
    for name, path in [('synth', 'out/eval/decision/decision/synth_test_known.json'),
                       ('public', 'out/eval/public/decision/public_test_known.json')]:
        report = json.loads(read(path))
        data[name] = {k: report[k] for k in ['n', 'env', 'metrics', 'calibrated']}
    sys.path.insert(0, str(ROOT))
    from model.inference import DecisionPredictor
    ckpt = 'out/repair_validation/quickstart/decision/decision.pth'
    tok = 'out/repair_validation/quickstart/tokenizer'
    predictor = DecisionPredictor(str(ROOT / ckpt), str(ROOT / tok), device='cpu')
    for path in [ckpt, tok + '/tokenizer.json']:
        sources[path] = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
    data['demo'] = []
    for kind in ['noul', 'choice', 'score']:
        request = json.loads(read(f'examples/{kind}.json'))
        data['demo'].append({'request': request, 'output': predictor.predict(request)})
    data['sources_sha256'] = sources
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'experiment_data.json').write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return data


def setup():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'axes.edgecolor': '#ccd6de', 'axes.labelcolor': '#42566d',
                         'text.color': '#142e4a', 'xtick.color': '#42566d',
                         'ytick.color': '#42566d', 'figure.facecolor': '#f7fafc',
                         'axes.facecolor': '#ffffff', 'savefig.facecolor': '#f7fafc'})


def finish(fig, name, title, subtitle, foot, bottom=.19):
    fig.suptitle(title, x=.06, y=.98, ha='left', fontsize=23, fontweight='bold')
    fig.text(.06, .90, subtitle, fontsize=11, color='#52677f')
    fig.text(.06, .025, foot, fontsize=9, color='#52677f', va='bottom')
    fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=bottom, wspace=.34)
    fig.savefig(OUT / (name + '.png'), dpi=170)
    plt.close(fig)


def trailing_mean(values, window=21):
    return [np.mean(values[max(0, i-window+1):i+1]) for i in range(len(values))]


def training(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    for ax, key, title, color in zip(axes[:2], ['mlm', 'decision'],
                                   ['MLM pretraining', 'Decision training'], COLORS):
        rows = data[key]['train']
        steps, losses = [r['step'] for r in rows], [r['loss'] for r in rows]
        ax.plot(steps, losses, color=color, alpha=.20, lw=.8, label='Logged batch loss')
        ax.plot(steps, trailing_mean(losses), color=color, lw=2.1, label='Trailing mean (21 records)')
        ax.set(title=title, xlabel='Optimizer step', ylabel='Loss')
        ax.grid(axis='y', alpha=.16)
        ax.legend(frameon=False, fontsize=8, loc='upper right')
        ax.ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
    rows = data['decision']['val']
    axes[2].plot([r['step'] for r in rows], [r['val_nll'] for r in rows], color=COLORS[2], lw=2)
    axes[2].set(title='Decision validation', xlabel='Optimizer step', ylabel='Validation NLL')
    axes[2].ticklabel_format(axis='x', style='sci', scilimits=(3, 3))
    axes[2].grid(axis='y', alpha=.16)
    finish(fig, 'training', 'Learning from the training logs',
           'HISTORICAL RUNS  |  MLM: 66,000 steps  /  Decision: 33,750 steps',
           'Source: out/{mlm,decision}/logs/train.log. Last run only; no interpolation.\nDecision run predates ordinal-loss repairs. Batch loss varies with task and candidate count; this is not a current-model benchmark.')


def evaluation(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    for ax, metric, title in zip(axes[:2], ['nll', 'brier'], ['Cross-entropy / NLL', 'Squared distribution error']):
        values = [data[k]['metrics']['all'][metric] for k in ['synth', 'public']]
        bars = ax.bar(['Synthetic\nn=18,000', 'Public text\nn=29,043'], values, color=COLORS[:2], width=.55)
        ax.bar_label(bars, fmt='%.3f', padding=6)
        ax.set(title=title, ylabel='Lower is better', ylim=(0,max(values)*1.23))
        ax.grid(axis='y', alpha=.15)
    s = data['synth']
    vals = [s['metrics']['all']['ece'], s['calibrated']['global']['all']['ece']]
    bars = axes[2].bar(['Raw', 'Global T'], vals, width=.55, color=[COLORS[2], COLORS[0]])
    axes[2].bar_label(bars, fmt='%.4f', padding=6)
    axes[2].set(title='Synthetic ECE: calibration', ylabel='Lower is better', ylim=(0,max(vals)*1.3))
    axes[2].grid(axis='y', alpha=.15)
    finish(fig, 'evaluation', 'What changes outside the training domain?',
           'HISTORICAL CHECKPOINT  |  Trained on synthetic tasks; evaluated on held-out synthetic and adapted public data',
           'Different datasets, candidate counts and label distributions: these bars show a domain gap, not a controlled causal comparison.\nCalibration panel uses the same synthetic test set. Historical results predate repairs; ordinal metrics are intentionally omitted.')


def api_comparison(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.7))
    entries = data['api']['summary']
    names = ['MiniSystemOne\n48/48 valid', 'Jev API\n48/48 valid', 'DeepSeek API\n37/48 valid']
    for ax, metric, title in zip(axes[:2], ['soft_acc', 'brier'], ['Soft accuracy (higher is better)', 'Distribution error (lower is better)']):
        vals = [entries[k][metric] for k in ['ours', 'jev', 'deepseek']]
        bars = ax.bar(names, vals, width=.55, color=COLORS)
        bars[2].set_hatch('//')
        ax.bar_label(bars, fmt='%.3f', padding=6)
        ax.set(title=title, ylim=(0,max(vals)*1.25))
        ax.tick_params(axis='x', labelsize=9)
        ax.grid(axis='y', alpha=.15)
    vals = [entries[k]['n'] for k in ['ours', 'jev', 'deepseek']]
    axes[2].bar(names, vals, width=.55, color=COLORS)
    axes[2].bar(names, [48-v for v in vals], bottom=vals, width=.55, color='#e4e9ef', hatch='//')
    axes[2].set(title='Valid responses / 48 attempts', ylim=(0,56), ylabel='Examples')
    axes[2].tick_params(axis='x', labelsize=9)
    for i,v in enumerate(vals):axes[2].text(i,v+1,str(v),ha='center')
    finish(fig, 'api-comparison', 'A small API comparison, with its limits visible',
           'HISTORICAL EXPLORATORY SAMPLE  |  48 synthetic examples  /  8 per source  /  K <= 8',
           'Source: out/compare_apis.json. DeepSeek metrics cover 37 valid responses; 11 parse failures were excluded: not a shared-sample ranking.\nModel/API versions are not recorded in the summary. No general superiority claim; local-vs-network latency is not compared.')


def demo(data):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5))
    titles = ['Noul  /  permission check', 'Choice  /  tool routing', 'Score  /  execution rating']
    questions = ['May this account proceed?', 'Which tool matches the request?', 'What grade should this run receive?']
    for ax, item, title, question in zip(axes, data['demo'], titles, questions):
        result = item['output']
        labels = [v['candidate'].replace('_',' ') for v in result['probabilities']]
        vals = [v['probability'] for v in result['probabilities']]
        ax.barh(labels, vals, color=COLORS[:len(vals)], height=.48)
        ax.invert_yaxis()
        for i,value in enumerate(vals): ax.text(value+.02,i,f'{value:.1%}', va='center',fontsize=11)
        ax.set_xlim(0,1)
        ax.set_xlabel('Predicted probability')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=40)
        ax.text(.0,1.045,question,transform=ax.transAxes,fontsize=9)
        ax.grid(axis='x',alpha=.15)
        detail = 'Human review (confidence < 0.8)' if result['primitive']=='choice' else (
            f"Expected score: {result['expected_score']:.3f} / 3" if result['primitive']=='score' else f"P(yes): {result['p_true']:.3f}")
        ax.text(0,-.25,detail,transform=ax.transAxes,fontsize=10,color='#138b99')
    finish(fig, 'task-demo', 'Three interfaces, actual model outputs',
           'REPAIRED QUICKSTART  |  128 x 2 encoder  /  5 MLM + 5 decision steps  /  CPU inference  /  no temperature',
           'Source: examples/{noul,choice,score}.json and the repaired quickstart checkpoint. The near-uniform outputs show an undertrained model.\nThis demonstrates the working interface and fallback route, not task competence. Threshold 0.8 is illustrative; no banking action is executed.', bottom=.30)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--refresh',action='store_true',help='Read original local logs and rerun the quickstart demo')
    args=parser.parse_args()
    data=refresh() if args.refresh else json.loads((OUT/'experiment_data.json').read_text(encoding='utf-8'))
    setup()
    for fn in [training,evaluation,api_comparison,demo]:fn(data)
    print('Wrote four figures to',OUT)


if __name__ == '__main__':
    main()
