# README visual sources

The overview is an AI-generated conceptual illustration. All other figures are deterministic Matplotlib plots of recorded data, not AI-generated experimental results.

## Reproduce the charts

From the repository root, with the project dependencies installed:

```shell
python scripts/make_readme_figures.py
```

This reads the committed [numeric snapshot](experiment_data.json); original weights and ignored `out/` files are not required. The snapshot records SHA-256 hashes of its original source files. To refresh from the original local run paths and rerun the CPU demonstration:

```shell
python scripts/make_readme_figures.py --refresh
```

Refreshing requires the local logs, evaluation reports, and repaired quickstart checkpoint. It replaces the snapshot and charts. Do not relabel a historical run as a new result without recording new sources and updating captions.

| Figure | Source and interpretation |
|---|---|
| `overview.png` | Built-in image generation; conceptual teaching illustration, no measured claims |
| `training.png` | `out/mlm/logs/train.log`, `out/decision/logs/train.log`; historical training before ordinal repairs. Retains the final run after a step counter reset, separates validation rows, shows raw logged loss and a trailing mean of 21 records. Logged batch loss is not an epoch mean. |
| `evaluation.png` | `out/eval/decision/decision/synth_test_known.json` and `out/eval/public/decision/public_test_known.json`; historical raw NLL/Brier and synthetic raw/global-temperature ECE. Datasets have different tasks and target distributions; this is not a controlled comparison. No outdated ordinal metric is plotted. |
| `api-comparison.png` | `out/compare_apis.json`; historical 48-example exploratory comparison, 8 per synthetic source, K ≤ 8. MiniSystemOne/Jev have 48 valid responses, DeepSeek 37 with 11 parse failures excluded from quality metrics. No common-success-subset results or model/API version identifiers are available in this summary. Do not interpret as a fair ranking or broad superiority claim. |
| `task-demo.png` | Actual CPU inference on all three repository example files using `out/repair_validation/quickstart/decision/decision.pth` and its tokenizer; 128 × 2 model with only five steps per stage. No temperature. These near-uniform outputs demonstrate the API and human-review fallback, not learned task competence. |

## Image generation prompt

Tool: built-in `image_gen.imagegen`. The tool does not expose a model-version selector, so no claim is made that this used a specific requested “image2.5” version.

```text
Use case: scientific-educational. Create a polished wide landscape README hero infographic for the open source educational project MiniSystemOne. White/very pale blue background, navy typography, teal and amber accents, restrained elegant scientific editorial illustration, crisp readable English text, generous whitespace. Large title 'MiniSystemOne', subtitle 'Build a decision model from scratch'. Show a consumer GPU on the left connected to a small layered neural encoder in the middle, then three clean output cards on the right labeled 'Noul', 'Choice', 'Score', with small icons: yes/no toggle, branching tool choices, ordinal rating blocks. Along the bottom show a clearly ordered teaching path with six steps and arrows: 'Data' → 'Tokenizer' → 'MLM' → 'Decision training' → 'Calibration' → 'Evaluation'. Small footer 'Conceptual overview • Not experimental results'. Do not draw numerical probabilities, benchmark bars, training curves, performance claims or fabricated metrics. Flat vector-like illustration with a subtle dimensional GPU, suitable as a premium GitHub README masthead. Aspect ratio approximately 2:1. All labels accurately spelled, no extra logos.
```

The original generated asset was copied into this directory; README rendering does not depend on any machine-local generation cache. Chart labels use English in both README versions, with localized explanations beside each image.
