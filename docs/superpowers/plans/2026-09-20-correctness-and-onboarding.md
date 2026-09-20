# Correctness and onboarding implementation plan

> Execute inline using executing-plans and test-driven-development. The user approved the review's repair sequence with “开始吧”.

**Goal:** Correct ordinal training and metrics, make training resumable, and provide a small usable inference/tutorial path.

**Architecture:** Keep the encoder and independent candidate scorer. Sort ordinal distributions by explicit levels only when computing ordinal losses/metrics; retain presentation order in model outputs. Share checkpoint/resume utilities between trainers and reuse canonical serialization for inference.

**Constraints:** No new runtime dependencies. Preserve old weights for inference, mark old reported experiments as historical. Do not overwrite the user's existing training artifacts. GPU training remains supported; CPU is supported for inference and regression tests.

- [x] Add failing unittest regressions for permutation/padding-safe ordinal loss, truncation priorities, abstention coverage, and Brier option propagation.
- [x] Repair model, dataset and metric implementations; run these regressions and architecture smoke tests.
- [x] Add resumable checkpoint state (epoch, next batch, RNG, configuration), deterministic epoch sampling, and partial accumulation flushing to both trainers. Test continuous versus interrupted training and reject incompatible resumes.
- [x] Add JSON inference CLI for Noul/Choice/Score using checkpoint config, tokenizer validation and chunked inference. Test valid and invalid requests.
- [x] Correct calibration explanations in both READMEs and technical docs, explain changed metrics and legacy weights, and add a minimal custom-task tutorial and isolated smoke commands.
- [x] Run regression suite, CPU architecture checks, small GPU training/resume and real-checkpoint CLI checks; inspect git diff for unintended changes.

Validation commands use `C:/Coding/Anaconda3/envs/minimind/python.exe`. Unit tests use `python -m unittest discover -s tests -v`; architecture tests use `python scripts/smoke_test.py --device cpu --skip_big`. New integration artifacts go under `out/repair_validation/`.

## Verified outcome

- 18 unittest tests passed with CUDA integration enabled (55.054 seconds). Both trainers produced identical full-precision weights after continuous versus interrupted/resumed training.
- CPU FP32 architecture smoke checks passed; large-K test was skipped.
- All seven QUICKSTART stages completed in out/repair_validation/quickstart, including fresh tokenizer, MLM, decision training, calibration, evaluation and CLI inference.
- Existing checkpoint inference and matching-temperature CLI inference passed. Existing flagship weights were not overwritten or retrained.
- git diff --check passed. Changes remain uncommitted on codex/correctness-and-onboarding.
