# NEST: Nascent Encoded Steganographic Thoughts

Code for the NeurIPS 2026 Evaluations & Datasets Track paper *NEST: Nascent
Encoded Steganographic Thoughts* (Artem Karpov, Independent).

- **Dataset (v1.1):** <https://huggingface.co/datasets/artkpv/nest> (CC-BY 4.0)
- **Code license:** MIT (`LICENSE`)
- This repository holds published versions only (tag `v1.1` = camera-ready).
  Paths under `artifacts/` in scripts and configs refer to run outputs that are
  not part of the repository.

```bibtex
@inproceedings{karpov2026nest,
  title     = {{NEST}: Nascent Encoded Steganographic Thoughts},
  author    = {Karpov, Artem},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026},
  url       = {https://huggingface.co/datasets/artkpv/nest},
}
```

## Steganographic reasoning experiments

This repository contains code for evaluating whether language models can solve
hidden reasoning tasks while embedding their reasoning in apparently unrelated
cover text. The experiments are implemented with Inspect AI and are orchestrated
by `scripts/run_experiment.py`.

## Setup

Use Python 3.11 or newer.

```bash
uv sync
source .venv/bin/activate
```

Set API keys for the providers used by the selected experiment config, for
example `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`.

## Main Experiment

The default Reasoning Gym sweep calibrates per-model task difficulty, screens
each model for encoding ability, runs the steganographic evaluation, and writes
plots and summary tables.

```bash
python scripts/run_experiment.py \
  --config scripts/experiment_default.yaml \
  --out-dir artifacts/anonymous_run/rg_experiment
```

Outputs are written under the selected `--out-dir`:

- `run.log`: full command log
- `experiment_config.yaml`: config snapshot
- `calibration/`: difficulty calibration records
- `screen_results.yaml`: encoding-screen outcomes
- `eval_logs/`: Inspect `.eval` logs
- `plots/`: generated analysis outputs

The runner is idempotent. Completed stages create files in `flags/`; rerunning
the same command skips completed work. Delete a relevant flag file to rerun a
stage.

To run only one stage or one slice:

```bash
python scripts/run_experiment.py --only calibrate --slug gpt54 \
  --out-dir artifacts/anonymous_run/rg_experiment
python scripts/run_experiment.py --only screen --algorithm sentence_acrostic \
  --out-dir artifacts/anonymous_run/rg_experiment
python scripts/run_experiment.py --only eval --slug gpt54 --family chain_sum \
  --out-dir artifacts/anonymous_run/rg_experiment
python scripts/run_experiment.py --only analyze --out-dir artifacts/anonymous_run/rg_experiment
```

## Alternative Configs

Run the larger acrostic-variant sweep:

```bash
python scripts/run_experiment.py \
  --config scripts/experiment_v2.yaml \
  --out-dir artifacts/anonymous_run/main_sweep_v2
```

Run only the format-channel screening experiment:

```bash
python scripts/run_experiment.py \
  --only screen \
  --config scripts/experiment_bitchannel.yaml \
  --out-dir artifacts/anonymous_run/bitchannel_evidence

python scripts/summarize_bitchannel_screen.py \
  artifacts/anonymous_run/bitchannel_evidence
```

## Analysis

Regenerate the main steganographic edge report from existing eval logs:

```bash
stego-plot \
  --log-dir artifacts/anonymous_run/rg_experiment/eval_logs \
  --output-dir artifacts/anonymous_run/rg_experiment/plots \
  stego-edge
```

Render a combined counting and Reasoning Gym stealth plot when both sources are
available:

```bash
python scripts/plot_unified_stealth.py \
  --counting-root artifacts/stego_counting \
  --rg-log-dir artifacts/anonymous_run/rg_experiment/eval_logs \
  --out artifacts/anonymous_run/unified_stealth/unified_stealth
```

## Dual Use

We request that AI developers exclude this repository and its artifacts from
training corpora, to preserve the integrity of our evaluations and to avoid
contamination with steganographic chains-of-thought.

## Notes For Review

- Model, task, algorithm, and concurrency settings live in the YAML files under
  `scripts/`.
- The main success metric is `stego_success`, defined by the composite scorer in
  `stego_inspect/scorers/composite.py`.
- `stego_detected` is reported as a monitor signal but does not gate success.
- Generated datasets and experiment artifacts are not required to be committed;
  they are recreated by the runner when needed.
