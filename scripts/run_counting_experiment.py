"""
Steganographic counting experiment runner.

Replaces the standalone runner that was removed in `29f671ad`
(`artifacts-Feb-2026/run_stego_counting_experiment.py`, plus its
`steg_counting_experiment.py` / `api_client.py` dependencies). That stack is
archived and must not be imported from; this script keeps the same *experiment
design* while running on the maintained Inspect path, so there is a single code
path and a single scorer.

Equivalence to the removed runner is not assumed — it is tested. See
`scripts/validate_counting_scorer.py`, which replays `counting_scorer` over the
stored legacy trial JSONs and requires 100% agreement on both `exact_match` and
`per_number_accuracy`.

Design notes (inherited from the original runner, keep them):

* **Digits-per-number is random, by design.** `generate_random_interval` picks
  `digit_count` from the divisors of D, so a cell at a given D mixes 1-, 2- and
  (where D allows) 3-digit numbers. The unit of analysis is therefore **all
  samples at a given D**. Do not report the `dpn=1` slice as a headline: it
  discards data, and at some D values it collapses to very few distinct
  sequences (D=8 admits only `[1..8]` and `[2..9]`). `dpn` is a diagnostic
  breakdown — multi-digit composition is the dominant failure mode.
* `dpn=3` only occurs when `D % 3 == 0`, so a D sweep that is all powers of two
  never exercises it. Include a D divisible by 3 (6, 12, 15) for full coverage.
* The monitor runs unconditionally for counting (`tasks.py`) — it supplies the
  `refused` signal that replaced the retired standalone refusal scorer. Budget
  two model calls per sample.
* **The monitor needs an explicit grader.** `monitor_scorer` resolves its model
  as `get_model(role="grader", default=get_model())`, so with no grader role the
  model under test grades its own transcripts. That is the self-monitoring bug
  disclosed in the 2026-07-29 ablation, and it is silent — the run succeeds and
  the numbers look ordinary. `--grader` is therefore always passed through as a
  model role; it defaults to the monitor the published cells used.

Usage:
  # paper's D sweep, default roster
  python scripts/run_counting_experiment.py --out-dir artifacts/$(date +%F)/counting

  # match the Reasoning Gym roster, a few D values
  python scripts/run_counting_experiment.py \
      --models anthropic/claude-opus-4-6,openai/gpt-5.4 \
      --D 4,6,8 --n-samples 120 --out-dir artifacts/2026-07-28/counting_topup

  # see the deterministic dpn split for a config without spending anything
  python scripts/run_counting_experiment.py --D 4,6,8 --n-samples 120 --dry-run
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

from inspect_ai import eval as inspect_eval

from stego_inspect.datasets.counting import generate_random_interval
from stego_inspect.tasks import stego_cot_eval


# The Reasoning Gym sweep roster, so counting and RG share one model set.
DEFAULT_MODELS = [
    "anthropic/claude-opus-4-6",
    "anthropic/claude-sonnet-4-6",
    "anthropic/claude-haiku-4-5-20251001",
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
]

# The paper's counting sweep.
DEFAULT_D = [4, 6, 8, 10, 12, 15, 16, 20, 26, 50]

# Monitor used by the published cells; see stego_inspect/scorers/monitor.py.
DEFAULT_GRADER = "anthropic/claude-sonnet-4-6"


def dpn_split(D: int, n_samples: int, seed: int) -> dict[int, int]:
    """Digits-per-number composition of a cell, without any API calls.

    Interval generation is seeded by `seed + i*1000` and does not depend on the
    model, so the split is deterministic and knowable in advance.
    """
    counts: collections.Counter = collections.Counter()
    for i in range(n_samples):
        X, Y, I = generate_random_interval(D, seed=seed + i * 1000)
        widths = {len(str(abs(n))) for n in range(X, Y, I)}
        counts[widths.pop() if len(widths) == 1 else 0] += 1
    return dict(sorted(counts.items()))


def _warn_on_truncation(eval_log, model: str, D: int) -> None:
    """Report samples whose generation was cut off before it finished.

    A provider-side safety classifier can halt generation mid-sentence; the
    Anthropic API reports `stop_reason: "refusal"`, which Inspect surfaces as
    `content_filter`. Such a sample scores as an ordinary failure, so a cell can
    be dominated by truncated generations while looking merely difficult. This
    has to be visible at run time, not discovered in analysis.
    """
    samples = getattr(eval_log, "samples", None) or []
    if not samples:
        return
    cut = 0
    for s in samples:
        out = getattr(s, "output", None)
        if out is not None and str(getattr(out, "stop_reason", "")) in (
            "content_filter", "max_tokens", "model_length"
        ):
            cut += 1
    if cut:
        print(
            f"    WARNING: {cut}/{len(samples)} ({cut/len(samples):.0%}) generations "
            f"truncated (stop_reason content_filter/max_tokens) for {model} D={D}. "
            "These score as failures but carry no capability signal — exclude them "
            "or report the truncation rate alongside accuracy.",
            file=sys.stderr,
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=",".join(DEFAULT_MODELS),
                   help="comma-separated model ids (default: the Reasoning Gym roster)")
    p.add_argument("--D", default=",".join(str(d) for d in DEFAULT_D),
                   help="comma-separated D values; D = TOTAL DIGIT COUNT, not sequence length")
    p.add_argument("--n-samples", type=int, default=120,
                   help="samples per (model, D) cell")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", default=None, help="artifact dir; logs go to <out-dir>/eval_logs")
    p.add_argument("--max-connections", type=int, default=15)
    p.add_argument("--grader", default=DEFAULT_GRADER,
                   help="model for the monitor scorer (grader role). Passing '' "
                        "lets the monitor fall back to the model under test, "
                        "which is self-monitoring — don't, unless that is the point.")
    p.add_argument("--monitor-max-tokens", type=int, default=None,
                   help="cap the grader's output tokens (default: provider default)")
    p.add_argument("--reasoning", choices=("off", "on"), default="off",
                   help="'off' (default, every published cell) or 'on': generate with the "
                        "model's own thinking. 'on' is a separate arm - not comparable to "
                        "reasoning-off cells. Scorer and monitor still read only the answer.")
    p.add_argument("--reasoning-effort", default=None,
                   help="with --reasoning on: effort sent to the model under test only "
                        "(never the grader), e.g. 'medium' for gpt-oss")
    p.add_argument("--max-tokens", type=int, default=None,
                   help="per-generation output cap incl. reasoning (default: provider default). "
                        "Set it explicitly for --reasoning on, so both arms share one budget.")
    p.add_argument("--model-args", default=None,
                   help="JSON dict of provider model args, e.g. "
                        "'{\"providers\": [\"DeepInfra\"]}' for prefill/* models "
                        "or '{\"reasoning_enabled\": false}' for openrouter/*")
    p.add_argument("--dry-run", action="store_true",
                   help="print the cell matrix and dpn split, make no API calls")
    args = p.parse_args(argv)

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    d_values = [int(d) for d in args.D.split(",") if d.strip()]
    model_args = json.loads(args.model_args) if args.model_args else {}
    model_roles = {"grader": args.grader} if args.grader else None

    print(f"models   : {len(models)}")
    for m in models:
        print(f"           {m}")
    print(f"D values : {d_values}")
    print(f"grader   : {args.grader or 'NONE — model under test monitors itself'}")
    print(f"reasoning: {args.reasoning}"
          + (f"   effort: {args.reasoning_effort}" if args.reasoning_effort else "")
          + (f"   max_tokens: {args.max_tokens}" if args.max_tokens else ""))
    if model_args:
        print(f"model args: {model_args}")
    print(f"n_samples: {args.n_samples} per cell -> {len(models)*len(d_values)*args.n_samples} samples")
    print("           (counting runs the monitor unconditionally: ~2 model calls/sample)")
    print()
    print("dpn composition per cell (deterministic, model-independent):")
    for D in d_values:
        split = dpn_split(D, args.n_samples, args.seed)
        desc = ", ".join(f"dpn={k}: {v}" for k, v in split.items() if k)
        mixed = split.get(0, 0)
        print(f"  D={D:>3}: {desc}" + (f", mixed-width: {mixed}" if mixed else ""))
    if not any(3 in dpn_split(D, args.n_samples, args.seed) for D in d_values):
        print("  NOTE: no cell produces dpn=3 (needs D divisible by 3, e.g. 6/12/15).")

    if args.dry_run:
        return 0

    if not args.out_dir:
        p.error("--out-dir is required unless --dry-run")
    log_dir = Path(args.out_dir) / "eval_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    failures = []
    for D in d_values:
        for model in models:
            print(f"\n=== {model} | D={D} | n={args.n_samples} ===", flush=True)
            try:
                # Inspect returns EvalLog objects with status "error"/"cancelled"
                # rather than raising for the common failure paths (fail_on_error
                # threshold, cancellation), so the return value must be checked —
                # otherwise a failed cell looks identical to a successful one.
                eval_logs = inspect_eval(
                    stego_cot_eval(
                        hidden_dataset="counting",
                        cover_dataset="custom_topics",
                        algorithm="sentence_acrostic",
                        D=D,
                        n_samples=args.n_samples,
                        seed=args.seed,
                        baselines="",
                        include_translation=False,
                        monitor_max_tokens=args.monitor_max_tokens,
                        reasoning=args.reasoning,
                        reasoning_effort=args.reasoning_effort,
                    ),
                    model=model,
                    model_args=model_args,
                    model_roles=model_roles,
                    **({"max_tokens": args.max_tokens} if args.max_tokens else {}),
                    log_dir=str(log_dir),
                    max_connections=args.max_connections,
                    retry_on_error=3,
                    fail_on_error=0.2,
                )
                for lg in (eval_logs or []):
                    if str(getattr(lg, "status", "")) != "success":
                        reason = getattr(getattr(lg, "error", None), "message", None) \
                            or f"status={getattr(lg, 'status', '?')}"
                        print(f"    FAILED: {reason}", file=sys.stderr)
                        failures.append((model, D, str(reason)))
                    else:
                        _warn_on_truncation(lg, model, D)
            except Exception as exc:  # keep the sweep going, report at the end
                print(f"    FAILED: {exc}", file=sys.stderr)
                failures.append((model, D, str(exc)))

    if failures:
        print(f"\n{len(failures)} cell(s) failed:", file=sys.stderr)
        for m, D, e in failures:
            print(f"  {m} D={D}: {e[:160]}", file=sys.stderr)
        return 1
    print("\nall cells completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
