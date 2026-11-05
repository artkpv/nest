#!/usr/bin/env python3
"""
Summarize a bit-channel screening run.

Usage:
    python scripts/summarize_bitchannel_screen.py \
        artifacts/<date>/bitchannel_evidence

Prints:
  - per (model, algorithm) summary table: mean, std, n
  - a few sample (target, extracted) pairs per cell for qualitative sanity
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stego_inspect.analysis.eval_reader import load_all_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "out_dir",
        type=Path,
        help="Experiment out-dir (contains screen_logs/ and screen_results.yaml).",
    )
    parser.add_argument(
        "--examples",
        type=int,
        default=3,
        help="Number of (target, extracted) examples per cell to print.",
    )
    args = parser.parse_args()

    log_dir = args.out_dir / "screen_logs"
    if not log_dir.exists():
        print(f"ERROR: {log_dir} does not exist.", file=sys.stderr)
        return 1

    df = load_all_results(log_dir=str(log_dir), task_name="stego_message_eval")
    if df.empty:
        print("No results found.", file=sys.stderr)
        return 1
    if "encoding_scorer" not in df.columns:
        print("No encoding_scorer column in results.", file=sys.stderr)
        return 1

    algo_col = (
        "task_arg__algorithm" if "task_arg__algorithm" in df.columns else None
    )
    if algo_col is None:
        print("No task_arg__algorithm column; is this a stego_message_eval log?")
        return 1

    summary = (
        df.groupby(["model", algo_col])
        .agg(
            mean=("encoding_scorer", "mean"),
            std=("encoding_scorer", "std"),
            n=("encoding_scorer", "size"),
        )
        .round(3)
        .sort_index()
    )
    print("=== Summary ===")
    print(summary.to_string())
    print()

    extracted_col = None
    for cand in ("encoding_scorer__extracted", "encoding_scorer__answer"):
        if cand in df.columns:
            extracted_col = cand
            break

    target_col = "meta__target_string" if "meta__target_string" in df.columns else None

    if extracted_col and target_col:
        print("=== Examples per cell ===")
        for (model, algo), g in df.groupby(["model", algo_col]):
            print(f"\n[{model} / {algo}]")
            sub = g[[target_col, extracted_col, "encoding_scorer"]].head(args.examples)
            for _, row in sub.iterrows():
                print(
                    f"  target={row[target_col]!r}  "
                    f"extracted={row[extracted_col]!r}  "
                    f"score={row['encoding_scorer']:.3f}"
                )
    else:
        print(
            "(target/extracted columns not found — skipping examples. "
            f"target_col={target_col!r}, extracted_col={extracted_col!r})"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
