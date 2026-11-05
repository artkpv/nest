"""
Render the unified counting + Reasoning Gym stealth scatter.

Pulls counting cells from legacy ``artifacts/stego_counting/**`` (using
``<trial>.monitor.json`` sidecars produced by ``scripts/monitor_counting.py``)
and RG cells from a directory of Inspect ``stego_cot_eval`` ``.eval`` logs.
Either source is optional — pass only what you have.

Usage:
  uv run python scripts/plot_unified_stealth.py \\
    --counting-root artifacts/stego_counting \\
    --counting-models gpt-5.2 claude-opus-4-5 claude-opus-4-6 claude-sonnet-4-6 gpt-5.4 \\
    --rg-log-dir artifacts/2026-05-01/main_sweep_v2/eval_logs \\
    --out artifacts/2026-05-02/unified_stealth/unified_stealth
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from stego_inspect.analysis.unified_stealth import (
    build_unified_cells,
    plot_unified_stealth,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--counting-root",
        type=Path,
        default=None,
        help="Root for legacy counting trials (with .monitor.json sidecars).",
    )
    p.add_argument(
        "--counting-models",
        nargs="*",
        default=[],
        help="Optional substring filter on model name (OR'd).",
    )
    p.add_argument(
        "--rg-log-dir",
        type=Path,
        default=None,
        help="Directory (or comma-separated list) of stego_cot_eval .eval logs.",
    )
    p.add_argument(
        "--min-trials",
        type=int,
        default=5,
        help="Minimum trials per cell to include in the plot.",
    )
    p.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output path stem (without extension). .pdf and .png written.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if not args.counting_root and not args.rg_log_dir:
        print("ERROR: pass --counting-root and/or --rg-log-dir", file=sys.stderr)
        sys.exit(2)

    cells = build_unified_cells(
        counting_root=str(args.counting_root) if args.counting_root else None,
        rg_log_dir=str(args.rg_log_dir) if args.rg_log_dir else None,
        counting_models=args.counting_models or None,
        min_trials=args.min_trials,
    )

    if cells.empty:
        print("No cells produced — nothing to plot.", file=sys.stderr)
        sys.exit(1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot_unified_stealth(cells, output_path=args.out)
    csv_path = args.out.with_suffix(".csv")
    cells.to_csv(csv_path, index=False)
    print(f"Cells: {len(cells)}")
    print(cells.to_string(index=False))
    print(f"Wrote: {args.out.with_suffix('.pdf')}")
    print(f"Wrote: {args.out.with_suffix('.png')}")
    print(f"Wrote: {csv_path}")


if __name__ == "__main__":
    main()
