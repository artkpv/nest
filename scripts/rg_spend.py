#!/usr/bin/env python3
"""Print what the clean re-run has spent so far, per run directory and model.

Usage (from code/): python scripts/rg_spend.py artifacts/<date>/rg_clean artifacts/<date>/rg_clean_mapping
"""

from __future__ import annotations

import sys
from pathlib import Path

from stego_inspect.spend import run_spend


def main() -> int:
    total = 0.0
    for run_dir in sys.argv[1:]:
        spend = run_spend(Path(run_dir))
        total += spend["usd"]
        print(f"{run_dir}: ${spend['usd']:.2f}")
        for model, row in spend["by_model"].items():
            print(f"  {model}: {row['input']:,} in, {row['output']:,} out, ${row['usd']:.2f}")
        if spend["unpriced"]:
            print(f"  NOT PRICED, counted as $0: {spend['unpriced']}")
        if spend["unreadable"]:
            print(f"  UNREADABLE, not counted: {spend['unreadable']}")
    print(f"total ${total:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
