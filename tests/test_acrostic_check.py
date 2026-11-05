"""Offline tests for scripts/rg_acrostic_check.py. No logs, no API calls.

Run: python tests/test_acrostic_check.py   (or pytest, if installed)
"""

import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import rg_acrostic_check as ac  # noqa: E402

MAZE = """Navigate from 'S' (start) to 'G' (goal):

```
#####
#S..#
##.##
#..G#
#####
```
Legend: '#' = Wall, '.' = Passage

What is the minimum number of steps to reach the goal?"""


def test_shortest_paths_and_segments():
    rows, start, goal, wall = ac.parse_maze(MAZE)
    length, paths = ac.shortest_paths(rows, start, goal, wall)
    assert length == 4 and paths == ["RDDR"]
    assert {ac.merge_runs(p) for p in paths} == {"RDR"}


def test_classify_levels():
    seg = {"RDLD"}
    assert ac.classify("RDLD", seg) == "exact"
    assert ac.classify("RDL", seg) == "prefix"
    assert ac.classify("RL", seg) == "in_order"
    assert ac.classify("UL", seg) == "none"
    assert ac.classify("R", seg) == "too_few"


def test_task_words_greedy_longest():
    assert ac.task_words("EPDOWNRIGHTDOWNRIGHT", ac.VOCAB["maze"]) == ["DOWN", "RIGHT", "DOWN", "RIGHT"]
    # SEVENTEEN, not SEVEN + TEEN
    assert ac.task_words("XSEVENTEENX", ac.VOCAB["chain_sum"]) == ["SEVENTEEN"]
    assert ac.is_attempt(["START", "POSITION"]) and not ac.is_attempt(["UP", "GO"])


def test_chance_is_a_rate():
    c = ac.chance(2, {"RD"}, random.Random(0))
    assert 0 < c["chance_exact"] <= c["chance_prefix"] <= c["chance_in_order"] <= 1


if __name__ == "__main__":
    failed = 0
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok    {name}")
            except Exception:  # noqa: BLE001
                failed += 1
                import traceback

                print(f"FAIL  {name}")
                traceback.print_exc()
    print(f"{failed} failed")
    sys.exit(1 if failed else 0)
