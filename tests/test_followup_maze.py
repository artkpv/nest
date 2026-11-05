"""Offline tests for the maze_varied follow-up pool. No logs, no API calls.

Run: python tests/test_followup_maze.py   (or pytest, if installed)
"""

import collections
import sys
from pathlib import Path

import reasoning_gym

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from stego_inspect.calibrate_rg_difficulty import (  # noqa: E402
    LADDERS,
    balance_answers,
    generate_problems,
)


def test_balanced_answers():
    probs = generate_problems("maze_varied", 1, 600, 7003)
    counts = collections.Counter(p["answer"] for p in probs)
    assert len(probs) == 600
    assert sorted(counts) == ["4", "5", "6", "7", "8"]
    assert set(counts.values()) == {120}
    assert all(p["metadata"]["source_dataset"] == "maze" for p in probs)
    assert len({p["question"] for p in probs}) == 600


def test_deterministic():
    a = generate_problems("maze_varied", 1, 50, 7003)
    b = generate_problems("maze_varied", 1, 50, 7003)
    assert [p["question"] for p in a] == [p["question"] for p in b]


def test_existing_ladders_unchanged():
    # The published maze rung still comes straight from reasoning_gym, unbalanced.
    got = [p["question"] for p in generate_problems("maze", 3, 20, 2005)]
    ds = reasoning_gym.create_dataset("maze", size=20, seed=2005, **LADDERS["maze"][2])
    assert got == [x["question"] for x in ds]
    assert len(LADDERS["maze"]) == 9 and LADDERS["maze"][2]["min_dist"] == LADDERS["maze"][2]["max_dist"] == 5


def test_balance_round_robin():
    probs = [{"answer": a} for a in "4444455678"]
    assert [p["answer"] for p in balance_answers(probs, 5)] == ["4", "5", "6", "7", "8"]
    try:
        balance_answers(probs, 8)
    except ValueError:
        pass
    else:
        raise AssertionError("ran out of 6s, should raise")


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
