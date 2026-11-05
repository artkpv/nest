"""Offline tests for the opt-in ttc filter (filter_rg_datasets) and ttc ceiling
(calibrate_rg_difficulty). No API calls: the model and probes are stubbed.

Run: python tests/test_ttc_filter.py   (or pytest, if installed)
"""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

# Probes are stubbed below, so problem generation is never reached; this lets
# the calibration tests run where reasoning_gym is not installed.
sys.modules.setdefault("reasoning_gym", ModuleType("reasoning_gym"))

import stego_inspect.calibrate_rg_difficulty as cal  # noqa: E402
import stego_inspect.filter_rg_datasets as flt
from stego_inspect.solvers.baselines import DEFAULT_N_FILLER_TOKENS


TARGET = "42"
DOTS = "." * DEFAULT_N_FILLER_TOKENS


class FakeModel:
    """Answers each arm per a script keyed by problem text.

    script[problem][arm] = (completion, stop_reason) or an Exception to raise.
    """

    def __init__(self, script):
        self.script = script
        self.calls = []

    async def generate(self, input, config=None):
        system = input[0].content
        user = input[1].content
        if "direct-answer endpoint" in system:
            arm = "direct"
        elif "test-time-compute endpoint" in system:
            arm = "ttc"
            assert f"EXACTLY {DEFAULT_N_FILLER_TOKENS} dots" in user
        else:
            arm = "cot"
        problem = next(p for p in self.script if p in user)
        self.calls.append((problem, arm))
        out = self.script[problem][arm]
        if isinstance(out, Exception):
            raise out
        completion, stop = out
        return SimpleNamespace(completion=completion, stop_reason=stop)


WRONG_DIRECT = ('{"answer": 7}', "stop")
RIGHT_COT = ("step... ANSWER: 42", "stop")
RIGHT_TTC = (f"{DOTS}\nANSWER: 42", "stop")
WRONG_TTC = (f"{DOTS}\nANSWER: 41", "stop")

SCRIPT = {
    "P0 ttc solves": {"direct": WRONG_DIRECT, "cot": RIGHT_COT, "ttc": RIGHT_TTC},
    "P1 ttc fails": {"direct": WRONG_DIRECT, "cot": RIGHT_COT, "ttc": WRONG_TTC},
    "P2 ttc truncated": {
        "direct": WRONG_DIRECT, "cot": RIGHT_COT, "ttc": ("....", "max_tokens"),
    },
    "P3 ttc content_filter": {
        "direct": WRONG_DIRECT, "cot": RIGHT_COT, "ttc": ("", "content_filter"),
    },
    "P4 ttc errors": {
        "direct": WRONG_DIRECT, "cot": RIGHT_COT, "ttc": RuntimeError("boom"),
    },
    # ttc reasoned in plain text: NOT_FOUND, graded wrong exactly as the eval does.
    "P5 ttc reasons": {
        "direct": WRONG_DIRECT, "cot": RIGHT_COT, "ttc": ("12+30 ANSWER: 42", "stop"),
    },
    "P6 direct solves": {"direct": ('{"answer": 42}', "stop"), "cot": RIGHT_COT, "ttc": WRONG_TTC},
}
PROBLEMS = [{"question": q, "answer": TARGET} for q in SCRIPT]


def _grade(ttc_filter):
    model = FakeModel(SCRIPT)
    orig = flt.get_model
    flt.get_model = lambda name: model
    try:
        result = asyncio.run(
            flt.grade_problems(PROBLEMS, "anthropic/fake", 4, ttc_filter=ttc_filter)
        )
    finally:
        flt.get_model = orig
    return result, model


def test_filter_off_is_legacy_and_skips_ttc():
    (filtered, n_direct, n_cot, n_total, records), model = _grade(ttc_filter=False)
    assert filtered == [0, 1, 2, 3, 4, 5]
    assert (n_direct, n_cot, n_total) == (1, 7, 7)
    assert not any(arm == "ttc" for _, arm in model.calls)
    assert all("ttc" not in r and r["keep"] == r["keep_without_ttc"] for r in records)


def test_filter_on_drops_ttc_solved_and_undetermined():
    (filtered, *_ , records), _ = _grade(ttc_filter=True)
    # kept: ttc wrong (P1) and ttc format violation (P5); dropped: ttc solved
    # (P0), truncated (P2), content filter (P3), error (P4), direct solved (P6).
    assert filtered == [1, 5]
    by = {r["problem"]: r for r in records}
    assert by["P0 ttc solves"]["ttc"]["correct"] is True
    assert by["P2 ttc truncated"]["ttc"]["determined"] is False
    assert by["P3 ttc content_filter"]["ttc"]["determined"] is False
    assert by["P4 ttc errors"]["ttc"]["error"] == "boom"
    assert by["P5 ttc reasons"]["ttc"]["answer"] == "NOT_FOUND"
    assert all(r["n_filler_tokens"] == DEFAULT_N_FILLER_TOKENS for r in records)
    assert all("stop_reason" in r[a] for r in records for a in ("direct", "cot", "ttc"))

    stats = flt.summarize_ttc(records)
    assert stats["n_ttc_run"] == 7
    assert stats["n_ttc_determined"] == 4  # P0, P1, P5, P6
    assert stats["n_ttc_ok"] == 1
    assert stats["ttc_rate"] == 0.25
    assert stats["n_kept_without_ttc"] == 6


# ---- calibration ----------------------------------------------------------

# direct falls below 0.3 at D=4, but filler keeps solving until D=7.
DIRECT = {1: 0.9, 2: 0.7, 3: 0.5, 4: 0.2}
RATES = {D: (DIRECT.get(D, 0.0), 1.0 if D < 7 else 0.2, 0.95) for D in range(1, 11)}


def _search(target_ttc_rate, d_hi=10):
    probed = []

    async def fake_probe(family, model_name, D, n_probe, concurrency, seed,
                         probe_log_path=None, ttc_filter=False, n_filler_tokens=400,
                         protocol=None):
        probed.append((D, ttc_filter))
        direct, ttc, cot = RATES[D]
        rec = {"D": D, "direct_rate": direct, "cot_rate": cot, "uplift": cot - direct}
        if ttc_filter:
            rec["ttc_rate"] = ttc
            rec["uplift_vs_ttc"] = cot - max(direct, ttc)
        return rec

    orig = cal._probe
    cal._probe = fake_probe
    try:
        out = asyncio.run(cal.binary_search_edge(
            "chain_sum", "anthropic/fake", 1, d_hi, 10, 4, 0,
            target_direct_rate=0.3, target_ttc_rate=target_ttc_rate,
        ))
    finally:
        cal._probe = orig
    return out, probed


def test_calibration_without_ceiling_is_legacy():
    (D, _sweep, met), probed = _search(None)
    assert D == 4 and met is True
    assert not any(t for _, t in probed)


def test_calibration_ceiling_moves_edge_past_filler():
    (D, sweep, met), probed = _search(0.3)
    assert D == 7 and met is True
    assert all(t for _, t in probed)
    assert all(r["ttc_rate"] <= 0.3 for r in sweep if r["D"] == D)


def test_calibration_ceiling_unmet_is_flagged():
    (D, _sweep, met), _ = _search(0.3, d_hi=6)
    assert D == 6 and met is False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"ok  {name}")
