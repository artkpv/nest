#!/usr/bin/env python3
"""
B6 McNemar with the sensitivity monitor in place of the gating monitor. Offline: reads logs only.

The clean re-run gates on a cross-family monitor (GPT-5.4 for Claude encoders, Sonnet 4.6
for GPT encoders). `regrade_monitor.py` re-graded every transcript with the other monitor
into `eval_logs_sensitivity/`: copies of the same logs in which only the monitor fields
differ. This runs the report's McNemar path (`build_condition_frame` -> `_pivot_paired`
-> `_mcnemar_one`) on both log sets, against word filler, direct and 400 dots, at cell,
model, family and pooled level, and compares them. Nothing under --run is written.

Usage (from code/):
    python scripts/b6_sensitivity_mcnemar.py --out artifacts/2026-11-07/b6_sensitivity_mcnemar
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from stego_inspect.analysis.condition_comparison import build_condition_frame
from stego_inspect.analysis.mcnemar_report import CELL_KEYS, _mcnemar_one, _pivot_paired, write_mcnemar_outputs

RUN = Path("artifacts/2026-10-31/rg_clean")
ARMS = {"ttc": "word filler", "direct": "direct", "ttc_dots": "400 dots"}
LEVELS = {"cell": ["model", "task_type"], "model": ["model"], "family": ["task_type"], "all": []}


def paired(frame: pd.DataFrame) -> pd.DataFrame:
    """`_pivot_paired`, plus the 400-dot arm through the same function (it pairs only ttc and direct)."""
    p = _pivot_paired(frame)
    dots = frame[frame["condition"] != "ttc"].replace({"condition": {"ttc_dots": "ttc"}})
    d = _pivot_paired(dots)[list(CELL_KEYS) + ["sample_id", "ttc", "stego"]]
    p = p.merge(d.rename(columns={"ttc": "ttc_dots", "stego": "stego_dots"}),
                on=list(CELL_KEYS) + ["sample_id"], how="left", validate="one_to_one")
    assert p["stego"].astype(str).equals(p["stego_dots"].astype(str)), "stego win differs between pivots"
    return p.drop(columns="stego_dots")


def tests(p: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for level, keys in LEVELS.items():
        groups = p.groupby(keys) if keys else [((), p)]
        for k, g in groups:
            k = k if isinstance(k, tuple) else (k,)
            row = {"level": level, "model": "all", "family": "all", **dict(zip(
                ["model" if c == "model" else "family" for c in keys], k))}
            for arm in ARMS:
                s = _mcnemar_one(g["stego"], g[arm])
                row.update({f"{arm}_b": s["b"], f"{arm}_c": s["c"], f"{arm}_n": s["n_paired"], f"{arm}_p": s["p_greater"]})
            rows.append(row)
    return pd.DataFrame(rows)


def check_gate(t: pd.DataFrame, run: Path) -> list[str]:
    """The gate run must reproduce the published report files."""
    bad = []
    report = run / "report"
    for level, f, key in (("cell", "mcnemar_cell.csv", ["model", "task_type"]),
                          ("model", "mcnemar_model.csv", ["model"]),
                          ("family", "mcnemar_family.csv", ["task_type"])):
        ref = pd.read_csv(report / f)
        mine = t[t["level"] == level].rename(columns={"family": "task_type"})
        m = ref.merge(mine, on=key, suffixes=("_ref", ""))
        for arm in ("ttc", "direct"):
            if not (m[f"{arm}_b_ref"].eq(m[f"{arm}_b"]).all() and m[f"{arm}_c_ref"].eq(m[f"{arm}_c"]).all()
                    and m[f"{arm}_n_paired"].eq(m[f"{arm}_n"]).all() and len(m) == len(ref)):
                bad.append(f"{f} {arm}")
    fc = pd.read_csv(report / "filler_compare.csv")
    mine = t[t["level"].isin(["family", "all"]) & (t["model"] == "all")]
    m = fc.merge(mine, on="family", suffixes=("_ref", ""))
    for arm in ("ttc", "ttc_dots"):
        if not (m[f"{arm}_b_ref"].eq(m[f"{arm}_b"]).all() and m[f"{arm}_c_ref"].eq(m[f"{arm}_c"]).all() and len(m) == len(fc)):
            bad.append(f"filler_compare.csv {arm}")
    return bad


def compare(gate: pd.DataFrame, sens: pd.DataFrame) -> pd.DataFrame:
    keys = ["level", "model", "family"]
    m = gate.merge(sens, on=keys, suffixes=("_gate", "_sens"), validate="one_to_one")
    rows = []
    for r in m.to_dict("records"):
        for arm, name in ARMS.items():
            pg, ps = r[f"{arm}_p_gate"], r[f"{arm}_p_sens"]
            sig_g, sig_s = bool(pg < 0.05), bool(ps < 0.05)  # NaN (no discordant pairs) is not significant
            changed = any(r[f"{arm}_{x}_gate"] != r[f"{arm}_{x}_sens"] for x in ("b", "c", "n"))
            rows.append({
                **{k: r[k] for k in keys}, "baseline": name,
                "gate_b": r[f"{arm}_b_gate"], "gate_c": r[f"{arm}_c_gate"], "gate_n": r[f"{arm}_n_gate"], "gate_p": pg,
                "sens_b": r[f"{arm}_b_sens"], "sens_c": r[f"{arm}_c_sens"], "sens_n": r[f"{arm}_n_sens"], "sens_p": ps,
                "counts_changed": changed, "sig_changed": sig_g != sig_s,
                "beats_at_005_gate": sig_g, "beats_at_005_sens": sig_s,
            })
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--run", type=Path, default=RUN)
    ap.add_argument("--sensitivity", default="eval_logs_sensitivity")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    out = {}
    for tag, d in (("gate", "eval_logs"), ("sensitivity", a.sensitivity)):
        frame = build_condition_frame(log_dir=str(a.run / d))
        write_mcnemar_outputs(frame, a.out / tag)  # the report's own tables and forest plots
        p = paired(frame)
        out[tag] = tests(p)
        out[tag].to_csv(a.out / f"mcnemar_all_levels_{tag}.csv", index=False, float_format="%.4f")
        p.to_csv(a.out / f"paired_{tag}.csv", index=False)

    bad = check_gate(out["gate"], a.run)
    if bad:
        raise SystemExit(f"gate run does not reproduce the report: {bad}")
    print("gate run reproduces report/mcnemar_{cell,model,family}.csv and filler_compare.csv")

    c = compare(out["gate"], out["sensitivity"])
    c.to_csv(a.out / "compare.csv", index=False, float_format="%.4f")
    flagged = c[c["sig_changed"] | c["beats_at_005_gate"] | c["beats_at_005_sens"]]
    flagged.to_csv(a.out / "flagged.csv", index=False, float_format="%.4f")
    pd.set_option("display.width", 220)
    print("\nrows whose counts changed:")
    print(c[c["counts_changed"]].to_string(index=False, float_format="%.4f"))
    print("\nflagged (p<0.05 status changes, or stego beats a baseline at p<0.05 under either monitor):")
    print(flagged.to_string(index=False, float_format="%.4f"))


if __name__ == "__main__":
    main()
