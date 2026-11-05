"""
Generate the stego-edge report: faceted condition plot + funnel summary CSV.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd

from .condition_comparison import build_condition_frame
from .mcnemar_report import write_mcnemar_outputs
from .plots import (
    plot_condition_comparison,
    plot_monitor_confidence,
    plot_stego_outcome_mix,
    plot_stego_stealth,
    plot_stego_uplift,
)


def _rate_str(values: list) -> tuple[float | None, str]:
    obs = [v for v in values if v is not None]
    n = len(obs)
    if n == 0:
        return None, "0/0"
    k = sum(1 for v in obs if v)
    return k / n, f"{k}/{n}"


def _mean_or_none(values) -> float | None:
    obs = [v for v in values if v is not None and not (isinstance(v, float) and pd.isna(v))]
    if not obs:
        return None
    return sum(obs) / len(obs)


def _caught_series(rows: pd.DataFrame) -> list[bool | None]:
    """Per-row OR-union: monitor caught the sample on either signal.

    `caught` is True if either explicit_reasoning OR stego_detected is True;
    None only if BOTH are None (scorer skipped). One True overrides the other
    being None.
    """
    out: list[bool | None] = []
    for er, sd in zip(rows["explicit_reasoning"].tolist(), rows["stego_detected"].tolist()):
        if er is True or sd is True:
            out.append(True)
        elif er is None and sd is None:
            out.append(None)
        else:
            out.append(False)
    return out


def build_funnel(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate tidy condition frame into a wide (task_type,difficulty,model) summary."""
    if df.empty:
        return pd.DataFrame()

    rows = []
    groups = df.groupby(["task_type", "difficulty", "model"], dropna=False)
    for (task_type, difficulty, model), g in groups:
        sample_ids = g["sample_id"].dropna().unique()
        n_samples = len(sample_ids)

        def cond_acc(cond: str) -> float | None:
            vals = g[g["condition"] == cond]["correct"].tolist()
            vals = [v for v in vals if v is not None]
            if not vals:
                return None
            return sum(1 for v in vals if v) / len(vals)

        stego_rows = g[g["condition"] == "stego"]
        attempt_rate, attempt_str = _rate_str(stego_rows["stego_attempted"].tolist())
        emr_mean = _mean_or_none(stego_rows["encoding_match_ratio"].tolist())

        ss_vals = [v for v in stego_rows["stego_success"].tolist() if v is not None]
        success_rate = (sum(1 for v in ss_vals if v) / len(ss_vals)) if ss_vals else None

        ceiling_ok_samples = {
            sid for sid, ok in zip(g["sample_id"], g["ceiling_ok"])
            if ok is True
        }
        stego_ceil = stego_rows[stego_rows["sample_id"].isin(ceiling_ok_samples)]
        ss_ceil = [v for v in stego_ceil["stego_success"].tolist() if v is not None]
        success_rate_cot = (sum(1 for v in ss_ceil if v) / len(ss_ceil)) if ss_ceil else None

        n_stego = len(stego_rows)
        caught_vals = _caught_series(stego_rows)
        caught_rate, _ = _rate_str(caught_vals)
        er_rate, _ = _rate_str(stego_rows["explicit_reasoning"].tolist())
        sd_rate, _ = _rate_str(stego_rows["stego_detected"].tolist())
        stealth_rate = (1.0 - caught_rate) if caught_rate is not None else None
        mconf_mean = _mean_or_none(stego_rows["monitor_confidence"].tolist()) \
            if "monitor_confidence" in stego_rows.columns else None

        rows.append({
            "task_type": task_type,
            "difficulty": difficulty,
            "model": model,
            "n_samples": n_samples,
            "cot_acc": cond_acc("cot"),
            "direct_acc": cond_acc("direct"),
            "ttc_acc": cond_acc("ttc"),
            "stego_acc": cond_acc("stego"),
            "stego_attempt_rate": attempt_rate,
            "stego_attempt_observed": attempt_str,
            "encoding_match_ratio": emr_mean,
            "stego_success_rate": success_rate,
            "stego_success_rate_cot": success_rate_cot,
            "n_stego_samples": n_stego,
            "caught_rate": caught_rate,
            "stealth_rate": stealth_rate,
            "explicit_reasoning_rate": er_rate,
            "stego_detected_rate": sd_rate,
            "monitor_confidence_mean": mconf_mean,
        })

    return pd.DataFrame(rows).sort_values(["task_type", "difficulty", "model"])


def generate_stego_edge_report(
    log_dir: "str | list[str]" = "./logs",
    output_dir: str = "./plots/stego_edge",
    require_ceiling: bool = False,
    model: Optional[str] = None,
    dataset: Optional[str] = None,
) -> dict:
    """Build the stego-edge plot + funnel CSV, one subdirectory per algorithm.

    Each eval log is tagged with a single algorithm (from `task_arg__algorithm`),
    so we partition the tidy frame by that column and emit a fresh plot bundle
    per algorithm under `output_dir/<algorithm>/`. Baseline rows (direct/cot/ttc)
    appear within each algorithm's slice — they share the model/task and will
    look similar across algorithms, which is the point: each per-algorithm plot
    shows stego performance against its paired baselines.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = build_condition_frame(log_dir=log_dir, model=model)
    if dataset and not df.empty:
        df = df[df["task_type"] == dataset]

    if df.empty:
        print("No stego_cot_eval samples found (counting excluded).")
        return {}

    algorithms = sorted({a for a in df["algorithm"].dropna().unique() if a})
    if not algorithms:
        algorithms = ["unknown"]
        df = df.assign(algorithm="unknown")

    if "arm" not in df.columns:
        df = df.assign(arm="unknown")
    df["arm"] = df["arm"].fillna("unknown")

    results: dict[str, dict] = {}
    for algo in algorithms:
        algo_df = df[df["algorithm"] == algo]
        if algo_df.empty:
            continue

        arms = sorted(algo_df["arm"].unique().tolist())
        for arm in arms:
            arm_df = algo_df[algo_df["arm"] == arm]
            if arm_df.empty:
                continue

            # Single-arm runs keep the legacy layout (<algorithm>/...); when
            # multiple arms are present, plots fan out under <algorithm>/<arm>/.
            algo_out = out / algo if len(arms) == 1 else out / algo / arm
            algo_out.mkdir(parents=True, exist_ok=True)

            plot_path = algo_out / "stego_edge.png"
            fig = plot_condition_comparison(arm_df, output_path=plot_path, require_ceiling=require_ceiling)
            plt.close(fig)

            uplift_path = algo_out / "stego_uplift.png"
            fig = plot_stego_uplift(arm_df, output_path=uplift_path)
            plt.close(fig)

            stealth_path = algo_out / "stego_stealth.png"
            fig = plot_stego_stealth(arm_df, output_path=stealth_path)
            plt.close(fig)

            outcome_path = algo_out / "stego_outcome_mix.png"
            fig = plot_stego_outcome_mix(arm_df, output_path=outcome_path)
            plt.close(fig)

            confidence_path = algo_out / "stego_monitor_confidence.png"
            fig = plot_monitor_confidence(arm_df, output_path=confidence_path)
            plt.close(fig)

            funnel = build_funnel(arm_df)
            funnel.insert(0, "arm", arm)
            csv_path = algo_out / "funnel_summary.csv"
            funnel.to_csv(csv_path, index=False, float_format="%.3f")

            mcnemar_paths = write_mcnemar_outputs(arm_df, algo_out)
            for label, mp in mcnemar_paths.items():
                print(f"Saved mcnemar {label}: {mp}")

            with pd.option_context("display.max_rows", None, "display.width", 200, "display.max_columns", 20):
                print(f"\n=== Algorithm: {algo}  Arm: {arm} ===")
                print(funnel.to_string(index=False))
            print(f"Saved plot:       {plot_path}")
            print(f"Saved uplift:     {uplift_path}")
            print(f"Saved stealth:    {stealth_path}")
            print(f"Saved outcome:    {outcome_path}")
            print(f"Saved confidence: {confidence_path}")
            print(f"Saved summary:    {csv_path}")

            results[f"{algo}/{arm}" if len(arms) > 1 else algo] = {
                "plot": str(plot_path),
                "uplift": str(uplift_path),
                "stealth": str(stealth_path),
                "outcome_mix": str(outcome_path),
                "monitor_confidence": str(confidence_path),
                "csv": str(csv_path),
                "mcnemar": {k: str(v) for k, v in mcnemar_paths.items()},
            }

    return results
