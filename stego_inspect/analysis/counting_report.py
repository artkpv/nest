"""
Top-level orchestration for the counting plot bundle.

Mirrors legacy `stego_counting_plots.generate_all_plots`. Takes a DataFrame
from `eval_reader.load_all_results`, reshapes it via
`counting_frame.to_counting_dataframe`, and writes the 5 standard PNGs plus a
refusal-analysis subdirectory.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from .counting_frame import get_condition_order_and_labels, to_counting_dataframe
from .plots import (
    plot_error_analysis,
    plot_family_comparison,
    plot_heatmap,
    plot_per_number_accuracy,
    plot_refusal_analysis_combined,
    plot_timeline,
)


def create_refusal_stats_table(df: pd.DataFrame, output_dir: Path) -> Path:
    """
    Per-(model, D) refusal/accuracy stats table, saved as CSV.

    Mirrors legacy `stego_counting_plots.create_refusal_stats_table`.
    """
    refusal_dir = Path(output_dir) / "refusal_analysis"
    refusal_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for (model, d), group in df.groupby(["model", "D"], dropna=False):
        if d is None or pd.isna(d):
            continue
        total = len(group)
        refusals = int(group["refused"].sum())
        exact_matches = int(group["exact_match"].sum())
        non_refused = group[~group["refused"]]
        avg_per_num = (
            float(non_refused["per_number_accuracy"].mean()) if not non_refused.empty else 0.0
        )
        rows.append(
            {
                "Model": model,
                "D": int(d),
                "Total Trials": total,
                "Refusals": refusals,
                "Refusal Rate": f"{(refusals / total) if total else 0:.2%}",
                "Exact Matches": exact_matches,
                "Exact Match Rate": f"{(exact_matches / total) if total else 0:.2%}",
                "Avg Per-Number Acc": f"{avg_per_num:.3f}",
            }
        )

    out_df = pd.DataFrame(rows).sort_values(["Model", "D"]) if rows else pd.DataFrame()
    csv_path = refusal_dir / "refusal_analysis_table.csv"
    out_df.to_csv(csv_path, index=False)
    return csv_path


def generate_counting_report(
    df: pd.DataFrame,
    output_dir: Path | str,
) -> None:
    """
    Render the full counting plot bundle for a counting DataFrame.

    Args:
        df: DataFrame — either raw from ``eval_reader.load_all_results``
            (will be transformed via ``to_counting_dataframe``) or already
            in the legacy counting schema (must have ``short_name`` column).
        output_dir: Directory to write PNGs and the refusal_analysis bundle.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If the DataFrame already has the legacy schema, skip transformation.
    if "short_name" in df.columns and "condition" in df.columns:
        counting_df = df
    else:
        counting_df = to_counting_dataframe(df)
    if counting_df.empty:
        print("No counting samples found in eval logs.")
        return

    condition_order, condition_labels = get_condition_order_and_labels(counting_df)
    print(f"Conditions: {condition_order}")
    print(f"Models: {sorted(counting_df['short_name'].unique())}")

    print("  1. Heatmap (exact match)")
    plot_heatmap(
        counting_df,
        output_dir / "1_heatmap_exact_match.png",
        condition_order,
        condition_labels,
    )
    plt.close("all")

    print("  2. Timeline")
    plot_timeline(counting_df, output_dir / "2_timeline.png")
    plt.close("all")
    # Models cover different D sets (the 2026-07 roster ran only D=4/8), so the
    # all-D mean above is not like-for-like across models. D=4 is the one level
    # every model ran; this is the version the paper uses.
    d4 = counting_df[counting_df["D"] == 4]
    if not d4.empty:
        plot_timeline(d4, output_dir / "2_timeline_D4.png", title_suffix=" (D=4)")
        plt.close("all")

    print("  3. Family comparison")
    plot_family_comparison(
        counting_df,
        output_dir / "3_family_comparison.png",
        condition_order,
        condition_labels,
    )
    plt.close("all")

    print("  4. Error analysis")
    plot_error_analysis(counting_df, output_dir / "4_error_analysis.png")
    plt.close("all")

    print("  5. Per-number accuracy heatmap")
    plot_per_number_accuracy(
        counting_df,
        output_dir / "5_heatmap_partial.png",
        condition_order,
        condition_labels,
    )
    plt.close("all")

    print("  Refusal analysis")
    n_refusal = plot_refusal_analysis_combined(counting_df, output_dir)
    csv_path = create_refusal_stats_table(counting_df, output_dir)
    print(f"  Saved {n_refusal} refusal plots and {csv_path.name}")

    # --- Digit-width stratified plots ------------------------------------
    if "digits_per_number" in counting_df.columns:
        dpn_values = sorted(counting_df["digits_per_number"].dropna().unique())
        for dpn in dpn_values:
            dpn = int(dpn)
            subset = counting_df[counting_df["digits_per_number"] == dpn]
            if len(subset) < 10:
                print(f"  Skipping dpn={dpn} (only {len(subset)} samples)")
                continue

            dpn_dir = output_dir / f"dpn_{dpn}"
            dpn_dir.mkdir(parents=True, exist_ok=True)
            cond_order, cond_labels = get_condition_order_and_labels(subset)
            suffix = f" ({dpn}-digit numbers)"
            print(f"\n  === digits_per_number={dpn} ({len(subset)} samples) ===")

            print(f"    1. Heatmap (exact match)")
            plot_heatmap(subset, dpn_dir / "1_heatmap_exact_match.png", cond_order, cond_labels, title_suffix=suffix)
            plt.close("all")

            print(f"    2. Timeline")
            plot_timeline(subset, dpn_dir / "2_timeline.png", title_suffix=suffix)
            plt.close("all")

            print(f"    3. Family comparison")
            plot_family_comparison(subset, dpn_dir / "3_family_comparison.png", cond_order, cond_labels, title_suffix=suffix)
            plt.close("all")

            print(f"    4. Error analysis")
            plot_error_analysis(subset, dpn_dir / "4_error_analysis.png", title_suffix=suffix)
            plt.close("all")

            print(f"    5. Per-number accuracy heatmap")
            plot_per_number_accuracy(subset, dpn_dir / "5_heatmap_partial.png", cond_order, cond_labels, title_suffix=suffix)
            plt.close("all")

            print(f"    Refusal analysis")
            plot_refusal_analysis_combined(subset, dpn_dir)
            create_refusal_stats_table(subset, dpn_dir)

    print(f"\nAll plots saved to: {output_dir}")
