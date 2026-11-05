#!/usr/bin/env python3
"""
Generate candidate JSONL datasets from Reasoning Gym.

Produces files in stego_inspect/data/rg_*.jsonl for use in stego_cot_eval
with hidden_dataset="reasoning_gym".

Usage:
    python stego_inspect/generate_rg_datasets.py [--size 200] [--seed 42]
"""

import argparse
import json
from pathlib import Path

import reasoning_gym


# Task definitions: (task_name, difficulty_param, difficulty_values, extra_config)
# For basic_arithmetic: terms = operands, so N terms = N-1 operations
# For chain_sum: terms = numbers in the chain
# For simple_equations: terms = number of terms in the equation
TASK_SPECS = [
    # basic_arithmetic: ops 1,2,3,5,7 → terms 2,3,4,6,8
    ("basic_arithmetic", "terms", [2, 3, 4, 6, 8], {}),
    # chain_sum: chain length 3,5,7,10
    ("chain_sum", "terms", [3, 5, 7, 10], {}),
    # simple_equations: terms 1,2,3,4
    ("simple_equations", "terms", [1, 2, 3, 4], {}),
]

# Map from our difficulty label (ops/length) to the file name suffix
FILENAME_LABELS = {
    "basic_arithmetic": {2: "ops1", 3: "ops2", 4: "ops3", 6: "ops5", 8: "ops7"},
    "chain_sum": {3: "len3", 5: "len5", 7: "len7", 10: "len10"},
    "simple_equations": {1: "terms1", 2: "terms2", 3: "terms3", 4: "terms4"},
}


def generate_dataset(
    task_name: str,
    terms: int,
    extra_config: dict,
    size: int,
    seed: int,
) -> list[dict]:
    """Generate samples from a Reasoning Gym task."""
    config = {"size": size, "seed": seed}

    if task_name == "simple_equations":
        config["min_terms"] = terms
        config["max_terms"] = terms
    else:
        config["min_terms"] = terms
        config["max_terms"] = terms

    config.update(extra_config)

    ds = reasoning_gym.create_dataset(task_name, **config)

    samples = []
    for item in ds:
        samples.append(
            {
                "question": item["question"],
                "answer": item["answer"],
                "metadata": item.get("metadata", {}),
            }
        )

    return samples


def main():
    parser = argparse.ArgumentParser(description="Generate Reasoning Gym datasets")
    parser.add_argument(
        "--size", type=int, default=200, help="Samples per task/difficulty"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: stego_inspect/data/)",
    )
    args = parser.parse_args()

    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else Path(__file__).parent.parent / "stego_inspect" / "data"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for task_name, _param_name, values, extra_config in TASK_SPECS:
        for val in values:
            label = FILENAME_LABELS[task_name][val]
            filename = f"rg_{task_name}_{label}.jsonl"
            filepath = output_dir / filename

            print(f"Generating {task_name} ({label}): {args.size} samples...")
            samples = generate_dataset(
                task_name, val, extra_config, args.size, args.seed
            )

            with open(filepath, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample) + "\n")

            print(f"  -> {filepath} ({len(samples)} samples)")

    print("\nDone!")


if __name__ == "__main__":
    main()
