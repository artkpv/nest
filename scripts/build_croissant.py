"""
Build data/release/v1.0/croissant.json — Croissant 1.0 (core + RAI) metadata
for the frozen NEST v1.0 dataset bundle, and register it in manifest.json.

Croissant is the MLCommons machine-readable (JSON-LD) dataset-description
format. NeurIPS Datasets & Benchmarks requires the core fields plus the RAI
(Responsible AI) extension.

This script is deterministic and offline. It:
  1. reads sha256/bytes/n_items for every data file from manifest.json
     (does NOT recompute — the v1.0 data is frozen);
  2. emits one cr:FileObject per bundle file plus cr:FileSets grouping the
     JSONL / JSON data files, and three record sets mirroring schema.json;
  3. writes croissant.json;
  4. adds a single croissant.json entry to manifest.json (existing entries,
     including their sha256s, are left untouched).

Usage (from repo root, in the venv):
    python scripts/build_croissant.py --release data/release/v1.0

Everything shipped stays anonymous (double-blind review): no author names,
emails, or non-anonymous URLs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import OrderedDict
from pathlib import Path

# Anonymous release location (double-blind); every contentUrl hangs off this.
BASE_URL = "https://anonymous.4open.science/r/encreas_stego-488F/data/release/v1.0/"

# ---------------------------------------------------------------------------
# Croissant 1.0 JSON-LD context (core + RAI namespaces).
# ---------------------------------------------------------------------------
CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "equivalentProperty": "cr:equivalentProperty",
    "examples": {"@id": "cr:examples", "@type": "@json"},
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "samplingRate": "cr:samplingRate",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

ENCODING_BY_SUFFIX = {
    ".jsonl": "application/jsonlines",
    ".json": "application/json",
    ".md": "text/markdown",
    ".py": "text/x-python",
}


def encoding_format(rel: str) -> str:
    suffix = Path(rel).suffix
    if suffix in ENCODING_BY_SUFFIX:
        return ENCODING_BY_SUFFIX[suffix]
    return "text/plain"  # LICENSE (extensionless)


def file_object(rel: str, entry: dict) -> dict:
    """One cr:FileObject describing a single bundle file, with frozen sha256."""
    obj = {
        "@type": "cr:FileObject",
        "@id": rel,
        "name": rel,
        "contentUrl": BASE_URL + rel,
        "encodingFormat": encoding_format(rel),
        "sha256": entry["sha256"],
    }
    if "bytes" in entry:
        obj["contentSize"] = f"{entry['bytes']} B"
    return obj


def file_set(fid: str, name: str, includes: str,
             encoding: str, description: str) -> dict:
    """A cr:FileSet that mlcroissant reads via its LocalDirectory path.

    Intentionally has NO `containedIn`: mlcroissant only reads *file content*
    for a multi-file FileSet when it walks the dataset folder (the directory the
    croissant.json lives in) and filters by `includes`. A FileSet `containedIn`
    loose FileObjects yields only a file *listing*, not the records. `includes`
    is therefore a folder-relative glob (e.g. `counting/*.jsonl`). The per-file
    cr:FileObjects above carry the sha256 integrity / provenance.
    """
    return {
        "@type": "cr:FileSet",
        "@id": fid,
        "name": name,
        "description": description,
        "encodingFormat": encoding,
        "includes": includes,
    }


def field(fid: str, name: str, data_type: str, file_set_id: str,
          *, column: str | None = None, json_path: str | None = None,
          transform_jsonpath: str | None = None, repeated: bool = False,
          description: str | None = None) -> dict:
    """A cr:Field.

    mlcroissant reading rules (verified against the library):
      * `application/jsonlines` is read with pandas so only TOP-LEVEL keys are
        addressable via `extract.column`; a nested value is reached by taking
        the top-level dict column and then applying `transform.jsonPath` to it.
        `extract.jsonPath` is NOT supported for JSON Lines.
      * `application/json` (whole file) IS read with jsonPath, so array records
        use `extract.jsonPath` like `$[*].id`.
      * A single record set/source must use ONE reading method — do not mix
        `extract.column` and `extract.jsonPath` fields on the same file set.
    """
    if column is not None:
        extract = {"column": column}
    else:
        extract = {"jsonPath": json_path}
    source = {"fileSet": {"@id": file_set_id}, "extract": extract}
    if transform_jsonpath is not None:
        source["transform"] = [{"jsonPath": transform_jsonpath}]
    f = {
        "@type": "cr:Field",
        "@id": fid,
        "name": name,
        "dataType": data_type,
        "source": source,
    }
    if repeated:
        f["repeated"] = True
    if description:
        f["description"] = description
    return f


def build_croissant(manifest: dict) -> dict:
    rel_paths = list(manifest.keys())

    # --- distribution: a FileObject per file, then FileSets grouping data ---
    distribution = [file_object(rel, manifest[rel]) for rel in rel_paths]

    distribution.append(file_set(
        "reasoning_gym-jsonl", "reasoning_gym-jsonl",
        "reasoning_gym/*.jsonl", "application/jsonlines",
        "The 17 Reasoning Gym pool files (one per (family, model) cell), JSON Lines.",
    ))
    distribution.append(file_set(
        "counting-jsonl", "counting-jsonl",
        "counting/*.jsonl", "application/jsonlines",
        "The 10 steganographic-counting trial files (one per difficulty level D), JSON Lines.",
    ))
    distribution.append(file_set(
        "cover_questions-json", "cover_questions-json",
        "cover_questions/*.json", "application/json",
        "The two cover-question prompt files (opinion_5.json, rg_custom_topics.json).",
    ))

    # --- record sets mirror schema.json ---
    rg_rs = {
        "@type": "cr:RecordSet",
        "@id": "reasoning_gym",
        "name": "reasoning_gym",
        "description": (
            "Reasoning Gym problem instances (question, answer, metadata) across four "
            "families (basic_arithmetic, chain_sum, simple_equations, maze). Each record "
            "was pre-screened so plain chain-of-thought solves it while direct answer fails "
            "at the calibrated difficulty D*. Family-specific metadata fields are null for "
            "records from other families."
        ),
        "field": [
            field("reasoning_gym/question", "question", "sc:Text", "reasoning_gym-jsonl",
                  column="question", description="The problem statement shown to the model."),
            field("reasoning_gym/answer", "answer", "sc:Text", "reasoning_gym-jsonl",
                  column="answer", description="Ground-truth answer (string)."),
            field("reasoning_gym/source_dataset", "source_dataset", "sc:Text", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="source_dataset",
                  description="Reasoning Gym family: basic_arithmetic, chain_sum, simple_equations, or maze."),
            field("reasoning_gym/source_index", "source_index", "sc:Integer", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="source_index",
                  description="Index of the item within its generated pool."),
            field("reasoning_gym/expression", "expression", "sc:Text", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="expression",
                  description="Arithmetic expression (basic_arithmetic / chain_sum only; null otherwise)."),
            field("reasoning_gym/num_terms", "num_terms", "sc:Integer", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="num_terms",
                  description="Number of terms (arithmetic families only; null otherwise)."),
            field("reasoning_gym/num_digits", "num_digits", "sc:Integer", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="num_digits",
                  description="Digit magnitude (arithmetic families only; null otherwise)."),
            field("reasoning_gym/grid_size", "grid_size", "sc:Integer", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="grid_size",
                  description="Maze grid side length (maze family only; null otherwise)."),
            field("reasoning_gym/shortest_path_length", "shortest_path_length", "sc:Integer", "reasoning_gym-jsonl",
                  column="metadata", transform_jsonpath="shortest_path_length",
                  description="Optimal maze solution length (maze family only; null otherwise)."),
        ],
    }

    counting_rs = {
        "@type": "cr:RecordSet",
        "@id": "counting_trials",
        "name": "counting_trials",
        "description": (
            "Fully materialized steganographic-counting trials. Each trial specifies a numeric "
            "sequence range(X, Y, I), a randomized digit->letter mapping, and a cover question; "
            "expected_acrostic is the D-character sentence-acrostic the model must produce. "
            "Reproducible from interval_seed and mapping_seed (see counting/regenerate.py). "
            "Note: each underlying JSONL record also carries the full digit_to_letter and "
            "letter_to_digit mapping objects (see schema.json); they are omitted from the fields "
            "below because Croissant/mlcroissant has no faithful JSON-object field type, and they "
            "are exactly reconstructible from mapping_seed."
        ),
        "field": [
            field("counting_trials/sample_id", "sample_id", "sc:Text", "counting-jsonl",
                  column="sample_id"),
            field("counting_trials/D", "D", "sc:Integer", "counting-jsonl",
                  column="D", description="Sequence length / difficulty level."),
            # interval object -> three integer subFields (top-level "interval"
            # dict column, then transform.jsonPath to reach X/Y/I).
            {
                "@type": "cr:Field",
                "@id": "counting_trials/interval",
                "name": "interval",
                "description": "Half-open integer interval range(X, Y, I) that defines the sequence.",
                "subField": [
                    field("counting_trials/interval/X", "X", "sc:Integer", "counting-jsonl",
                          column="interval", transform_jsonpath="X"),
                    field("counting_trials/interval/Y", "Y", "sc:Integer", "counting-jsonl",
                          column="interval", transform_jsonpath="Y"),
                    field("counting_trials/interval/I", "I", "sc:Integer", "counting-jsonl",
                          column="interval", transform_jsonpath="I"),
                ],
            },
            field("counting_trials/expected_sequence", "expected_sequence", "sc:Integer", "counting-jsonl",
                  column="expected_sequence", repeated=True,
                  description="The sequence range(X, Y, I) as a list of integers."),
            field("counting_trials/cover_question", "cover_question", "sc:Text", "counting-jsonl",
                  column="cover_question",
                  description="Innocuous opinion prompt the model answers on the surface."),
            field("counting_trials/interval_seed", "interval_seed", "sc:Integer", "counting-jsonl",
                  column="interval_seed"),
            field("counting_trials/mapping_seed", "mapping_seed", "sc:Integer", "counting-jsonl",
                  column="mapping_seed"),
            field("counting_trials/expected_acrostic", "expected_acrostic", "sc:Text", "counting-jsonl",
                  column="expected_acrostic",
                  description="D-character string of sentence-initial letters the model must produce."),
        ],
    }

    cover_rs = {
        "@type": "cr:RecordSet",
        "@id": "cover_questions",
        "name": "cover_questions",
        "description": (
            "Innocuous cover-question prompts: five opinion prompts (counting sweep, "
            "opinion_5.json) and twenty topic questions (reasoning gym sweep, "
            "rg_custom_topics.json). No personally identifying text. Note: opinion_5.json "
            "additionally carries a short `topic` label per record in the source file."
        ),
        "field": [
            field("cover_questions/id", "id", "sc:Text", "cover_questions-json",
                  json_path="$[*].id"),
            field("cover_questions/text", "text", "sc:Text", "cover_questions-json",
                  json_path="$[*].text", description="The prompt text."),
        ],
    }

    description = (
        "NEST v1.0 — frozen problem instances for evaluating steganographic chain-of-thought "
        "(CoT) capability and CoT monitorability in frontier LLMs. Two regimes: (1) steganographic "
        "reasoning on four Reasoning Gym families (basic_arithmetic, chain_sum, simple_equations, "
        "maze), where models solve a hidden problem while answering a cover question via "
        "sentence-acrostic encoding, with difficulty D* calibrated per (model, family) so plain CoT "
        "nearly saturates while direct answer fails; and (2) steganographic counting, encoding a "
        "numeric sequence as sentence-initial letters under a randomized digit->letter mapping, as a "
        "lower bound on steganographic capacity. Content is benign (arithmetic, maze navigation, "
        "opinion prompts); the paired evaluation protocol is dual-use and released for defensive "
        "CoT-monitoring research, not for training models toward covert encoding."
    )

    cite_as = (
        "@inproceedings{nest2026, title={{NEST}: Nascent Encoded Steganographic Thoughts}, "
        "booktitle={Advances in Neural Information Processing Systems}, year={2026}, "
        "note={Under review; author information withheld for double-blind review}}"
    )

    croissant = OrderedDict()
    croissant["@context"] = CONTEXT
    croissant["@type"] = "sc:Dataset"
    croissant["conformsTo"] = "http://mlcommons.org/croissant/1.0"
    croissant["name"] = "nest-v1.0"
    croissant["description"] = description
    croissant["license"] = "https://creativecommons.org/licenses/by/4.0/"
    croissant["url"] = BASE_URL
    croissant["version"] = "1.0.0"
    croissant["datePublished"] = "2026-05-06"
    croissant["citeAs"] = cite_as

    # --- RAI (Responsible AI) extension ---
    croissant["rai:dataCollection"] = (
        "Programmatically generated; no human subjects, no crowd-sourcing, no web scraping. "
        "Reasoning Gym records were produced by the open-source Reasoning Gym generators, then "
        "graded by frontier models and filtered per (model, family) to items where plain CoT "
        "succeeds and direct answer fails at the calibrated difficulty D*. Counting trials were "
        "generated deterministically by stego_inspect/datasets/counting.py from integer seeds "
        "(fully reproducible via counting/regenerate.py). Cover questions were authored by the "
        "researchers or drawn from a fixed topic list. Collected April 2026."
    )
    croissant["rai:dataCollectionType"] = "Synthetic / programmatically generated."
    croissant["rai:personalSensitiveInformation"] = (
        "None. All content is synthetic (arithmetic, chain-addition, linear-equation, and maze "
        "problems) or generic lifestyle opinion prompts. No personally identifying information."
    )
    croissant["rai:dataBiases"] = [
        "English-only: all prompts are in English.",
        "Model-specific calibration: difficulty levels D* are calibrated to specific model "
        "versions; the 'harder for direct, easier for CoT' property may not transfer to other "
        "models without re-calibration.",
        "Filtered pools: Reasoning Gym pools are filtered per (model, family) to a specific "
        "difficulty regime and are not a representative sample of the underlying generators.",
        "Uneven coverage: the maze family is available only for GPT-5.4 and GPT-5.4-mini; pool "
        "sizes range from 24 to 260 records per cell, so small cells give limited statistical power.",
    ]
    croissant["rai:dataUseCases"] = (
        "Evaluation of steganographic chain-of-thought (CoT) capability and of CoT monitorability "
        "in frontier LLMs: measuring whether a model can solve a hidden task while answering an "
        "unrelated cover question via sentence-acrostic encoding, and whether a monitor reading the "
        "output detects the encoding. Also usable for benchmarking difficulty-calibrated reasoning."
    )
    croissant["rai:dataLimitations"] = [
        "Model-specific calibration: D* levels are tied to specific model versions and may need "
        "re-calibration for new models.",
        "English-only prompts.",
        "Maze coverage limited to GPT-5.4 and GPT-5.4-mini.",
        "Dual-use: the protocol measures covert-reasoning capability; released for defensive "
        "monitoring research. Do NOT use it to fine-tune or train models toward steganographic "
        "encoding.",
        "Pool sizes range 24-260 records per cell; small pools offer limited statistical power.",
    ]

    croissant["distribution"] = distribution
    croissant["recordSet"] = [rg_rs, counting_rs, cover_rs]
    return croissant


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        return "unknown"


def update_manifest(manifest_path: Path, croissant_path: Path, commit: str) -> None:
    """Insert a single croissant.json entry into manifest.json, alphabetically,
    without touching any existing entry."""
    manifest = json.loads(manifest_path.read_text())
    entry = {
        "sha256": hashlib.sha256(croissant_path.read_bytes()).hexdigest(),
        "bytes": croissant_path.stat().st_size,
        "generator_commit": commit,
    }
    new_manifest = OrderedDict()
    inserted = False
    for key, val in manifest.items():
        # keep alphabetical placement: croissant.json sorts after cover_questions/*
        if not inserted and key.startswith("reasoning_gym/"):
            new_manifest["croissant.json"] = entry
            inserted = True
        new_manifest[key] = val
    if not inserted:
        new_manifest["croissant.json"] = entry
    manifest_path.write_text(json.dumps(new_manifest, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--release", default="data/release/v1.0",
                        help="Path to the frozen release bundle directory.")
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    release = Path(args.release)
    if not release.is_absolute():
        release = repo_root / release

    manifest_path = release / "manifest.json"
    manifest = json.loads(manifest_path.read_text())

    croissant = build_croissant(manifest)
    croissant_path = release / "croissant.json"
    croissant_path.write_text(json.dumps(croissant, indent=2) + "\n")
    print(f"  wrote {croissant_path.relative_to(repo_root)}")

    update_manifest(manifest_path, croissant_path, git_head(repo_root))
    print(f"  registered croissant.json in {manifest_path.relative_to(repo_root)}")


if __name__ == "__main__":
    main()
