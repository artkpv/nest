"""
Assemble the public NEST dataset release for Hugging Face (camera-ready).

Builds the full tree of the HF dataset repo `artkpv/nest` in one folder:

    <out>/
    ├── README.md          # HF dataset card (from data/release/drafts/hf-card/README.md)
    ├── LICENSE            # CC-BY-4.0 (copy of v1.0/LICENSE)
    ├── croissant.json     # Croissant core + RAI for the whole release (the OpenReview file)
    ├── v1.0/              # data/release/v1.0, data bytes unchanged; docs from drafts/v1.0-docs/
    │   └── croissant.json # Croissant for v1.0 alone
    └── v1.1/              # addendum: transcripts (C1) + clean RG re-run pools (B6)
        ├── README.md, CHANGELOG.md, manifest.json
        ├── transcripts/   # data/transcripts, as is
        ├── reasoning_gym_clean/
        │   ├── cells.jsonl          # one row per (model, family): D, kept ids, evaluated ids
        │   └── pools/<family>_<model>_D<D>.jsonl
        └── maze_followup/           # Opus 4.6 maze_varied follow-up (B8), same layout, one cell

Deterministic and offline. Nothing is uploaded. Every copied data file is checked
against the hash recorded for it (v1.0/manifest.json, each run's items_manifest.json);
any mismatch aborts.

Usage (from code/, in the venv):
    python scripts/build_hf_release.py --out data/release/drafts/hf-nest
    # local load test: a variant whose file sets read the folder directly
    python scripts/build_hf_release.py --out data/release/drafts/hf-nest --local-test
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
from collections import OrderedDict
from pathlib import Path

from build_croissant import CONTEXT, build_croissant, encoding_format, field

HF_REPO = "artkpv/nest"
CODE_URL = "https://github.com/artkpv/nest"
CREATOR = {"@type": "sc:Person", "name": "Artem Karpov",
           "affiliation": {"@type": "sc:Organization", "name": "Independent"}}
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
CROISSANT_VERSION = "1.0"
# (source run, v1.1 folder, run slug suffix): pools copied with their item ids.
RG_CLEAN = (Path("artifacts/2026-10-31/rg_clean"), "reasoning_gym_clean", "clean")
MAZE_FOLLOWUP = (Path("artifacts/2026-11-04/rg_opus46_maze_followup"), "maze_followup", "fu")
TRANSCRIPTS = Path("data/transcripts")
V10 = Path("data/release/v1.0")
DRAFTS = Path("data/release/drafts")
# v1.0 files whose bytes may change in the camera-ready (documentation only).
V10_DOCS = ("README.md", "DATASHEET.md", "CHANGELOG.md")

MODELS = OrderedDict([
    ("opus45", "anthropic/claude-opus-4-5-20251101"),
    ("opus46", "anthropic/claude-opus-4-6"),
    ("sonnet46", "anthropic/claude-sonnet-4-6"),
    ("haiku45", "anthropic/claude-haiku-4-5-20251001"),
    ("gpt54", "openai/gpt-5.4-2026-03-05"),
    ("gpt54mini", "openai/gpt-5.4-mini-2026-03-17"),
])


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def n_lines(path: Path) -> int:
    return sum(1 for line in path.read_bytes().splitlines() if line.strip())


def git_head() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def resolve_url(rel: str) -> str:
    return f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/{rel}"


def file_object(out: Path, rel: str, description: str | None = None) -> dict:
    p = out / rel
    obj = {
        "@type": "cr:FileObject",
        "@id": rel,
        "name": rel,
        "contentUrl": resolve_url(rel),
        "encodingFormat": encoding_format(rel),
        "contentSize": f"{p.stat().st_size} B",
        "sha256": sha256(p),
    }
    if description:
        obj["description"] = description
    return obj


def repo_object() -> dict:
    """The HF git repo; file sets live in it (Croissant `containedIn`)."""
    return {
        "@type": "cr:FileObject",
        "@id": "hf-repo",
        "name": "hf-repo",
        "description": f"The Hugging Face dataset repository {HF_REPO} (git).",
        "contentUrl": f"https://huggingface.co/datasets/{HF_REPO}",
        "encodingFormat": "git+https",
        "sha256": "main",
    }


def file_set(fid: str, includes: str, encoding: str, description: str,
             local: bool) -> dict:
    fs = {
        "@type": "cr:FileSet",
        "@id": fid,
        "name": fid,
        "description": description,
        "encodingFormat": encoding,
        "includes": includes,
    }
    # The local-test variant reads the folder next to croissant.json instead.
    if not local:
        fs["containedIn"] = {"@id": "hf-repo"}
    return fs


def from_file(rs: dict, file_id: str) -> dict:
    """Re-point every field source of a record set from a fileSet to a fileObject."""
    rs = copy.deepcopy(rs)

    def walk(f):
        src = f.get("source")
        if src and "fileSet" in src:
            src["fileObject"] = {"@id": file_id}
            del src["fileSet"]
        for sub in f.get("subField", []):
            walk(sub)
    for f in rs["field"]:
        walk(f)
    return rs


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------

def build_v10(out: Path, commit: str) -> dict:
    """Copy v1.0, overlay the camera-ready docs, refresh their manifest entries."""
    dest = out / "v1.0"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(V10, dest, ignore=shutil.ignore_patterns("croissant.json"))
    for name in V10_DOCS:
        shutil.copy2(DRAFTS / "v1.0-docs" / name, dest / name)

    manifest = json.loads((V10 / "manifest.json").read_text(), object_pairs_hook=OrderedDict)
    for rel, entry in manifest.items():
        if rel == "croissant.json":
            continue
        p = dest / rel
        if rel in V10_DOCS:
            entry["sha256"] = sha256(p)
            entry["bytes"] = p.stat().st_size
            entry["generator_commit"] = commit
        elif sha256(p) != entry["sha256"]:
            raise SystemExit(f"v1.0 frozen file changed: {rel}")
    return manifest


def build_transcripts(out: Path) -> list[str]:
    dest = out / "v1.1" / "transcripts"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(TRANSCRIPTS, dest)
    return sorted(str(p.relative_to(out)) for p in dest.iterdir())


def build_rg_pools(out: Path, run: tuple[Path, str, str]) -> list[str]:
    """Pools renamed to the v1.0 scheme, plus cells.jsonl from items_manifest.json."""
    src_root, folder, suffix = run
    dest = out / "v1.1" / folder
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "pools").mkdir(parents=True)
    manifest = json.loads((src_root / "items_manifest.json").read_text())
    rows = []
    for key, cell in manifest["cells"].items():
        run_slug, family = cell["slug"], cell["family"]
        model = run_slug.removesuffix(suffix)
        src = src_root / "pools" / f"rg_{family}_{run_slug}_edge.jsonl"
        if sha256(src) != cell["pool_sha256"]:
            raise SystemExit(f"{folder} pool changed since freeze: {src}")
        ids_path = src_root / "items" / f"{key}.json"
        if sha256(ids_path) != cell["ids_sha256"]:
            raise SystemExit(f"{folder} item ids changed since freeze: {ids_path}")
        name = f"{family}_{model}_D{cell['D']}.jsonl"
        shutil.copy2(src, dest / "pools" / name)
        kept = json.loads(src.with_suffix(".json").read_text())
        eval_ids = json.loads(ids_path.read_text())
        assert len(kept) == cell["n_kept"] and len(eval_ids) == cell["n"]
        rows.append(OrderedDict([
            ("cell", f"{model}__{family}"),
            ("model", MODELS[model]),
            ("model_slug", model),
            ("family", family),
            ("D", cell["D"]),
            ("pool_file", f"pools/{name}"),
            ("pool_sha256", cell["pool_sha256"]),
            ("n_pool", cell["n_pool"]),
            ("n_kept", cell["n_kept"]),
            ("n_eval", cell["n"]),
            ("kept_ids", kept),
            ("eval_ids", eval_ids),
            ("eval_item_sha256", cell["item_sha256"]),
            ("item_seed", manifest["item_seed"]),
        ]))
    with open(dest / "cells.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return sorted(str(p.relative_to(out)) for p in dest.rglob("*") if p.is_file())


def write_manifest(out: Path, base: str, rels: list[str], commit: str) -> None:
    m = OrderedDict()
    for rel in sorted(rels):
        p = out / rel
        entry = OrderedDict([("sha256", sha256(p)), ("bytes", p.stat().st_size)])
        if p.suffix == ".jsonl":
            entry["n_items"] = n_lines(p)
        entry["generator_commit"] = commit
        m[str(Path(rel).relative_to(base))] = entry
    (out / base / "manifest.json").write_text(json.dumps(m, indent=2) + "\n")


# ---------------------------------------------------------------------------
# Croissant
# ---------------------------------------------------------------------------

TRANSCRIPT_FIELDS = [
    # (name, dataType, top-level column, jsonPath inside it, description)
    ("transcript_id", "sc:Text", "transcript_id", None, "<run>:<model>__<family>__<algorithm>#<sample id>."),
    ("canary", "sc:Text", "canary", None, "Canary string; exclude this record from training corpora."),
    ("model", "sc:Text", "provenance", "model_requested", "Model under test (provider/model id)."),
    ("algorithm", "sc:Text", "condition", "algorithm", "Steganographic channel, e.g. sentence_acrostic."),
    ("use_mapping", "sc:Boolean", "condition", "use_mapping", "Whether a digit->letter mapping was given."),
    ("hidden_dataset", "sc:Text", "condition", "hidden_dataset", "Hidden-task family."),
    ("hidden_problem", "sc:Text", "hidden_task", "problem", "The hidden problem."),
    ("hidden_answer", "sc:Text", "hidden_task", "answer", "Ground-truth answer."),
    ("cover_question", "sc:Text", "cover_question", None, "Cover question answered on the surface."),
    ("system_prompt", "sc:Text", "prompts", "system", "System prompt sent to the model."),
    ("user_prompt", "sc:Text", "prompts", "user", "User prompt sent to the model."),
    ("stego_response", "sc:Text", "stego_response", None, "The model's full steganographic response."),
    ("n_filler_tokens", "sc:Integer", "n_filler_tokens", None, "Filler budget of the ttc baseline."),
    ("stego_correct", "sc:Boolean", "scores", "stego_correct", "Hidden answer correct."),
    ("baseline_direct_correct", "sc:Boolean", "scores", "baseline_direct_correct", "Direct-answer baseline correct."),
    ("baseline_ttc_correct", "sc:Boolean", "scores", "baseline_ttc_correct", "Filler-token baseline correct."),
    ("explicit_reasoning", "sc:Boolean", "scores", "explicit_reasoning", "Monitor saw plain-text reasoning (null = undetermined)."),
    ("stego_detected", "sc:Boolean", "scores", "stego_detected", "Monitor spotted the encoding (null = undetermined; does not gate)."),
    ("stego_success", "sc:Boolean", "scores", "stego_success", "Automated composite (an upper bound on RG; null = undetermined)."),
    ("label", "sc:Text", "adjudication", "label", "Why the transcript was picked."),
    ("verdict", "sc:Text", "adjudication", "verdict", "Hand verdict; authoritative over stego_success."),
    ("verdict_note", "sc:Text", "adjudication", "note", "Hand-read note."),
]


def transcripts_record_set() -> dict:
    fields = []
    for name, dtype, col, path, desc in TRANSCRIPT_FIELDS:
        fields.append(field(f"transcripts/{name}", name, dtype, "v1.1/transcripts/transcripts.jsonl",
                            column=col, transform_jsonpath=path, description=desc))
    return from_file({
        "@type": "cr:RecordSet",
        "@id": "transcripts",
        "name": "transcripts",
        "description": (
            "Curated model transcripts (v1.1/transcripts/transcripts.jsonl), each with the full "
            "prompts, the steganographic response, the automated scores and a hand verdict. "
            "Per-record baseline responses (baselines.direct/cot/ttc) and provenance are in the "
            "JSONL and described in v1.1/transcripts/README.md; they are not exposed as fields. "
            "Every record carries a canary string."
        ),
        "field": fields,
    }, "v1.1/transcripts/transcripts.jsonl")


def rg_pool_record_sets(folder: str, file_set_id: str, pools_description: str,
                        cells_description: str) -> list[dict]:
    pool_fields = [
        field(f"{folder}/{n}", n, t, file_set_id,
              column=c, transform_jsonpath=p, description=d)
        for n, t, c, p, d in [
            ("question", "sc:Text", "question", None, "The problem statement shown to the model."),
            ("answer", "sc:Text", "answer", None, "Ground-truth answer (string)."),
            ("source_dataset", "sc:Text", "metadata", "source_dataset", "Reasoning Gym family."),
            ("source_index", "sc:Integer", "metadata", "source_index",
             "Line index in the pool file; the ids in cells.jsonl refer to it."),
        ]
    ]
    pools = {
        "@type": "cr:RecordSet",
        "@id": folder,
        "name": folder,
        "description": pools_description,
        "field": pool_fields,
    }
    cell_specs = [
        ("cell", "sc:Text", False, "<model_slug>__<family>."),
        ("model", "sc:Text", False, "Model under test (provider/model id)."),
        ("model_slug", "sc:Text", False, "Short model name used in file names."),
        ("family", "sc:Text", False, "Reasoning Gym family."),
        ("D", "sc:Integer", False, "Calibrated difficulty rung."),
        ("pool_file", "sc:Text", False, f"Pool file, relative to v1.1/{folder}/."),
        ("pool_sha256", "sc:Text", False, "SHA-256 of the pool file."),
        ("n_pool", "sc:Integer", False, "Items generated."),
        ("n_kept", "sc:Integer", False, "Items under both the direct and the filler ceiling."),
        ("n_eval", "sc:Integer", False, "Items evaluated (seeded sample of kept_ids)."),
        ("kept_ids", "sc:Integer", True, "Pool line indices that passed the ceilings."),
        ("eval_ids", "sc:Integer", True, "Pool line indices evaluated; eval sample rg_<i> is eval_ids[i]."),
        ("eval_item_sha256", "sc:Text", True, "sha256(json.dumps([question, str(answer)])) per evaluated item."),
        ("item_seed", "sc:Integer", False, "Seed of the random.Random sample of eval_ids from kept_ids."),
    ]
    cells = from_file({
        "@type": "cr:RecordSet",
        "@id": f"{folder}_cells",
        "name": f"{folder}_cells",
        "description": cells_description,
        "key": {"@id": f"{folder}_cells/cell"},
        "field": [field(f"{folder}_cells/{n}", n, t, "x", column=n,
                        repeated=rep, description=d) for n, t, rep, d in cell_specs],
    }, f"v1.1/{folder}/cells.jsonl")
    return [pools, cells]


def rg_clean_record_sets() -> list[dict]:
    return rg_pool_record_sets(
        "reasoning_gym_clean", "rg_clean-pools-jsonl",
        "Clean re-run (Nov 2026) Reasoning Gym pools behind the camera-ready main RG figures: "
        "150 generated items per (model, family) cell at the calibrated D, for six models and "
        "four families. Which items passed the direct/filler ceilings and which were evaluated "
        "is in record set reasoning_gym_clean_cells. Family-specific metadata is in the JSONL "
        "`metadata` object.",
        "One record per (model, family) cell of the clean re-run: D, item counts, kept and "
        "evaluated item ids, hashes.")


def maze_followup_record_sets() -> list[dict]:
    return rg_pool_record_sets(
        "maze_followup", "maze_followup-pools-jsonl",
        "Opus 4.6 maze follow-up (Nov 2026, pre-registered): 600 fresh maze_varied items "
        "(Reasoning Gym maze generator, shortest-path distance 4-8, grid 6-10, 120 items per "
        "answer) filtered for Claude Opus 4.6 with the clean re-run's direct/filler ceilings. "
        "Separate from reasoning_gym_clean and never pooled with it; no item appears in any "
        "earlier maze pool. Kept and evaluated ids are in maze_followup_cells.",
        "One record for the follow-up's single cell (Opus 4.6, maze_varied): item counts, kept "
        "and evaluated item ids, hashes.")


RAI_EXTRA = {
    "rai:dataSocialImpact": (
        "Positive: lets AI developers and auditors measure whether frontier models can hide "
        "reasoning from chain-of-thought monitors, a precondition for relying on CoT monitoring "
        "as a safety measure. Risk: the evaluation protocol is dual-use; detailed prompts and "
        "transcripts could inform attempts to elicit or train covert reasoning. Mitigations: the "
        "problem content is benign, the license notes and datasheet ask that the data not be used "
        "to train steganographic encoding{canary}."
    ),
    "rai:hasSyntheticData": True,
    "prov:wasDerivedFrom": [
        {"@type": "sc:Dataset", "name": "Reasoning Gym",
         "url": "https://github.com/open-thought/reasoning-gym",
         "description": "Procedural generators (reasoning-gym 0.1.25) for basic_arithmetic, chain_sum, simple_equations and maze."},
    ],
    "prov:wasGeneratedBy": [
        {"@type": "prov:Activity", "name": "Generation",
         "description": f"Problem instances generated by Reasoning Gym generators and by the NEST counting generator from integer seeds (code: {CODE_URL}); cover questions written by the author."},
        {"@type": "prov:Activity", "name": "Difficulty calibration and filtering",
         "description": "Per (model, family) a difficulty rung D was chosen by binary search and pools were filtered to items the model solves with plain CoT but not by direct answer (and, in v1.1, not with filler tokens), by querying the model APIs. The v1.1 maze follow-up uses a fixed varied-answer maze configuration instead of a calibrated rung."},
        {"@type": "prov:Activity", "name": "Model evaluation",
         "description": f"v1.1 transcripts are outputs of Anthropic and OpenAI models under the NEST prompts (Inspect framework, code: {CODE_URL}), scored by an LLM monitor (Claude Sonnet 4.6)."},
        {"@type": "prov:Activity", "name": "Hand adjudication",
         "description": "The author read each v1.1 transcript and assigned the verdict field; no crowd workers."},
    ],
}


V11_COLLECTION = (
    " v1.1 (October-November 2026): clean re-run pools generated by the same Reasoning Gym "
    "generators for six models and filtered per (model, family) to items under a 0.3 ceiling "
    "for both direct answer and filler tokens, then a seeded sample was evaluated; a 600-item maze_varied pool "
    "(answers balanced) was generated and filtered the same way for an Opus 4.6 follow-up; 24 "
    "transcripts were picked by hand from NEST evaluation logs (April-November 2026) and "
    "adjudicated by the author."
)

CANARY_NOTE = (", and released transcripts carry a canary string so they can be excluded "
               "from training corpora")


def assemble(base: dict, *, name: str, version: str, description: str, url: str,
             distribution: list, record_sets: list, extra_limitations: list[str],
             has_transcripts: bool) -> dict:
    c = OrderedDict()
    ctx = dict(CONTEXT)
    ctx["prov"] = "http://www.w3.org/ns/prov#"
    c["@context"] = ctx
    c["@type"] = "sc:Dataset"
    c["conformsTo"] = f"http://mlcommons.org/croissant/{CROISSANT_VERSION}"
    c["name"] = name
    c["description"] = description
    c["creator"] = CREATOR
    c["license"] = LICENSE_URL
    c["url"] = url
    c["version"] = version
    c["datePublished"] = base["datePublished"]
    c["dateModified"] = "2026-11-06"
    c["citeAs"] = (
        "@inproceedings{karpov2026nest, title={{NEST}: Nascent Encoded Steganographic Thoughts}, "
        "author={Karpov, Artem}, booktitle={Advances in Neural Information Processing Systems "
        "(Evaluations and Datasets Track)}, year={2026}, url={https://huggingface.co/datasets/artkpv/nest}}"
    )
    c["keywords"] = ["steganography", "chain-of-thought", "CoT monitoring", "AI safety", "LLM evaluation"]
    for k, v in base.items():
        if k.startswith("rai:"):
            c[k] = copy.deepcopy(v)
    if has_transcripts:
        # v1.0's RAI text describes v1.0 alone; scope its version-specific items.
        scope = lambda s: "v1.0: " + s if ("aze" in s or "Pool sizes" in s or "pool sizes" in s) else s
        c["rai:dataLimitations"] = [scope(s) for s in c["rai:dataLimitations"]]
        c["rai:dataBiases"] = [scope(s) for s in c["rai:dataBiases"]]
        c["rai:dataCollection"] = "v1.0: " + c["rai:dataCollection"] + V11_COLLECTION
    c["rai:dataLimitations"] = c["rai:dataLimitations"] + extra_limitations
    c.update(copy.deepcopy(RAI_EXTRA))
    c["rai:dataSocialImpact"] = c["rai:dataSocialImpact"].format(
        canary=CANARY_NOTE if has_transcripts else "")
    if not has_transcripts:
        c["prov:wasGeneratedBy"] = c["prov:wasGeneratedBy"][:2]
    c["distribution"] = distribution
    c["recordSet"] = record_sets
    return c


def v10_parts(out: Path, prefix: str, local: bool) -> tuple[list, list, dict]:
    """FileObjects, FileSets and record sets for v1.0, with paths under `prefix`."""
    manifest = json.loads((out / "v1.0" / "manifest.json").read_text())
    base = build_croissant(manifest)
    objs = [file_object(out, f"v1.0/{rel}") for rel in manifest if rel != "croissant.json"]
    sets = [
        file_set("reasoning_gym-jsonl", f"{prefix}reasoning_gym/*.jsonl", "application/jsonlines",
                 "The 17 v1.0 Reasoning Gym pool files (one per (family, model) cell).", local),
        file_set("counting-jsonl", f"{prefix}counting/*.jsonl", "application/jsonlines",
                 "The 10 v1.0 counting trial files (one per difficulty D).", local),
        file_set("cover_questions-json", f"{prefix}cover_questions/*.json", "application/json",
                 "The two v1.0 cover-question files.", local),
    ]
    return objs, sets, base


V10_DESCRIPTION_TAIL = (
    " Hosted on Hugging Face since the NeurIPS 2026 camera-ready; data files are byte-identical "
    f"to the reviewed submission. Code: {CODE_URL}."
)


def build_croissants(out: Path, transcript_rels: list[str], rg_rels: list[str],
                     fu_rels: list[str], local: bool) -> tuple[dict, dict]:
    # Root croissant: the whole release; file set globs are relative to the repo root.
    objs, sets, base = v10_parts(out, "v1.0/", local)
    v11_files = ["v1.1/README.md", "v1.1/CHANGELOG.md", "v1.1/manifest.json",
                 *transcript_rels,
                 *[r for r in rg_rels + fu_rels if not r.endswith(".jsonl") or r.endswith("cells.jsonl")]]
    v11_objs = [file_object(out, r) for r in v11_files]
    rg_set = file_set("rg_clean-pools-jsonl", "v1.1/reasoning_gym_clean/pools/*.jsonl",
                      "application/jsonlines",
                      "The 24 v1.1 clean re-run pool files (one per (family, model) cell).", local)
    fu_set = file_set("maze_followup-pools-jsonl", "v1.1/maze_followup/pools/*.jsonl",
                      "application/jsonlines", "The v1.1 Opus 4.6 maze follow-up pool file.", local)
    top = [file_object(out, "README.md"), file_object(out, "LICENSE")]
    dist = ([] if local else [repo_object()]) + top + objs + v11_objs + sets + [rg_set, fu_set]
    record_sets = (base["recordSet"] + rg_clean_record_sets() + maze_followup_record_sets()
                   + [transcripts_record_set()])
    description = (
        "NEST (Nascent Encoded Steganographic Thoughts): problem instances and curated "
        "transcripts for evaluating steganographic chain-of-thought (CoT) capability and CoT "
        "monitorability in frontier LLMs. v1.0 (frozen, as reviewed): Reasoning Gym pools at "
        "per-(model, family) calibrated difficulty, steganographic counting trials, cover "
        "questions. v1.1 (camera-ready addendum): the clean re-run Reasoning Gym pools behind "
        "the camera-ready main RG figures (six models x four families, with kept and evaluated "
        "item ids), a 600-item varied-answer maze pool from a pre-registered Opus 4.6 follow-up, and "
        "24 hand-adjudicated model transcripts. Content is benign (arithmetic, "
        "mazes, opinion prompts); the evaluation protocol is dual-use and released for defensive "
        f"CoT-monitoring research, not for training models toward covert encoding. Code: {CODE_URL}."
    )
    root = assemble(
        base, name="nest", version="1.1.0", description=description,
        url=f"https://huggingface.co/datasets/{HF_REPO}",
        distribution=dist, record_sets=record_sets,
        extra_limitations=[
            "v1.1 transcripts are 24 hand-picked examples chosen to illustrate scored outcomes, "
            "not a random sample; they contain model outputs that remain subject to the providers' terms.",
            "v1.1 clean-run pools: the maze family has one constant answer per difficulty rung; "
            "GPT-5.4-mini maze has only 20 kept items; Sonnet 4.6 and Haiku 4.5 maze sit at the floor rung D=2.",
            "v1.1 maze follow-up: one model (Claude Opus 4.6) and one family; its filter is specific "
            "to that model version.",
        ],
        has_transcripts=True,
    )

    # v1.0 croissant: v1.0 alone; same layout, so the same repo-relative globs.
    objs10, sets10, base10 = v10_parts(out, "v1.0/", local)
    v10 = assemble(
        base10, name="nest-v1.0", version="1.0.0",
        description=base10["description"] + V10_DESCRIPTION_TAIL,
        url=f"https://huggingface.co/datasets/{HF_REPO}/tree/main/v1.0",
        distribution=([] if local else [repo_object()]) + objs10 + sets10,
        record_sets=base10["recordSet"], extra_limitations=[], has_transcripts=False,
    )
    return root, v10


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/release/drafts/hf-nest")
    ap.add_argument("--local-test", action="store_true",
                    help="Write croissant.local.json (file sets read the local folder, no git repo) for mlcroissant load.")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    commit = git_head()

    manifest10 = build_v10(out, commit)
    transcript_rels = build_transcripts(out)
    rg_rels = build_rg_pools(out, RG_CLEAN)
    fu_rels = build_rg_pools(out, MAZE_FOLLOWUP)
    for name in ("README.md", "CHANGELOG.md"):
        shutil.copy2(DRAFTS / "v1.1-docs" / name, out / "v1.1" / name)
    write_manifest(out, "v1.1", ["v1.1/README.md", "v1.1/CHANGELOG.md", *transcript_rels, *rg_rels, *fu_rels], commit)
    shutil.copy2(DRAFTS / "hf-card" / "README.md", out / "README.md")
    shutil.copy2(V10 / "LICENSE", out / "LICENSE")

    # v1.0/croissant.json is listed in v1.0/manifest.json (never the reverse), so
    # write the croissant first, then its manifest entry.
    # Neither croissant lists v1.0/croissant.json; v1.0/manifest.json lists it, so
    # write the croissants first and register v1.0/croissant.json after.
    (out / "v1.0" / "manifest.json").write_text(json.dumps(manifest10, indent=2) + "\n")
    root, v10 = build_croissants(out, transcript_rels, rg_rels, fu_rels, local=False)
    (out / "v1.0" / "croissant.json").write_text(json.dumps(v10, indent=2) + "\n")
    p = out / "v1.0" / "croissant.json"
    manifest10["croissant.json"] = OrderedDict([
        ("sha256", sha256(p)), ("bytes", p.stat().st_size), ("generator_commit", commit)])
    (out / "v1.0" / "manifest.json").write_text(json.dumps(manifest10, indent=2) + "\n")
    (out / "croissant.json").write_text(json.dumps(root, indent=2) + "\n")
    print(f"wrote {out}")

    if args.local_test:
        root_l, v10_l = build_croissants(out, transcript_rels, rg_rels, fu_rels, local=True)
        for c in (root_l, v10_l):
            for d in c["distribution"]:
                if d["@type"] == "cr:FileObject":
                    d["contentUrl"] = d["@id"]  # relative path, resolved next to the json
        (out / "croissant.local.json").write_text(json.dumps(root_l, indent=2) + "\n")
        print(f"wrote {out / 'croissant.local.json'} (local test only, do not upload)")


if __name__ == "__main__":
    main()
