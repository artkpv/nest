"""
Model metadata registry for plotting.

Maps Inspect AI model identifiers to display metadata (short name, family,
release date) used to label and order plots. Ported from
`api_client.MODEL_INFO`, with `get_model_info` tolerant of provider-prefixed
Inspect identifiers like `anthropic/claude-opus-4-6`.
"""

from __future__ import annotations


MODEL_INFO: dict[str, dict] = {
    # Anthropic
    "claude-opus-4-7": {
        "family": "Claude",
        "release_date": "2026-04",
        "short_name": "Claude-4.7-Opus",
    },
    "claude-opus-4-6": {
        "family": "Claude",
        "release_date": "2026-02",
        "short_name": "Claude-4.6-Opus",
    },
    "claude-opus-4-6-20260205": {
        "family": "Claude",
        "release_date": "2026-02",
        "short_name": "Claude-4.6-Opus",
    },
    "claude-sonnet-4-6": {
        "family": "Claude",
        "release_date": "2026-02",
        "short_name": "Claude-4.6-Sonnet",
    },
    "claude-opus-4-5-20251101": {
        "family": "Claude",
        "release_date": "2025-11",
        "short_name": "Claude-4.5-Opus",
    },
    "claude-haiku-4-5-20251001": {
        "family": "Claude",
        "release_date": "2025-10",
        "short_name": "Claude-4.5-Haiku",
    },
    "claude-sonnet-4-5-20250929": {
        "family": "Claude",
        "release_date": "2025-09",
        "short_name": "Claude-4.5-Sonnet",
    },
    "claude-opus-4-1-20250805": {
        "family": "Claude",
        "release_date": "2025-08",
        "short_name": "Claude-4.1-Opus",
    },
    "claude-opus-4-20250514": {
        "family": "Claude",
        "release_date": "2025-05",
        "short_name": "Claude-4-Opus",
    },
    "claude-sonnet-4-20250514": {
        "family": "Claude",
        "release_date": "2025-05",
        "short_name": "Claude-4-Sonnet",
    },
    "claude-3-7-sonnet-20250219": {
        "family": "Claude",
        "release_date": "2025-02",
        "short_name": "Claude-3.7-Sonnet",
    },
    "claude-3-5-haiku-20241022": {
        "family": "Claude",
        "release_date": "2024-10",
        "short_name": "Claude-3.5-Haiku",
    },
    "claude-3-haiku-20240307": {
        "family": "Claude",
        "release_date": "2024-03",
        "short_name": "Claude-3-Haiku",
    },
    # OpenAI
    "gpt-3.5-turbo-0125": {
        "family": "GPT",
        "release_date": "2024-01",
        "short_name": "GPT-3.5-Turbo",
    },
    "gpt-4-0613": {
        "family": "GPT",
        "release_date": "2023-06",
        "short_name": "GPT-4",
    },
    "gpt-4-turbo-2024-04-09": {
        "family": "GPT",
        "release_date": "2024-04",
        "short_name": "GPT-4-Turbo",
    },
    "gpt-4o-2024-05-13": {
        "family": "GPT",
        "release_date": "2024-05",
        "short_name": "GPT-4o",
    },
    "gpt-4o-2024-08-06": {
        "family": "GPT",
        "release_date": "2024-08",
        "short_name": "GPT-4o-Aug",
    },
    "gpt-4o-2024-11-20": {
        "family": "GPT",
        "release_date": "2024-11",
        "short_name": "GPT-4o-Nov",
    },
    "gpt-4o-mini-2024-07-18": {
        "family": "GPT",
        "release_date": "2024-07",
        "short_name": "GPT-4o-mini",
    },
    "gpt-4.1-2025-04-14": {
        "family": "GPT",
        "release_date": "2025-04",
        "short_name": "GPT-4.1",
    },
    "gpt-4.1-mini-2025-04-14": {
        "family": "GPT",
        "release_date": "2025-04",
        "short_name": "GPT-4.1-mini",
    },
    "gpt-4.1-nano-2025-04-14": {
        "family": "GPT",
        "release_date": "2025-04",
        "short_name": "GPT-4.1-nano",
    },
    "gpt-5-2025-08-07": {
        "family": "GPT",
        "release_date": "2025-08",
        "short_name": "GPT-5",
    },
    "gpt-5-mini-2025-08-07": {
        "family": "GPT",
        "release_date": "2025-08",
        "short_name": "GPT-5-mini",
    },
    "gpt-5-nano-2025-08-07": {
        "family": "GPT",
        "release_date": "2025-08",
        "short_name": "GPT-5-nano",
    },
    "gpt-5.1-2025-11-13": {
        "family": "GPT",
        "release_date": "2025-11",
        "short_name": "GPT-5.1",
    },
    "gpt-5.2-2025-12-11": {
        "family": "GPT",
        "release_date": "2025-12",
        "short_name": "GPT-5.2",
    },
    "gpt-5.4": {
        "family": "GPT",
        "release_date": "2026-02",
        "short_name": "GPT-5.4",
    },
    "gpt-5.4-mini": {
        "family": "GPT",
        "release_date": "2026-02",
        "short_name": "GPT-5.4-mini",
    },
    "gpt-6-astra": {
        "family": "GPT",
        "release_date": "2026-09",
        "short_name": "GPT-6-Astra",
    },
    # Open-weight (OpenRouter / Together)
    "openai/gpt-oss-20b": {
        "family": "GPT-OSS",
        "release_date": "2025-08",
        "short_name": "gpt-oss-20B",
    },
    "qwen/qwen3.5-9b": {
        "family": "Qwen",
        "release_date": "2026-02",
        "short_name": "Qwen3.5-9B",
    },
    # CoT-controllability study (SPAR, Cot-control-spar/cot-control-treated-models),
    # served on one vLLM instance: base = INT4 weights without the adapter,
    # treated = same weights + the `qwen3.8-27b-A` LoRA (many-rule training).
    "RedHatAI/Qwen3.8-27B-INT4": {
        "family": "Qwen",
        "release_date": "2026-08",
        "short_name": "Qwen3.8-27B-INT4",
    },
    "qwen3.8-27b-A": {
        "family": "Qwen",
        "release_date": "2026-08",
        "short_name": "Qwen3.8-27B-INT4+A",
    },
    # Few-rule control arm (6 ReasonIF rules) - isolates many-rule training from
    # fine-tuning on reasoning-format rules in general (SPAR README).
    "qwen3.8-27b-Q5": {
        "family": "Qwen",
        "release_date": "2026-08",
        "short_name": "Qwen3.8-27B-INT4+Q5",
    },
    # gpt-oss arms: attention-only LoRAs served on the original MXFP4 weights;
    # the untreated arm is "openai/gpt-oss-20b" itself (entry above).
    "gpt-oss-20b-A": {
        "family": "GPT-OSS",
        "release_date": "2025-08",
        "short_name": "gpt-oss-20B+A",
    },
    "gpt-oss-20b-Q5": {
        "family": "GPT-OSS",
        "release_date": "2025-08",
        "short_name": "gpt-oss-20B+Q5",
    },
    # OpenRouter / others
    "deepseek/deepseek-v3.2": {
        "family": "DeepSeek",
        "release_date": "2025-12",
        "short_name": "DeepSeek-V3.2",
    },
    "deepseek/deepseek-chat-v3-0324": {
        "family": "DeepSeek",
        "release_date": "2025-03",
        "short_name": "DeepSeek-V3",
    },
    "qwen/qwen3-235b-a22b-2507": {
        "family": "Qwen",
        "release_date": "2025-07",
        "short_name": "Qwen3-235B",
    },
    "moonshotai/kimi-k2": {
        "family": "Kimi",
        "release_date": "2025-07",
        "short_name": "Kimi-K2",
    },
    "google/gemini-2.5-pro": {
        "family": "Gemini",
        "release_date": "2025-06",
        "short_name": "Gemini-2.5-Pro",
    },
    "google/gemini-3-pro-preview": {
        "family": "Gemini",
        "release_date": "2025-11",
        "short_name": "Gemini-3-Pro",
    },
    "x-ai/grok-code-fast-1": {
        "family": "Grok",
        "release_date": "2025-08",
        "short_name": "Grok-Code-Fast",
    },
    "x-ai/grok-4.1-fast": {
        "family": "Grok",
        "release_date": "2025-11",
        "short_name": "Grok-4.1-Fast",
    },
}


# Aliases: legacy / shorthand IDs that should resolve to the same canonical
# short_name as the dated record above. Without these, `get_model_info` falls
# back to the raw ID and produces duplicate rows in heatmaps for the same model.
_ALIASES: dict[str, str] = {
    # Claude undated / shorthand forms used in old stego_counting trials.
    # These are what remain after provider-prefix stripping.
    "gpt-5.4-2026-03-05": "gpt-5.4",
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
    "claude-haiku-4-5": "claude-haiku-4-5-20251001",
    "claude-sonnet-4-5": "claude-sonnet-4-5-20250929",
    "claude-opus-4-5": "claude-opus-4-5-20251101",
    "claude-opus-4-1": "claude-opus-4-1-20250805",
    "claude-opus-4": "claude-opus-4-20250514",
    "claude-sonnet-4": "claude-sonnet-4-20250514",
    "claude-3-7-sonnet": "claude-3-7-sonnet-20250219",
    "claude-3-5-sonnet": "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku": "claude-3-5-haiku-20241022",
    "claude-3-haiku": "claude-3-haiku-20240307",
    "sonnet-3-7": "claude-3-7-sonnet-20250219",
    "sonnet-3-5": "claude-3-5-sonnet-20241022",
    "haiku-3-5": "claude-3-5-haiku-20241022",
    "haiku-3": "claude-3-haiku-20240307",
    "opus-4": "claude-opus-4-20250514",
    "opus-4-1": "claude-opus-4-1-20250805",
    "opus-4-5": "claude-opus-4-5-20251101",
    "opus-4-6": "claude-opus-4-6",
    "sonnet-4": "claude-sonnet-4-20250514",
    "sonnet-4-5": "claude-sonnet-4-5-20250929",
    "haiku-4-5": "claude-haiku-4-5-20251001",
    # Anthropic 3-5-sonnet (added because trials reference it).
    "claude-3-5-sonnet-20241022": "claude-3-5-sonnet-20241022",
    # Gemini provider-prefix variant -> google/ canonical.
    "gemini/gemini-2.5-pro": "google/gemini-2.5-pro",
    # OpenAI shorthand IDs (no date) -> dated canonical entry where possible.
    "openai/gpt-5.2": "gpt-5.2-2025-12-11",
}

# Add Claude-3.5-Sonnet canonical entry if not already present.
if "claude-3-5-sonnet-20241022" not in MODEL_INFO:
    MODEL_INFO["claude-3-5-sonnet-20241022"] = {
        "family": "Claude",
        "release_date": "2024-10",
        "short_name": "Claude-3.5-Sonnet",
    }


def canonical_model_id(inspect_model: str) -> str:
    """Return the canonical MODEL_INFO key for *inspect_model*, or the original string."""
    if inspect_model in MODEL_INFO:
        return inspect_model
    if inspect_model in _ALIASES:
        return _ALIASES[inspect_model]
    parts = inspect_model.split("/")
    for i in range(1, len(parts)):
        candidate = "/".join(parts[i:])
        if candidate in MODEL_INFO:
            return candidate
        if candidate in _ALIASES:
            return _ALIASES[candidate]
    return inspect_model


def get_model_info(inspect_model: str) -> dict:
    """
    Look up display metadata for an Inspect AI model identifier.

    Inspect identifiers are typically `provider/model` (e.g.
    `anthropic/claude-opus-4-6`, `openai/gpt-4o-2024-11-20`). This function
    tries the full string first, then progressively strips leading
    provider segments, then falls back to a generic record.
    """
    if inspect_model in MODEL_INFO:
        return MODEL_INFO[inspect_model]

    # Resolve legacy/shorthand alias to canonical key.
    if inspect_model in _ALIASES:
        return MODEL_INFO[_ALIASES[inspect_model]]

    # Strip provider prefixes one at a time (e.g. `openrouter/deepseek/deepseek-v3.2`).
    parts = inspect_model.split("/")
    for i in range(1, len(parts)):
        candidate = "/".join(parts[i:])
        if candidate in MODEL_INFO:
            return MODEL_INFO[candidate]
        if candidate in _ALIASES:
            return MODEL_INFO[_ALIASES[candidate]]

    return {
        "family": "Unknown",
        "release_date": "1970-01",
        "short_name": inspect_model,
    }


def parse_model_name_and_version(short_name: str) -> tuple[str, str, float]:
    """
    Parse a display short_name to (company, model_name, version).

    Examples:
        "Claude-4.5-Opus" -> ("Anthropic", "Opus", 4.5)
        "GPT-5.1" -> ("OpenAI", "GPT", 5.1)
        "GPT-5-mini" -> ("OpenAI", "Mini", 5.0)
        "GPT-4.1-nano" -> ("OpenAI", "Nano", 4.1)
        "Gemini-2.5-Pro" -> ("Google", "Gemini", 2.5)

    Ported verbatim from legacy `stego_counting_plots.parse_model_name_and_version`.
    """
    parts = short_name.split("-")

    if parts[0] == "Claude":
        company = "Anthropic"
        name = parts[-1]
        try:
            version = float(parts[1])
        except (ValueError, IndexError):
            version = 0.0
        return (company, name, version)

    family = parts[0]
    company_map = {
        "GPT": "OpenAI",
        "Gemini": "Google",
        "DeepSeek": "DeepSeek",
        "Qwen": "Qwen",
        "Qwen3": "Qwen",
        "Kimi": "Moonshot AI",
        "Grok": "xAI",
    }
    company = company_map.get(family, family)

    if family == "GPT":
        tier = "GPT"
        if parts[-1].lower() == "mini":
            tier = "Mini"
            parts = parts[:-1]
        elif parts[-1].lower() == "nano":
            tier = "Nano"
            parts = parts[:-1]

        version = 0.0
        if len(parts) > 1:
            version_str = parts[1]
            if version_str.endswith("o"):
                try:
                    version = float(version_str[:-1])
                except ValueError:
                    version = 0.0
            else:
                try:
                    version = float(version_str)
                except ValueError:
                    version = 0.0
        return (company, tier, version)

    if len(parts) > 1:
        version_str = parts[1]
        if version_str.endswith("o"):
            try:
                version = float(version_str[:-1])
            except ValueError:
                version = 0.0
        elif version_str.startswith("K"):
            try:
                version = float(version_str[1:])
            except ValueError:
                version = 0.0
        elif version_str.startswith("V"):
            try:
                version = float(version_str[1:])
            except ValueError:
                version = 0.0
        else:
            try:
                version = float(version_str)
            except ValueError:
                version = 0.0
    else:
        version = 0.0

    return (company, family, version)
