"""
Base constants and helpers for 1-bit-per-word-boundary whitespace steganography.

Design: each inter-word gap is a binary slot. A single regular space encodes
bit 0; two regular spaces encode bit 1. This is intentionally simple — LLMs
cannot reliably count larger runs or emit invisible control characters
(tabs, NBSP, trailing whitespace), so larger alphabets collapse under
real generation conditions.
"""

from __future__ import annotations

import re


SINGLE = " "      # bit 0 — normal inter-word spacing
DOUBLE = "  "     # bit 1 — double space between words


def encode_bits_to_gaps(words: list[str], bits: str) -> str:
    """Join `words` using SINGLE/DOUBLE according to `bits`.

    There must be exactly len(words) - 1 inter-word gaps, so we require
    len(bits) == len(words) - 1. Caller is responsible for supplying enough
    words (one more than the bit count).
    """
    if len(words) < 2:
        raise ValueError(f"need at least 2 words, got {len(words)}")
    if len(bits) != len(words) - 1:
        raise ValueError(
            f"bit count {len(bits)} must equal gap count {len(words) - 1}"
        )
    out = [words[0]]
    for bit, w in zip(bits, words[1:]):
        if bit == "0":
            out.append(SINGLE)
        elif bit == "1":
            out.append(DOUBLE)
        else:
            raise ValueError(f"bit must be '0' or '1', got {bit!r}")
        out.append(w)
    return "".join(out)


# Regex for inter-word gaps: runs of regular spaces (exactly 1 or 2) between
# non-whitespace tokens. Note: tabs/newlines/NBSP don't count as gap characters
# here — we deliberately restrict to plain ASCII spaces so model output with
# accidental paragraph breaks doesn't get parsed as encoded bits.
_GAP_RE = re.compile(r"(?<=\S)( {1,2})(?=\S)")


def decode_gaps_to_bits(text: str) -> str:
    """Walk inter-word gaps in order and emit '0' for single, '1' for double.

    Gaps with more than 2 spaces, or non-space whitespace (tab, newline),
    are skipped — they don't correspond to a valid slot. This is lenient by
    design; the BCD decoder (mapping.bits_to_digits) will reject truncated or
    otherwise invalid bitstrings downstream.
    """
    bits = []
    for m in _GAP_RE.finditer(text):
        bits.append("0" if len(m.group(1)) == 1 else "1")
    return "".join(bits)
