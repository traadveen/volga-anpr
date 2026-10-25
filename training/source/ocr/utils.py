"""
Utility helpers shared across OCR training and data-preparation scripts.

The functions in this module focus on sanitising and validating Russian
number-plate labels so that both training and inference stages operate on a
consistent alphabet (AБЕКМHOPCTYX + digits, mapped to their Latin lookalikes).
"""

from __future__ import annotations

import re
from typing import Iterable

PLATE_ALPHABET = "ABEKMHOPCTYX"
ALLOWED_CHARS = set(PLATE_ALPHABET + "0123456789")

# Mapping Cyrillic letters that look like Latin ones to the Latin alphabet used
# by the OCR model. We encode characters via their Unicode codepoints to avoid
# accidental corruption when saving files in a non-UTF8 editor.
CYR_EQ = "\u0410\u0412\u0415\u041A\u041C\u041D\u041E\u0420\u0421\u0422\u0423\u0425"
CYR_TO_LAT = str.maketrans(
    {c: l for c, l in zip(CYR_EQ, PLATE_ALPHABET)}
    | {c.lower(): l for c, l in zip(CYR_EQ, PLATE_ALPHABET)}
)

# When we see digits that look like letters (and vice versa) we do not attempt
# to coerce them automatically here; that logic is handled elsewhere when the
# expected pattern is known. The goal of this module is to keep characters
# within the valid alphabet and get rid of obvious noise.


def pattern_for_len(length: int) -> str | None:
    """Return the canonical RU plate pattern for the provided length."""
    if length == 8:  # e.g. A001BC77
        return "LDDDLLDD"
    if length == 9:  # e.g. A001BC777
        return "LDDDLLDDD"
    return None


def matches_plate_pattern(text: str, allow_partial: bool = False) -> bool:
    """
    Check whether ``text`` fits the RU licence plate pattern.

    When ``allow_partial`` is True the function treats ``#`` as a wildcard
    character (matching any required symbol). This mirrors the evaluation
    scripts that tolerate missing characters.
    """
    if not text:
        return False
    pattern = pattern_for_len(len(text))
    if not pattern:
        return False
    for ch, kind in zip(text, pattern):
        if allow_partial and ch == "#":
            continue
        if kind == "D":
            if ch not in "0123456789":
                return False
        else:
            if ch not in PLATE_ALPHABET:
                return False
    return True


PLATE_REGEX = re.compile(rf"[{PLATE_ALPHABET}][0-9]{{3}}[{PLATE_ALPHABET}]{{2}}[0-9]{{2,3}}")


def is_valid_plate(text: str) -> bool:
    """Validate that the text matches the canonical RU plate pattern."""
    return bool(PLATE_REGEX.fullmatch(text or ""))


def sanitise_plate_characters(text: str) -> str:
    """
    Helper that transliterates and strips illegal characters, returning only
    digits or letters from ``PLATE_ALPHABET``. The function does not enforce the
    final pattern or length. This lossy character filter is not a GT-label
    normalizer; use ``sanitize_plate_label`` for annotations.
    """
    if not text:
        return ""
    cleaned = text.strip().replace(" ", "")
    cleaned = cleaned.translate(CYR_TO_LAT)
    cleaned = cleaned.upper()
    return "".join(ch for ch in cleaned if ch in ALLOWED_CHARS)


def sanitize_plate_label(text: str, *, allow_partial: bool = True) -> str:
    """
    High-level sanitiser used by dataset loaders. It transliterates the label,
    preserves explicit unknown slots and rejects unsupported characters. Only
    whitespace and hyphens at known plate-group boundaries are removed, never
    uncertain symbols or filename suffixes.

    Parameters
    ----------
    text:
        Raw label string that may contain Cyrillic symbols, whitespace or noise.
    allow_partial:
        When False the returned string must fully match the RU plate pattern,
        otherwise an empty string is returned. When True explicit ``#`` slots
        are retained without checking the pattern; partial labels must not
        become complete CTC training targets.
    """
    if not isinstance(text, str):
        return ""
    normalized = text.translate(CYR_TO_LAT).upper()
    cleaned = "".join(ch for ch in normalized if not ch.isspace())
    if "-" in cleaned:
        letter = rf"[{PLATE_ALPHABET}#]"
        digit = r"[0-9#]"
        white = rf"{letter}-?{digit}{{3}}-?{letter}{{2}}-?{digit}{{2,3}}"
        yellow = rf"{letter}{{2}}-?{digit}{{3}}-?{digit}{{2,3}}"
        if not (re.fullmatch(white, cleaned) or re.fullmatch(yellow, cleaned)):
            return ""
        cleaned = cleaned.replace("-", "")
    if any(ch not in ALLOWED_CHARS and ch != "#" for ch in cleaned):
        return ""
    if not cleaned:
        return ""
    if allow_partial:
        return cleaned
    return cleaned if matches_plate_pattern(cleaned, allow_partial=False) else ""


def drop_invalid_labels(labels: Iterable[str]) -> list[str]:
    """
    Convenience helper that sanitises an iterable of labels and keeps only those
    that match the RU format.
    """
    result: list[str] = []
    for label in labels:
        cleaned = sanitize_plate_label(label, allow_partial=False)
        if cleaned:
            result.append(cleaned)
    return result


__all__ = [
    "ALLOWED_CHARS",
    "PLATE_ALPHABET",
    "sanitize_plate_label",
    "sanitise_plate_characters",
    "matches_plate_pattern",
    "pattern_for_len",
    "is_valid_plate",
    "drop_invalid_labels",
]
