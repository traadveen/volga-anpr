"""
Utilities for training and using the custom CRNN-based OCR recognizer.

The module exposes helpers shared between the training script and runtime
inference inside ``run.py``. Model APIs load lazily so label utilities stay
independent of the training framework.
"""

_MODEL_EXPORTS = {"CRNNRecognizer", "create_crnn_model", "load_crnn_checkpoint",
                  "PlateOCRDataset", "alphabet_from_tsv", "CTCLabelConverter"}
from .utils import (  # noqa: F401
    ALLOWED_CHARS,
    PLATE_ALPHABET,
    sanitize_plate_label,
    sanitise_plate_characters,
    matches_plate_pattern,
    pattern_for_len,
    is_valid_plate,
    drop_invalid_labels,
)

__all__ = [
    "CRNNRecognizer",
    "create_crnn_model",
    "load_crnn_checkpoint",
    "PlateOCRDataset",
    "alphabet_from_tsv",
    "CTCLabelConverter",
    "ALLOWED_CHARS",
    "PLATE_ALPHABET",
    "sanitize_plate_label",
    "sanitise_plate_characters",
    "matches_plate_pattern",
    "pattern_for_len",
    "is_valid_plate",
    "drop_invalid_labels",
]


def __getattr__(name):
    if name not in _MODEL_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module
    value = getattr(import_module(".crnn", __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
