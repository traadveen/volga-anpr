"""
Utilities for training and using the custom CRNN-based OCR recognizer.

The module exposes helpers shared between the training script and runtime
inference inside ``run.py``.
"""

from .crnn import (  # noqa: F401
    CRNNRecognizer,
    create_crnn_model,
    load_crnn_checkpoint,
    PlateOCRDataset,
    alphabet_from_tsv,
    CTCLabelConverter,
)
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
