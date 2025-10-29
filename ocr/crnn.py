from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import Dataset

from .utils import matches_plate_pattern, sanitize_plate_label

DEFAULT_ALPHABET = "0123456789ABCEHKMOPTXY"


def alphabet_from_tsv(tsv_path: str) -> str:
    """Scan labels.tsv and return sorted unique characters."""
    chars = set()
    with open(tsv_path, "r", encoding="utf-8") as fh:
        for line in fh:
            try:
                _, label = line.rstrip("\n").split("\t", 1)
            except ValueError:
                continue
            chars.update(label.strip())
    sorted_chars = "".join(sorted(chars))
    return sorted_chars or DEFAULT_ALPHABET


def _normalize_image(gray: np.ndarray, image_height: int, image_width: int) -> np.ndarray:
    """Resize and pad the image to a fixed canvas while keeping aspect ratio."""
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        raise ValueError("Empty image encountered during preprocessing.")
    scale = image_height / float(h)
    new_w = int(round(w * scale))
    new_w = max(1, min(image_width, new_w))
    resized = cv2.resize(gray, (new_w, image_height), interpolation=cv2.INTER_AREA if new_w < w else cv2.INTER_CUBIC)
    canvas = np.zeros((image_height, image_width), dtype=np.float32)
    canvas[:, :new_w] = resized.astype(np.float32)
    canvas /= 255.0
    canvas = (canvas - 0.5) / 0.5  # normalize to roughly [-1, 1]
    return canvas


def _read_gray_image(path: str) -> np.ndarray:
    """
    Read an image from ``path`` in grayscale using a Unicode-safe fallback.

    OpenCV on Windows struggles with non-ASCII paths when using ``imread``;
    for such cases we fall back to decoding from raw bytes.
    """
    needs_fallback = os.name == "nt" and any(ord(ch) > 127 for ch in path)
    if not needs_fallback:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except (OSError, ValueError) as exc:
        raise FileNotFoundError(f"Failed to read image: {path}") from exc
    if data.size == 0:
        raise FileNotFoundError(f"Empty image data: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Failed to decode image: {path}")
    return img


class CTCLabelConverter:
    """Bidirectional mapping between text labels and CTC targets."""

    def __init__(self, alphabet: str):
        if len(alphabet) == 0:
            raise ValueError("Alphabet must contain at least one symbol.")
        self.alphabet = alphabet
        self.blank_idx = len(alphabet)
        self.char_to_idx = {ch: idx for idx, ch in enumerate(alphabet)}

    def encode(self, texts: Sequence[str]) -> Tuple[Tensor, Tensor]:
        lengths = torch.tensor([len(t) for t in texts], dtype=torch.long)
        if lengths.numel() == 0:
            return torch.empty(0, dtype=torch.long), lengths
        total_length = int(lengths.sum().item())
        targets = torch.empty(total_length, dtype=torch.long)
        offset = 0
        for t in texts:
            for ch in t:
                if ch not in self.char_to_idx:
                    raise ValueError(f"Character '{ch}' is not in the alphabet.")
                targets[offset] = self.char_to_idx[ch]
                offset += 1
        return targets, lengths

    def _decode_sequence(
        self,
        sequence: Iterable[int],
        probs: Iterable[float],
    ) -> Tuple[str, float, List[float]]:
        result_chars: List[str] = []
        char_probs: List[float] = []
        prev = self.blank_idx
        for idx, prob in zip(sequence, probs):
            if idx == self.blank_idx:
                prev = self.blank_idx
                continue
            if idx == prev:
                if char_probs:
                    char_probs[-1] = max(char_probs[-1], float(prob))
                continue
            result_chars.append(self.alphabet[idx])
            char_probs.append(float(prob))
            prev = idx
        text = "".join(result_chars)
        confidence = float(np.mean(char_probs)) if char_probs else 0.0
        return text, confidence, char_probs

    def decode_logits(self, logits: Tensor) -> Tuple[List[str], List[float]]:
        probs = torch.softmax(logits, dim=2)
        max_prob, max_idx = probs.max(dim=2)  # T x B
        max_idx = max_idx.transpose(0, 1).cpu().numpy()
        max_prob = max_prob.transpose(0, 1).cpu().numpy()
        texts: List[str] = []
        confs: List[float] = []
        per_char: List[List[float]] = []
        for seq, seq_prob in zip(max_idx, max_prob):
            text, conf, chars = self._decode_sequence(seq, seq_prob)
            texts.append(text)
            confs.append(conf)
            per_char.append(chars)
        return texts, confs, per_char


@dataclass
class PlateSample:
    path: str
    label: str
    raw_label: str


class PlateOCRDataset(Dataset):
    """Dataset reading plate crops and labels from a TSV file."""

    def __init__(
        self,
        root_dir: str,
        tsv_file: str,
        converter: CTCLabelConverter,
        image_height: int = 32,
        image_width: int = 160,
    ):
        self.root_dir = root_dir
        self.converter = converter
        self.image_height = image_height
        self.image_width = image_width
        self.samples: List[PlateSample] = []
        self.warnings: List[str] = []
        self.stats: Dict[str, int] = {
            "rows": 0,
            "kept": 0,
            "missing_files": 0,
            "invalid_label": 0,
            "invalid_pattern": 0,
        }

        tsv_path = os.path.join(root_dir, tsv_file)
        if not os.path.exists(tsv_path):
            raise FileNotFoundError(f"labels TSV not found: {tsv_path}")

        with open(tsv_path, "r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, 1):
                line = line.rstrip("\n")
                if not line:
                    continue
                self.stats["rows"] += 1
                try:
                    rel_path, label = line.split("\t", 1)
                except ValueError:
                    self.stats["invalid_label"] += 1
                    if len(self.warnings) < 20:
                        self.warnings.append(f"[line {line_no}] Malformed row -> skipped")
                    continue
                rel_path = rel_path.strip()
                raw_label = label.strip()
                if not rel_path or not raw_label:
                    self.stats["invalid_label"] += 1
                    if len(self.warnings) < 20:
                        self.warnings.append(f"[line {line_no}] Empty path or label -> skipped")
                    continue
                sanitized = sanitize_plate_label(raw_label, allow_partial=True)
                if not sanitized:
                    self.stats["invalid_label"] += 1
                    if len(self.warnings) < 20:
                        self.warnings.append(
                            f"[line {line_no}] Label '{raw_label}' -> empty after sanitisation"
                        )
                    continue
                if not matches_plate_pattern(sanitized, allow_partial=False):
                    self.stats["invalid_pattern"] += 1
                    if len(self.warnings) < 20:
                        self.warnings.append(
                            f"[line {line_no}] Label '{raw_label}' sanitised to '{sanitized}' but fails RU pattern"
                        )
                    continue
                full_path = os.path.normpath(os.path.join(root_dir, rel_path))
                if not os.path.isfile(full_path):
                    self.stats["missing_files"] += 1
                    if len(self.warnings) < 20:
                        self.warnings.append(f"[line {line_no}] Missing image: {full_path}")
                    continue
                self.samples.append(
                    PlateSample(path=full_path, label=sanitized, raw_label=raw_label)
                )

        if not self.samples:
            raise RuntimeError(
                f"No valid samples found in {tsv_path} (rows={self.stats['rows']}, kept=0)."
            )
        self.stats["kept"] = len(self.samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        sample = self.samples[idx]
        img = _read_gray_image(sample.path)
        img = _normalize_image(img, self.image_height, self.image_width)
        tensor = torch.from_numpy(img).unsqueeze(0)  # 1 x H x W
        encoded, lengths = self.converter.encode([sample.label])
        return tensor, encoded, lengths[0], sample.label


class CRNN(nn.Module):
    """CRNN backbone with a simple ResNet-like feature extractor and BiLSTM head."""

    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 256,
        lstm_layers: int = 2,
    ):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)),

            nn.Conv2d(512, 512, kernel_size=2, padding=0),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
        )

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        b, c, h, w = features.size()
        if h != 1:
            raise RuntimeError(f"Expected feature map height 1, got {h}")
        features = features.squeeze(2)  # B x C x W
        features = features.permute(2, 0, 1)  # W x B x C
        recurrent, _ = self.lstm(features)
        logits = self.classifier(recurrent)  # W x B x C
        return logits


def create_crnn_model(
    alphabet: str,
    hidden_size: int = 256,
    lstm_layers: int = 2,
) -> CRNN:
    num_classes = len(alphabet) + 1  # include CTC blank
    return CRNN(num_classes=num_classes, hidden_size=hidden_size, lstm_layers=lstm_layers)


def load_crnn_checkpoint(
    checkpoint_path: str,
    device: torch.device | str = "cpu",
):
    ckpt = torch.load(checkpoint_path, map_location=device)
    if "alphabet" not in ckpt or "model" not in ckpt:
        raise KeyError("Checkpoint must contain 'alphabet' and 'model' keys.")
    alphabet = ckpt["alphabet"]
    cfg = ckpt.get("config", {})
    model = create_crnn_model(
        alphabet=alphabet,
        hidden_size=cfg.get("hidden_size", 256),
        lstm_layers=cfg.get("lstm_layers", 2),
    )
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()
    converter = CTCLabelConverter(alphabet)
    image_height = cfg.get("image_height", 32)
    image_width = cfg.get("image_width", 160)
    return model, converter, image_height, image_width


class CRNNRecognizer:
    """Runtime wrapper exposing EasyOCR-like readtext() API for the trained CRNN."""

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.model, self.converter, self.image_height, self.image_width = load_crnn_checkpoint(
            checkpoint_path, device=self.device
        )

    @torch.inference_mode()
    def readtext(self, img, detail: int = 1, **kwargs):
        if isinstance(img, np.ndarray):
            gray = img
        else:
            raise TypeError("CRNNRecognizer expects numpy.ndarray inputs.")
        norm = _normalize_image(gray, self.image_height, self.image_width)
        tensor = torch.from_numpy(norm).unsqueeze(0).unsqueeze(0).to(self.device)
        logits = self.model(tensor)  # W x B x C
        texts, confs, per_char = self.converter.decode_logits(logits)
        text = texts[0] if texts else ""
        conf = confs[0] if confs else 0.0
        extras = {"char_probs": per_char[0]} if per_char else {}
        if detail:
            return [((0, 0, 0, 0), text, conf, extras)]
        return [text]
