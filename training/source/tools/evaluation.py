"""Auditable internal ANPR diagnostics. No model or third-party dependency imports."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable

VERSION = "1.0.0"
TARGET_TYPES = ("type1", "type1a", "type1b")
PLATE_TYPES = (*TARGET_TYPES, "other")
CONDITIONS = ("day", "night", "angle", "blur", "motion_blur", "glare", "dirt", "small_plate")
ERROR_CATEGORIES = ("detection", "ocr", "false_positive", "wrong_type", "bad_crop")
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Record:
    image: str
    plate_num: str
    plate_type: str
    line: int = 0
    is_vehicle: bool = True
    confidence: float | None = None
    bbox: Box | None = None
    conditions: tuple[str, ...] = ()
    quad: tuple[tuple[float, float], ...] | None = None
    raw_ocr: str | None = None
    postprocessed_ocr: str | None = None
    crop_bbox: Box | None = None
    det_conf: float | None = None

    @property
    def target(self) -> bool:
        return self.plate_type in TARGET_TYPES and self.is_vehicle

    @property
    def placeholder(self) -> bool:
        return not self.plate_num and not self.plate_type


def safe_image_name(value: str) -> str:
    """Normalize separators, but never discard directories or accept traversal."""
    value = value.strip().replace("\\", "/")
    if not value or "\x00" in value or ":" in value:
        raise ValueError(f"Invalid image name: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or not path.name or path.name == ".":
        raise ValueError(f"Image path must stay relative to images-root: {value!r}")
    return str(path)


def finite(value: str | float, label: str, probability: bool = False) -> float:
    try:
        number = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{label}: expected a finite number") from exc
    if not math.isfinite(number) or (probability and not 0 <= number <= 1):
        raise ValueError(f"{label}: expected {'a probability in [0, 1]' if probability else 'a finite number'}")
    return number


def parse_box(value: str, label: str) -> Box | None:
    if not value.strip():
        return None
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: expected JSON [x, y, width, height]") from exc
    if not isinstance(raw, list) or len(raw) != 4 or any(type(x) not in (int, float) for x in raw):
        raise ValueError(f"{label}: expected JSON [x, y, width, height]")
    box = tuple(finite(x, label) for x in raw)
    if box[2] <= 0 or box[3] <= 0:
        raise ValueError(f"{label}: width and height must be positive")
    if not math.isfinite(box[0] + box[2]) or not math.isfinite(box[1] + box[3]):
        raise ValueError(f"{label}: box endpoints overflow")
    if not math.isfinite(box[2] * box[3]) or box[2] * box[3] == 0:
        raise ValueError(f"{label}: box area overflows or underflows")
    return box  # type: ignore[return-value]


def parse_quad(value: str, label: str) -> tuple[tuple[float, float], ...] | None:
    if not value.strip():
        return None
    try:
        raw = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}: expected four [x, y] points") from exc
    if not isinstance(raw, list) or len(raw) != 4:
        raise ValueError(f"{label}: expected four [x, y] points")
    points = []
    for point in raw:
        if not isinstance(point, list) or len(point) != 2 or any(type(x) not in (int, float) for x in point):
            raise ValueError(f"{label}: expected four [x, y] points")
        points.append(tuple(finite(x, label) for x in point))
    return tuple(points)


def boolean(value: str, label: str) -> bool:
    if value.lower() in ("1", "true"):
        return True
    if value.lower() in ("0", "false"):
        return False
    raise ValueError(f"{label}: expected 0/1 or false/true")


def read_rows(path: str | Path, required: Iterable[str]) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        fields = reader.fieldnames or []
        if len(fields) != len(set(fields)):
            raise ValueError(f"{path}: duplicate CSV column names")
        missing = set(required) - set(fields)
        if missing:
            raise ValueError(f"{path}: missing columns: {', '.join(sorted(missing))}")
        result = []
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"{path}:{reader.line_num}: malformed CSV row")
            clean = {key: value.strip() for key, value in row.items()}
            clean["_line"] = str(reader.line_num)
            result.append(clean)
        return result


def _common(row: dict[str, str]) -> dict:
    line = int(row["_line"])
    tags = set(filter(None, re.split(r"[,|\s]+", row.get("conditions", ""))))
    for condition in CONDITIONS:
        if row.get(condition, "") and boolean(row[condition], f"line {line} {condition}"):
            tags.add(condition)
    return {
        "image": safe_image_name(row["image"]), "plate_num": row["plate_num"],
        "plate_type": row["plate_type"], "line": line,
        "bbox": parse_box(row.get("bbox", ""), f"line {line} bbox"),
        "quad": parse_quad(row.get("quad", ""), f"line {line} quad"),
        "conditions": tuple(sorted(tags)),
    }


def load_ground_truth(path: str | Path) -> list[Record]:
    result: list[Record] = []
    seen = set()
    per_image: dict[str, list[Record]] = defaultdict(list)
    for row in read_rows(path, ("image", "plate_num", "plate_type", "is_vehicle")):
        common = _common(row)
        placeholder = not row["plate_num"] and not row["plate_type"]
        if not placeholder and row["plate_type"] not in PLATE_TYPES:
            raise ValueError(f"line {row['_line']}: unknown ground-truth plate_type")
        if row["plate_type"] in TARGET_TYPES and not row["plate_num"]:
            raise ValueError(f"line {row['_line']}: target plate_num must not be empty")
        vehicle = boolean(row["is_vehicle"], f"line {row['_line']} is_vehicle") if row["is_vehicle"] else False
        if not placeholder and not row["is_vehicle"]:
            raise ValueError(f"line {row['_line']}: is_vehicle is required for a labeled object")
        if placeholder and (common["bbox"] is not None or common["quad"] is not None):
            raise ValueError(f"line {row['_line']}: empty-image row must not contain geometry")
        record = Record(**common, is_vehicle=vehicle)
        key = (record.image, record.plate_num, record.plate_type, vehicle, record.bbox, record.quad)
        if key in seen:
            raise ValueError(f"line {record.line}: duplicate ground-truth row")
        seen.add(key)
        result.append(record)
        per_image[record.image].append(record)
    for name, records in per_image.items():
        if any(r.placeholder for r in records) and len(records) > 1:
            raise ValueError(f"{name}: empty-image row conflicts with another ground-truth row")
    return result


def load_predictions(path: str | Path) -> list[Record]:
    result = []
    for row in read_rows(path, ("image", "plate_num", "plate_type", "confidence")):
        common = _common(row)
        if row["plate_type"] not in PLATE_TYPES:
            raise ValueError(f"line {row['_line']}: unknown prediction plate_type")
        # Empty target OCR output is allowed: it is an error, not a discarded detection.
        result.append(Record(
            **common, confidence=finite(row["confidence"], f"line {row['_line']} confidence", True),
            raw_ocr=row.get("raw_ocr") or None,
            postprocessed_ocr=row.get("postprocessed_ocr") or None,
            crop_bbox=parse_box(row.get("crop_bbox", ""), f"line {row['_line']} crop_bbox"),
            det_conf=finite(row["det_conf"], f"line {row['_line']} det_conf", True) if row.get("det_conf") else None,
        ))
    return result


def load_image_list(path: str | Path) -> list[str]:
    """Read a UTF-8 file with one relative image path per line (no CSV header)."""
    names = [safe_image_name(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(names) != len(set(names)):
        raise ValueError("image-list contains duplicate image names")
    return names


def intersection(a: Box, b: Box) -> float:
    width = max(0.0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    height = max(0.0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    return width * height


def iou(a: Box, b: Box) -> float:
    overlap = intersection(a, b)
    area_a, area_b = a[2] * a[3], b[2] * b[3]
    scale = max(area_a, area_b)
    # Scaling avoids overflow when two individually finite areas are added.
    return (overlap / scale) / (area_a / scale + area_b / scale - overlap / scale)


def spatial_matching(gt: list[Record], predictions: list[Record], threshold: float) -> dict[int, int]:
    """Maximum-cardinality bipartite matching; text/type never participate.

    Candidate order is descending IoU then input index. Augmenting paths ensure
    maximum cardinality, unlike a greedy highest-IoU selection. Total IoU is not
    the optimization objective; stable input order resolves equivalent matches.
    """
    edges: dict[int, list[int]] = {}
    for g, actual in enumerate(gt):
        candidates = [(iou(actual.bbox, pred.bbox), p) for p, pred in enumerate(predictions)]
        edges[g] = [p for overlap, p in sorted(candidates, key=lambda item: (-item[0], item[1])) if overlap >= threshold]
    owners: dict[int, int] = {}

    def augment(g: int, seen: set[int]) -> bool:
        for p in edges[g]:
            if p in seen:
                continue
            seen.add(p)
            if p not in owners or augment(owners[p], seen):
                owners[p] = g
                return True
        return False

    for g in sorted(edges, key=lambda index: (len(edges[index]), index)):
        augment(g, set())
    return {g: p for p, g in owners.items()}


def levenshtein(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for index, ca in enumerate(a, 1):
        current = [index]
        for j, cb in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def ratio(numerator: int | float, denominator: int | float) -> float | None:
    return numerator / denominator if denominator else None


def _target_metrics(outcomes: list[dict], spatial: bool) -> dict:
    matches = [outcome for outcome in outcomes if outcome["matched"]]
    accepted = [outcome for outcome in matches if outcome["accepted_target"]]
    char_total = sum(outcome["gt_characters"] for outcome in matches)
    edits = sum(outcome["edit_distance"] for outcome in matches) if spatial else None
    cer = ratio(edits, char_total) if spatial else None
    return {
        "target_gt_count": len(outcomes),
        "matched_target_count": len(matches) if spatial else None,
        "accepted_localized_target_count": len(accepted) if spatial else None,
        "e2e_exact_count": sum(outcome["e2e_exact"] for outcome in outcomes),
        "e2e_exact_accuracy": ratio(sum(outcome["e2e_exact"] for outcome in outcomes), len(outcomes)),
        "target_localization_recall": ratio(len(matches), len(outcomes)) if spatial else None,
        "detection_recall": ratio(len(accepted), len(outcomes)) if spatial else None,
        "ocr_full_plate_accuracy": ratio(sum(outcome["text_exact"] for outcome in matches), len(matches)) if spatial else None,
        "type_accuracy": ratio(sum(outcome["type_exact"] for outcome in matches), len(matches)) if spatial else None,
        "ocr_reference_characters": char_total if spatial else None,
        "ocr_edit_distance": edits,
        "ocr_cer": cer,
        "ocr_character_accuracy": max(0.0, 1.0 - cer) if cer is not None else None,
    }


def evaluate(
    gt: list[Record], predictions: list[Record], image_list: Iterable[str] | None = None,
    iou_threshold: float = 0.5, confidence_threshold: float = 0.0,
) -> dict:
    """Evaluate exhaustive image annotations. Unknown predictions are rejected.

    An image-list adds explicitly empty images, not unknown/unlabeled images.
    All rows must provide bbox for spatial mode; partial geometry falls back to
    the exact text/type multiset diagnostic for the entire run.
    """
    finite(iou_threshold, "iou_threshold", True)
    finite(confidence_threshold, "confidence_threshold", True)
    if iou_threshold == 0:
        raise ValueError("iou_threshold must be > 0 to exclude non-overlapping boxes")
    known = {record.image for record in gt}
    if image_list is not None:
        known.update(safe_image_name(name) for name in image_list)
    unknown = sorted({record.image for record in predictions} - known)
    if unknown:
        raise ValueError(f"Predictions refer to unknown images: {', '.join(unknown[:10])}")
    active = [record for record in predictions if record.confidence is None or record.confidence >= confidence_threshold]
    objects = [record for record in gt if not record.placeholder]
    spatial = all(record.bbox is not None for record in objects + active)
    # With no objects at all, negatives are observable but localization/OCR are not.
    spatial = spatial and bool(objects or active)
    mode = "spatial_iou" if spatial else "text_matching_diagnostic"
    gt_by_image: dict[str, list[Record]] = defaultdict(list)
    pred_by_image: dict[str, list[Record]] = defaultdict(list)
    for record in objects:
        gt_by_image[record.image].append(record)
    for record in active:
        pred_by_image[record.image].append(record)
    outcomes: list[dict] = []
    errors: list[dict] = []
    image_outcomes: list[dict] = []
    other_count = sum(record.plate_type == "other" for record in objects)
    non_vehicle_count = sum(not record.is_vehicle for record in objects)
    other_accepted = non_vehicle_accepted = 0
    other_accepted_by_type = Counter()
    accepted_correct_localizations = 0
    target_pred_count = sum(record.plate_type in TARGET_TYPES for record in active)

    def error(category: str, name: str, actual: Record | None, pred: Record | None, reason: str, **extra) -> None:
        errors.append({"category": category, "image": name, "reason": reason,
                       "gt": asdict(actual) if actual else None, "prediction": asdict(pred) if pred else None,
                       "image_ground_truth": [asdict(record) for record in gt_by_image[name]],
                       "image_predictions": [asdict(record) for record in pred_by_image[name]],
                       "image_context_note": "All same-image rows for review; these lists do not assert a pairing.", **extra})

    for name in sorted(known):
        actuals, preds = gt_by_image[name], pred_by_image[name]
        if spatial:
            matching = spatial_matching(actuals, preds, iou_threshold)
        else:
            available: dict[tuple[str, str], list[int]] = defaultdict(list)
            for p, pred in enumerate(preds):
                if pred.plate_type in TARGET_TYPES:
                    available[pred.plate_num, pred.plate_type].append(p)
            matching = {}
            for g, actual in enumerate(actuals):
                bucket = available[actual.plate_num, actual.plate_type]
                if actual.target and bucket:
                    matching[g] = bucket.pop(0)
        used = set(matching.values())
        image_targets = sum(actual.target for actual in actuals)
        image_accepted = sum(pred.plate_type in TARGET_TYPES for pred in preds)
        image_outcomes.append({"image": name, "target_gt_count": image_targets,
                               "accepted_prediction_count": image_accepted,
                               "negative_image": image_targets == 0,
                               "negative_image_false_positive": image_targets == 0 and image_accepted > 0})
        for g, actual in enumerate(actuals):
            pred = preds[matching[g]] if g in matching else None
            accepted = pred is not None and pred.plate_type in TARGET_TYPES
            if not actual.target:
                if spatial and accepted:
                    other_accepted += actual.plate_type == "other"
                    if actual.plate_type == "other":
                        other_accepted_by_type[pred.plate_type] += 1
                    non_vehicle_accepted += not actual.is_vehicle
                    error("false_positive", name, actual, pred, "Target prediction matched a non-target object", iou=iou(actual.bbox, pred.bbox))
                continue
            text_exact = pred is not None and actual.plate_num == pred.plate_num
            type_exact = pred is not None and actual.plate_type == pred.plate_type
            outcomes.append({"image": name, "gt_line": actual.line, "prediction_line": pred.line if pred else None,
                             "plate_type": actual.plate_type, "conditions": list(actual.conditions),
                             "prediction_type": pred.plate_type if pred else None,
                             "matched": pred is not None, "accepted_target": accepted,
                             "text_exact": text_exact, "type_exact": type_exact,
                             "e2e_exact": text_exact and type_exact,
                             "gt_characters": len(actual.plate_num),
                             "edit_distance": levenshtein(actual.plate_num, pred.plate_num) if pred and spatial else None,
                             "iou": iou(actual.bbox, pred.bbox) if pred and spatial else None})
            if spatial:
                accepted_correct_localizations += accepted
                if pred is None:
                    error("detection", name, actual, None, "No spatially matched prediction")
                else:
                    if not text_exact:
                        error("ocr", name, actual, pred, "Literal text mismatch on a spatially matched target", edit_distance=levenshtein(actual.plate_num, pred.plate_num), iou=iou(actual.bbox, pred.bbox))
                    if not type_exact:
                        error("wrong_type", name, actual, pred, "Wrong type on a spatially matched target", iou=iou(actual.bbox, pred.bbox))
                    if pred.crop_bbox is not None:
                        coverage = intersection(actual.bbox, pred.crop_bbox) / (actual.bbox[2] * actual.bbox[3])
                        if coverage < 0.95:
                            error("bad_crop", name, actual, pred, "Recorded crop covers <95% of GT bbox; diagnostic only, not proof of OCR cause", crop_gt_coverage=coverage)
            elif pred is None:
                # Do not invent detection/OCR/type causes from text-only correspondence.
                error("text_unmatched", name, actual, None, "No exact text/type multiset match; cause is unknown without boxes")
        for p, pred in enumerate(preds):
            if p not in used and pred.plate_type in TARGET_TYPES:
                error("false_positive", name, None, pred, "Unmatched accepted prediction" if spatial else "Unmatched output in exact text/type diagnostic; localization FP is unknown")

    metrics = _target_metrics(outcomes, spatial)
    metrics.update({
        "accepted_prediction_count": target_pred_count,
        "detection_precision": ratio(accepted_correct_localizations, target_pred_count) if spatial else None,
        "detection_f1": ratio(2 * accepted_correct_localizations, target_pred_count + len(outcomes)) if spatial else None,
        "e2e_exact_precision": ratio(metrics["e2e_exact_count"], target_pred_count),
        "negative_image_count": sum(row["negative_image"] for row in image_outcomes),
        "negative_image_fp_count": sum(row["negative_image_false_positive"] for row in image_outcomes),
        "other_gt_count": other_count,
        "other_to_target_count": other_accepted if spatial else None,
        "other_to_target_acceptance_rate": ratio(other_accepted, other_count) if spatial else None,
        "non_vehicle_gt_count": non_vehicle_count,
        "non_vehicle_to_target_count": non_vehicle_accepted if spatial else None,
        "non_vehicle_to_target_acceptance_rate": ratio(non_vehicle_accepted, non_vehicle_count) if spatial else None,
    })
    metrics["negative_image_fpr"] = ratio(metrics["negative_image_fp_count"], metrics["negative_image_count"])
    per_type = {ptype: _target_metrics([row for row in outcomes if row["plate_type"] == ptype], spatial) for ptype in TARGET_TYPES}
    for ptype, subset in per_type.items():
        predicted = sum(record.plate_type == ptype for record in active)
        typed_tp = sum(row["matched"] and row["type_exact"] for row in outcomes if row["plate_type"] == ptype)
        subset.update({
            "accepted_prediction_count": predicted,
            "typed_detection_precision": ratio(typed_tp, predicted) if spatial else None,
            "typed_detection_recall": ratio(typed_tp, subset["target_gt_count"]) if spatial else None,
            "typed_detection_f1": ratio(2 * typed_tp, predicted + subset["target_gt_count"]) if spatial else None,
        })
    # Outputting other is optional: score false target acceptance, never demand an other prediction.
    per_type["other"] = {
        "gt_count": other_count,
        "to_target_count": other_accepted if spatial else None,
        "to_target_acceptance_rate": ratio(other_accepted, other_count) if spatial else None,
        "rejection_rate": ratio(other_count - other_accepted, other_count) if spatial else None,
        "to_type_count": {ptype: other_accepted_by_type[ptype] if spatial else None for ptype in TARGET_TYPES},
    }
    per_condition = {tag: _target_metrics([row for row in outcomes if tag in row["conditions"]], spatial) for tag in CONDITIONS}
    known_image_tags: dict[str, set[str]] = defaultdict(set)
    for record in gt:
        known_image_tags[record.image].update(record.conditions)
    for tag, subset in per_condition.items():
        # Image-level FP slices require an explicit tag on an image annotation.
        negatives = [row for row in image_outcomes if row["negative_image"] and tag in known_image_tags[row["image"]]]
        subset["tagged_negative_image_count"] = len(negatives)
        subset["tagged_negative_image_fp_count"] = sum(row["negative_image_false_positive"] for row in negatives)
        subset["tagged_negative_image_fpr"] = ratio(subset["tagged_negative_image_fp_count"], len(negatives))
    missing = [tag for tag in CONDITIONS if per_condition[tag]["target_gt_count"] == 0]
    return {
        "schema_version": VERSION, "metric_family": "internal_literal_not_official",
        "matching_mode": mode,
        "matching_description": "Maximum cardinality IoU-only matching, deterministic descending-IoU candidates and row-order ties" if spatial else "Exact (image, plate_num, plate_type) multiset diagnostic; no localization or OCR attribution",
        "config": {"iou_threshold": iou_threshold, "confidence_threshold": confidence_threshold,
                   "unreadable_policy": "# is a literal character", "bbox_format": "absolute pixels [x,y,width,height]",
                   "crop_gt_coverage_warning_threshold": 0.95},
        "counts": {"images": len(known), "gt_rows": len(gt), "gt_objects": len(objects),
                   "prediction_rows": len(predictions), "evaluated_prediction_rows": len(active),
                   "filtered_prediction_rows": len(predictions) - len(active),
                   "optional_other_predictions": sum(record.plate_type == "other" for record in active),
                   "gt_objects_missing_bbox": sum(record.bbox is None for record in objects),
                   "predictions_missing_bbox": sum(record.bbox is None for record in active)},
        "metrics": metrics, "per_type": per_type, "per_condition": per_condition,
        "coverage": {"missing_target_types": [ptype for ptype in TARGET_TYPES if not per_type[ptype]["target_gt_count"]],
                     "missing_target_condition_strata": missing,
                     "target_gt_without_condition_tags": sum(not row["conditions"] for row in outcomes)},
        "warnings": (["Text-only diagnostic: a missing exact match cannot distinguish detection, OCR, type, or crop errors."] if not spatial else []) +
                    (["Some requested condition strata have no annotated target samples."] if missing else []),
        "targets": outcomes, "images": image_outcomes, "errors": errors,
        "error_counts": dict(Counter(error["category"] for error in errors)),
    }


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
