"""Evaluate semicolon ANPR CSVs without importing the inference stack.

Run either ``python tools/evaluate.py --help`` or ``python -m tools.evaluate --help``.
Only diagnostic image export needs Pillow; the scoring module is standard-library only.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import subprocess
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tools.evaluation import (
    ERROR_CATEGORIES, VERSION, evaluate, file_sha256, load_ground_truth,
    load_image_list, load_predictions,
)


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def image_path(root: Path, name: str) -> Path:
    path = (root / name).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Image resolves outside images-root: {name}")
    if not path.is_file():
        raise ValueError(f"Annotated/listed image does not exist: {path}")
    return path


def input_manifest(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha256(path), "bytes": path.stat().st_size}


def canonical_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode("utf-8")).hexdigest()


def export_crop(frame, box, path: Path) -> dict | None:
    if box is None:
        return None
    x, y, width, height = box
    clipped = (max(0, min(frame.width, math.floor(x))), max(0, min(frame.height, math.floor(y))),
               max(0, min(frame.width, math.ceil(x + width))), max(0, min(frame.height, math.ceil(y + height))))
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return {"path": None, "requested_xywh": list(box), "warning": "Box does not intersect image pixels"}
    frame.crop(clipped).save(path)
    return {"path": str(path.resolve()), "clipped_xyxy": list(clipped), "requested_xywh": list(box)}


def provenance(args: argparse.Namespace, names: list[str], command: list[str], report: dict) -> dict:
    root = Path(__file__).resolve().parents[1]
    files = {"ground_truth": input_manifest(args.ground_truth), "predictions": input_manifest(args.predictions)}
    if args.image_list is not None:
        files["image_list"] = input_manifest(args.image_list)
    # Include an inference manifest when present, but do not trust an unrelated one.
    inference_path = args.predictions.with_suffix(".manifest.json")
    if inference_path.exists():
        manifest = json.loads(inference_path.read_text(encoding="utf-8-sig"))
        if manifest.get("prediction_sha256") != files["predictions"]["sha256"]:
            raise ValueError("Adjacent prediction manifest does not match prediction SHA-256")
        files["inference_manifest"] = input_manifest(inference_path)
    sources = {str(path.relative_to(root)).replace("\\", "/"): file_sha256(path)
               for path in [root / "tools/evaluation.py", Path(__file__).resolve()]}
    images = [{"image": name, "sha256": file_sha256(image_path(args.images_root, name)),
               "bytes": image_path(args.images_root, name).stat().st_size} for name in names]
    config = {**report["config"], "images_root": str(args.images_root),
              "image_population": "GT image names union explicit image-list; image-list asserts exhaustive annotations",
              "crop_preview": "Raw stored pixels; no EXIF transpose; bbox clipped to image boundaries",
              "errors_root": str(args.errors_root.resolve())}
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    identity = {"input_hashes": {key: value["sha256"] for key, value in files.items()},
                "config_sha256": canonical_hash(config), "source_hashes": sources,
                "images_sha256": canonical_hash(images)}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(), "command": command,
        "python": sys.version, "platform": platform.platform(), "git_head": commit,
        "evaluator_version": VERSION, "inputs": files, "configuration": config,
        "images": images, **identity, "run_id": canonical_hash(identity)[:16],
    }


def export_errors(report: dict, images_root: Path, errors_root: Path, run_id: str) -> dict:
    from PIL import Image, __version__ as pillow_version

    descriptions = {
        "detection": "Unmatched target GT at the IoU threshold. Requires GT and prediction boxes.",
        "ocr": "Literal OCR mismatch on a spatial match. This records the error, not its root cause.",
        "false_positive": "Unmatched accepted output or accepted non-target object. In text mode this is an output mismatch, not a proven localization false positive.",
        "wrong_type": "Wrong predicted type on a spatial match. Requires GT and prediction boxes.",
        "bad_crop": "Recorded crop covers less than 95% of the matched GT box. This is a warning, not proof that the crop caused an OCR error.",
    }
    errors_root.mkdir(parents=True, exist_ok=True)
    for category in ERROR_CATEGORIES:
        destination = errors_root / category
        destination.mkdir(exist_ok=True)
        (destination / "README.md").write_text(
            f"# {category}\n\n{descriptions[category]}\n\n"
            f"Current run: `{run_id}`. The authoritative case list is `../index.json`. "
            "Folders from earlier runs are retained, and are not part of the current report.\n\n"
            "Missing diagnostics mean unavailable evidence, never zero errors. Without GT boxes, "
            "unmatched GT causes remain unknown and are listed in `../unclassified/`. "
            "JSON records include both annotations and all available raw/postprocessed OCR, confidence, and crop geometry.\n",
            encoding="utf-8")
    preview_dir = errors_root / "previews" / run_id
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_cache: dict[str, str | None] = {}
    context_cache: dict[str, str] = {}
    warnings: list[str] = []
    index = []
    for number, error in enumerate(report["errors"], 1):
        category = error["category"] if error["category"] in ERROR_CATEGORIES else "unclassified"
        folder = errors_root / category / run_id
        folder.mkdir(parents=True, exist_ok=True)
        stem = f"case_{number:05d}"
        original_path = image_path(images_root, error["image"])
        exported = {**error, "case_id": stem, "image_path": str(original_path),
                    "run_id": run_id, "crops": {}, "image_preview": None, "media_warnings": []}
        try:
            with Image.open(original_path) as opened:
                # Annotation coordinates describe stored pixels, matching cv2.imread.
                frame = opened.convert("RGB")
            if error["image"] not in preview_cache:
                preview = frame.copy()
                preview.thumbnail((1280, 1280))
                preview_path = preview_dir / (hashlib.sha256(error["image"].encode("utf-8")).hexdigest()[:20] + ".jpg")
                preview.save(preview_path, quality=90)
                preview_cache[error["image"]] = str(preview_path.resolve())
            exported["image_preview"] = preview_cache[error["image"]]
            if error["image"] not in context_cache:
                context_dir = errors_root / "context" / run_id
                context_dir.mkdir(parents=True, exist_ok=True)
                image_key = hashlib.sha256(error["image"].encode("utf-8")).hexdigest()[:20]
                context = {"image": error["image"], "image_path": str(original_path),
                           "ground_truth": error["image_ground_truth"], "predictions": [],
                           "note": "All same-image predictions are candidates for human review. No pairing is inferred."}
                for pred_number, candidate in enumerate(error["image_predictions"], 1):
                    context["predictions"].append({
                        **candidate,
                        "bbox_crop": export_crop(frame, candidate["bbox"], context_dir / f"{image_key}_pred_{pred_number}_bbox.png"),
                        "ocr_crop": export_crop(frame, candidate["crop_bbox"], context_dir / f"{image_key}_pred_{pred_number}_ocr.png"),
                    })
                context_path = context_dir / f"{image_key}.json"
                write_json(context_path, context)
                context_cache[error["image"]] = str(context_path.resolve())
            exported["image_context_file"] = context_cache[error["image"]]
            actual, prediction = error["gt"] or {}, error["prediction"] or {}
            for label, box in [("gt_bbox", actual.get("bbox")), ("prediction_bbox", prediction.get("bbox")),
                               ("recorded_ocr_crop", prediction.get("crop_bbox"))]:
                crop_path = folder / f"{stem}_{label}.png"
                crop = export_crop(frame, box, crop_path)
                exported["crops"][label] = crop
                if crop and crop.get("warning"):
                    exported["media_warnings"].append(f"{label}: {crop['warning']}")
        except (OSError, ValueError, Image.DecompressionBombError) as exc:
            exported["media_warnings"].append(f"Cannot export image diagnostics: {type(exc).__name__}: {exc}")
        if exported["media_warnings"]:
            warnings.extend(f"{error['image']}: {warning}" for warning in exported["media_warnings"])
        case_path = folder / f"{stem}.json"
        write_json(case_path, exported)
        index.append({"case_id": stem, "category": error["category"], "image": error["image"],
                      "json": str(case_path.resolve())})
    write_json(errors_root / "index.json", {"run_id": run_id, "cases": index, "case_count": len(index)})
    (errors_root / "README.md").write_text(
        "# Error evidence\n\nUse `index.json` to inspect the current run. A single GT/prediction pair can have several diagnoses, "
        "so category counts are not an additive decomposition of end-to-end errors.\n\n"
        "`unclassified/` holds unmatched target GT in text-only mode. Its root cause cannot be inferred. "
        "Images and crops are diagnostic exports, not additions to a training set. No text normalization or inferred labels are applied.\n",
        encoding="utf-8")
    return {"root": str(errors_root.resolve()), "index": str((errors_root / "index.json").resolve()),
            "pillow_version": pillow_version, "case_count": len(index), "warnings": sorted(set(warnings))}


def write_metrics_csv(path: Path, report: dict) -> None:
    """Long format, with explicit null (not an accidental zero or empty cell)."""
    def rows(scope: str, group: str, metrics: dict):
        for name, value in metrics.items():
            if isinstance(value, dict):
                for subtype, number in value.items():
                    yield scope, group, f"{name}.{subtype}", "null" if number is None else number
            else:
                yield scope, group, name, "null" if value is None else value

    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["scope", "group", "metric", "value"])
        writer.writerows(rows("overall", "all", report["metrics"]))
        writer.writerows(rows("counts", "all", report["counts"]))
        for section in ["per_type", "per_condition"]:
            for group, metrics in report[section].items():
                writer.writerows(rows(section, group, metrics))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--images-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("reports/baseline"), help="Output stem (or .json path)")
    parser.add_argument("--image-list", type=Path, help="Exhaustive UTF-8 relative image list; unannotated entries assert zero objects")
    parser.add_argument("--errors-root", type=Path, help="Default: <output parent>/errors")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    args = parser.parse_args(argv)
    args.images_root = args.images_root.resolve()
    prefix = args.output.with_suffix("") if args.output.suffix.lower() in (".json", ".csv") else args.output
    args.errors_root = args.errors_root or prefix.parent / "errors"
    try:
        if not args.images_root.is_dir():
            raise ValueError(f"images-root is not a directory: {args.images_root}")
        inputs = {path.resolve() for path in [args.ground_truth, args.predictions, args.image_list] if path is not None}
        adjacent_manifest = args.predictions.with_suffix(".manifest.json")
        if adjacent_manifest.exists():
            inputs.add(adjacent_manifest.resolve())
        if any(Path(str(prefix) + suffix).resolve() in inputs for suffix in (".json", ".csv")):
            raise ValueError("Output path would overwrite an input file")
        gt, predictions = load_ground_truth(args.ground_truth), load_predictions(args.predictions)
        names = load_image_list(args.image_list) if args.image_list is not None else None
        report = evaluate(gt, predictions, names, args.iou_threshold, args.confidence_threshold)
        command = [sys.executable, *sys.argv] if argv is None else [sys.executable, str(Path(__file__).resolve()), *argv]
        report["provenance"] = provenance(args, [row["image"] for row in report["images"]], command, report)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        report["error_artifacts"] = export_errors(report, args.images_root, args.errors_root, report["provenance"]["run_id"])
        report["warnings"].extend(report["error_artifacts"]["warnings"])
        json_path, csv_path = Path(str(prefix) + ".json"), Path(str(prefix) + ".csv")
        write_metrics_csv(csv_path, report)
        report["artifacts"] = {"metrics_csv": input_manifest(csv_path), "report_json": str(json_path.resolve())}
        write_json(json_path, report)
    except (ValueError, OSError, ImportError) as exc:
        parser.exit(2, f"Evaluation failed: {exc}\n")
    print(json.dumps({"matching_mode": report["matching_mode"], "counts": report["counts"],
                      "metrics": report["metrics"], "report": str(json_path.resolve())}, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
