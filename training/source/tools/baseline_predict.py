"""Instrument the frozen legacy pipeline on independent images; no model changes.

This is a diagnostic adapter, NOT reproduction of the historical video score.
Each image passes through the original process_frames with a fresh tracker.
Legacy output lacks type/confidence: type1 is an explicit constant baseline and
confidence is the existing OCR sequence mean (uncalibrated), never track votes.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import random
import socket
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_weight(path):
    path = Path(path).resolve()
    if not path.is_file():
        raise ValueError(f'Local weight missing: {path}; no automatic download allowed')
    with path.open('rb') as stream:
        if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
            raise ValueError(f'Git LFS pointer, not a weight: {path}; fetch the LFS object first')
    return path


@contextlib.contextmanager
def deny_network():
    """Block Python TCP connections during imports/model loading/inference."""
    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create = socket.create_connection

    def blocked(*args, **kwargs):
        raise RuntimeError('Network access is disabled for baseline inference')

    socket.socket.connect = blocked
    socket.socket.connect_ex = blocked
    socket.create_connection = blocked
    try:
        yield
    finally:
        socket.socket.connect = original_connect
        socket.socket.connect_ex = original_connect_ex
        socket.create_connection = original_create


def xywh(box):
    x1, y1, x2, y2 = box[:4]
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def summary(values):
    if not values:
        return None
    ordered = sorted(values)

    def percentile(q):
        x = (len(ordered) - 1) * q
        lo = int(x)
        return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (x - lo)

    return dict(count=len(values), mean=statistics.mean(values), median=statistics.median(values),
                p90=percentile(.9), p95=percentile(.95), min=min(values), max=max(values))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--detector', type=Path, default=ROOT / 'models/yolo11_plate.pt')
    parser.add_argument('--ocr', type=Path, default=ROOT / 'models/ocr_crnn_nomeroff_clean.pt')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--conf', type=float, default=.5, help='README CRNN example baseline')
    parser.add_argument('--iou', type=float, default=.5)
    parser.add_argument('--char-conf', type=float, default=.5)
    parser.add_argument('--letter-conf', type=float, default=.7)
    parser.add_argument('--digit-conf', type=float, default=.5)
    args = parser.parse_args(argv)
    for key in ['conf', 'iou', 'char_conf', 'letter_conf', 'digit_conf']:
        if not 0 <= getattr(args, key) <= 1:
            parser.error(f'{key} must be in [0,1]')
    if args.threads < 1:
        parser.error('--threads must be positive')
    try:
        detector_path, ocr_path = check_weight(args.detector), check_weight(args.ocr)
    except ValueError as error:
        parser.error(str(error))
    if not args.input.is_dir():
        parser.error('--input must be an image directory')
    images = sorted(p for p in args.input.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    # Filename field of the organizer format is a basename; ambiguity is an error.
    if len({p.name for p in images}) != len(images):
        parser.error('Duplicate image basenames in nested folders')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    trace_path = args.output.with_suffix('.trace.jsonl')
    manifest_path = args.output.with_suffix('.manifest.json')
    trace_images = []
    wall_start = time.perf_counter()
    os.environ['YOLO_OFFLINE'] = 'true'
    os.environ['YOLO_AUTOINSTALL'] = 'false'
    os.environ['YOLO_CONFIG_DIR'] = str(ROOT / '.baseline_yolo')
    with deny_network():
        import cv2
        import numpy as np
        import torch
        import run as legacy

        if args.device == 'cuda' and not torch.cuda.is_available():
            parser.error('CUDA explicitly requested but unavailable; refusing silent CPU fallback')
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.set_num_threads(args.threads)
        cv2.setNumThreads(1)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        def sync():
            if args.device == 'cuda':
                torch.cuda.synchronize()

        class RecordingRecognizer(legacy.CRNNRecognizer):
            def readtext(self, *a, **kw):
                result = super().readtext(*a, **kw)
                self.last_raw = result
                return result

        class RecordingDetector:
            def __init__(self, model):
                self.model = model
                self.last = []

            def predict(self, *a, **kw):
                sync()
                t0 = time.perf_counter()
                result = self.model.predict(*a, **kw)
                sync()
                self.elapsed = (time.perf_counter() - t0) * 1000
                self.last = []
                if result and result[0].boxes is not None:
                    for box in result[0].boxes:
                        bounds = box.xyxy[0].cpu().tolist()
                        width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
                        self.last.append(dict(bbox=xywh(bounds), confidence=float(box.conf[0].cpu()),
                                              class_id=int(box.cls[0].cpu()),
                                              legacy_geometry_pass=1.6 <= max(1., width) / max(1., height) <= 6.5
                                              and max(1., width) >= 50 and max(1., height) >= 14))
                return result

        load_start = time.perf_counter()
        yolo = legacy.YOLO(str(detector_path))
        detector = RecordingDetector(yolo)
        reader = RecordingRecognizer(str(ocr_path), device=args.device)
        sync()
        load_ms = (time.perf_counter() - load_start) * 1000
        row_count = 0
        headers = ['image', 'plate_num', 'plate_type', 'confidence', 'bbox', 'raw_ocr',
                   'postprocessed_ocr', 'crop_bbox', 'det_conf', 'legacy_track_score']
        with args.output.open('w', encoding='utf-8', newline='') as csvfile, trace_path.open('w', encoding='utf-8') as trace:
            writer = csv.DictWriter(csvfile, fieldnames=headers, delimiter=';')
            writer.writeheader()
            for index, path in enumerate(images):
                sync()
                image_start = time.perf_counter()
                info = dict(image=path.name, relative_path=path.relative_to(args.input).as_posix(),
                            sha256=sha256(path), index=index, status='ok')
                observations = []
                try:
                    # imdecode supports Windows Unicode paths. Pixel BGR content is unchanged.
                    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
                    if frame is None:
                        raise ValueError('Image decode failed')
                    info['shape'] = list(frame.shape)
                    info['read_ms'] = (time.perf_counter() - image_start) * 1000

                    def observed_ocr(recognizer, source, bbox):
                        recognizer.last_raw = []
                        sync()
                        start = time.perf_counter()
                        text, conf, meta = legacy.ocr_plate(
                            recognizer, source, bbox, char_conf_thr=args.char_conf,
                            char_conf_thr_letters=args.letter_conf, char_conf_thr_digits=args.digit_conf,
                            return_crop=True)
                        sync()
                        raw = recognizer.last_raw[0] if recognizer.last_raw else None
                        obs = dict(bbox=xywh(bbox), det_conf=float(bbox[4]), raw_ocr=raw[1] if raw else '',
                                   ocr_confidence=float(raw[2]) if raw else 0., postprocessed_ocr=text,
                                   ocr_ms=(time.perf_counter() - start) * 1000,
                                   char_probs=meta.get('char_probs'),
                                   crop_bbox=xywh(meta['crop_bbox']) if 'crop_bbox' in meta else None)
                        observations.append(obs)
                        return text, conf, meta

                    tracker = legacy.SimpleTracker(iou_match=.3, max_misses=12, lock_conf_thresh=legacy.TRACK_LOCK_CONF)
                    legacy_args = argparse.Namespace(conf=args.conf, iou=args.iou, show=False, ocr_every_n=1)
                    with contextlib.redirect_stdout(io.StringIO()):
                        rows = legacy.process_frames(
                            [(0, 0, frame, {'source_path': str(path)})], tracker, detector, reader,
                            observed_ocr, legacy_args, args.device, None, str(path), legacy.EmissionLimiter(200))
                    info['detector_ms'] = detector.elapsed
                    info['detector_boxes_before_geometry_filter'] = detector.last
                    info['observations'] = observations
                    info['emissions'] = []
                    unused = list(tracker.tracks.values())
                    for row in rows:
                        chosen = next(t for t in unused if (t.best_plate or '########') == row['plate_num'])
                        unused.remove(chosen)
                        obs = observations[chosen.id - 1]
                        exported = dict(image=path.name, plate_num=row['plate_num'], plate_type='type1',
                                        confidence=obs['ocr_confidence'], bbox=json.dumps(obs['bbox']),
                                        raw_ocr=obs['raw_ocr'], postprocessed_ocr=row['plate_num'],
                                        crop_bbox=json.dumps(obs['crop_bbox']), det_conf=obs['det_conf'],
                                        legacy_track_score=chosen.best_conf)
                        writer.writerow(exported)
                        info['emissions'].append(exported)
                        row_count += 1
                except Exception as error:
                    info.update(status='error', error=f'{type(error).__name__}: {error}', observations=observations)
                sync()
                info['image_ms_including_read_csv'] = (time.perf_counter() - image_start) * 1000
                trace.write(json.dumps(info, ensure_ascii=False) + '\n')
                trace.flush()
                trace_images.append(info)
                if (index + 1) % 10 == 0 or index + 1 == len(images):
                    print(f'{index + 1}/{len(images)} images; {row_count} predictions', flush=True)

        try:
            commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
        versions = {}
        for package in ['torch', 'torchvision', 'ultralytics', 'opencv-python', 'easyocr', 'numpy', 'pandas']:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        manifest = dict(
            status='completed' if all(x['status'] == 'ok' for x in trace_images) else 'completed_with_errors',
            experiment='legacy_isolated_images_readme_crnn', historical_video_baseline_reproduced=False,
            type_policy='constant type1; legacy has no classifier',
            confidence_policy='existing CRNN sequence mean, uncalibrated; track score stored separately',
            config={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            command=sys.argv, baseline_commit=commit, python=sys.version, platform=platform.platform(),
            packages=versions, cuda=torch.version.cuda, device=args.device,
            network='Python socket TCP connections blocked during import/loading/inference',
            models={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p): dict(sha256=sha256(p), bytes=p.stat().st_size)
                    for p in [detector_path, ocr_path]},
            detector_names=yolo.names, detector_yaml=yolo.model.yaml,
            ocr=dict(alphabet=reader.converter.alphabet, image_height=reader.image_height, image_width=reader.image_width,
                     parameters=sum(p.numel() for p in reader.model.parameters())),
            source_hashes={str(p.relative_to(ROOT)): sha256(p) for p in [ROOT/'run.py', ROOT/'ocr/crnn.py', ROOT/'ocr/utils.py', Path(__file__)]},
            images=len(images), predictions=row_count, failed_images=[x['image'] for x in trace_images if x['status'] != 'ok'],
            model_load_ms=load_ms, total_wall_ms=(time.perf_counter()-wall_start)*1000,
            image_timing_ms=summary([x['image_ms_including_read_csv'] for x in trace_images]),
            timing_protocol='CPU wall time incl read, image hashing and CSV row serialization; first image cold; model load separate; trace serialization excluded; not reference hardware benchmark',
            prediction_sha256=sha256(args.output), trace_sha256=sha256(trace_path))
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'Predictions: {args.output}; manifest: {manifest_path}', flush=True)
    return 1 if manifest['failed_images'] else 0


if __name__ == '__main__':
    raise SystemExit(main())
