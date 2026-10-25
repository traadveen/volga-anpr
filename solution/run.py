"""FORMAT-CONSISTENCY-20261025 candidate adapter: independent images, offline, per-file decode errors."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import os
import sys
import tempfile
import time
from importlib.metadata import version

# Packaged as run.py at the export root, or invoked from tools in the project.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent if HERE.name == 'tools' else HERE
sys.path.insert(0, str(ROOT))
from src.inference_io import IMAGE_EXTENSIONS, file_sha256, load_runtime_config, offline_network
from tools.validate_output import COLUMNS, validate_rows


def inputs(source):
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError('Input must be JPG/JPEG/PNG/BMP')
        return [source]
    if not source.is_dir():
        raise ValueError('Input does not exist: ' + str(source))
    return sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def runtime_config(root):
    from tools.verify_candidate import verify_runtime
    return verify_runtime(root)


from tools.output_transaction import publish


def recognize(paths, config, device, show=False):
    traces, predictions, errors = [], [], []
    initialization_ms = 0.
    with offline_network():
        import cv2
        import numpy as np
        import torch
        from src.image_anpr_f import DetectorF, OrdinaryFallbackDetector
        from src.image_anpr_g import OCRG, ImagePipelineG
        from src.multiline_g import ConfigG
        from src.plate_formats import apply_format_consistent_profile
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; select --device cpu')
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        pipeline = None
        preview = None
        if show:
            from tools.image_preview import ImagePreview
            preview = ImagePreview()
        try:
            for index, path in enumerate(paths, start=1):
                started = time.perf_counter()
                try:
                    encoded = np.fromfile(path, dtype=np.uint8)
                    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
                    if image is None:
                        raise ValueError('Unreadable image')
                except (OSError, ValueError, cv2.error) as error:
                    record = dict(image=path.name, status='input_error', error=str(error))
                    errors.append(record)
                    traces.append(record)
                    continue
                decode_ms = (time.perf_counter() - started) * 1000
                if pipeline is None:
                    loading = time.perf_counter()
                    detector = DetectorF(config['detector'], device, config['imgsz'], config['confidence'], config['iou'])
                    if config.get('ordinary_detector'):
                        reference = DetectorF(config['ordinary_detector'], device, config['imgsz'], config['confidence'], config['iou'])
                        detector = OrdinaryFallbackDetector(detector, reference)
                    ocr = OCRG(config['ocr'], device)
                    pipeline = ImagePipelineG(detector, ocr, ConfigG(**config['split']), multi=config['multi'],
                                              postprocess=config['postprocess'], expansion=config['expansion'])
                    initialization_ms = (time.perf_counter() - loading) * 1000
                inference_started = time.perf_counter()
                raw_result = pipeline.process(image, image_name=path.name)
                raw_result['ordinary_reference_detection_indices'] = list(getattr(detector, 'last_added_indices', []))
                result = apply_format_consistent_profile(raw_result)
                result['submission_pipeline_ms'] = (time.perf_counter() - inference_started) * 1000
                result['input_sha256'] = hashlib.sha256(encoded.tobytes()).hexdigest()
                result['read_decode_ms'] = decode_ms
                traces.append(result)
                predictions.extend(result['predictions'])
                if preview is not None:
                    preview.show(image, result, path, index, len(paths))
        finally:
            if preview is not None:
                preview.close()
        performance = dict(model_initialization_ms=initialization_ms,
            gpu_name=torch.cuda.get_device_name(0) if device == 'cuda' else None,
            peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20 if device == 'cuda' else None,
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device == 'cuda' else None)
    return predictions, traces, errors, performance


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--show', action='store_true', help='Показывать каждое фото с рамками и результатом распознавания; клавиша — следующее фото')
    args = parser.parse_args(argv)
    try:
        if args.output.suffix.lower() != '.csv':
            raise ValueError('--output must be CSV')
        paths = inputs(args.input)
        config_path = runtime_config(ROOT)
        config_hash = file_sha256(config_path)
        config = load_runtime_config(config_path, ROOT)
        previous = {k: os.environ.get(k) for k in ('YOLO_OFFLINE', 'YOLO_AUTOINSTALL')}
        os.environ.update(YOLO_OFFLINE='true', YOLO_AUTOINSTALL='false')
        try:
            values = recognize(paths, config, args.device, show=args.show) if paths else ([], [], [], dict(model_initialization_ms=0.))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        predictions, traces, errors, performance = values
        if paths and len(errors) == len(paths):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix('.failed-run.json').write_text(json.dumps(dict(errors=errors), ensure_ascii=False, indent=2), encoding='utf-8')
            raise ValueError('All input images are unreadable; previous CSV is preserved')
        if file_sha256(config_path) != config_hash:
            raise ValueError('Configuration changed during inference')
        model_keys = ('detector', 'ocr') + (('ordinary_detector',) if 'ordinary_detector' in config else ())
        for key in model_keys:
            if file_sha256(Path(config[key])) != config[key + '_sha256']:
                raise ValueError('Checkpoint changed during inference')
        manifest = dict(name=config['name'], confidence_semantics='uncalibrated mean character probability; not plate correctness probability', device=args.device, recognition_profile='format_consistency_v1',
            input_images=len(paths), processed_images=len(paths)-len(errors), skipped_images=len(errors),
            prediction_rows=len(predictions), model_paths={k: config[k] for k in model_keys},
            model_sha256={k: config[k + '_sha256'] for k in model_keys}, config_sha256=config_hash,
            runtime_versions={k: version(k) for k in ('torch', 'torchvision', 'ultralytics', 'numpy', 'opencv-python')},
            initialization_ms=performance['model_initialization_ms'], performance=performance, ground_truth_input=False,
            network='Python TCP/DNS/UDP blocked during imports, loading and recognition')
        publish(args.output, predictions, traces, errors, manifest)
        print(f'Processed {len(paths)-len(errors)} images; skipped {len(errors)} unreadable; predictions {len(predictions)}')
        return 0
    except (OSError, ValueError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    raise SystemExit(main())
