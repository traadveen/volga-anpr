"""FINAL-I submission adapter: independent images, offline, per-file decode errors."""
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
    root = Path(root)
    if (root / 'bundle_manifest.json').exists():
        from tools.final_i_runtime import verify_runtime
        return verify_runtime(root)
    from tools.build_final_i_bundle import verify_freeze
    verify_freeze(root)
    return root / 'configs/phase_i/final_i.json'


def publish(output, predictions, traces, errors, manifest):
    validation = validate_rows([{k: row[k] for k in COLUMNS} for row in predictions],
                               {r['image'] for r in traces})
    if not validation['valid']:
        raise ValueError('Model output failed CSV validation: ' + '; '.join(validation['errors'][:3]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='final-output-', dir=output.parent) as temporary:
        stage = Path(temporary)
        csv_path = stage / 'result.csv'
        with csv_path.open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter=';', lineterminator='\n')
            writer.writeheader()
            writer.writerows({k: row[k] for k in COLUMNS} for row in predictions)
        payloads = {'.trace.json': dict(records=traces, ground_truth_input=False),
                    '.errors.json': dict(errors=errors, skipped_images=len(errors)),
                    '.manifest.json': dict(manifest, output_validation=validation)}
        for suffix, data in payloads.items():
            path = stage / ('result' + suffix)
            path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            os.replace(path, output.with_suffix(suffix))
        os.replace(csv_path, output)


def recognize(paths, config, device):
    traces, predictions, errors = [], [], []
    initialization_ms = 0.
    with offline_network():
        import cv2
        import numpy as np
        import torch
        from src.image_anpr_f import DetectorF
        from src.image_anpr_g import OCRG, ImagePipelineG
        from src.multiline_g import ConfigG
        from src.plate_formats import apply_russian_profile
        torch.set_num_threads(4)
        cv2.setNumThreads(1)
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA is unavailable; select --device cpu')
        if device == 'cuda':
            torch.cuda.reset_peak_memory_stats()
        pipeline = None
        for path in paths:
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
                ocr = OCRG(config['ocr'], device)
                pipeline = ImagePipelineG(detector, ocr, ConfigG(**config['split']), multi=config['multi'],
                                          postprocess=config['postprocess'], expansion=config['expansion'])
                initialization_ms = (time.perf_counter() - loading) * 1000
            inference_started = time.perf_counter()
            result = apply_russian_profile(pipeline.process(image, image_name=path.name))
            result['submission_pipeline_ms'] = (time.perf_counter() - inference_started) * 1000
            result['input_sha256'] = hashlib.sha256(encoded.tobytes()).hexdigest()
            result['read_decode_ms'] = decode_ms
            traces.append(result)
            predictions.extend(result['predictions'])
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
            values = recognize(paths, config, args.device) if paths else ([], [], [], dict(model_initialization_ms=0.))
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
        predictions, traces, errors, performance = values
        if paths and len(errors) == len(paths):
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix('.errors.json').write_text(json.dumps(dict(errors=errors), ensure_ascii=False, indent=2), encoding='utf-8')
            raise ValueError('All input images are unreadable; previous CSV is preserved')
        if file_sha256(config_path) != config_hash:
            raise ValueError('Configuration changed during inference')
        for key in ('detector', 'ocr'):
            if file_sha256(Path(config[key])) != config[key + '_sha256']:
                raise ValueError('Checkpoint changed during inference')
        manifest = dict(name='FINAL-I', device=args.device, recognition_profile='russian_v1',
            input_images=len(paths), processed_images=len(paths)-len(errors), skipped_images=len(errors),
            prediction_rows=len(predictions), model_paths={k: config[k] for k in ('detector', 'ocr')},
            model_sha256={k: config[k + '_sha256'] for k in ('detector', 'ocr')}, config_sha256=config_hash,
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
