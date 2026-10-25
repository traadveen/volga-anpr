"""Measure complete submission CLI wall time on provided non-test images."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.baseline_predict import summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--rounds', type=int, default=1)
    args = parser.parse_args(argv)
    if args.output.exists() or not 1 <= args.rounds <= 3:
        parser.error('Use a new report path and 1..3 rounds')
    entry = ROOT / 'run.py' if (ROOT / 'bundle_manifest.json').exists() else ROOT / 'tools/run_final_submission.py'
    directory = args.output.with_suffix('')
    if directory.exists():
        parser.error('Benchmark trace directory already exists')
    directory.mkdir(parents=True)
    rounds = []
    image_ms = []
    for index in range(args.rounds):
        output = directory / f'round{index:02d}.csv'
        command = [sys.executable, '-I', str(entry), '--input', str(args.input),
                   '--output', str(output), '--device', args.device]
        started = time.perf_counter()
        result = subprocess.run(command, capture_output=True, text=True, encoding='utf-8', errors='replace')
        elapsed = (time.perf_counter() - started) * 1000
        (directory / f'round{index:02d}.log').write_text(result.stdout + result.stderr, encoding='utf-8')
        if result.returncode != 0:
            raise ValueError(f'Benchmark CLI failed; see round{index:02d}.log')
        manifest = json.loads(output.with_suffix('.manifest.json').read_text(encoding='utf-8'))
        traces = json.loads(output.with_suffix('.trace.json').read_text(encoding='utf-8'))['records']
        if not traces or manifest['skipped_images'] or manifest['ground_truth_input']:
            raise ValueError('Benchmark requires nonempty valid images without GT input')
        if any(r.get('cached_detections') is not False for r in traces):
            raise ValueError('Benchmark requires fresh detections on every image')
        if len(traces) != manifest['processed_images']:
            raise ValueError('Benchmark image count mismatch')
        current = [r['read_decode_ms'] + r['submission_pipeline_ms'] for r in traces]
        image_ms.extend(current)
        rounds.append(dict(round=index, images=len(traces), entire_CLI_wall_ms=elapsed,
            entire_CLI_ms_per_image=elapsed/len(traces),
            model_initialization_ms=manifest['initialization_ms'], performance=manifest['performance'],
            prediction_rows=manifest['prediction_rows']))
    report = dict(device=args.device, python=sys.version, input=str(args.input.resolve()), rounds=rounds,
        entire_CLI_amortized_ms_per_image=summary([r['entire_CLI_ms_per_image'] for r in rounds]),
        individual_read_decode_pipeline_ms=summary(image_ms),
        scope='Entire subprocess: interpreter, imports, integrity checks, model load, image I/O, inference, output validation and CSV/JSON writes. Per-image distribution excludes shared startup/output writes.',
        reference_i5_7600_GTX1050Ti_verified=False, no_GT_input=True, detector_cache_used=False,
        performance_only_no_accuracy_claim=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=True, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
