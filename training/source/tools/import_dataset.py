"""Import explicitly reviewed, already-unpacked HF data into a new local collection."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.importers import hf_detection, hf_ocr
from dataset.importers.common import license_review, publish
from tools.validate_local import protected_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=['hf-ocr', 'hf-detection'])
    parser.add_argument('--source-root', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path, help='New collection directory; existing datasets cannot be overwritten')
    parser.add_argument('--license-review', required=True, type=Path)
    parser.add_argument('--splits', nargs='+', choices=['train', 'val', 'test'], default=['train'])
    parser.add_argument('--plate-type', choices=['type1', 'type1a', 'type1b'])
    parser.add_argument('--class-map', type=Path, help='JSON class ID -> reviewed type or empty string')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    args = parser.parse_args(argv)
    try:
        if len(args.splits) != len(set(args.splits)):
            raise ValueError('Source splits must not be repeated')
        source = hf_ocr if args.kind == 'hf-ocr' else hf_detection
        review, evidence = license_review(args.license_review, source.SOURCE_ID)
        if source is hf_ocr:
            pairs = source.prepare(args.source_root, args.splits, review, args.plate_type)
        else:
            if not args.class_map:
                raise ValueError('Detection import requires --class-map')
            pairs = source.prepare(args.source_root, args.splits, review, json.loads(args.class_map.read_text(encoding='utf-8')))
        result = publish(args.output, pairs, review, evidence, excluded=protected_manifests(args.exclude_manifest), seed=args.seed)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Import failed: {error}\n')
    print(json.dumps({'output': str(args.output.resolve()), 'splits': result['splits']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
