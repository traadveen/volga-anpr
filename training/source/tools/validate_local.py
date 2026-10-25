"""Additional local dataset validator; does not replace organizer validation."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.validation import validate_dataset
from dataset.schema import FORMAT_PROFILES


def protected_manifests(explicit):
    manifests = list(explicit)
    default = ROOT.parent / 'external_data/organizer_debug/source_manifest.json'
    if default.exists() and default not in manifests:
        manifests.append(default)
    return manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--report', type=Path, default=ROOT / 'reports/dataset_validation.json')
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    parser.add_argument('--no-splits', action='store_true')
    parser.add_argument('--format-profile', choices=FORMAT_PROFILES, default='legacy_v2')
    args = parser.parse_args(argv)
    try:
        root = args.dataset.resolve()
        output = args.report.resolve()
        # Keep validation read-only even if a caller points its report at data.
        if output.is_relative_to(root):
            raise ValueError('Report must be outside the dataset directory')
        exclusions = protected_manifests(args.exclude_manifest)
        if output in {Path(path).resolve() for path in exclusions}:
            raise ValueError('Report would overwrite a protected evaluation manifest')
        result = validate_dataset(args.dataset, check_splits=not args.no_splits,
                                  excluded=exclusions, format_profile=args.format_profile)
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    except (OSError, ValueError) as error:
        parser.exit(2, f'Validation failed: {error}\n')
    print(json.dumps({k: result[k] for k in ['valid', 'ready_for_e2e_evaluation', 'counts', 'coverage', 'errors', 'warnings']}, indent=2, ensure_ascii=False))
    return 0 if result['valid'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
