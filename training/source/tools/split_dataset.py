"""Create leakage-checked train/val/test metadata with a fixed seed."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dataset.splitting import make_split
from dataset.schema import FORMAT_PROFILES
from tools.validate_local import protected_manifests


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, default=ROOT / 'dataset')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--ratios', type=float, nargs=3, default=[.7, .15, .15])
    parser.add_argument('--exclude-manifest', type=Path, action='append', default=[])
    parser.add_argument('--format-profile', choices=FORMAT_PROFILES, default='legacy_v2')
    args = parser.parse_args(argv)
    try:
        result = make_split(args.dataset, seed=args.seed, ratios=args.ratios,
                            excluded=protected_manifests(args.exclude_manifest), format_profile=args.format_profile)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Split failed: {error}\n')
    print(json.dumps({'seed': result['seed'], 'format_profile': result['format_profile'], 'splits': result['splits'], 'warnings': result['warnings']}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
