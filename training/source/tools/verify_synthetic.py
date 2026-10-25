"""Independent raster/layout/geometry/hash checks for PHASE B generated samples."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dataset.generator.verify import verify_collection


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--report',required=True,type=Path)
    args=parser.parse_args(argv)
    try:
        if args.report.resolve().is_relative_to(args.dataset.resolve()):raise ValueError('Report must be outside dataset')
        report=verify_collection(args.dataset)
        args.report.parent.mkdir(parents=True,exist_ok=True)
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    except (OSError,ValueError) as error:parser.exit(2,f'Verification failed: {error}\n')
    print(json.dumps(report,ensure_ascii=False,indent=2));return 0 if report['valid'] else 1


if __name__=='__main__':raise SystemExit(main())
