"""Generate PHASE B only: versioned synthetic plate crops with pixel ground-truth checks."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dataset.generator.config import DEFAULT
from dataset.generator.generate import generate
from tools.validate_local import protected_manifests


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--count',type=int,required=True,help='Number of images, not identities')
    parser.add_argument('--plate-type',choices=['all','type1','type1a','type1b'],default='all')
    parser.add_argument('--output',type=Path,required=True,help='New collection directory')
    parser.add_argument('--config',type=Path,default=DEFAULT)
    parser.add_argument('--variants',type=int,default=2,help='Augmented views per unique plate; all stay in one split')
    parser.add_argument('--profile',choices=['mixed','clean'],default='mixed')
    parser.add_argument('--exclude-manifest',type=Path,action='append',default=[])
    args=parser.parse_args(argv)
    try:result=generate(args.output,args.count,args.seed,args.plate_type,args.config,args.variants,args.profile,protected_manifests(args.exclude_manifest))
    except (OSError,ValueError) as error:parser.exit(2,f'Generation failed: {error}\n')
    print(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False));return 0


if __name__=='__main__':raise SystemExit(main())
