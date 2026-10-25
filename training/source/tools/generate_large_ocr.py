import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset.generator.large import generate_large

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--workers',type=int,default=3)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    gpu=json.loads(Path('reports/phase_d/cuda_validation.json').read_text(encoding='utf-8'))
    if gpu['status']!='PASS':raise RuntimeError('D5 CUDA gate not passed')
    result=generate_large(args.output,args.workers,{'type1':100,'type1a':100,'type1b':100} if args.pilot else None)
    print(json.dumps(result,indent=2))
