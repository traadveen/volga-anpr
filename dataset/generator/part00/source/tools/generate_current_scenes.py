"""Generate a new procedural scene collection with current Russian formats."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset.generator.scenes_current import generate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--count',type=int,default=32);p.add_argument('--seed',type=int,default=20261015)
    p.add_argument('--profile',choices=['clean','mixed'],default='mixed');args=p.parse_args()
    m=generate(args.output,args.count,args.seed,args.profile)
    print(json.dumps({k:m[k] for k in ('count','plates_by_type','negative_scenes','rejected_attempts','final_split_created')}))

if __name__=='__main__':main()
