"""Generate current-format authored yellow crops without touching frozen v2."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
from dataset.generator.type1b_v1 import generate

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);p.add_argument('--count',type=int,default=24);p.add_argument('--seed',type=int,default=20261015);a=p.parse_args()
    result=generate(a.output,a.count,a.seed);print('Generated',result['count'],'pixel-checked yellow control crops; no training or final split')

if __name__=='__main__':main()
