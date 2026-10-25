"""V2 CRNN training. Old trainer and checkpoints remain available."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from training.crnn_v2 import TrainConfig,train


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--resume',type=Path)
    parser.add_argument('--stop-after-epoch',type=int)
    args=parser.parse_args()
    train(TrainConfig.load(args.config),args.resume,args.stop_after_epoch)


if __name__=='__main__':main()
