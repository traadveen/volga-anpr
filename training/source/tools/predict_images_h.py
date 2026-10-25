"""GT-free FINAL-H inference with explicit frozen recognition profile."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import json
from src.inference_io import file_sha256


def main():
    root=Path(__file__).resolve().parents[1]
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    args=parser.parse_args()
    config=root/'configs/phase_h/final_h.json'
    freeze=json.loads((root/'configs/phase_h/final_h_freeze.json').read_text('utf-8'))
    for path,sha in freeze['hashes'].items():
        if file_sha256(Path(path))!=sha:
            raise ValueError('FINAL-H frozen file changed: '+path)
    profile=json.loads(config.read_text('utf-8'))['recognition_profile']
    from tools.predict_images_g import main as predict
    predict(['--input',str(args.input),'--output',str(args.output),'--config',str(config),
             '--device',args.device,'--recognition-profile',profile])


if __name__=='__main__':
    main()
