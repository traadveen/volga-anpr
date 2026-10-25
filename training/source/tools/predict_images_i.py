"""GT-free FINAL-I inference with the frozen FINAL-H OCR and recognition profile."""
from pathlib import Path
import argparse,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inference_io import file_sha256


def main():
    root=Path(__file__).resolve().parents[1]
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu');args=p.parse_args()
    config=root/'configs/phase_i/final_i.json'
    freeze=json.loads((root/'configs/phase_i/final_i_freeze.json').read_text(encoding='utf-8'))
    if freeze['status']!='FINAL-I-FROZEN':raise ValueError('FINAL-I is not frozen')
    for path,sha in freeze['hashes'].items():
        if file_sha256(Path(path))!=sha:raise ValueError('FINAL-I frozen input changed: '+path)
    profile=json.loads(config.read_text(encoding='utf-8'))['recognition_profile']
    from tools.predict_images_g import main as predict
    predict(['--input',str(args.input),'--output',str(args.output),'--config',str(config),
        '--device',args.device,'--recognition-profile',profile])


if __name__=='__main__':main()
