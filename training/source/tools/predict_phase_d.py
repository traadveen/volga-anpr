"""Legacy real evaluation with one explicit runtime deskew ablation."""
import argparse
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.baseline_predict import deny_network


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--ocr',required=True)
    parser.add_argument('--device',default='cpu')
    parser.add_argument('--no-deskew',action='store_true')
    args=parser.parse_args()
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).resolve().parents[1]/'.baseline_yolo')
    with deny_network():
        import run as legacy
        from tools import baseline_predict
        if args.no_deskew:
            legacy._deskew_estimate=lambda gray:0.
        status=baseline_predict.main(['--input',args.input,'--output',args.output,
            '--ocr',args.ocr,'--device',args.device])
    import json
    report=Path(args.output).with_suffix('.ablation.json')
    report.write_text(json.dumps({'checkpoint':args.ocr,'deskew':'none' if args.no_deskew else 'old',
        'device':args.device,'scope':'Original legacy pipeline; only runtime deskew switch and specified OCR weights differ',
        'model_selection_allowed':False},indent=2),encoding='utf-8')
    return status


if __name__=='__main__':raise SystemExit(main())
