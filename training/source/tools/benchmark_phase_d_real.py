"""Sequential identical CPU real trials, after all training; scoring unchanged."""
import json
import argparse
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluation import load_ground_truth,load_predictions,evaluate
from dataset.schema import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--resume',action='store_true');args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    public=Path('D:/volga/external_data/organizer_debug/Полуфинал - public')
    output=root/'reports/phase_d/real_benchmark';output.mkdir(exist_ok=True)
    gt=load_ground_truth(public/'debug_labels.csv');result={'device':'cpu','scope':'Sequential uncaptured full100-image real evaluation after training, original sources/YOLO/preprocessing/postprocess fixed. First image cold, model load separate. Actual i5-6500, not reference hardware.',
        'trials':{},'public_model_selection_allowed':False}
    definitions=(('old','models/ocr_crnn_nomeroff_clean.pt',False),
        ('smoke','models/phase_d/smoke/best.pt',False),
        ('finetune','models/phase_d/exp_d_01_finetune/best.pt',False),
        ('random','models/phase_d/exp_d_02_random/best.pt',False),
        ('finetune_no_deskew','models/phase_d/exp_d_01_finetune/best.pt',True),
        ('old_no_deskew','models/ocr_crnn_nomeroff_clean.pt',True))
    for name,weights,no_deskew in definitions:
        predictions=output/f'{name}.csv'
        if predictions.exists() and not args.resume:raise ValueError('Use --resume for previously completed trials')
        command=[sys.executable,'-X','utf8',str(root/'tools'/('predict_phase_d.py' if no_deskew else 'baseline_predict.py')),
            '--input',str(public/'debug set'),'--output',str(predictions),'--ocr',str(root/weights),'--device','cpu']
        if no_deskew:command.append('--no-deskew')
        if not predictions.exists():
            run=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8')
            (output/f'{name}.run.log').write_text(run.stdout+run.stderr,encoding='utf-8')
            if run.returncode:raise RuntimeError(f'{name} prediction failed: {run.stderr[-1000:]}')
        else:
            saved=json.loads(predictions.with_suffix('.manifest.json').read_text())
            if saved['status']!='completed' or saved['prediction_sha256']!=sha256(predictions) or not any(m['sha256']==sha256(root/weights) for m in saved['models'].values()):
                raise ValueError('Completed trial provenance changed')
        score=output/f'{name}_official'
        command=[sys.executable,'-X','utf8',str(root/'tools/score_official.py'),'--scorer',str(public/'evaluate.py'),
            '--ground-truth',str(public/'debug_labels.csv'),'--predictions',str(predictions),
            '--images',str(public/'debug set'),'--output',str(score)]
        run=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8')
        if run.returncode:raise RuntimeError(run.stderr)
        manifest=json.loads(predictions.with_suffix('.manifest.json').read_text())
        literal=evaluate(gt,load_predictions(predictions))
        (output/f'{name}_literal.json').write_text(json.dumps(literal,indent=2),encoding='utf-8')
        official=json.loads(score.with_suffix('.json').read_text())
        result['trials'][name]={'checkpoint':weights,'checkpoint_sha256':sha256(root/weights),
            'deskew':'none' if no_deskew else 'old','official':official,'literal_metrics':literal['metrics'],
            'latency':manifest['image_timing_ms'],'model_load_ms':manifest['model_load_ms'],
            'predictions':str(predictions),'predictions_sha256':sha256(predictions),'source_hashes':manifest['source_hashes'],
            'manifest':str(predictions.with_suffix('.manifest.json'))}
        (output/'comparison.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'trial':name,'correct':official['full_match_total'],'score':official['primary_score'],
            'timing':manifest['image_timing_ms']}),flush=True)


if __name__=='__main__':main()
