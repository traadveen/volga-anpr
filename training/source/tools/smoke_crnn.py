"""Bounded D3 smoke + real resume evidence; not a quality experiment."""
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from ocr.crnn import CRNNRecognizer
from training.crnn_v2 import TrainConfig,OCRDatasetV2,train
from tools.benchmark_multiline import timing


def main():
    config=TrainConfig.load('configs/crnn_smoke.yaml')
    report_dir=Path('reports/phase_d');report_dir.mkdir(parents=True,exist_ok=True)
    first=train(config,stop_after_epoch=2)
    last=Path(config.output_dir)/'last.pt'
    # Preserve an explicit before/after snapshot of resume evidence, not a
    # model-only restart. The immutable original weights are untouched.
    before=torch.load(last,weights_only=True,map_location='cpu')
    second=train(config,resume=last)
    after=torch.load(last,weights_only=True,map_location='cpu')
    dataset=OCRDatasetV2(config,config.validation_split,False)
    reader=CRNNRecognizer(str(Path(config.output_dir)/'best.pt'),device=second['device'])
    inputs=[dataset.input_for(i,False).crop for i in range(len(dataset))]
    for _ in range(5):reader.readtext(inputs[0])
    durations=[];predictions=[]
    for image in inputs:
        if second['device']=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();result=reader.readtext(image)
        if second['device']=='cuda':torch.cuda.synchronize()
        durations.append((time.perf_counter()-start)*1000);predictions.append(result[0][1])
    evidence=second['resume']
    evidence['checks']={'epoch':before['epoch']==2 and after['epoch']==3,
        'optimizer_step_restored_and_advanced':evidence['optimizer_step_after']>evidence['optimizer_step_before']>0,
        'scheduler_restored_and_advanced':evidence['scheduler_before']==before['scheduler_state_dict'] and evidence['scheduler_after']['last_epoch']>evidence['scheduler_before']['last_epoch'],
        'all_types_loaded':set(first['train_types'])=={'type1','type1a','type1b'},
        'best_created':(Path(config.output_dir)/'best.pt').exists(),'last_created':last.exists(),
        'legacy_inference_loads':len(predictions)==len(inputs),'finite_train_validation_loss':all(np.isfinite(r['train_loss']) and np.isfinite(r['validation']['loss']) for r in second['history'])}
    report={**second,'epochs':after['epoch'],'dataset_total_samples':600,
        'training_time_seconds':first['training_time_seconds']+second['training_time_seconds'],
        'first_segment':first,'resume_verified':all(evidence['checks'].values()),
        'inference_latency_ms':timing(durations),'latency_scope':'warm frozen best.pt CRNN normalization+forward+greedy on prepared validation crops; no IO/detector/rectification',
        'CER':second['validation']['cer'],'Full_Plate_Accuracy':second['validation']['full_plate_accuracy'],
        'purpose':'mechanical smoke only; not production checkpoint selection or quality benchmark',
        'GPU_used':second['device']=='cuda'}
    (report_dir/'smoke_training.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({'resume_verified':report['resume_verified'],'inference_latency_ms':report['inference_latency_ms']},indent=2))
    if not report['resume_verified']:raise RuntimeError('Smoke gate failed')


if __name__=='__main__':main()
