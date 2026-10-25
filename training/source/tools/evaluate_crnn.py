"""Frozen split OCR-only comparisons; never selects weights from test/public data."""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from torch.utils.data import DataLoader
from ocr.crnn import load_crnn_checkpoint
from training.crnn_v2 import TrainConfig,OCRDatasetV2,collate,evaluate,seed_everything
from dataset.schema import sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True)
    parser.add_argument('--split',choices=('val','test'),required=True)
    parser.add_argument('--weights',nargs='+',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args();config=TrainConfig.load(args.config)
    seed_everything(config.seed);torch.set_num_threads(config.threads)
    device=torch.device(config.device)
    data=OCRDatasetV2(config,f'splits/{args.split}.csv')
    loader=DataLoader(data,batch_size=config.batch_size,shuffle=False,num_workers=0,collate_fn=collate)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    result={'split':args.split,'dataset_fingerprint':data.fingerprint,'dataset_meta_sha256':sha256(data.root/'meta.csv'),
            'device':str(device),'type_counts':dict(Counter(r.plate_type for r in data.rows)),
            'purpose':'Frozen OCR-only evaluation; oracle generator rectification and Phase C stitch',
            'results':{},'selection_allowed':args.split=='val'}
    for index,path in enumerate(args.weights):
        model,converter,_,_=load_crnn_checkpoint(path,device=device)
        if device.type=='cuda':torch.cuda.synchronize()
        start=time.perf_counter();metrics,cases=evaluate(model,loader,converter,device,config.ctc_on_cpu)
        if device.type=='cuda':torch.cuda.synchronize()
        seconds=time.perf_counter()-start
        case_path=output.with_name(output.stem+f'.model{index}.predictions.json')
        case_path.write_text(json.dumps(cases,indent=2),encoding='utf-8')
        timing=[]
        # Fixed first100 inputs, 5 warmups,3 repeats: cached tensors -> device -> model -> greedy.
        for i in range(min(100,len(data))):
            tensor=data[i][0].unsqueeze(0)
            for repeat in range(3 if i>=5 else 4):
                if device.type=='cuda':torch.cuda.synchronize()
                t0=time.perf_counter()
                with torch.inference_mode():converter.decode_logits(model(tensor.to(device)))
                if device.type=='cuda':torch.cuda.synchronize()
                if repeat or i>=5:timing.append((time.perf_counter()-t0)*1000)
        result['results'][path]={'checkpoint_sha256':sha256(Path(path)),'metrics':metrics,
            'seconds_batch_evaluation':seconds,'images_per_sec_batch_evaluation':len(data)/seconds,
            'latency_ms':{'mean':float(np.mean(timing)),'p95':float(np.percentile(timing,95))},
            'latency_scope':'prepared normalized tensor transfer + single-image forward + greedy; no IO/geometry/detection',
            'predictions':str(case_path),'by_difficulty':{tier:{'count':len(group),'full_plate_accuracy':
                sum(r['ground_truth']==r['prediction'] for r in group)/len(group)}
                for tier in ('clean','medium','hard')
                if (group:=[r for r in cases if data.samples[r['image']].get('difficulty')==tier])}}
        output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        print(json.dumps({'weights':path,'metrics':metrics,'seconds':seconds}),flush=True)
        del model
        if device.type=='cuda':torch.cuda.empty_cache()


if __name__=='__main__':main()
