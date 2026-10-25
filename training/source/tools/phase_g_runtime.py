"""Warm uncached full-pipeline timing, actual predictions and runtime parity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,time
import cv2,numpy as np,torch
from tools.phase_g_common import ROOT,REPORT,IMAGES,read,write
from tools.phase_e import load_image,summary
from tools.phase_g_experiments import evaluate_rows
from src.image_anpr_f import DetectorF
from src.image_anpr_g import OCRG,ImagePipelineG
from src.multiline_g import ConfigG
from dataset.schema import sha256

def main():
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','cuda'],default='cuda');p.add_argument('--rounds',type=int,default=2);args=p.parse_args()
    print('Loading actual G',args.device,'pipeline',flush=True);torch.set_num_threads(4);cv2.setNumThreads(1)
    best=read(REPORT/'provisional_best_g.json');spec=best['result']['config'];f=read(ROOT/'configs/phase_f/best_f.json')
    detector=DetectorF(f['detector'],args.device,640,spec['threshold'],.5);ocr=OCRG(best['checkpoint'],args.device)
    pipeline=ImagePipelineG(detector,ocr,ConfigG(**spec['split']),multi=spec['multi'],postprocess=spec['postprocess'],expansion=spec['expansion'])
    frames={p.name:load_image(p) for p in sorted(IMAGES.glob('*.jpg'))}
    for n in list(frames)[:5]:pipeline.process(frames[n],image_name=n)
    for count in range(1,6):ocr.recognize_candidates([np.zeros((32,160),dtype=np.float32)]*count)
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats()
    times=[];stages=[];traces=[];predictions=[]
    for repeat in range(args.rounds):
        current=[]
        for name,frame in frames.items():
            ocr.synchronize();start=time.perf_counter();r=pipeline.process(frame,image_name=name);ocr.synchronize();elapsed=(time.perf_counter()-start)*1000
            times.append(elapsed);stages.append({'total':elapsed,'detector':r['detector_ms'],'candidate_count':sum(d.get('candidate_count',1) for d in r['detections'])});current.append(r)
        if repeat==0:traces=current;predictions=[p for r in current for p in r['predictions']]
        else:assert [r['predictions'] for r in current]==[r['predictions'] for r in traces],'Nondeterministic repeated inference'
        print(args.device,'round',repeat+1,'mean',round(np.mean(times[-100:]),2),flush=True)
    folder=REPORT/'runtime'/args.device;folder.mkdir(parents=True,exist_ok=True)
    evaluation=evaluate_rows(predictions,folder)
    result={'device':args.device,'torch':torch.__version__,'times':summary(times),'stages':stages,'rounds':args.rounds,
        'scope':'Warm uncached detection+OBB warp+label-free split candidates+batched frozen-architecture OCR+CTC score+postprocess; all100 predecoded real scenes. Includes CPU preprocessing/greedy/NMS/GPU synchronization; excludes files, startup, model/DLL load and warmup.',
        'detector_cached':False,'VRAM_peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if args.device=='cuda' else None,
        'VRAM_peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2 if args.device=='cuda' else None,
        'GPU_name':torch.cuda.get_device_name(0) if args.device=='cuda' else None,'GPU_capability':list(torch.cuda.get_device_capability(0)) if args.device=='cuda' else None,
        'checkpoint_sha256':sha256(Path(best['checkpoint'])),'evaluation':evaluation}
    write(folder/'trace.json',{'records':traces});write(folder/'benchmark.json',result)
    print('Completed',args.device,result['times'],flush=True)

if __name__=='__main__':main()
