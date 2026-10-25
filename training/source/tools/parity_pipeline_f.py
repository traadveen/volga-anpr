"""Compare live independent-image CPU/GPU outputs, including bbox and OCR text."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from src.geometry_f import bbox_iou
from tools.phase_e import ROOT,read,write,official
from tools.phase_f_setup import REPORT

def main():
    p=argparse.ArgumentParser();p.add_argument('--cpu',required=True,type=Path);p.add_argument('--gpu',required=True,type=Path);a=p.parse_args()
    cpu=read(a.cpu.with_suffix('.trace.json'))['records'];gpu=read(a.gpu.with_suffix('.trace.json'))['records']
    assert [r['image'] for r in cpu]==[r['image'] for r in gpu]
    pairs=[];mismatches=[]
    for c,g in zip(cpu,gpu):
        used=set()
        for d in c['detections']:
            overlap,i,e=max(((bbox_iou(d['bbox'],e['bbox']),i,e) for i,e in enumerate(g['detections']) if i not in used),default=(0,None,None))
            if e is not None:used.add(i)
            same=e is not None and overlap>.99 and d.get('plate_type')==e.get('plate_type') and d.get('final_ocr')==e.get('final_ocr') and d.get('status')==e.get('status')
            normalized_same=e is not None and d.get('normalized_sha256')==e.get('normalized_sha256')
            corner_delta=float(np.max(np.abs(np.array(d['observed_quad'])-np.array(e['observed_quad'])))) if e and d.get('observed_quad') and e.get('observed_quad') else None
            row={'image':c['image'],'bbox_IoU':overlap,'same_type_text_status':same,'same_normalized_pixels':normalized_same,'max_corner_delta_px':corner_delta}
            pairs.append(row)
            if not same:mismatches.append({'summary':row,'cpu':d,'gpu':e})
        if len(c['detections'])!=len(g['detections']):mismatches.append({'image':c['image'],'counts':[len(c['detections']),len(g['detections'])]})
    strings=lambda traces:sorted((r['image'],p['plate_num'],p['plate_type']) for r in traces for p in r['predictions'])
    exact=strings(cpu)==strings(gpu);cs,_=official(a.cpu,REPORT/'parity/cpu_score');gs,_=official(a.gpu,REPORT/'parity/gpu_score')
    result={'status':'PASS' if exact and not mismatches else 'DIFFERENCES','images':len(cpu),'pairs':pairs,'mismatches':mismatches,
        'text_and_type_exact':exact,'CPU_score':cs,'GPU_score':gs,
        'CPU_benchmark':read(a.cpu.with_suffix('.manifest.json')),'GPU_benchmark':read(a.gpu.with_suffix('.manifest.json')),
        'tolerances':'BBox IoU>0.99, exact OCR text/type/status; normalized pixels reported separately, not required byte equality between hardware',
        'scope':'Live CPU and CUDA forward passes on same independent images, frozen checkpoint/preprocessing; no temporal voting'}
    write(REPORT/'parity/cpu_gpu.json',result);print(result['status'],'exact text/type',exact,'mismatches',len(mismatches),flush=True)

if __name__=='__main__':main()
