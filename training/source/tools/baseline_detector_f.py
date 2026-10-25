"""Pre-training old detector spatial baseline on reviewed complete real scenes."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
import torch
from src.geometry_f import bbox_from_quad
from src.image_anpr_f import DetectorF
from tools.evaluate_detector_f import detection_metrics,serialize_detections
from tools.phase_e import ROOT,IMAGES,read,write,load_image
from tools.phase_f_setup import REPORT
import csv


def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    quads=read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')['records'];group={}
    for r in quads:group.setdefault(r['image'],[]).append(r)
    with (IMAGES.parent/'debug_labels.csv').open(encoding='utf-8-sig') as f:
        counts=Counter(r['image'] for r in csv.DictReader(f,delimiter=';'))
    records=[]
    for name,rs in group.items():
        if len(rs)!=counts[name]:continue
        records.append({'image':name,'annotations':[{'bbox':bbox_from_quad(r['manual_quad']),
            'quad':r['manual_quad'],'plate_type':r['plate_type']} for r in rs]})
    detector=DetectorF(ROOT/'models/yolo11_plate.pt',imgsz=1024,confidence=.001)
    predictions={}
    for r in records:
        im=load_image(IMAGES/r['image']);predictions[r['image']]=serialize_detections(im,detector.detect(im))
    result={'imgsz':1024,'real_complete_scenes':len(records),'real_plate_targets':sum(len(r['annotations']) for r in records),
        'metrics':detection_metrics(records,predictions),'typed_geometry_router_metrics':detection_metrics(records,predictions,typed=True),
        'scope':'21 completely annotated convenience-selected public development scenes; visible previously detected plates. Not full115 detector recall or independent test.',
        'type1b':'N/A — no real type1b GT','completed_before_training':True,
        'protocol':'Confidence0.5 for P/R/F1; AP from confidence0.001; IoU0.5:0.05:0.95; all GT within each chosen frame included',
        'GT_frames':records,'predictions':predictions}
    write(REPORT/'detector_baseline.json',result)
    print({k:v for k,v in result.items() if k not in ('GT_frames','predictions','typed_geometry_router_metrics')},flush=True)


if __name__=='__main__':main()
