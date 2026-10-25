"""Diagnostic proposals only; threshold selection happens on explicit val later."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataclasses import asdict
import torch,cv2
from tools.phase_g_common import freeze,REPORT,IMAGES,write
from tools.phase_e import load_image
from src.image_anpr_f import DetectorF

def main():
    c=freeze()['baseline'];torch.set_num_threads(4);cv2.setNumThreads(1)
    det=DetectorF(c['detector'],'cuda',640,.01,.5);rows=[]
    for i,p in enumerate(sorted(IMAGES.glob('*.jpg'))):
        rows.append({'image':p.name,'detections':[asdict(d) for d in det.detect(load_image(p))]})
        if i%20==0:print(i+1,'/100',flush=True)
    write(REPORT/'audit/lowconf_proposals.json',{'threshold':.01,'diagnostic_only':True,'records':rows})
    print('Saved low-confidence spatial proposals',flush=True)

if __name__=='__main__':main()
