"""Scientific export of detector recall / paired FPA by actual target size."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
from tools.phase_e import ROOT,read,write
from tools.phase_f_setup import REPORT,DATA
from src.geometry_f import bbox_iou

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args()
    folder=REPORT/'experiments'/a.experiment;result=read(folder/'result.json')
    manifest=read(DATA/'manifest.json');cases=read(folder/'validation_e2e_cases.json')
    records={r['image']:r for r in manifest['records'] if r['split']=='val'}
    predictions=read(folder/'validation_predictions.json');groups={}
    for case in cases:
        r=records[case['image']];width=case['GT_width'];input_width=width*result['config']['imgsz']/max(r['width'],r['height'])
        target=next(t for t in r['annotations'] if t['plate_num']==case['GT'])
        box=target['bbox'];area=(box[2]-box[0])*(box[3]-box[1])/(r['width']*r['height'])
        detected=max((bbox_iou(d['bbox'],box) for d in predictions[r['image']] if d['confidence']>=.5),default=0)>=.5
        for protocol,value,bounds in [('source_width',width,(40,80,160)),('detector_input_width',input_width,(16,32,64))]:
            bin_name=('tiny','small','medium','large')[sum(value>=b for b in bounds)]
            groups.setdefault((case['source'],protocol,bin_name),[]).append(dict(case,detected=detected,input_width=input_width,normalized_bbox_area=area))
    output=[]
    for (source,protocol,bin_name),rr in groups.items():
        output.append({'source':source,'protocol':protocol,'size':bin_name,'targets':len(rr),
            'detector_recall':sum(r['detected'] for r in rr)/len(rr),'paired_FPA':sum(r['correct'] for r in rr)/len(rr),
            'mean_input_width':float(np.mean([r['input_width'] for r in rr])),
            'mean_normalized_bbox_area':float(np.mean([r['normalized_bbox_area'] for r in rr]))})
    write(REPORT/'size_analysis.json',{'experiment':a.experiment,'rows':output,
        'source_thresholds_px':[40,80,160],'detector_input_thresholds_px':[16,32,64],
        'matching':'Detector conf>=.5 and IoU>=.5; FPA requires IoU-matched exact text AND type',
        'scope':'Validation only; nine real GT plates are exposed convenience sample, 207 synthetic GT. No real generalization.'})
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    order=('tiny','small','medium','large')
    for ax,source in zip(axes,('synthetic_val','real_development_val')):
        rows={r['size']:r for r in output if r['source']==source and r['protocol']=='detector_input_width'}
        x=np.arange(4);rec=[rows[k]['detector_recall'] if k in rows else np.nan for k in order];fpa=[rows[k]['paired_FPA'] if k in rows else np.nan for k in order]
        ax.plot(x,rec,'o-',label='Detection recall @IoU .5');ax.plot(x,fpa,'s-',label='Paired full-plate accuracy')
        ax.set_xticks(x,[f'{k}\nn={rows[k]["targets"] if k in rows else 0}' for k in order]);ax.set_ylim(-.04,1.04)
        ax.set_title(source+' (development)');ax.grid(alpha=.2);ax.legend(fontsize=8)
        ax.set_xlabel('Width at detector input: <16 / <32 / <64 / >=64 px')
    fig.savefig(REPORT/'plate_size_recall_fpa.png',dpi=170);plt.close(fig)

if __name__=='__main__':main()
