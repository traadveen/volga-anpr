"""Physical real crop distribution and a bounded targeted synthetic recipe."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,cv2,numpy as np
from tools.phase_g_common import REPORT,CONFIG,read,write

def main():
    rows=[]
    for r in read(REPORT/'error_summary.json')['rows']:
        if not r['crop']:continue
        image=cv2.imdecode(np.fromfile(r['crop'],dtype=np.uint8),cv2.IMREAD_GRAYSCALE)
        q=np.array(r['OBB']);b=r['bbox'];h,w=image.shape;edge=q[1]-q[0]
        dif=np.abs(np.diff(image.astype(float),axis=1));boundary=dif[:,7::8].mean() if w>8 else 0
        interior=np.delete(dif,np.arange(7,dif.shape[1],8),axis=1).mean() if w>8 else 0
        rows.append({'image':r['image'],'plate_type':r['plate_type'],'GT':r['ground_truth'],
            'width':w,'height':h,'aspect':w/h,'angle_deg':float(np.degrees(np.arctan2(edge[1],edge[0]))),
            'brightness':float(image.mean()),'contrast':float(image.std()),'laplacian_variance':float(cv2.Laplacian(image,cv2.CV_64F).var()),
            'JPEG_block_proxy':float(boundary/max(interior,1e-6)),'input_good':r['good_input_verified']})
    stats={}
    for kind in ('ALL','type1','type1a','type1b'):
        group=[r for r in rows if kind=='ALL' or r['plate_type']==kind];stats[kind]={'count':len(group)}
        for field in ('width','height','aspect','angle_deg','brightness','contrast','laplacian_variance','JPEG_block_proxy'):
            stats[kind][field]={f'p{p}':float(np.percentile([r[field] for r in group],p)) for p in (10,25,50,75,90)} if group else None
    write(REPORT/'real_crop_distribution.json',{'rows':rows,'statistics':stats,'scope':'Observed source-resolution bbox crops, including failed inputs; detection-conditioned, no detector-miss crops. JPEG proxy is edge-grid ratio, not recovered JPEG quality.'})
    with (REPORT/'real_crop_distribution.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    recipe={'seed':20261004,'samples':360,'split':{'train':180,'val':90,'test':90},'plate_mask':'LDDDLLRR/RRR',
        'layout':'Canonical exact body/region character order, two rows only for type1a; all character ink preserved before degradation',
        'font_variants':['frozen_authored_condensed','Windows_Arial_condensed_to_same_cells'],
        'glyph_variations':{'thickness_pixels':[-1,0,1],'spacing_shift_px':[-2,0,2],'region_relative_width':[.96,1,1.04]},
        'source_resolution_width_by_type':{t:None if stats[t]['width'] is None else [max(48,round(stats[t]['width']['p10'])),max(80,round(stats[t]['width']['p75']))] for t in ('type1','type1a')},
        'degradation':{'JPEG_quality':[60,80,95],'gaussian_sigma':[0,.35,.65],'brightness_scale':[.85,1.,1.15]},
        'type1b':'No real distribution; use authored yellow one-line control, never claim real robustness',
        'ground_truth_checks':['independent nearest-glyph raster decoding before degradation','all character cells inside image','non-overlapping ordered cells','all ink retained','two-row ordering','source identity-disjoint from all D/F/public labels'],
        'test_policy':'Freeze new test before selection; terminal OCR-only evaluation only, never E2E or independent real quality. This recipe is prepared for targeted augmentation, not an implicit claim that it matches real font distribution.'}
    write(CONFIG/'targeted_synthetic.json',recipe)
    print(stats['type1a'],flush=True)

if __name__=='__main__':main()
