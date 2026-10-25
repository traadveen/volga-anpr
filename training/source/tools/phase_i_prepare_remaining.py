"""Generate train/val geometry proposals; no GT or reviewer-state mutation."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import math
import hashlib
import json
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_i_review import Review,atomic_json,geometry
from src.geometry_f import order_obb,bbox_iou
from src.inference_io import file_sha256
from tools.phase_h_common import now


def edge_candidates(frame,row):
    """Fallback for yellow/white plates touching a similarly colored car body."""
    b=np.array(row['bbox'],float);bw,bh=b[2:]-b[:2];h,w=frame.shape[:2]
    lo=np.maximum([0,0],b[:2]-[max(6,bw*.15),max(6,bh*.35)]).astype(int)
    hi=np.minimum([w,h],b[2:]+[max(6,bw*.15),max(6,bh*.35)]).astype(int)
    gray=cv2.cvtColor(frame[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2GRAY)
    scale=2 if bh<70 else 1
    gray=cv2.resize(gray,None,fx=scale,fy=scale,interpolation=cv2.INTER_CUBIC)
    result=[]
    for enhanced in (gray,cv2.createCLAHE(clipLimit=2,tileGridSize=(4,4)).apply(gray)):
        for low,high in [(25,90),(50,130)]:
            edges=cv2.Canny(enhanced,low,high)
            for contour in cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)[0]:
                hull=cv2.convexHull(contour);area=cv2.contourArea(hull)/(scale*scale)
                if not .30*bw*bh<area<1.20*bw*bh:
                    continue
                for eps in (.015,.025,.035,.05):
                    poly=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True)
                    if len(poly)!=4:
                        continue
                    try:
                        q=order_obb(poly.reshape(4,2)).astype(float)/scale+lo
                        geometry(q.tolist(),row['image_size'])
                    except ValueError:
                        continue
                    envelope=np.r_[q.min(0),q.max(0)]
                    score=bbox_iou(envelope,b)-.08*abs(math.log(max(area/(bw*bh),.01)/.75))
                    if score>.45:
                        result.append({'quad':q.tolist(),'source':'plate_edge_contour','score':score,'confidence':None})
                    break
    return sorted(result,key=lambda item:item['score'],reverse=True)


def candidates(frame,row):
    b=np.array(row['bbox'],float)
    x,y,x2,y2=b; bw,bh=x2-x,y2-y
    h,w=frame.shape[:2]
    lo=np.maximum([0,0],b[:2]-[max(6,bw*.18),max(6,bh*.4)]).astype(int)
    hi=np.minimum([w,h],b[2:]+[max(6,bw*.18),max(6,bh*.4)]).astype(int)
    crop=frame[lo[1]:hi[1],lo[0]:hi[0]]
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    options=[]
    if row.get('proposal_quad'):
        try:
            q,_,_=geometry(row['proposal_quad'],row['image_size'])
            q=np.array(q)
            envelope=np.r_[q.min(0),q.max(0)]
            options.append({'quad':q.tolist(),'source':'frozen_BEST_G_OBB','score':bbox_iou(envelope,b),
                            'confidence':row.get('proposal_confidence')})
        except ValueError:
            pass
    yellow=row['plate_type']=='type1b'
    thresholds=[(60,45),(90,65),(120,85)] if yellow else [(100,110),(80,150),(65,185)]
    for sat,val in thresholds:
        mask=cv2.inRange(hsv,(8,sat,val),(44,255,255)) if yellow else cv2.inRange(hsv,(0,0,val),(179,sat,255))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        for contour in cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]:
            hull=cv2.convexHull(contour); area=cv2.contourArea(hull)
            if not .23*bw*bh<area<1.3*bw*bh:
                continue
            for eps in [.015,.02,.03,.04,.05]:
                poly=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True)
                if len(poly)!=4:
                    continue
                try:
                    q=order_obb(poly.reshape(4,2)).astype(float)+lo
                    geometry(q.tolist(),row['image_size'])
                except ValueError:
                    continue
                local=q-lo
                if (np.any(local[:,0]<=1) or np.any(local[:,1]<=1) or
                    np.any(local[:,0]>=crop.shape[1]-2) or np.any(local[:,1]>=crop.shape[0]-2)):
                    continue
                envelope=np.r_[q.min(0),q.max(0)]
                score=bbox_iou(envelope,b)-.10*abs(math.log(max(area/(bw*bh),.01)/.8))
                if score>.35:
                    options.append({'quad':q.tolist(),'source':'yellow_contour' if yellow else 'white_contour',
                                    'score':score,'confidence':None})
                break
    options.sort(key=lambda item:item['score'],reverse=True)
    unique=[]
    for item in options:
        if all(not np.allclose(item['quad'],old['quad'],atol=1) for old in unique):
            unique.append(item)
    masks=[item for item in unique if item['source']!='frozen_BEST_G_OBB' and item['score']>=.55]
    model=next((item for item in unique if item['source']=='frozen_BEST_G_OBB' and item['score']>=.60),None)
    selected=(masks[0] if masks else model) if yellow else (model or (masks[0] if masks else None))
    if selected is None:
        edges=edge_candidates(frame,row)
        unique.extend(edges[:4])
        unique.sort(key=lambda item:item['score'],reverse=True)
        if edges and edges[0]['score']>.55:
            selected=edges[0]
    return selected,unique[:6]


def prepare(folder):
    review=Review()
    folder=Path(folder)
    folder.mkdir(parents=True,exist_ok=False)
    snapshot=review.state_path.read_bytes()
    state=json.loads(snapshot)
    (folder/'state_at_start.json').write_bytes(snapshot)
    rows=[r for r in review.rows if r['review_id'] not in state['decisions']]
    result=[]
    for n,row in enumerate(rows,1):
        if file_sha256(Path(row['image']))!=row['source_sha256']:
            raise ValueError('Source changed: '+row['review_id'])
        frame=cv2.imdecode(np.fromfile(row['image'],np.uint8),cv2.IMREAD_COLOR)
        if [frame.shape[1],frame.shape[0]]!=row['image_size']:
            raise ValueError('Source dimensions changed')
        selected,alternatives=candidates(frame,row)
        b=row['bbox']; fallback=[[b[0],b[1]],[b[2],b[1]],[b[2],b[3]],[b[0],b[3]]]
        result.append({**{k:row[k] for k in ['review_id','plate_num','plate_type','image','image_size','bbox','split','source_sha256','source']},
            'n':n,'quad':selected['quad'] if selected else None,'old_quad':fallback,
            'method':selected['source'] if selected else 'no_candidate',
            'heuristic_score':selected['score'] if selected else None,'alternatives':alternatives,
            'visual_accepted':False,'visual_reviewed_by':None,'human_confirmed':False,'quad_verified':False})
        if n%60==0:
            print('Prepared',n,'/',len(rows),flush=True)
    value={'created_at':now(),'index_sha256':review.index_sha,
        'state_snapshot_sha256':hashlib.sha256(snapshot).hexdigest(),'mode':'new_samples_only',
        'GT_created':0,'training_started':False,'test_read':False,'records':result}
    atomic_json(folder/'candidates.json',value)
    render(folder)
    print('Output:',folder,'objects:',len(result),'candidates:',sum(r['quad'] is not None for r in result),flush=True)


def render(folder):
    folder=Path(folder)
    rows=json.loads((folder/'candidates.json').read_bytes())['records']
    font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
    small=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',14)
    panels=[]; mappings={}
    for r in rows:
        frame=cv2.imdecode(np.fromfile(r['image'],np.uint8),cv2.IMREAD_COLOR)
        h,w=frame.shape[:2];q=np.array(r['quad'] or r['old_quad']);b=np.array(r['bbox'])
        lo=np.minimum(q.min(0),b[:2]);hi=np.maximum(q.max(0),b[2:]);bw,bh=hi-lo
        lo=np.maximum([0,0],lo-[max(7,bw*.12),max(7,bh*.28)]).astype(int)
        hi=np.minimum([w,h],hi+[max(7,bw*.12),max(7,bh*.28)]).astype(int)
        clean=Image.fromarray(cv2.cvtColor(frame[lo[1]:hi[1],lo[0]:hi[0]],cv2.COLOR_BGR2RGB))
        factor=min(410/clean.width,98/clean.height)
        im=clean.resize((max(1,round(clean.width*factor)),max(1,round(clean.height*factor))),Image.Resampling.LANCZOS)
        x=(430-im.width)//2;y=45+(98-im.height)//2
        mappings[str(r['n'])]={'lo':lo.tolist(),'scale_x':im.width/clean.width,'scale_y':im.height/clean.height,'image_origin':[x,y]}
        panel=Image.new('RGB',(430,255),'#f4f7f8');draw=ImageDraw.Draw(panel)
        draw.text((6,3),f'{r["n"]:03d} {r["plate_num"]} {r["plate_type"]}',fill='#20343d',font=font)
        draw.text((6,24),f'{r["method"]} | {b[2]-b[0]:.0f}x{b[3]-b[1]:.0f}px',fill='#526e80',font=small)
        for k in [0,1]:
            panel.paste(im,(x,y+k*104))
            if k:
                pts=[(x+float(p[0]-lo[0])*im.width/clean.width,y+104+float(p[1]-lo[1])*im.height/clean.height) for p in q]
                draw.line(pts+[pts[0]],fill='#00d5ac' if r['quad'] else '#e8a01c',width=2)
                for i,(u,v) in enumerate(pts):
                    draw.ellipse((u-2,v-2,u+2,v+2),fill='#ed384c')
                    draw.text((u+3,v-14),str(i+1),fill='#ed384c',font=small)
        panel.save(folder/f'{r["n"]:03d}.png');panels.append(panel)
    for s in range(math.ceil(len(panels)/24)):
        sheet=Image.new('RGB',(1720,1530),'#c8d5dc')
        for j,p in enumerate(panels[s*24:(s+1)*24]):
            sheet.paste(p,((j%4)*430,(j//4)*255))
        sheet.save(folder/f'sheet_{s+1:02d}.png')
    atomic_json(folder/'crop_mapping.json',mappings)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--folder',type=Path,required=True)
    parser.add_argument('--render-only',action='store_true')
    args=parser.parse_args()
    render(args.folder) if args.render_only else prepare(args.folder)
