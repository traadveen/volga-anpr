"""Bounded label-free type1a split/stitch candidates, frozen one-line path."""
from dataclasses import dataclass,replace
import hashlib,math
import cv2,numpy as np
from src.multiline import MultilineConfig,stitch_rows,find_split
from src.ocr_preprocess import prepare_ocr_input
from ocr.crnn import _normalize_image

METHODS=('A_current_F','B_projection','C_adaptive','D_components','E_projection_prior')

@dataclass(frozen=True)
class ConfigG:
    method:str='B_projection'
    search_min:float=.35
    search_max:float=.65
    row_margin:float=.025
    stitch_height:int=64
    stitch_gap:int=10
    stitch_padding:int=4
    max_candidates:int=4
    def __post_init__(self):
        if self.method not in METHODS:raise ValueError('Unknown split method')
        if not 0<self.search_min<self.search_max<1:raise ValueError('Invalid central window')
        if not 0<=self.row_margin<.1:raise ValueError('Unsafe row margin')
        if self.stitch_height<16 or self.stitch_gap<0 or self.stitch_padding<0:raise ValueError('Invalid stitch dimensions')
        if type(self.max_candidates)!=int or not 1<=self.max_candidates<=5:raise ValueError('At most five candidates')

def projection_split(gray,config):
    if gray.dtype!=np.uint8 or gray.ndim!=2 or min(gray.shape)<16:raise ValueError('Expected nonempty uint8 gray crop')
    h,w=gray.shape
    if config.method=='C_adaptive':
        block=max(3,round(min(h,w)*.18)//2*2+1)
        mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,block,12)
    else:_,mask=cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV|cv2.THRESH_OTSU)
    left,right=round(w*.12),max(round(w*.62),round(w*.12)+1)
    profile=(mask[:,left:right]>0).mean(axis=1)
    smooth=np.convolve(profile,np.ones(max(3,round(h*.018)))/max(3,round(h*.018)),mode='same')
    lo,hi=round(h*config.search_min),round(h*config.search_max)
    positions=np.arange(lo,hi);cost=smooth[lo:hi].copy()
    if config.method=='E_projection_prior':cost+=.09*np.abs(positions/h-.47)
    if config.method=='D_components':
        split=find_split(gray,'connected_components',replace(MultilineConfig(),min_component_height=.08,min_component_width=.008))
        if not split.reliable:return None,{'reliable':False,'reason':'component evidence insufficient','confidence':0}
        cut=split.cut_y
    else:
        # An empty band before the first row is not an inter-row valley.
        # Among near-minimum valleys with ink on both sides, use the midpoint
        # of the longest contiguous band, so cuts do not graze stroke edges.
        feasible=[]
        for y,score in zip(positions,cost):
            u=smooth[round(h*.08):max(round(h*.08)+1,y-round(h*.035))]
            l=smooth[min(h-2,y+round(h*.035)):round(h*.9)]
            if len(u) and len(l) and min(u.max(),l.max())>.075:feasible.append((int(y),float(score)))
        if not feasible:return None,{'reliable':False,'reason':'no two-row valley','confidence':0}
        minimum=min(score for y,score in feasible);acceptable=[y for y,score in feasible if score<=minimum+.01]
        runs=[]
        for y in acceptable:
            if not runs or y!=runs[-1][-1]+1:runs.append([y])
            else:runs[-1].append(y)
        band=max(runs,key=lambda r:(len(r),-abs(np.mean(r)/h-.47)))
        cut=round((band[0]+band[-1])/2)
    upper=smooth[round(h*.08):max(round(h*.08)+1,cut-round(h*.035))]
    lower=smooth[min(h-2,cut+round(h*.035)):round(h*.9)]
    if not len(upper) or not len(lower):return None,{'reliable':False,'reason':'empty row evidence','confidence':0}
    upper_peak,lower_peak=float(upper.max()),float(lower.max());peak=min(upper_peak,lower_peak)
    valley=float(smooth[cut]);depth=max(0.,1-valley/max(peak,1e-8))
    balance=min(upper_peak,lower_peak)/max(upper_peak,lower_peak,1e-8)
    distance=min(cut/h-config.search_min,config.search_max-cut/h)/(config.search_max-config.search_min)
    foreground=float((mask[round(h*.05):round(h*.95),left:right]>0).mean())
    confidence=float(np.clip(depth*.6+balance*.25+max(0,distance)*.3,0,1))
    reliable=peak>.075 and depth>.22 and balance>.25 and .035<foreground<.7
    return (cut if reliable else None),{'method':config.method,'cut_y':cut,'reliable':reliable,'confidence':confidence,
        'valley_depth':depth,'height_ratio':cut/max(1,h-cut),'foreground':foreground,'row_balance':balance,
        'distance_from_window_edge':distance,'reason':'accepted two-row foreground' if reliable else 'insufficient row/valley evidence'}

def prepare_g(gray,plate_type,config=None):
    config=config or ConfigG()
    if plate_type!='type1a' or config.method=='A_current_F':
        p=prepare_ocr_input(gray,plate_type)
        return p.normalized,p.crop,dict(p.diagnostics,method='A_current_F',structural_confidence=1 if p.diagnostics['effective_mode']=='split_and_stitch' else 0)
    cut,diag=projection_split(gray,config)
    if cut is None:
        p=prepare_ocr_input(gray,plate_type);return p.normalized,p.crop,dict(p.diagnostics,method=config.method,proposal=diag,fallback='A_current_F',structural_confidence=0)
    h,w=gray.shape;margin=max(1,round(h*config.row_margin))
    boxes=((0,margin,w,cut),(0,cut,w,h-margin))
    rows=tuple(gray[b:e,a:c] for a,b,c,e in boxes)
    if min(r.shape[0] for r in rows)<8:raise ValueError('Empty/unsafe row split')
    stitch_cfg=replace(MultilineConfig(),stitch_height=config.stitch_height,stitch_gap=config.stitch_gap,stitch_padding=config.stitch_padding)
    image,matrices=stitch_rows(rows,stitch_cfg)
    return _normalize_image(image,32,160),image,{'method':config.method,'effective_mode':'split_and_stitch',
        'row_boxes':boxes,'split':diag,'structural_confidence':diag['confidence'],'stitch_transforms':[m.tolist() for m in matrices]}

def candidates(gray,plate_type,config=None):
    cfg=config or ConfigG();output=[];seen=set()
    methods=['A_current_F','B_projection','C_adaptive','D_components','E_projection_prior'] if plate_type=='type1a' else ['A_current_F']
    for method in methods:
        normalized,image,diag=prepare_g(gray,plate_type,replace(cfg,method=method));key=hashlib.sha256(normalized.tobytes()).hexdigest()
        if key in seen:continue
        seen.add(key);output.append((normalized,image,diag))
        if len(output)>=cfg.max_candidates:break
    return output

def candidate_score(raw,sequence_log_probability,structural_confidence,valid_mask):
    # CTC log probability per decoded character; no label supplied. The mask
    # tests the observed string only; geometry is a modest structural bonus.
    if not raw or not math.isfinite(sequence_log_probability):return -math.inf
    return sequence_log_probability/max(1,len(raw))+.75*bool(valid_mask)+.05*structural_confidence
