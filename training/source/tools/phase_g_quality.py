"""Controlled OCR comparison, actual-input review and CPU/GPU parity."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,hashlib
from collections import Counter
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
from tools.phase_g_common import ROOT,REPORT,IMAGES,read,write,scorer
from tools.phase_e import load_image
from tools.phase_g_metrics import ocr_metrics,waterfall
from tools.phase_g_audit import artifacts,paste,FONT
from src.image_anpr import save_image,current_postprocess
from src.image_anpr_g import OCRG,gray_for_ocr
from src.multiline_g import ConfigG,candidates
from src.geometry_f import canonical_rectify,bbox_iou
from src.real_data_g import confidence_bins

def main():
    torch.set_num_threads(4);cv2.setNumThreads(1)
    cuda=read(REPORT/'runtime/cuda/trace.json')['records'];cpu=read(REPORT/'runtime/cpu/trace.json')['records']
    differences=[];ious=[];conf=[]
    for a,b in zip(cuda,cpu):
        if [(p['plate_num'],p['plate_type']) for p in a['predictions']]!=[(p['plate_num'],p['plate_type']) for p in b['predictions']]:differences.append(a['image'])
        if len(a['detections'])!=len(b['detections']):differences.append(a['image']+' detection count')
        for x,y in zip(a['detections'],b['detections']):
            ious.append(bbox_iou(x['bbox'],y['bbox']));conf.append(abs(x.get('ocr_confidence',0)-y.get('ocr_confidence',0)))
    write(REPORT/'CPU_GPU_parity.json',{'status':'PASS' if not differences and min(ious)>.99 else 'FAIL','text_type_differences':differences,'bbox_IoU_min':min(ious),'OCR_confidence_max_delta':max(conf),'images':100,'detection_rows':len(ious),'scope':'Actual uncached full pipeline; CPU torch2.14 versus CUDA torch2.6; confidence numerical differences allowed.'})
    baseline=read(REPORT/'error_summary.json')['rows'];traces={r['image']:r for r in cuda}
    details=list(csv.DictReader((REPORT/'runtime/cuda/official.details.csv').open(encoding='utf-8-sig'),delimiter=';'))
    successes={(r['image'],r['gt_plate'],r['gt_type']) for r in details if r['status']=='ok' and r['plate_ok']=='1' and r['type_ok']=='1'}
    cfg=ConfigG(**read(REPORT/'provisional_best_g.json')['result']['config']['split']);rows=[]
    for old in baseline:
        row=dict(old);row['official_success']=(old['image'],old['ground_truth'],old['plate_type']) in successes
        d=next((d for d in traces[old['image']]['detections'] if d['detection_index']==old['detector_index']),None)
        if d:
            row.update(raw_OCR=d.get('raw_ocr'),postprocessed_OCR=d.get('postprocessed_ocr'),final_OCR=d.get('final_ocr'),split_status=d.get('preprocessing',{}).get('effective_mode'),ocr_confidence=d.get('ocr_confidence'))
        if row['official_success']:row['primary_failure_category']='SUCCESS'
        elif old['primary_failure_category'] in ('UNKNOWN','DETECTOR_MISS','LOW_DETECTOR_CONFIDENCE','UNREADABLE','BAD_RECTIFICATION'):pass
        elif d and row['split_status']=='whole_plate' and old['plate_type']=='type1a':row['primary_failure_category']='TYPE1A_SPLIT_FAILURE'
        elif d:row['primary_failure_category']='OCR_FAILURE_GOOD_INPUT'
        else:row['primary_failure_category']='UNKNOWN'
        if not row['official_success'] and d:
            folder=REPORT/'remaining/inputs'/Path(old['image']).stem/str(d['detection_index']);folder.mkdir(parents=True,exist_ok=True)
            frame=load_image(IMAGES/old['image'])
            if d.get('G_fallback'):paths=artifacts(frame,d,folder)
            else:
                crop,_=canonical_rectify(frame,d['observed_quad'],d['plate_type'],'observed OBB');gray=gray_for_ocr(crop)
                normalized,stitch,diag=candidates(gray,d['plate_type'],cfg)[d['selected_candidate']]
                assert hashlib.sha256(normalized.tobytes()).hexdigest()==d['normalized_sha256']
                save_image(folder/'rectified.png',crop);save_image(folder/'stitched.png',stitch)
                save_image(folder/'normalized.png',np.clip((normalized+1)*127.5,0,255).astype(np.uint8));np.save(folder/'normalized.npy',normalized)
                for name,box in zip(('top','bottom'),diag['row_boxes']):
                    a,b,c,e=box;save_image(folder/(name+'.png'),gray[b:e,a:c])
                paths={name:str(folder/(name+'.png')) if (folder/(name+'.png')).exists() else None for name in ('rectified','stitched','normalized','top','bottom')}
            row['G_artifacts']=paths
        rows.append(row)
    failed=[r for r in rows if not r['official_success']]
    for page in range((len(failed)+7)//8):
        canvas=Image.new('RGB',(1600,1000),'white');draw=ImageDraw.Draw(canvas)
        for j,r in enumerate(failed[page*8:page*8+8]):
            x=j%2*800;y=j//2*250
            draw.text((x+5,y+5),r['image']+' '+r['ground_truth']+' '+r['plate_type'],font=FONT,fill='black')
            draw.text((x+5,y+30),'raw '+str(r.get('raw_OCR'))+' final '+str(r.get('final_OCR')),font=FONT,fill='black')
            for name,box in [('rectified',(x+5,y+65,245,155)),('stitched',(x+260,y+65,530,85)),('normalized',(x+260,y+160,530,80))]:
                path=r.get('G_artifacts',{}).get(name)
                if path:paste(canvas,path,box)
            if not r.get('G_artifacts'):draw.text((x+5,y+100),r['primary_failure_category'],font=FONT,fill='red')
        canvas.save(REPORT/'remaining'/f'review_{page+1}.jpg',quality=95)
    write(REPORT/'remaining_best_g_failures.json',{'rows':rows,'failed':len(failed),'categories_provisional':dict(Counter(r['primary_failure_category'] for r in failed)),'visual_review':'PENDING exact G input sheets','associations':'Frozen0.5 detector physical associations reviewed in F audit; detector checkpoint unchanged. Official correctness uses frozen scorer.'})
    # This is the identical71 F tensors, not newly improved G preprocessing.
    ocr=OCRG(read(REPORT/'provisional_best_g.json')['checkpoint'],'cuda');goods=[]
    for r in read(REPORT/'real_good_input_ocr.json')['rows']:
        raw,confidence,probs=ocr.recognize(np.load(Path(r['normalized_input']).with_suffix('.npy')))
        goods.append(dict(r,F_raw=r['raw_OCR'],F_post=r['postprocessed_OCR'],raw_OCR=raw,postprocessed_OCR=current_postprocess(raw,probs)))
    write(REPORT/'real_good_input_ocr_after.json',{'scope':'Same71 visually reviewed frozen F input tensors, complete labels only. Isolates changed OCR weights, not split selection. Exposed convenience subset, not generalization.', 'metrics':{k:{field:ocr_metrics([r for r in goods if k=='ALL' or r['plate_type']==k],field) for field in ('F_raw','F_post','raw_OCR','postprocessed_OCR')} for k in ('ALL','type1','type1a','type1b')},'rows':goods})
    module=scorer();train=set(read(ROOT/'reports/phase_f/dataset_composition.json')['real_import']['original_train_images']);bins=[];statuses=Counter();pertype={k:Counter() for k in ('type1','type1a','type1b')}
    for r in rows:
        if r['image'] in train or r['detector_index'] is None:continue
        d=next(d for d in traces[r['image']]['detections'] if d['detection_index']==r['detector_index'])
        raw=module.compare_plates(r['ground_truth'],d.get('raw_ocr') or '',r['plate_type'])[1] and d.get('plate_type')==r['plate_type']
        post=module.compare_plates(r['ground_truth'],d.get('postprocessed_ocr') or '',r['plate_type'])[1] and d.get('plate_type')==r['plate_type']
        status='FIXED' if post and not raw else 'BROKEN' if raw and not post else 'UNCHANGED';statuses[status]+=1;pertype[r['plate_type']][status]+=1
        bins.append({'image':r['image'],'correct':bool(r['official_success']),'detector_confidence':d.get('detector_confidence'),'ocr_confidence':d.get('ocr_confidence'),'final_confidence':d.get('ocr_confidence')})
    write(REPORT/'postprocess_analysis.json',{'scope':'Exposed80-frame real_val physically associated detected targets; GT discrepancies included per organizer literal. Missing targets excluded.', 'counts':dict(statuses),'per_type':{k:dict(v) for k,v in pertype.items()}})
    write(REPORT/'confidence_bins.json',{'scope':'Exposed associated real_val outputs, outcome is E2E text+type correct; detector confidence bins are not spatial detection calibration. Final confidence explicitly aliases mean OCR character confidence; no arbitrary multiplication.', 'bins':{k:confidence_bins(bins,k) for k in ('detector_confidence','ocr_confidence','final_confidence')},'rows':bins})
    # Retain all38 oracle comparison; exclude known clipped/degraded inputs from its good-input ceiling.
    oracle=read(REPORT/'oracle/comparison.json');eligible=[r for r in oracle['rows'] if r['mode']=='GT_QUAD_MANUAL_SPLIT' and r['image'] not in ('img_018.jpg','img_030.jpg')]
    write(REPORT/'oracle/good_input_eligibility_correction.json',{'count':len(eligible),'raw_correct':sum(r['raw']==r['ground_truth'] for r in eligible),'post_correct':sum(r['post']==r['ground_truth'] for r in eligible),'excluded':{'img_018.jpg':'source degradation','img_030.jpg':'Approximate manual quad clips upper strokes'},'policy':'All38 comparison remains a geometry diagnostic; only36 are good-input eligible. No manual split in production or training.'})
    print('Parity',not differences,'remaining',len(failed),'same input OCR',ocr_metrics(goods,'raw_OCR'),flush=True)

if __name__=='__main__':main()
