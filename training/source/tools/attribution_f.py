"""Conservative causal accounting; missing spatial GT never becomes detector recall."""
import argparse
import hashlib
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import cv2
import numpy as np
from PIL import Image,ImageDraw,ImageFont
from tools.phase_e import ROOT,IMAGES,read,write,official,load_image
from tools.phase_f_setup import REPORT
from src.geometry_f import bbox_iou,bbox_from_quad,canonical_rectify
from src.image_anpr import padded_crop
from src.ocr_preprocess import prepare_ocr_input

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args()
    exp=REPORT/'experiments'/args.experiment;result=read(exp/'result.json');traces=read(exp/'trace.json')['records']
    (REPORT/'attribution').mkdir(parents=True,exist_ok=True)
    score,matches=official(exp/'predictions.csv',REPORT/'attribution/official')
    original=read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')['records']
    spatial={(r['image'],r['ground_truth']):{'quad':r['manual_quad'],'type':r['plate_type']} for r in original}
    for r in read(REPORT/'real_annotation_v2/manual_annotations.json')['records']:
        for a in r['annotations']:
            spatial[(r['original_image'],a.get('organizer_literal_plate_num',a['plate_num']))]={'quad':a['quad'],'type':a['plate_type'],'visible_identity':a['plate_num']}
    by_image={r['image']:r for r in traces};rows=[];previews=[]
    for match in matches:
        if not match.gt.is_target:continue
        g=match.gt;ref=spatial.get((g.image,g.plate));ds=by_image[g.image]['detections'];category='SUCCESS' if match.full_ok else 'UNKNOWN_REQUIRES_NEW_SPATIAL_REVIEW'
        chosen=None;overlap=None;coverage=None;input_review=None
        if ref:
            box=bbox_from_quad(ref['quad']);overlap,chosen=max(((bbox_iou(d['bbox'],box),d) for d in ds),key=lambda x:x[0],default=(0,None))
            if not match.full_ok:
                if overlap<.1:category='DETECTOR_MISS_ON_MANUAL_GT'
                elif overlap<.5:category='BAD_LOCALIZATION_ON_MANUAL_GT'
                elif chosen.get('status')!='ocr':category='FILTER_REJECTION_ON_MANUAL_GT'
                elif chosen.get('plate_type')!=g.ptype:category='ROUTING_ON_MANUAL_GT'
                else:category='INPUT_REVIEW_REQUIRED'
            if chosen:
                b=chosen['bbox'];inter=max(0,min(b[2],box[2])-max(b[0],box[0]))*max(0,min(b[3],box[3])-max(b[1],box[1]))
                coverage=inter/((box[2]-box[0])*(box[3]-box[1]))
                if not match.full_ok and category=='INPUT_REVIEW_REQUIRED' and not chosen.get('perspective_rectification') and coverage<.94:
                    category='CLIPPED_BBOX_ON_MANUAL_GT'
                source=load_image(IMAGES/g.image)
                try:
                    if chosen.get('perspective_rectification'):
                        crop,_=canonical_rectify(source,chosen['observed_quad'],chosen['plate_type'],'model_diagnostic')
                    else:crop,_=padded_crop(source,chosen['bbox'],0)
                    h,w=crop.shape[:2];scale=max(480/w,160/h)
                    resized=cv2.resize(crop,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC) if scale>1 else crop
                    kind=chosen['plate_type'] if result['config'].get('stitch',True) else 'type1'
                    normalized=prepare_ocr_input(cv2.cvtColor(resized,cv2.COLOR_BGR2GRAY),kind)
                    if chosen.get('normalized_sha256'):
                        assert hashlib.sha256(normalized.normalized.tobytes()).hexdigest()==chosen['normalized_sha256'],'Diagnostic image must exactly equal the processed OCR input'
                    inputs=REPORT/'attribution/inputs';inputs.mkdir(parents=True,exist_ok=True)
                    stem=f'{len(rows):03}_{Path(g.image).stem}'
                    np.save(inputs/f'{stem}_normalized.npy',normalized.normalized)
                    Image.fromarray(np.rint(np.clip((normalized.normalized+1)*127.5,0,255)).astype(np.uint8)).save(inputs/f'{stem}_normalized.png')
                    Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB)).save(inputs/f'{stem}_crop.png')
                    input_review={'normalized_npy':str(inputs/f'{stem}_normalized.npy'),
                        'normalized_png':str(inputs/f'{stem}_normalized.png'),'crop_png':str(inputs/f'{stem}_crop.png'),
                        'normalized_tensor_sha256':hashlib.sha256(normalized.normalized.tobytes()).hexdigest()}
                    # No canonical/downstream output is used to rewrite production predictions.
                    previews.append((len(rows),g,crop,np.clip((normalized.normalized+1)*127.5,0,255).astype(np.uint8),category,chosen))
                except (ValueError,cv2.error):pass
        if ref and ref.get('visible_identity') and ref['visible_identity']!=g.plate and not match.full_ok:
            category='SOURCE_GT_TRANSCRIPTION_DISCREPANCY'
        rows.append({'image':g.image,'ground_truth':g.plate,'plate_type':g.ptype,'official_success':match.full_ok,
            'spatial_GT_available':bool(ref),'bbox_IoU':overlap,'GT_envelope_coverage':coverage,'category':category,
            'chosen':chosen,'input_review':input_review,'visible_identity':ref.get('visible_identity') if ref else None})
    folder=REPORT/'attribution';folder.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',17);paths=[]
    for page in range((len(previews)+5)//6):
        canvas=Image.new('RGB',(1200,1050),'white');draw=ImageDraw.Draw(canvas)
        for i,(index,g,crop,norm,cat,d) in enumerate(previews[page*6:(page+1)*6]):
            x=i%2*600;y=i//2*350
            draw.text((x+5,y+5),f'{index}: {g.image} {g.plate} {g.ptype}',font=font,fill='black')
            draw.text((x+5,y+28),cat,font=font,fill='black')
            im=Image.fromarray(cv2.cvtColor(crop,cv2.COLOR_BGR2RGB));im.thumbnail((575,220));canvas.paste(im,(x+5,y+55))
            im=Image.fromarray(norm).resize((480,96));canvas.paste(im,(x+5,y+235))
            draw.text((x+5,y+332),str(d.get('final_ocr'))+' '+d.get('preprocessing',{}).get('effective_mode','none'),font=font,fill='black')
        path=folder/f'input_review_{page+1:02}.jpg';canvas.save(path);paths.append(str(path))
    output={'experiment':args.experiment,'targets':len(rows),'official_correct':score['full_match_total'],'rows':rows,
        'categories':dict(Counter(r['category'] for r in rows)),
        'spatial_GT_targets':sum(r['spatial_GT_available'] for r in rows),'input_review_paths':paths,'review_status':'PENDING',
        'scope':'Manual quad subset only for geometry attribution; all other failed public targets remain UNKNOWN. Official text matching is not spatial detector recall. Training-scene GT/transcription discrepancy identified separately.',
        'OCR_attribution_policy':'Do not assign OCR_FAILURE until actual normalized input has been manually verified for correct type/all rows and non-clipped characters. Structural split acceptance alone is insufficient.'}
    assert len(rows)==115;write(folder/'remaining_failures.json',output);print(output['categories'],flush=True)

if __name__=='__main__':main()
