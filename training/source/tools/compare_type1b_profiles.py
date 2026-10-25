"""Read-only OCR crop diagnostics; never independent E2E or automatic GT."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,hashlib,json,time
from collections import Counter

from src.inference_io import load_runtime_config,file_sha256,offline_network
from src.plate_formats import valid_plate,yellow_postprocess

ROOT=Path(__file__).resolve().parents[1]


def read_records(verified_index,synthetic_manifest):
    from dataset.validation import excluded_hashes
    protected=ROOT.parent/'external_data/organizer_debug/source_manifest.json'
    excluded=excluded_hashes([protected]) if protected.exists() else set()
    records=[];seen=set();skipped=Counter()
    with Path(verified_index).open(encoding='utf-8-sig',newline='') as stream:
        for row in csv.DictReader(stream):
            yes=lambda key:row.get(key,'').lower() in ('true','1')
            if not (yes('verified') and yes('ocr_usable') and yes('geometry_GT_verified') and yes('is_vehicle')
                    and not yes('is_synthetic') and row.get('review_status') in ('accepted','corrected')
                    and valid_plate(row['plate_num'],row['plate_type'])):
                skipped['not_full_human_ocr_record']+=1;continue
            key=(row['source_sha256'],row['crop_sha256'],row['plate_num'])
            if key in seen:skipped['duplicate_human_crop']+=1;continue
            seen.add(key)
            if row['source_sha256'] in excluded or row['crop_sha256'] in excluded:
                skipped['protected_evaluation_bytes']+=1;continue
            if file_sha256(Path(row['image']))!=row['source_sha256'] or file_sha256(Path(row['crop']))!=row['crop_sha256']:
                raise ValueError('Human record source/crop bytes changed')
            records.append({'id':row['review_id'],'crop':row['crop'],'crop_sha256':row['crop_sha256'],
                'source_sha256':row['source_sha256'],'plate_num':row['plate_num'],'plate_type':row['plate_type'],
                'scope':'human_verified_development_crop','source_family':row['source_family'],
                'duplicate_group_id':row['duplicate_group_id'],'original_image_group_id':row['original_image_group_id']})
    manifest_path=Path(synthetic_manifest);manifest=json.loads(manifest_path.read_text(encoding='utf-8'))
    for row in manifest['records']:
        path=manifest_path.parent/row['image']
        if file_sha256(path)!=row['image_sha256'] or row['raster_decoded']!=row['plate_num']:raise ValueError('Authored synthetic control changed')
        records.append({'id':row['image'],'crop':str(path),'crop_sha256':row['image_sha256'],
                        'plate_num':row['plate_num'],'plate_type':'type1b','scope':'authored_format_control_crop'})
    return records,dict(skipped)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verified-index',type=Path,default=ROOT.parent/'datasets/real_verified/index.csv')
    parser.add_argument('--synthetic-manifest',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda');args=parser.parse_args()
    if args.output.resolve().is_relative_to(args.verified_index.resolve().parent):raise ValueError('Report must stay outside verified data')
    records,skipped=read_records(args.verified_index,args.synthetic_manifest)
    index_hash=file_sha256(args.verified_index);config_path=ROOT/'configs/phase_g/best_g.json';c=load_runtime_config(config_path,ROOT)
    config_hash=file_sha256(config_path);results=[]
    with offline_network():
        import cv2,numpy as np,torch
        from src.image_anpr import current_postprocess
        from src.image_anpr_g import OCRG,gray_for_ocr
        from src.multiline_g import ConfigG,candidates,candidate_score
        from src.real_data_g import valid_complete_plate
        torch.set_num_threads(4);cv2.setNumThreads(1);ocr=OCRG(c['ocr'],args.device)
        for row in records:
            image=cv2.imdecode(np.fromfile(row['crop'],dtype=np.uint8),cv2.IMREAD_COLOR)
            if image is None:raise ValueError('Unreadable crop')
            ocr.synchronize();start=time.perf_counter()
            # Human type is intentionally oracle assistance: crop-only diagnostics.
            prepared=candidates(gray_for_ocr(image),row['plate_type'],ConfigG(**c['split']))
            decoded=ocr.recognize_candidates([p[0] for p in prepared])
            scores=[candidate_score(v[0],v[3],p[2]['structural_confidence'],valid_complete_plate(v[0])) for p,v in zip(prepared,decoded)]
            selected=max(range(len(scores)),key=lambda i:scores[i]);raw,confidence,probabilities,logp=decoded[selected]
            old=current_postprocess(raw,probabilities)
            if row['plate_type']=='type1b':new,reason=yellow_postprocess(raw,probabilities)
            else:new,reason=old,'frozen_white_path'
            ocr.synchronize();elapsed=(time.perf_counter()-start)*1000
            results.append({**row,'raw':raw,'frozen_output':old,'russian_output':new,'typed_reason':reason,
                'raw_exact':raw==row['plate_num'],'frozen_exact':old==row['plate_num'],'russian_exact':new==row['plate_num'],
                'confidence':confidence,'elapsed_crop_ms':elapsed,'selected_candidate':selected,
                'normalized_sha256':hashlib.sha256(prepared[selected][0].tobytes()).hexdigest()})
    if file_sha256(args.verified_index)!=index_hash or file_sha256(config_path)!=config_hash:raise ValueError('Human/config state changed')
    groups={}
    for scope,kind in sorted({(r['scope'],r['plate_type']) for r in results}):
        rows=[r for r in results if r['scope']==scope and r['plate_type']==kind]
        groups[scope+'::'+kind]={'samples':len(rows),'raw_exact':sum(r['raw_exact'] for r in rows),
            'frozen_exact':sum(r['frozen_exact'] for r in rows),'russian_exact':sum(r['russian_exact'] for r in rows),
            'changed':sum(r['frozen_output']!=r['russian_output'] for r in rows)}
    args.output.mkdir(parents=True,exist_ok=True)
    report={'scope':'GT-assisted crop-only development diagnostics; not independent E2E, not a reference hardware benchmark',
            'groups':groups,'skipped':skipped,'records':results,'verified_index_sha256':index_hash,
            'config_sha256':config_hash,'ocr_sha256':c['ocr_sha256'],'device':args.device,
            'training_started':False,'new_GT_created':0,'public_debug_images_read':False,'final_split_created':False}
    (args.output/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    exposure=[{k:r[k] for k in ('id','plate_num','plate_type','source_sha256','source_family','duplicate_group_id','original_image_group_id')}
              for r in results if r['scope']=='human_verified_development_crop']
    (args.output/'exposure_manifest.json').write_text(json.dumps({'records':exposure,'hold_from_future_untouched_test':True,
                 'reason':'Human labels and OCR outputs compared during development','split':'unassigned'},ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(groups,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
