"""Snapshot human geometry and export isolated train/val OBB crops for a bounded pilot."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from collections import Counter
import json
import cv2
import numpy as np
from tools.phase_i_review import Review, atomic_json, geometry, training_region, sha_json
from tools.phase_h_common import ROOT, read, image, now
from src.inference_io import file_sha256
from src.plate_formats import valid_plate

REPORT=ROOT/'reports/phase_i/pilot_20261019'
DATA=ROOT.parent/'generated_data/phase_i_pilot_20261019'
CLASSES={'type1':0,'type1a':1,'type1b':2}


def eligible(row,decision,excluded):
    return (row['review_id'] not in excluded and decision.get('status')=='accepted'
        and all(decision.get(k) is True for k in ('human_confirmed','quad_verified','obb_verified','training_region_complete'))
        and decision.get('has_other_plates') is False and row.get('detector_usable') in ('True',True))


def crop_record(row,decision):
    if decision['source_sha256']!=row['source_sha256']:
        raise ValueError('Decision/source binding changed')
    q,obb,_=geometry(decision['quad'],row['image_size'])
    if not np.allclose(obb,decision['obb'],rtol=0,atol=.001):
        raise ValueError('Derived OBB differs from human corners')
    region=training_region(q,row['image_size'])
    if region!=decision['training_region']:
        raise ValueError('Human completeness applies to a different region')
    frame=image(row['image'],row['source_sha256'])
    x,y,x2,y2=region
    crop=frame[y:y2,x:x2]
    points=np.array(obb)-[x,y]
    normalized=points/[x2-x,y2-y]
    if not np.isfinite(normalized).all() or normalized.min()<0 or normalized.max()>1:
        raise ValueError('OBB does not fit the confirmed training region')
    return crop,normalized.tolist(),region


def write_example(folder,split,name,frame,kind,quad):
    images=folder/'images'/split;labels=folder/'labels'/split
    images.mkdir(parents=True,exist_ok=True);labels.mkdir(parents=True,exist_ok=True)
    path=images/(name+'.png')
    if path.exists():raise ValueError('Never overwrite an exported example')
    if not cv2.imwrite(str(path),frame):raise ValueError('Could not export crop')
    label=labels/(name+'.txt')
    label.write_text(str(CLASSES[kind])+' '+' '.join(f'{v:.9f}' for p in quad for v in p)+'\n',encoding='ascii')
    return {'export_image':str(path),'export_sha256':file_sha256(path),'export_label':str(label),
            'label_sha256':file_sha256(label),'shape':list(frame.shape[1::-1])}


def yaml_file(folder):
    path=folder/'data.yaml'
    path.write_text('path: '+folder.as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: type1\n  1: type1a\n  2: type1b\n',encoding='utf-8')
    return str(path)


def verify_selected(manifest,review=None):
    review=review or Review()
    review.guard.verify(force=True)
    if manifest['index_sha256']!=review.index_sha:raise ValueError('Immutable annotation queue changed')
    reservation=ROOT.parent/'datasets/reports/data3_holdout_reservations.json'
    if manifest.get('holdout_reservation_sha256') and file_sha256(reservation)!=manifest['holdout_reservation_sha256']:
        raise ValueError('Holdout reservation metadata changed after export')
    synthetic=ROOT.parent/'generated_data/detector_f_v2/manifest.json'
    if manifest.get('source_synthetic_manifest_sha256') and file_sha256(synthetic)!=manifest['source_synthetic_manifest_sha256']:
        raise ValueError('Synthetic source provenance changed after export')
    state=review.state()
    for ident,expected in manifest['selected_decision_sha256'].items():
        if sha_json(state['decisions'][ident])!=expected:raise ValueError('Selected human decision changed: '+ident)
    for row in manifest['records']:
        for path,key in [('export_image','export_sha256'),('export_label','label_sha256')]:
            if file_sha256(Path(row[path]))!=row[key]:raise ValueError('Export changed: '+row[path])
    return True


def prepare(report=REPORT,data=DATA,statement=None):
    # Each annotation round has its own immutable export and human snapshot.
    report=Path(report);data=Path(data)
    target=report/'dataset_manifest.json'
    if target.exists():
        m=read(target);verify_selected(m);return m
    if data.exists():raise ValueError('Incomplete export exists; inspect it before continuing')
    review=Review();state=review.state()
    qc_path=ROOT/'reports/phase_i/deep_qc_20261019/close_visual_audit.json'
    qc=read(qc_path)
    if qc['index_sha256']!=review.index_sha:raise ValueError('QC binds another immutable queue')
    # Retain exclusion IDs/reasons across new confirmations. New state checksum is bound below.
    excluded={r['review_id']:r['qc_reason'] for r in qc['records'] if not r['proposal_pool_eligible']}
    selected=[r for r in review.rows if eligible(r,state['decisions'][r['review_id']],excluded)]
    for row in selected:
        decision=state['decisions'][row['review_id']]
        if decision['index_sha256']!=review.index_sha or any(decision[k]!=row[k] for k in ('plate_type','plate_num','split')):
            raise ValueError('Human decision is not bound to the frozen label/split')
    train=[r for r in selected if r['split']=='train'];val=[r for r in selected if r['split']=='val']
    if not train or not val:raise ValueError('Human-verified train AND val geometry required')
    for key in ('plate_num','source_sha256','duplicate_group_id','original_image_group_id'):
        if {r[key] for r in train}&{r[key] for r in val}:raise ValueError('Train/val leakage: '+key)
    reservation=ROOT.parent/'datasets/reports/data3_holdout_reservations.json'
    reserved=read(reservation)['records']
    for key in ('review_id','plate_num','source_sha256','duplicate_group_id','original_image_group_id'):
        blocked={r.get(key) for r in reserved if r.get(key)}
        if blocked&{r.get(key) for r in selected if r.get(key)}:
            raise ValueError('Reserved holdout identity in development export: '+key)
    report.mkdir(parents=True,exist_ok=True)
    atomic_json(report/'geometry_snapshot.json',state)
    atomic_json(report/'user_sampling_statement.json',{'created_at':now(),
        'statement':statement or 'я проверил выборку из 50, там все было правильно, давай работать дальше',
        'reported_sample_size':50,'individually_identified_by_statement':False,
        'policy':'Count actual recorded human decisions only; no blanket GT acceptance inferred.'})
    records=[]
    for row in selected:
        crop,q,region=crop_record(row,state['decisions'][row['review_id']])
        # Equal budgets in both variants: three train images per real object.
        repeats=3 if row['split']=='train' else 1
        for variant in ('real_only','replay'):
            for repeat in range(repeats if variant=='real_only' else 1):
                name='real_'+row['review_id']+(f'_r{repeat}' if repeat else '')
                exported=write_example(data/variant,row['split'],name,crop,row['plate_type'],q)
                records.append({**exported,'variant':variant,'split':row['split'],'domain':'human_real',
                    'review_id':row['review_id'],'plate_type':row['plate_type'],'plate_num':row['plate_num'],
                    'source_image':row['image'],'source_sha256':row['source_sha256'],'training_region':region,
                    'duplicate_group_id':row['duplicate_group_id'],'original_image_group_id':row['original_image_group_id'],
                    'source_family':row['source_family'],'quad_verified':True,'region_complete':True,
                    'repeat':repeat,'geometry_provenance':'explicit human quad + explicit human padded-region completeness'})
    source=ROOT.parent/'generated_data/detector_f_v2'
    source_manifest=source/'manifest.json';synthetic=read(source_manifest)
    if synthetic['license']!='CC-BY-4.0 authored procedural assets':raise ValueError('Unexpected replay license/provenance')
    real_numbers={r['plate_num'] for r in review.rows}|{r['plate_num'] for r in reserved if r.get('plate_num')}
    replay_counts=Counter()
    for row in synthetic['records']:
        annotations=row.get('annotations',[])
        if not (row.get('is_synthetic') is True and row['split']=='train' and
                row.get('domain')=='authored_procedural_vehicle_scene' and len(annotations)==1):continue
        a=annotations[0];kind=a['plate_type']
        if kind not in ('type1','type1a') or replay_counts[kind]>=len(train) or a['plate_num'] in real_numbers:continue
        if not valid_plate(a['plate_num'],kind) or a.get('occlusion'):continue
        frame=image(source/'obb/images/train'/row['image'],row['sha256'])
        q,obb,_=geometry(a['quad'],[row['width'],row['height']])
        normalized=np.array(obb)/[row['width'],row['height']]
        if normalized.min()<0 or normalized.max()>1:continue
        exported=write_example(data/'replay','train','synthetic_'+Path(row['image']).stem,frame,kind,normalized.tolist())
        records.append({**exported,'variant':'replay','split':'train','domain':'authored_synthetic',
            'plate_type':kind,'plate_num':a['plate_num'],'source_image':str(source/'obb/images/train'/row['image']),
            'source_sha256':row['sha256'],'source_family':'authored_procedural_vehicle_scene',
            'geometry_provenance':'exact generator quad; minAreaRect OBB','license':synthetic['license']})
        replay_counts[kind]+=1
    if any(replay_counts[k]!=len(train) for k in ('type1','type1a')):raise ValueError('Insufficient safe replay')
    ys={k:yaml_file(data/k) for k in ('real_only','replay')}
    counts={split:dict(Counter(r['plate_type'] for r in selected if r['split']==split)) for split in ('train','val')}
    sizes={split:dict(Counter(r['size_bin_640'] for r in selected if r['split']==split)) for split in ('train','val')}
    m={'created_at':now(),'name':'PHASE-I human geometry pilot','scope':'Bounded pilot; coverage is recorded per type below.',
       'index_sha256':review.index_sha,'state_revision':state['revision'],
       'selected_decision_sha256':{r['review_id']:sha_json(state['decisions'][r['review_id']]) for r in selected},
       'qc_audit_sha256':file_sha256(qc_path),'excluded_ids_and_reasons':excluded,
       'source_synthetic_manifest_sha256':file_sha256(source_manifest),'holdout_reservation_sha256':file_sha256(reservation),
       'holdout_identity_overlap':0,'real_counts':counts,'source_size_bins_640':sizes,
       'real_unique_plates':{split:len({r['plate_num'] for r in selected if r['split']==split}) for split in ('train','val')},
       'replay_counts':dict(replay_counts),'data_yamls':ys,'records':records,
       'fixed_training_budget':{'epochs':4,'batch':4,'images_per_epoch':3*len(train),'frozen_layers':10,
          'checkpoint_selection':'fixed final epoch last.pt; full frozen VAL is primary'},
       'limitations':['Crop scale differs from full-scene scale; full-frame validation is primary.',
          'Class coverage is limited to explicitly confirmed live decisions, recorded in real_counts.',
          'Real-only repeats real train three times; replay adds two white synthetic examples per real example.',
          'Historical synthetic type1b excluded because its old text format is incompatible with current yellow plates.'],
       'training_started':False,'test_read':False,'public_debug_used':False,'human_GT_created':0}
    verify_selected(m,review);atomic_json(target,m)
    print(json.dumps({k:m[k] for k in ('real_counts','real_unique_plates','source_size_bins_640','replay_counts','fixed_training_budget')},indent=2))
    return m


if __name__=='__main__':
    prepare()
