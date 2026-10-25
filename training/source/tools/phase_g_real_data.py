"""Immutable exposed real partitions plus empty future independent test."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,shutil
from collections import Counter
import cv2,numpy as np
from tools.phase_g_common import ROOT,REPORT,CONFIG,DATA,IMAGES,read,write,freeze
from dataset.schema import sha256
from src.real_data_g import image_fingerprints,verify_partition,connected_groups

def main():
    freeze();train=set(read(ROOT/'reports/phase_f/dataset_composition.json')['real_import']['original_train_images'])
    spatial={}
    from tools.phase_g_common import spatial_references
    for r in spatial_references():spatial[(r['image'],r.get('organizer_ground_truth',r['ground_truth']))]=r
    originals=list(csv.DictReader((IMAGES.parent/'debug_labels.csv').open(encoding='utf-8-sig'),delimiter=';'))
    fingerprints={};rows=[]
    for p in sorted(IMAGES.glob('*.jpg')):
        image=cv2.imdecode(np.fromfile(p,dtype=np.uint8),cv2.IMREAD_COLOR);fingerprints[p.name]=image_fingerprints(image)
        split='real_train' if p.name in train else 'real_val';dest=DATA/split/'images'/p.name;dest.parent.mkdir(parents=True,exist_ok=True)
        if not dest.exists():shutil.copyfile(p,dest)
        assert sha256(dest)==sha256(p)
    for r in originals:
        ref=spatial.get((r['image'],r['plate_num']));split='real_train' if r['image'] in train else 'real_val'
        rows.append(dict(r,split=split,source='organizer_public_debug_exposed',source_capture_id=None,sequence_id=None,vehicle_id=None,
            previously_exposed=True,identity_reviewed=bool(ref),visible_plate_num=ref['ground_truth'] if ref else None,
            quad=ref['quad'] if ref else None,bbox=None if not ref else np.r_[np.min(ref['quad'],axis=0),np.max(ref['quad'],axis=0)].tolist(),
            bbox_format='absolute_xyxy',quad_order='TL_TR_BR_BL',geometry_source='manual_E_F' if ref else 'UNANNOTATED',
            conditions=['public_debug','capture_conditions_unconfirmed'],license='organizer evaluation package; external redistribution not authorized',
            **fingerprints[r['image']]))
    near=[];names=sorted(fingerprints)
    for i,a in enumerate(names):
        for b in names[i+1:]:
            distance=(int(fingerprints[a]['dhash128'],16)^int(fingerprints[b]['dhash128'],16)).bit_count()
            if distance<=5:near.append({'left':a,'right':b,'distance':distance,'review_status':'PENDING'})
    checks=verify_partition(rows)
    for split in ('real_train','real_val','real_test'):
        (DATA/split/'images').mkdir(parents=True,exist_ok=True);write(DATA/split/'annotations.json',[r for r in rows if r['split']==split])
    result={'records':rows,'checks':checks,'near_duplicate_candidates':near,'near_duplicate_policy':'dHash is a review candidate, never proof of different images. Confirmed candidates union into groups; unresolved cross-split candidate blocks selection.',
        'exposure':'All100 public frames previously exposed; real_val is engineering development, no source-disjoint or independent accuracy claim.',
        'source_disjoint':False,'unknown_capture_groups':100,'real_test_policy':'EMPTY. New unexposed reviewed capture/vehicle/plate/sequence/source groups required; test manifest freezes before selection and one terminal evaluation. No test tuning.',
        'images':{'real_train':20,'real_val':80,'real_test':0},'independent_status':'GENERALIZATION NOT YET VERIFIED'}
    write(DATA/'manifest.json',result);write(REPORT/'real_data_manifest.json',result)
    write(DATA/'real_test/freeze.json',{'state':'AWAITING_UNEXPOSED_DATA','files':{},'user_confirmed_no_new_images':True})
    write(CONFIG/'selection_policy.json',{'primary':'Official E2E FPA on explicitly exposed real_val80frames87targets; never independent test accuracy',
        'control_correct':37,'control_GT':87,'type1_floor_all':11,'type1a_floor_all':40,'overall_floor_all':51,
        'latency_mean_ms_max':100,'latency_preferred_ms':60,'peak_torch_allocated_MiB_max':4096,'ties':'simpler and faster; bounded candidates, no oracle/GT in inference',
        'allowed_real_thresholds':[.2,.35,.5,.65,.8],'OBB_total_size_expansion':[0,.02,.04,.06],
        'no_real_type1b':'No real type1b present; keep user mask LDDDLLRR/RRR; only synthetic regression support',
        'new_test':'Unavailable; retain honest engineering-only designation'})
    print(checks,'near candidates',near,flush=True)

if __name__=='__main__':main()
