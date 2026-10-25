"""Group/split and mixture safeguards for real-domain experiments."""
import math,random,re,hashlib,json
from collections import defaultdict
import cv2,numpy as np

MASK=re.compile(r'^[ABEKMHOPCTYX][0-9]{3}[ABEKMHOPCTYX]{2}[0-9]{2,3}$')

def valid_complete_plate(text):return bool(MASK.fullmatch(text))

def image_fingerprints(image):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY) if image.ndim==3 else image
    tiny=cv2.resize(gray,(17,8),interpolation=cv2.INTER_AREA)
    bits=(tiny[:,1:]>tiny[:,:-1]).ravel()
    return {'pixel_sha256':hashlib.sha256(image.tobytes()).hexdigest(),
            'dhash128':f'{int("".join(map(str,bits.astype(int))),2):032x}'}

def connected_groups(records,near_pairs=()):
    names=sorted({r['image'] for r in records});parent={n:n for n in names}
    def find(n):
        while parent[n]!=n:parent[n]=parent[parent[n]];n=parent[n]
        return n
    def union(a,b):
        x,y=find(a),find(b)
        if x!=y:parent[max(x,y)]=min(x,y)
    keys={}
    for r in records:
        values=[]
        for field in ('pixel_sha256','vehicle_id','sequence_id','source_capture_id'):
            if r.get(field):values.append(field+':'+r[field])
        if valid_complete_plate(r.get('visible_plate_num') or r.get('plate_num','')):
            values.append('plate:'+(r.get('visible_plate_num') or r['plate_num']))
        for k in values:
            if k in keys:union(r['image'],keys[k])
            else:keys[k]=r['image']
    for a,b in near_pairs:
        if a not in parent or b not in parent:raise ValueError('Unknown near duplicate image')
        union(a,b)
    groups=defaultdict(list)
    for n in names:groups[find(n)].append(n)
    return list(groups.values())

def verify_partition(records,near_pairs=()):
    splits={}
    for r in records:
        if r['image'] in splits and splits[r['image']]!=r['split']:raise ValueError('Same image crosses splits')
        splits[r['image']]=r['split']
    for group in connected_groups(records,near_pairs):
        if len({splits[n] for n in group})>1:raise ValueError('Connected identity/content/capture group crosses splits: '+','.join(group))
    tests=[r for r in records if r['split']=='real_test']
    for r in tests:
        if r.get('previously_exposed') or not r.get('source_capture_id') or not r.get('identity_reviewed'):
            raise ValueError('Independent real test requires unexposed reviewed identity and capture provenance')
    return {'groups':len(connected_groups(records,near_pairs)),'real_test_records':len(tests),'independent_test_available':bool(tests)}

def mixed_indices(real_count,synthetic_count,total,real_fraction,seed):
    if isinstance(total,bool) or not isinstance(total,int) or total<1 or not math.isfinite(real_fraction) or not 0<real_fraction<1:
        raise ValueError('Finite mixture in (0,1) and positive integer total required')
    if min(real_count,synthetic_count)<1:raise ValueError('Both domains required')
    rng=random.Random(seed);n=round(total*real_fraction)
    result=[('real',rng.randrange(real_count)) for _ in range(n)]+[('synthetic',rng.randrange(synthetic_count)) for _ in range(total-n)]
    rng.shuffle(result);return result

def confidence_bins(rows,key,edges=(0,.2,.4,.6,.8,1.000001)):
    bins=[]
    for a,b in zip(edges,edges[1:]):
        group=[r for r in rows if r.get(key) is not None and a<=r[key]<b]
        bins.append({'low':a,'high':min(1,b),'count':len(group),
            'correct':sum(bool(r['correct']) for r in group),'empirical_accuracy':sum(bool(r['correct']) for r in group)/len(group) if group else None})
    if sum(b['count'] for b in bins)!=sum(r.get(key) is not None for r in rows):raise ValueError('Confidence outside [0,1]')
    return bins
