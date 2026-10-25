"""Bounded Phase D OCR fine-tuning with real data and current-format replay."""
from pathlib import Path
import sys,os
os.environ.update(CUBLAS_WORKSPACE_CONFIG=':4096:8',YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
from collections import Counter,defaultdict
import random
import time
import cv2
import numpy as np
import torch
from tools.phase_h_common import (ROOT,REPORT,CONFIG,MODELS,D_WEIGHTS,SEED,SPLIT,read,write,
    load_split,guard_check,image,target_choices,decode_choices,now,yes)
from tools.phase_h_evaluate import match_objects
from src.image_anpr_f import DetectorF
from src.image_anpr_g import OCRG
from src.plate_formats import valid_plate
from src.inference_io import offline_network,file_sha256
from ocr.crnn import load_crnn_checkpoint

SYNTHETIC = ROOT.parent/'generated_data/current_scenes_5000_v1_20261015'


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def prepare():
    guard_check()
    if (REPORT/'training/prepared.json').exists():
        value = read(REPORT/'training/prepared.json')
        if value['tensor_sha256'] != file_sha256(REPORT/'training/arrays.npz'):
            raise ValueError('Prepared tensors changed')
        return value
    cfg = read(ROOT/'configs/phase_g/best_g.json')
    detector = DetectorF(cfg['detector'],'cuda',cfg['imgsz'],cfg['confidence'],cfg['iou'])
    arrays = []
    records = []
    detection_cache = {}
    def add(row,choices,domain,geometry):
        if not valid_plate(row['plate_num'],row['plate_type']):
            return
        ids = []
        for n,_,diag in choices:
            ids.append(len(arrays))
            arrays.append(n)
        records.append({'id':row['review_id'],'text':row['plate_num'],'type':row['plate_type'],
            'domain':domain,'arrays':ids,'geometry':geometry,'image':row['image'],
            'source_sha256':row['source_sha256'],
            'candidate_methods':[c[2]['method'] for c in choices],
            'candidate_modes':[c[2].get('effective_mode') for c in choices],
            'candidate_diagnostics':[c[2] for c in choices]})
    real = load_split('train')
    for index,row in enumerate(real):
        frame = image(row['image'],row['source_sha256'])
        sha = row['source_sha256']
        if sha not in detection_cache:
            detection_cache[sha] = detector.detect(frame)
        ds = detection_cache[sha]
        matched = match_objects([row],ds)
        q = ds[matched[0]].quad if 0 in matched else None
        geometry = 'verified_bbox_only_no_orientation_GT'
        if q is not None:
            try:
                choices = target_choices(frame,row,cfg,q)
                geometry = 'frozen_detector_observed_OBB_not_GT'
            except (ValueError,cv2.error):
                choices = target_choices(frame,row,cfg)
        else:
            choices = target_choices(frame,row,cfg)
        add(row,choices,'real_train',geometry)
        if index%50==0:
            print('Prepared real',index+1,'/',len(real),flush=True)
    # No label from test is used for learning or selection. Opaque identity
    # hashes in the reservation ledger are consulted solely to forbid replay collisions.
    ledger_path = ROOT.parent/'datasets/reports/data3_holdout_reservations.json'
    identity_hash = lambda text:hashlib.sha256(text.encode('ascii')).hexdigest()
    forbidden = {identity_hash(r['plate_num']) for r in load_split('val')}
    forbidden.update(identity_hash(r['plate_num']) for r in read(ledger_path)['records']
                     if yes(r.get('verified')) and r.get('plate_num'))
    manifest_path = SYNTHETIC/'manifest.json'
    source = read(manifest_path)
    pool = list(source['records'])
    random.Random(SEED).shuffle(pool)
    selected = Counter()
    seen = set()
    skipped_collision = 0
    for scene in pool:
        domain = 'synthetic_control' if int(scene['sha256'][:8],16)%5==0 else 'synthetic_train'
        needed = 30 if domain=='synthetic_control' else 400
        annotations = []
        for a in scene['annotations']:
            key = (domain,a['plate_type'])
            if selected[key]>=needed or a['plate_num'] in seen:
                continue
            if identity_hash(a['plate_num']) in forbidden:
                skipped_collision += 1
                continue
            region = a['plate_num'][5 if a['plate_type']=='type1b' else 6:]
            if (not valid_plate(a['plate_num'],a['plate_type']) or region in ('00','000')
                    or a.get('canonical_read')!=a['plate_num']):
                continue
            annotations.append(a)
        if not annotations:
            continue
        path = SYNTHETIC/scene['image']
        if file_sha256(path) != scene['sha256']:
            raise ValueError('Replay source changed')
        frame = image(path)
        for a in annotations:
            kind = a['plate_type']
            if selected[(domain,kind)]>=needed:
                continue
            r = {'review_id':'synthetic_'+scene['sha256'][:16]+'_'+str(len(records)),
                 'plate_num':a['plate_num'],'plate_type':kind,'bbox':a['bbox_xyxy'],
                 'source_sha256':scene['sha256'],'image':str(path)}
            add(r,target_choices(frame,r,cfg,a['quad']),domain,'exact_generator_quad')
            selected[(domain,kind)] += 1
            seen.add(a['plate_num'])
        if all(selected[(d,k)]>=n for d,n in [('synthetic_train',400),('synthetic_control',30)]
               for k in ('type1','type1a','type1b')):
            break
    if len([r for r in records if r['domain']=='synthetic_train'])!=1200:
        raise ValueError('Insufficient valid replay')
    # Cache all val candidates once, with the same GT-bbox OCR diagnostic scope.
    for row in load_split('val'):
        add(row,target_choices(image(row['image'],row['source_sha256']),row,cfg),'real_val','verified_bbox_only_no_orientation_GT')
    folder = REPORT/'training'
    folder.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(folder/'arrays.npz',inputs=np.stack(arrays).astype(np.float32))
    value = {'created_at':now(),'records':records,'tensor_sha256':file_sha256(folder/'arrays.npz'),
        'split_checksum':file_sha256(SPLIT),'preprocessing':cfg,
        'synthetic_manifest_sha256':file_sha256(manifest_path),
        'reservation_ledger_sha256':file_sha256(ledger_path),
        'reserved_identity_access':'Opaque membership-exclusion hashes only; no test images or predictions',
        'replay_identity_collisions_excluded':skipped_collision,
        'counts':dict(Counter(r['domain'] for r in records)),
        'real_geometry':dict(Counter(r['geometry'] for r in records if r['domain']=='real_train')),
        'representation':'Shared BEST-G rectification/candidates/split/stitch/32x160; no GT-guided candidate choice',
        'test_inference':False,'original_real_files_modified':False}
    write(folder/'prepared.json',value)
    # Inspection montage of unaugmented training inputs, evenly by real type.
    panel = np.full((720,1000,3),240,np.uint8)
    for col,kind in enumerate(('type1','type1a','type1b')):
        subset = [r for r in records if r['domain']=='real_train' and r['type']==kind][:6]
        for j,r in enumerate(subset):
            n = arrays[r['arrays'][0]]
            tile = cv2.cvtColor(np.clip((n+1)*127.5,0,255).astype(np.uint8),cv2.COLOR_GRAY2BGR)
            tile = cv2.resize(tile,(320,64))
            x,y=col*333,j*120
            panel[y+30:y+94,x:x+320] = tile
            cv2.putText(panel,kind+' '+r['text'],(x+4,y+23),cv2.FONT_HERSHEY_SIMPLEX,.55,(20,20,20),1)
    cv2.imencode('.jpg',panel)[1].tofile(folder/'real_inputs_review.jpg')
    guard_check()
    print('Training preparation',value['counts'],flush=True)
    return value


def augment(n,rng):
    gray = np.clip((n+1)*127.5,0,255).astype(np.uint8)
    if rng.random()<.2:
        gray = cv2.GaussianBlur(gray,(3,3),float(rng.uniform(.2,.5)))
    if rng.random()<.15:
        ok,encoded = cv2.imencode('.jpg',gray,[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(75,96))])
        if not ok:
            raise ValueError('JPEG augmentation failed')
        gray = cv2.imdecode(encoded,cv2.IMREAD_GRAYSCALE)
    if rng.random()<.15:
        h,w = gray.shape
        src = np.float32([[0,0],[w-1,0],[w-1,h-1],[0,h-1]])
        dst = src+rng.uniform(-.2,.2,(4,2)).astype(np.float32)
        gray = cv2.warpPerspective(gray,cv2.getPerspectiveTransform(src,dst),(w,h),borderMode=cv2.BORDER_REPLICATE)
    value = gray.astype(np.float32)
    value = (value-127.5)*float(rng.uniform(.92,1.08))+127.5+float(rng.uniform(-8,8))
    if rng.random()<.2:
        value += rng.normal(0,1.5,value.shape)
    return (np.clip(value,0,255)/127.5-1).astype(np.float32)


def training_view(records):
    """Exclude unsafe unsplit type1a inputs from optimization, never from val."""
    result=[]
    excluded=[]
    for source in records:
        r=dict(source)
        if r['domain'] in ('real_train','synthetic_train') and r['type']=='type1a':
            keep=[i for i,d in enumerate(r['candidate_diagnostics'])
                  if d.get('effective_mode')=='split_and_stitch']
            if not keep:
                excluded.append({'id':r['id'],'domain':r['domain'],'reason':'No safe automatic split/stitch input'})
                continue
            for key in ('arrays','candidate_methods','candidate_modes','candidate_diagnostics'):
                r[key]=[r[key][i] for i in keep]
        result.append(r)
    return result,excluded


def ocr_metrics(ocr,records,arrays,domain):
    correct = Counter()
    counts = Counter()
    for r in records:
        if r['domain'] != domain:
            continue
        # Selection uses the frozen automatic candidate scoring, never GT text.
        choices = [(arrays[i],None,diag) for i,diag in zip(r['arrays'],r['candidate_diagnostics'])]
        decoded,_,_ = decode_choices(ocr,choices)
        counts[r['type']] += 1
        correct[r['type']] += decoded[0] == r['text']
    return {'FPA':sum(correct.values())/sum(counts.values()),'correct':sum(correct.values()),
            'count':sum(counts.values()),'per_type':{k:{'count':counts[k],'correct':correct[k],
            'FPA':correct[k]/counts[k] if counts[k] else None} for k in ('type1','type1a','type1b')}}


def train(name,real_fraction):
    prepared = read(REPORT/'training/prepared.json')
    review = read(REPORT/'training/input_review.json')
    if review.get('status') != 'PASS_WITH_PREPROCESS_ELIGIBILITY_FILTER' or review.get('prepared_sha256') != file_sha256(REPORT/'training/prepared.json'):
        raise ValueError('Prepared input visual review missing or stale')
    if file_sha256(REPORT/'training/arrays.npz') != prepared['tensor_sha256']:
        raise ValueError('Training tensors changed')
    guard_check()
    seed_all(SEED)
    arrays = np.load(REPORT/'training/arrays.npz',allow_pickle=False)['inputs']
    records,excluded = training_view(prepared['records'])
    write(REPORT/'training/training_view_eligibility.json',{'excluded_from_optimization_only':excluded,
          'counts':dict(Counter(r['domain'] for r in records)),
          'original_membership_GT_unchanged':True,'val_excluded':0})
    model,converter,h,w = load_crnn_checkpoint(str(D_WEIGHTS),'cuda')
    if (h,w)!=(32,160):
        raise ValueError('Checkpoint representation mismatch')
    ocr = OCRG.__new__(OCRG)
    ocr.torch=torch
    ocr.device='cuda'
    ocr.model=model
    ocr.converter=converter
    spec = {'name':name,'real_fraction':real_fraction,'epochs_max':12,'patience':3,
        'samples_per_epoch':1536,'batch_size':32,'lr':.0001,'weight_decay':.0001,'clip_grad':5.,
        'seed':SEED,'optimizer':'AdamW','ctc_on_cpu':True,'amp':False,
        'initial_checkpoint':str(D_WEIGHTS),'initial_sha256':file_sha256(D_WEIGHTS),
        'data_sha256':file_sha256(REPORT/'training/prepared.json'),'split_sha256':file_sha256(SPLIT),
        'augmentation':'brightness +/-8, contrast .92..1.08, blur sigma .2..5, JPEG75..95, noise1.5, perspective <=.2pixel',
        'selection':'Real val raw OCR FPA; ties retain earlier checkpoint',
        'max_training_seconds':1200,'class_sampling':'Uniform type, then uniform sample; uniform candidate within sample',
        'inference':prepared['preprocessing']}
    write(CONFIG/(name+'.json'),spec)
    folder = MODELS/name
    folder.mkdir(parents=True,exist_ok=False)
    optimizer = torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay'])
    criterion = torch.nn.CTCLoss(blank=converter.blank_idx,reduction='mean',zero_infinity=False)
    groups = {(domain,kind):[r for r in records if r['domain']==domain and r['type']==kind]
              for domain in ('real_train','synthetic_train') for kind in ('type1','type1a','type1b')}
    for pool in groups.values():
        if not pool:
            raise ValueError('Empty type/domain training stratum')
    initial = ocr_metrics(ocr,records,arrays,'real_val')
    synthetic_initial = ocr_metrics(ocr,records,arrays,'synthetic_control')
    write(REPORT/'training/H_OCR_0_D_unchanged.json',{'real_val':initial,'synthetic_control':synthetic_initial})
    history=[]
    best_correct=initial['correct']
    stale=0
    start=time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    # Preserve the unchanged checkpoint as a legitimate no-improvement outcome.
    def save_checkpoint(path,epoch,metrics):
        state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        torch.save({'model':state,'alphabet':converter.alphabet,
            'config':{'image_height':32,'image_width':160,'hidden_size':256,'lstm_layers':2},
            'phase_h':spec,'epoch':epoch,'selection_metrics':metrics,
            'optimizer':optimizer.state_dict(),'rng_state':torch.get_rng_state(),
            'cuda_rng_state':torch.cuda.get_rng_state_all()},path)
    save_checkpoint(folder/'best.pt',0,initial)
    for epoch in range(1,spec['epochs_max']+1):
        rng=np.random.default_rng(SEED+epoch)
        model.train()
        losses=[]
        for _ in range(spec['samples_per_epoch']//spec['batch_size']):
            batch=[]
            inputs=[]
            for j in range(spec['batch_size']):
                domain='real_train' if rng.random()<real_fraction else 'synthetic_train'
                kind=('type1','type1a','type1b')[int(rng.integers(3))]
                pool=groups[(domain,kind)]
                r=pool[int(rng.integers(len(pool)))]
                batch.append(r)
                inputs.append(augment(arrays[r['arrays'][int(rng.integers(len(r['arrays'])))]],rng))
            x=torch.from_numpy(np.stack(inputs)).unsqueeze(1).cuda()
            targets,lengths=converter.encode([r['text'] for r in batch])
            optimizer.zero_grad(set_to_none=True)
            logits=model(x)
            t=logits.shape[0]
            if any(t<len(r['text'])+sum(a==b for a,b in zip(r['text'],r['text'][1:])) for r in batch):
                raise ValueError('CTC alignment impossible')
            loss=criterion(logits.log_softmax(2).cpu(),targets,
                torch.full((len(batch),),t,dtype=torch.long),lengths)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite CTC loss')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.)
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        val=ocr_metrics(ocr,records,arrays,'real_val')
        train_scores=ocr_metrics(ocr,records,arrays,'real_train')
        synth=ocr_metrics(ocr,records,arrays,'synthetic_control')
        improved=val['correct']>best_correct
        if improved:
            best_correct=val['correct']
            stale=0
            save_checkpoint(folder/'best.pt',epoch,val)
        else:
            stale+=1
        record={'epoch':epoch,'loss':float(np.mean(losses)),'real_train':train_scores,
            'real_val':val,'synthetic_control':synth,'improved':improved,'stale_epochs':stale,
            'elapsed_seconds':time.perf_counter()-start,
            'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2}
        history.append(record)
        write(REPORT/'training'/name/'history.json',history)
        save_checkpoint(folder/'last.pt',epoch,val)
        print(name,'epoch',epoch,'train',train_scores['FPA'],'val',val['FPA'],'loss',record['loss'],flush=True)
        if stale>=spec['patience'] or record['elapsed_seconds']>spec['max_training_seconds']:
            break
    summary={'name':name,'spec':spec,'initial_val':initial,'initial_synthetic_control':synthetic_initial,
        'history':history,'best_correct':best_correct,'best_checkpoint':str(folder/'best.pt'),
        'best_sha256':file_sha256(folder/'best.pt'),'last_sha256':file_sha256(folder/'last.pt'),
        'stop_reason':'early_stopping' if stale>=spec['patience'] else 'time_limit' if history[-1]['elapsed_seconds']>spec['max_training_seconds'] else 'epochs_max'}
    write(REPORT/'training'/name/'summary.json',summary)
    guard_check()
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--prepare',action='store_true')
    parser.add_argument('--train',choices=['H_OCR_1_real_heavy','H_OCR_2_balanced'])
    args=parser.parse_args()
    torch.set_num_threads(4)
    cv2.setNumThreads(1)
    with offline_network():
        if args.prepare:
            prepare()
        if args.train:
            train(args.train,.75 if args.train=='H_OCR_1_real_heavy' else .5)
