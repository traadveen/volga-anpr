"""Small real-domain OCR tournament; detector and architecture remain frozen."""
from pathlib import Path
import os,sys
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,random,time,hashlib,argparse
from collections import Counter
from dataclasses import asdict
import cv2,numpy as np,torch
from tools.phase_g_common import ROOT,REPORT,CONFIG,DATA,IMAGES,SEED,read,write,freeze
from tools.phase_e import load_image
from tools.phase_g_experiments import cached_detections,evaluate_rows
from src.geometry_f import canonical_rectify
from src.image_anpr_g import OCRG,ImagePipelineG,gray_for_ocr
from src.multiline_g import ConfigG,prepare_g
from src.real_data_g import mixed_indices,valid_complete_plate
from ocr.crnn import load_crnn_checkpoint
from dataset.schema import sha256

SYNTHETIC=ROOT.parent/'generated_data/synthetic_d_large_v1'

def eligible(r):
    return r['full']['per_type']['type1']['full_match']>=11 and r['full']['per_type']['type1a']['full_match']>=40 and r['full']['full_match_total']>=51

def prepare_training(config):
    train=read(DATA/'real_train/annotations.json');real=[];excluded=[]
    for r in train:
        if not r['quad'] or not r['identity_reviewed'] or not valid_complete_plate(r.get('visible_plate_num') or ''):continue
        if r['image'] in ('img_018.jpg','img_030.jpg') and r['plate_type']=='type1a':excluded.append((r['image'],'source degradation or approximate quad clips top strokes'));continue
        crop,_=canonical_rectify(load_image(DATA/'real_train/images'/r['image']),r['quad'],r['plate_type'],'verified_train_manual_quad')
        gray=gray_for_ocr(crop);n,image,diag=prepare_g(gray,r['plate_type'],ConfigG(**config['split']))
        if r['plate_type']=='type1a' and diag['effective_mode']!='split_and_stitch':excluded.append((r['image'],'unsafe split'));continue
        real.append({'normalized':n,'label':r['visible_plate_num'],'type':r['plate_type'],'image':r['image'],
            'tensor_sha256':hashlib.sha256(n.tobytes()).hexdigest()})
    forbidden={r['plate_num'] for r in read(DATA/'real_val/annotations.json') if valid_complete_plate(r['plate_num'])}
    contract=read(SYNTHETIC/'generator/ocr_inputs.json');index={n:i for i,n in enumerate(contract['image_order'])}
    print('Verifying frozen D replay cache bytes',flush=True)
    assert sha256(SYNTHETIC/'generator/ocr_inputs.npy')==contract['cache_sha256']
    assert sha256(SYNTHETIC/'meta.csv')==contract['meta_sha256']
    cache=np.load(SYNTHETIC/'generator/ocr_inputs.npy',mmap_mode='r',allow_pickle=False)
    synthetic=[];counts=Counter();rng=random.Random(SEED)
    rows=list(csv.DictReader((SYNTHETIC/'splits/train.csv').open(encoding='utf-8-sig'),delimiter=';'));rng.shuffle(rows)
    for r in rows:
        kind=r['plate_type']
        if counts[kind]>=400 or r['plate_num'] in forbidden:continue
        # Selected raw source bytes and actual tensors bind replay provenance.
        n=np.array(cache[index[r['image']],0],copy=True)
        synthetic.append({'normalized':n,'label':r['plate_num'],'type':kind,'image':r['image'],'source_sha256':sha256(SYNTHETIC/r['image']),
            'tensor_sha256':hashlib.sha256(n.tobytes()).hexdigest()});counts[kind]+=1
        if len(synthetic)==1200:break
    assert len(synthetic)==1200 and len(real)>=15
    manifest={'real_count':len(real),'synthetic_count':len(synthetic),'synthetic_types':dict(counts),'excluded_real':excluded,
        'forbidden_val_identity_count':len(forbidden),'real_records':[{k:v for k,v in r.items() if k!='normalized'} for r in real],
        'synthetic_records':[{k:v for k,v in r.items() if k!='normalized'} for r in synthetic],
        'synthetic_source':'Frozen D TRAIN only; no synthetic val/test replay; real_val identities filtered',
        'real_source':'Only supervised real_train quad/type/identity, automatic label-free G split; no oracle manual split supplied',
        'provenance':{'real_manifest_sha256':sha256(DATA/'manifest.json'),'D_cache_contract_sha256':sha256(SYNTHETIC/'generator/ocr_inputs.json')},'preprocessing':config}
    write(REPORT/'training/replay_manifest.json',manifest)
    from src.image_anpr import save_image
    from tools.phase_g_audit import FONT,paste
    from PIL import Image,ImageDraw
    paths=[]
    for i,r in enumerate(real):
        path=REPORT/'training/real_inputs'/f'{i:02}_{r["label"]}.png';save_image(path,np.clip((r['normalized']+1)*127.5,0,255).astype(np.uint8));paths.append(path)
    for page in range((len(real)+7)//8):
        canvas=Image.new('RGB',(1200,800),'white');draw=ImageDraw.Draw(canvas)
        for j,(r,path) in enumerate(zip(real[page*8:(page+1)*8],paths[page*8:(page+1)*8])):
            x=j%2*600;y=j//2*200;draw.text((x,y),r['image']+' '+r['label']+' '+r['type'],font=FONT,fill='black');paste(canvas,path,(x,y+35,580,116))
        canvas.save(REPORT/'training'/f'real_review_{page+1:02}.jpg',quality=95)
    np.savez_compressed(REPORT/'training/selected_tensors.npz',real=np.stack([r['normalized'] for r in real]),synthetic=np.stack([r['normalized'] for r in synthetic]))
    manifest['selected_tensor_sha256']=sha256(REPORT/'training/selected_tensors.npz');write(REPORT/'training/replay_manifest.json',manifest)
    return real,synthetic

def evaluate_checkpoint(ocr,config,frames,cache,folder):
    pipeline=ImagePipelineG(None,ocr,ConfigG(**config['split']),multi=config['multi'],postprocess=config['postprocess'],expansion=config['expansion'])
    predictions=[];traces=[]
    for name,frame in frames.items():
        r=pipeline.process(frame,image_name=name,detections=cache[name]);traces.append(r);predictions.extend(r['predictions'])
    result=evaluate_rows(predictions,folder);write(folder/'trace.json',{'records':traces});return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare-only',action='store_true');args=parser.parse_args()
    print('Loading bounded real OCR fine-tuning runtime',flush=True)
    gate=read(REPORT/'training_gate.json');assert gate['state']=='AUTHORIZED_BY_MEASURED_GAP_AFTER_DIAGNOSTIC_GATES'
    baseline=freeze()['baseline'];torch.set_num_threads(4);cv2.setNumThreads(1);torch.manual_seed(SEED);np.random.seed(SEED);random.seed(SEED)
    pool=read(REPORT/'split_tournament.json')+read(REPORT/'followup_tournament.json')
    best=max([r for r in pool if eligible(r)],key=lambda r:(r['real_val_exposed']['full_match_total'],-r['cached_prepared_OCR_latency']['mean']))
    cfg=best['config'];write(REPORT/'training/inference_control.json',best)
    if args.prepare_only:
        real,synthetic=prepare_training(cfg)
        print('Prepared verified train inputs',len(real),'real and',len(synthetic),'synthetic; visual review required before training',flush=True);return
    review=read(REPORT/'training/real_input_review.json');assert review['status']=='PASS'
    assert review['manifest_sha256']==sha256(REPORT/'training/replay_manifest.json')
    manifest=read(REPORT/'training/replay_manifest.json');assert manifest['preprocessing']==cfg
    assert sha256(REPORT/'training/selected_tensors.npz')==manifest['selected_tensor_sha256']
    arrays=np.load(REPORT/'training/selected_tensors.npz',allow_pickle=False)
    real=[dict(r,normalized=n) for r,n in zip(manifest['real_records'],arrays['real'])]
    synthetic=[dict(r,normalized=n) for r,n in zip(manifest['synthetic_records'],arrays['synthetic'])]
    frames={p.name:load_image(p) for p in sorted(IMAGES.glob('*.jpg'))};cache=cached_detections();results=[]
    init_model,converter,_,_=load_crnn_checkpoint(baseline['ocr'],'cpu')
    original={k:v.clone() for k,v in init_model.state_dict().items()};alphabet=converter.alphabet
    criterion=torch.nn.CTCLoss(blank=converter.blank_idx,reduction='mean',zero_infinity=False)
    for name,ratio in (('real_heavy',.75),('balanced',.5),('synthetic_heavy',.25)):
        torch.manual_seed(SEED);rng=np.random.default_rng(SEED);model=init_model.to('cuda');model.load_state_dict(original)
        assert all(torch.equal(v.detach().cpu(),original[k]) for k,v in model.state_dict().items())
        spec={'name':'G06_'+name,'real_fraction':ratio,'epochs_max':4,'samples_per_epoch':1024,'batch_size':32,
            'lr':.0001,'weight_decay':.0001,'seed':SEED,'initial_checkpoint':baseline['ocr'],'initial_sha256':sha256(Path(baseline['ocr'])),
            'initial_state_exact':True,'inference':cfg,'train_replay_manifest_sha256':sha256(REPORT/'training/replay_manifest.json'),
            'stop_seconds_per_run':180,'selection':'Exposed real_val E2E, type1>=11/15, type1a>=40/100, no architecture change'}
        write(CONFIG/(spec['name']+'.json'),spec);folder=ROOT/'models/phase_g'/spec['name'];folder.mkdir(parents=True,exist_ok=True)
        optimizer=torch.optim.AdamW(model.parameters(),lr=spec['lr'],weight_decay=spec['weight_decay']);start=time.perf_counter();history=[];best_key=(-1,-1);best_path=None
        torch.cuda.reset_peak_memory_stats()
        for epoch in range(1,spec['epochs_max']+1):
            model.train();mix=mixed_indices(len(real),len(synthetic),1024,ratio,SEED+epoch);losses=[]
            for offset in range(0,len(mix),32):
                batch=[(real if domain=='real' else synthetic)[i] for domain,i in mix[offset:offset+32]]
                arrays=[]
                for r in batch:
                    n=r['normalized'].copy()
                    # Mild photometric changes only; no shape, glyph or label edits.
                    n=np.clip((n+1)*float(rng.uniform(.88,1.12))-1+float(rng.uniform(-.05,.05)),-1,1).astype(np.float32);arrays.append(n)
                x=torch.from_numpy(np.stack(arrays)).unsqueeze(1).to('cuda');targets,lengths=converter.encode([r['label'] for r in batch])
                optimizer.zero_grad(set_to_none=True);logits=model(x);t=logits.shape[0]
                assert all(t>=len(r['label'])+sum(a==b for a,b in zip(r['label'],r['label'][1:])) for r in batch)
                loss=criterion(logits.log_softmax(2).cpu(),targets,torch.full((len(batch),),t,dtype=torch.long),lengths)
                if not torch.isfinite(loss):raise ValueError('Nonfinite CTC loss')
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
                if torch.cuda.max_memory_allocated()>4096*1024**2:raise RuntimeError('4GiB allocation budget exceeded')
            model.eval();ocr=OCRG.__new__(OCRG);ocr.torch=torch;ocr.device='cuda';ocr.model=model;ocr.converter=converter
            result=evaluate_checkpoint(ocr,cfg,frames,cache,REPORT/'training'/spec['name']/f'epoch_{epoch}')
            key=(result['real_val_exposed']['full_match_total'],result['full']['full_match_total']) if eligible(result) else (-1,-1)
            record={'epoch':epoch,'loss_mean':float(np.mean(losses)),'real_val_correct':result['real_val_exposed']['full_match_total'],
                'full_correct':result['full']['full_match_total'],'type1':result['full']['per_type']['type1']['full_match'],
                'eligible':eligible(result),'elapsed_seconds':time.perf_counter()-start,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2}
            history.append(record);print(spec['name'],record,flush=True)
            if key>best_key:
                best_key=key;best_path=folder/'best.pt';state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
                torch.save({'model':state,'alphabet':alphabet,'config':{'image_height':32,'image_width':160,'hidden_size':256,'lstm_layers':2},
                    'phase_g':spec,'epoch':epoch,'selection_metrics':record},best_path)
                write(folder/'best_result.json',result)
            write(REPORT/'training'/spec['name']/'history.json',history)
            if time.perf_counter()-start>spec['stop_seconds_per_run']:break
        results.append({'name':spec['name'],'history':history,'best_checkpoint':str(best_path) if best_path else None,'best_key':best_key,'control_val_correct':best['real_val_exposed']['full_match_total']})
        # The next ratio starts from the exact same frozen D weights, not this run.
        optimizer=None;model.to('cpu');torch.cuda.empty_cache()
    write(REPORT/'training/tournament.json',results)
    print('Three-ratio bounded fine-tuning finished',flush=True)

if __name__=='__main__':main()
