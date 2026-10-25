"""Strict CRNN training with complete, weights-only-compatible checkpoints."""
from __future__ import annotations
from collections import Counter
from dataclasses import asdict,dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader,Dataset
import yaml

from dataset.schema import read_metadata,plate_error,resolve_image,sha256
from dataset.validation import validate_dataset
from ocr.crnn import DEFAULT_ALPHABET,CTCLabelConverter,create_crnn_model
from src.multiline import MultilineConfig
from src.ocr_preprocess import prepare_ocr_input
from tools.benchmark_multiline import distance


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: str = 'D:/volga/generated_data/synthetic_b_v3'
    train_split: str = 'splits/train.csv'
    validation_split: str = 'splits/val.csv'
    output_dir: str = 'models/phase_d/smoke'
    seed: int = 42
    alphabet: str = DEFAULT_ALPHABET
    image_height: int = 32
    image_width: int = 160
    hidden_size: int = 256
    lstm_layers: int = 2
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = .0001
    weight_decay: float = .0001
    clip_grad: float = 5.
    scheduler_step_size: int = 10
    scheduler_gamma: float = .5
    patience: int = 0
    min_delta: float = 0.
    device: str = 'auto'
    threads: int = 4
    num_workers: int = 0
    deterministic: bool = True
    amp: bool = False
    ctc_on_cpu: bool = True
    augmentation_brightness: tuple = (1.,1.)
    augmentation_contrast: tuple = (1.,1.)
    multiline_config: str = 'configs/multiline_v2.json'
    cache_images: bool = True
    clean_only: bool = False
    samples_per_type: int = 0
    sanity_same_split: bool = False
    sanity_target_accuracy: float = .98
    max_training_seconds: float = 180.
    init_weights: str | None = 'models/ocr_crnn_nomeroff_clean.pt'
    normalized_cache: str | None = None
    balanced_sampling: bool = False
    difficulty_levels: tuple = ()

    def __post_init__(self):
        if self.alphabet!=DEFAULT_ALPHABET:raise ValueError('Alphabet/order must match frozen inference checkpoint')
        if (self.image_height,self.image_width,self.hidden_size,self.lstm_layers)!=(32,160,256,2):
            raise ValueError('This phase preserves the existing CRNN architecture/input dimensions')
        for name in ('epochs','batch_size','threads','scheduler_step_size'):
            if type(getattr(self,name)) is not int or getattr(self,name)<=0:raise ValueError(f'Invalid {name}')
        for name in ('patience','num_workers','samples_per_type'):
            if type(getattr(self,name)) is not int or getattr(self,name)<0:raise ValueError(f'Invalid {name}')
        if self.device not in ('auto','cpu','cuda'):raise ValueError('Invalid device')
        if self.normalized_cache and (list(self.augmentation_brightness)!=[1.,1.] or list(self.augmentation_contrast)!=[1.,1.]):
            raise ValueError('Normalized cache requires no additional stochastic photometric augmentation')
        if any(t not in ('clean','medium','hard') for t in self.difficulty_levels):raise ValueError('Unknown difficulty level')
        if self.sanity_same_split and self.train_split!=self.validation_split:raise ValueError('Sanity same-split flag requires the same metadata path')
        for name in ('learning_rate','clip_grad','scheduler_gamma','max_training_seconds'):
            v=getattr(self,name)
            if not math.isfinite(v) or v<=0:raise ValueError(f'Invalid {name}')
        if not 0<=self.weight_decay or not 0<=self.min_delta or not 0<self.sanity_target_accuracy<=1:raise ValueError('Invalid regularization/metric setting')
        for values in (self.augmentation_brightness,self.augmentation_contrast):
            if len(values)!=2 or not 0<values[0]<=values[1] or not all(math.isfinite(x) for x in values):raise ValueError('Invalid augmentation range')

    @classmethod
    def load(cls,path):return cls(**yaml.safe_load(Path(path).read_text(encoding='utf-8')))


def seed_everything(seed,deterministic=True):
    os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(deterministic)
    torch.backends.cudnn.deterministic=deterministic
    torch.backends.cudnn.benchmark=False
    cv2.setNumThreads(1)


def seed_worker(_):
    seed=torch.initial_seed()%2**32
    np.random.seed(seed);random.seed(seed)


def capture_rng():
    numpy_state=np.random.get_state()
    return {'python':random.getstate(),'numpy':[numpy_state[0],numpy_state[1].tolist(),
            int(numpy_state[2]),int(numpy_state[3]),float(numpy_state[4])],
            'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']);n=state['numpy']
    np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),n[2],n[3],n[4]))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():torch.cuda.set_rng_state_all(state['cuda'])


def validate_label(text,converter):
    error=plate_error(text)
    if error or '#' in text:raise ValueError(f'Invalid complete training label {text!r}: {error or "unknown character"}')
    target,length=converter.encode([text])
    if int(length[0])!=len(text) or len(target)!=len(text) or torch.any(target==converter.blank_idx):raise ValueError('Invalid encoded target/blank')
    minimum=len(text)+sum(a==b for a,b in zip(text,text[1:]))
    return target,minimum


class OCRDatasetV2(Dataset):
    def __init__(self,config,split,training=False,verified_images=None):
        self.root=Path(config.dataset_root).resolve();self.config=config;self.training=training
        self.converter=CTCLabelConverter(config.alphabet)
        self.multiline=MultilineConfig.load(config.multiline_config)
        if any(word in str(self.root).lower() for word in ('organizer_debug','reports/phase_c','reports\\phase_c')):
            raise ValueError('Evaluation-only public data cannot be used for training')
        rows,issues=read_metadata(self.root/split)
        if issues:raise ValueError(f'Invalid labels before training: {issues}')
        manifest=json.loads((self.root/'generator/manifest.json').read_text(encoding='utf-8'))
        self.samples={r['image']:r for r in manifest['samples']}
        if training and config.difficulty_levels:
            rows=[r for r in rows if self.samples[r.image].get('difficulty') in config.difficulty_levels]
        if config.clean_only:rows=[r for r in rows if self.samples[r.image]['clean']]
        if config.samples_per_type:
            rng=random.Random(config.seed);chosen=[]
            for kind in ('type1','type1a','type1b'):
                group=sorted((r for r in rows if r.plate_type==kind),key=lambda r:r.image);rng.shuffle(group)
                if len(group)<config.samples_per_type:raise ValueError(f'Not enough {kind} samples')
                chosen.extend(group[:config.samples_per_type])
            rows=sorted(chosen,key=lambda r:r.image)
        if not rows:raise ValueError('Empty split')
        self.rows=rows;self.images={};self.minimum_ctc=[]
        for row in rows:
            if row.plate_type not in ('type1','type1a','type1b'):raise ValueError('OCR target type required')
            _,minimum=validate_label(row.plate_num,self.converter);self.minimum_ctc.append(minimum)
            path=resolve_image(self.root,row.image);sample=self.samples[row.image]
            if config.normalized_cache and verified_images is not None:
                # The caller has just checked ALL current raw bytes. The cache
                # contract below independently binds the actual training tensors.
                if verified_images[row.image]['sha256']!=sample['image_sha256']:
                    raise ValueError(f'Image provenance changed: {row.image}')
            elif sha256(path)!=sample['image_sha256']:raise ValueError(f'Image changed: {row.image}')
            if sample['plate_type']!=row.plate_type:raise ValueError('Layout provenance mismatch')
            # A frozen normalized cache is validated below. Raw bytes are still
            # hash-checked above; decoding them again is unnecessary for training.
            if not config.normalized_cache or config.cache_images:
                with Image.open(path) as im:image=np.asarray(im.convert('RGB'))[:,:,::-1].copy()
                if config.cache_images:self.images[row.image]=image
        # Identity includes actual selected images/labels/geometry/config and bytes.
        entries=[(r.image,r.plate_num,r.plate_type,self.samples[r.image]) for r in rows]
        self.fingerprint=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest()
        self.normalized=None;self.cache_index={}
        if config.normalized_cache:
            cache=Path(config.normalized_cache)
            contract=json.loads(cache.with_suffix('.json').read_text(encoding='utf-8'))
            if contract['meta_sha256']!=sha256(self.root/'meta.csv') or contract['cache_sha256']!=sha256(cache):
                raise ValueError('Normalized cache dataset/pixels changed')
            for path,digest in contract['preprocessing_hashes'].items():
                if sha256(Path(path))!=digest:raise ValueError('Normalized cache preprocessing changed')
            self.normalized=np.load(cache,mmap_mode='r',allow_pickle=False)
            if self.normalized.shape!=(len(contract['image_order']),1,32,160):raise ValueError('Cache shape mismatch')
            self.cache_index={name:i for i,name in enumerate(contract['image_order'])}
            if len(self.cache_index)!=len(contract['image_order']) or any(r.image not in self.cache_index for r in rows):
                raise ValueError('Cache image population mismatch')
            self.fingerprint=hashlib.sha256((self.fingerprint+json.dumps(contract,sort_keys=True)).encode()).hexdigest()

    def __len__(self):return len(self.rows)

    def input_for(self,index,augment=False):
        row=self.rows[index];sample=self.samples[row.image]
        if row.image in self.images:image=self.images[row.image].copy()
        else:
            with Image.open(resolve_image(self.root,row.image)) as im:image=np.asarray(im.convert('RGB'))[:,:,::-1].copy()
        if augment:
            brightness=np.random.uniform(*self.config.augmentation_brightness)
            contrast=np.random.uniform(*self.config.augmentation_contrast)
            image=np.clip((image.astype(float)-127.5)*contrast+127.5,0,255)*brightness
            image=np.clip(image,0,255).astype(np.uint8)
        with Image.open(resolve_image(self.root,sample['canonical'])) as im:canonical_size=im.size
        prepared=prepare_ocr_input(image,row.plate_type,homography=sample['homography'],canonical_size=canonical_size,
                                   multiline_config=self.multiline,image_height=self.config.image_height,image_width=self.config.image_width)
        if row.plate_type=='type1a' and prepared.diagnostics['effective_mode']!='split_and_stitch':
            raise ValueError(f'type1a OCR training requires stitch; unreliable geometry: {row.image}')
        return prepared

    def __getitem__(self,index):
        row=self.rows[index]
        if self.normalized is not None:
            pixels=self.normalized[self.cache_index[row.image]].copy()
            return torch.from_numpy(pixels),row.plate_num,row.plate_type,row.image
        prepared=self.input_for(index,self.training)
        return torch.from_numpy(prepared.normalized).unsqueeze(0),row.plate_num,row.plate_type,row.image


def collate(batch):
    images,texts,types,ids=zip(*batch)
    converter=CTCLabelConverter(DEFAULT_ALPHABET)
    targets,lengths=converter.encode(texts)
    return torch.stack(images),targets,lengths,list(texts),list(types),list(ids)


def checked_ctc(logits,targets,lengths,texts,converter,ctc_on_cpu=True):
    if logits.ndim!=3 or logits.shape[1]!=len(texts) or logits.shape[2]!=len(converter.alphabet)+1:
        raise ValueError('CTC logits/alphabet/batch mismatch')
    if len(set(converter.alphabet))!=len(converter.alphabet) or converter.blank_idx!=len(converter.alphabet):raise ValueError('Invalid alphabet/blank')
    expected,expected_lengths=converter.encode(texts)
    if not torch.equal(expected,targets.cpu()) or not torch.equal(expected_lengths,lengths.cpu()):raise ValueError('CTC targets or target lengths mismatch')
    for text in texts:
        _,minimum=validate_label(text,converter)
        if minimum>logits.shape[0]:raise ValueError('Impossible CTC alignment including adjacent repeats')
    if not torch.isfinite(logits).all():raise FloatingPointError('Nonfinite logits')
    probabilities=logits.float().log_softmax(2)
    if ctc_on_cpu:probabilities=probabilities.cpu();targets=targets.cpu();lengths=lengths.cpu()
    else:targets=targets.to(logits.device);lengths=lengths.to(logits.device)
    input_lengths=torch.full((len(texts),),logits.shape[0],dtype=torch.long,device=probabilities.device)
    loss=torch.nn.CTCLoss(blank=converter.blank_idx,zero_infinity=False)(probabilities,targets,input_lengths,lengths)
    if not torch.isfinite(loss):raise FloatingPointError('Nonfinite CTC loss; never hidden with zero_infinity')
    return loss


def evaluate(model,loader,converter,device,ctc_on_cpu=True):
    model.eval();loss_sum=0.;cases=[]
    with torch.inference_mode():
        for images,targets,lengths,texts,types,ids in loader:
            logits=model(images.to(device))
            loss=checked_ctc(logits,targets,lengths,texts,converter,ctc_on_cpu)
            loss_sum+=float(loss)*len(texts)
            predictions,confidence,probs=converter.decode_logits(logits)
            raw=logits.argmax(2).T.cpu().tolist()
            for gt,pred,conf,kind,id_,prob,tokens in zip(texts,predictions,confidence,types,ids,probs,raw):
                cases.append({'image':id_,'ground_truth':gt,'prediction':pred,'confidence':conf,
                              'plate_type':kind,'char_probs':prob,'raw_ctc_tokens':tokens})
    count=len(cases);chars=sum(len(r['ground_truth']) for r in cases)
    cer=sum(distance(r['ground_truth'],r['prediction']) for r in cases)/chars
    return {'count':count,'loss':loss_sum/count,'full_plate_accuracy':sum(r['ground_truth']==r['prediction'] for r in cases)/count,
            'cer':cer,'character_accuracy':1-cer,'by_type':{t:sum(r['ground_truth']==r['prediction'] for r in cases if r['plate_type']==t)/sum(r['plate_type']==t for r in cases)
                     for t in sorted({r['plate_type'] for r in cases})}},cases


def atomic_save(checkpoint,path):
    path=Path(path);temporary=path.with_suffix('.tmp')
    torch.save(checkpoint,temporary);os.replace(temporary,path)


def train(config:TrainConfig,resume=None,stop_after_epoch=None):
    seed_everything(config.seed,config.deterministic);torch.set_num_threads(config.threads)
    device=torch.device('cuda' if config.device=='auto' and torch.cuda.is_available() else 'cpu' if config.device=='auto' else config.device)
    if device.type=='cuda' and not torch.cuda.is_available():raise RuntimeError('CUDA requested but unavailable; no silent fallback')
    root=Path(config.dataset_root)
    provenance=json.loads((root/'generator/manifest.json').read_text(encoding='utf-8'))
    frozen_images=None
    if provenance.get('generator_version')=='phase-d-large-v1':
        gate_path=root/'generator/acceptance.json'
        if not gate_path.exists():raise ValueError('Large dataset visual/automatic acceptance gate not passed')
        gate=json.loads(gate_path.read_text(encoding='utf-8'))
        if gate.get('meta_sha256')!=sha256(root/'meta.csv') or not all(gate.get('gates',{}).get(k) is True for k in ('automatic_all','visual_100_per_type','type1a_stitch')):
            raise ValueError('Large dataset acceptance is stale or incomplete')
        if gate.get('split_manifest_sha256')!=sha256(root/'splits/manifest.json'):
            raise ValueError('Frozen split/pixel manifest changed after review')
        frozen_images=json.loads((root/'splits/manifest.json').read_text(encoding='utf-8'))['image_manifest']
    validated=validate_dataset(root,frozen_images=frozen_images)
    if not validated['valid']:raise ValueError(validated['errors'])
    train_data=OCRDatasetV2(config,config.train_split,True,verified_images=validated['images'])
    val_data=OCRDatasetV2(config,config.validation_split,False,verified_images=validated['images'])
    if not config.sanity_same_split and ({r.plate_num for r in train_data.rows}&{r.plate_num for r in val_data.rows}):raise ValueError('Train/validation plate identity leakage')
    output=Path(config.output_dir)
    if output.resolve().is_relative_to(Path(config.dataset_root).resolve()):raise ValueError('Checkpoint output must be outside immutable dataset')
    if not resume and (output/'last.pt').exists():raise ValueError('Output exists; use resume or a new experiment directory')
    output.mkdir(parents=True,exist_ok=True)
    generator=torch.Generator().manual_seed(config.seed)
    val_generator=torch.Generator().manual_seed(config.seed+1)
    options={'batch_size':config.batch_size,'num_workers':config.num_workers,'collate_fn':collate,
             'worker_init_fn':seed_worker,'persistent_workers':False,'pin_memory':device.type=='cuda'}
    sampler=None
    if config.balanced_sampling:
        from torch.utils.data import WeightedRandomSampler
        counts=Counter(r.plate_type for r in train_data.rows)
        sampler=WeightedRandomSampler([1/counts[r.plate_type] for r in train_data.rows],len(train_data),replacement=True,generator=generator)
    train_loader=DataLoader(train_data,shuffle=sampler is None,sampler=sampler,generator=generator,**options)
    val_loader=DataLoader(val_data,shuffle=False,generator=val_generator,**options)
    converter=CTCLabelConverter(config.alphabet)
    model=create_crnn_model(config.alphabet,config.hidden_size,config.lstm_layers).to(device)
    model.eval()  # Shape probing must not update random-init BatchNorm statistics.
    with torch.inference_mode():shape=model(torch.zeros(1,1,config.image_height,config.image_width,device=device)).shape
    if max(train_data.minimum_ctc+val_data.minimum_ctc)>shape[0] or shape[2]!=converter.blank_idx+1:raise ValueError('Dataset/CTC shape incompatible')
    optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
    scheduler=torch.optim.lr_scheduler.StepLR(optimizer,step_size=config.scheduler_step_size,gamma=config.scheduler_gamma)
    scaler=torch.amp.GradScaler('cuda',enabled=config.amp and device.type=='cuda')
    start_epoch=1;best=None;best_epoch=None;stale=0;history=[];resume_evidence=None
    data_hashes={'train':train_data.fingerprint,'validation':val_data.fingerprint,'multiline':sha256(Path(config.multiline_config))}
    if resume:
        checkpoint=torch.load(resume,map_location='cpu',weights_only=True)
        required=('model_state_dict','optimizer_state_dict','scheduler_state_dict','epoch','best_metric','alphabet','config','seed','rng_state','dataloader_rng','scaler_state_dict','data_hashes')
        if any(key not in checkpoint for key in required):raise ValueError('Not a full V2 resume checkpoint; use init_weights for legacy weights')
        if checkpoint['data_hashes']!=data_hashes:raise ValueError('Resume dataset/preprocessing changed')
        for key,value in asdict(config).items():
            saved_value=checkpoint['config'].get(key,getattr(TrainConfig(),key))
            if key not in ('epochs','max_training_seconds') and saved_value!=value:
                # YAML lists and dataclass tuple defaults are equivalent.
                if not isinstance(value,(tuple,list)) or list(saved_value)!=list(value):raise ValueError(f'Resume config changed: {key}')
        if checkpoint['alphabet']!=config.alphabet:raise ValueError('Resume alphabet changed')
        model.load_state_dict(checkpoint['model_state_dict']);optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # Optimizer.load_state_dict maps moment tensors to each parameter device
        # and preserves the CPU scalar step for non-capturable AdamW. Moving
        # every state tensor to CUDA changes the uninterrupted optimizer contract.
        scheduler.load_state_dict(checkpoint['scheduler_state_dict']);scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch=checkpoint['epoch']+1;best=checkpoint['best_metric'];best_epoch=checkpoint['best_epoch'];stale=checkpoint['epochs_no_improve'];history=checkpoint['history']
        generator.set_state(checkpoint['dataloader_rng'].cpu());val_generator.set_state(checkpoint['validation_dataloader_rng'].cpu());restore_rng(checkpoint['rng_state'])
        resume_evidence={'loaded_epoch':checkpoint['epoch'],'next_epoch':start_epoch,'optimizer_step_before':max(float(s.get('step',0)) for s in optimizer.state.values()),
                         'scheduler_before':scheduler.state_dict(),'best_metric_restored':best,'rng_restored':True}
        if start_epoch>config.epochs:raise ValueError('Resume total epochs must exceed saved epoch')
    elif config.init_weights:
        checkpoint=torch.load(config.init_weights,map_location='cpu',weights_only=True)
        if checkpoint['alphabet']!=config.alphabet:raise ValueError('Initialization alphabet/order mismatch')
        model.load_state_dict(checkpoint['model'])
    started=time.perf_counter();reason='epochs_completed'
    if device.type=='cuda':torch.cuda.reset_peak_memory_stats()
    for epoch in range(start_epoch,config.epochs+1):
        epoch_start=time.perf_counter();model.train();total=0.;processed=0
        for batch_index,(images,targets,lengths,texts,_,_) in enumerate(train_loader,1):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type,enabled=config.amp and device.type=='cuda'):
                logits=model(images.to(device))
            loss=checked_ctc(logits,targets,lengths,texts,converter,config.ctc_on_cpu)
            scaler.scale(loss).backward();scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(),config.clip_grad,error_if_nonfinite=True)
            scaler.step(optimizer);scaler.update();total+=float(loss.detach())*len(texts)
            processed+=len(texts)
            if batch_index%200==0:
                print(json.dumps({'event':'training_progress','epoch':epoch,'images':processed,
                    'total':len(train_data),'mean_train_loss':total/processed,
                    'images_per_sec':processed/(time.perf_counter()-epoch_start)}),flush=True)
        metrics,cases=evaluate(model,val_loader,converter,device,config.ctc_on_cpu)
        primary_improved=best is None or metrics['full_plate_accuracy']>best['full_plate_accuracy']+config.min_delta
        improved=best is None or (metrics['full_plate_accuracy'],-metrics['cer'])>(best['full_plate_accuracy'],-best['cer'])
        if improved:best=metrics.copy();best_epoch=epoch
        stale=0 if primary_improved else stale+1
        scheduler.step()
        record={'epoch':epoch,'train_loss':total/len(train_data),'validation':metrics,'epoch_seconds':time.perf_counter()-epoch_start,'learning_rate':optimizer.param_groups[0]['lr']}
        history.append(record)
        state=model.state_dict()
        checkpoint={'checkpoint_version':2,'model_state_dict':state,'model':state,'optimizer_state_dict':optimizer.state_dict(),
                    'scheduler_state_dict':scheduler.state_dict(),'scaler_state_dict':scaler.state_dict(),'epoch':epoch,
                    'best_metric':best,'best_epoch':best_epoch,'epochs_no_improve':stale,'alphabet':config.alphabet,'config':asdict(config),
                    'seed':config.seed,'rng_state':capture_rng(),'dataloader_rng':generator.get_state(),'validation_dataloader_rng':val_generator.get_state(),
                    'data_hashes':data_hashes,'history':history,'metrics':metrics,'model_architecture':'original CRNN'}
        atomic_save(checkpoint,output/'last.pt')
        if improved:
            atomic_save(checkpoint,output/'best.pt')
            (output/'best_predictions.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
        print(json.dumps(record),flush=True)
        if stop_after_epoch and epoch>=stop_after_epoch:reason='requested_epoch_stop';break
        if config.sanity_same_split and metrics['full_plate_accuracy']>=config.sanity_target_accuracy:reason='sanity_target_reached';break
        if config.patience and stale>=config.patience:reason='early_stopping';break
        if time.perf_counter()-started>=config.max_training_seconds:reason='time_budget';break
    if resume_evidence:
        resume_evidence.update({'saved_epoch':epoch,'optimizer_step_after':max(float(s.get('step',0)) for s in optimizer.state.values()),
                               'scheduler_after':scheduler.state_dict(),'actually_continued':epoch>=start_epoch})
    report={'device':str(device),'start_epoch':start_epoch,'last_epoch':epoch,'epochs_completed_this_run':epoch-start_epoch+1,
            'dataset_size':{'train':len(train_data),'validation':len(val_data)},'train_types':dict(Counter(r.plate_type for r in train_data.rows)),
            'train_loss':history[-1]['train_loss'],'validation':history[-1]['validation'],'training_time_seconds':time.perf_counter()-started,
            'best_epoch':best_epoch,'best_metric':best,'stop_reason':reason,'resume':resume_evidence,'history':history,
            'data_hashes':data_hashes,'config':asdict(config),'sanity_only':config.sanity_same_split,'checkpoints':{name:sha256(output/name) for name in ('last.pt','best.pt')}}
    report['GPU_memory_usage']={'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
        'peak_reserved_bytes':torch.cuda.max_memory_reserved()} if device.type=='cuda' else None
    report['images_per_sec_including_validation_and_checkpoints']=len(train_data)*(epoch-start_epoch+1)/report['training_time_seconds']
    (output/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
