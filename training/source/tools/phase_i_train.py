"""Four-epoch detector-only pilot, loading the frozen BEST-G/F03 checkpoint exactly."""
from pathlib import Path
import os,sys
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import hashlib
import shutil
import torch
from ultralytics import YOLO
from tools.phase_i_dataset import REPORT,DATA,prepare,verify_selected
from tools.phase_h_common import ROOT,read,now
from tools.phase_i_review import atomic_json
from src.inference_io import file_sha256,offline_network


def train(variant,report=REPORT,data=DATA,name=None):
    report=Path(report)
    manifest=prepare(report,data);verify_selected(manifest)
    name=name or {'real_only':'I_DET_1_type1b_pilot','replay':'I_DET_2_type1b_replay_pilot'}[variant]
    output=report/'training'/f'{name}.json'
    if output.exists():
        saved=read(output)
        if file_sha256(Path(saved['checkpoint']))!=saved['checkpoint_sha256']:raise ValueError('Pilot checkpoint changed')
        return saved
    run=ROOT/'runs/phase_i'/name
    if run.exists():raise ValueError('Existing/incomplete training run must not be overwritten')
    cfg=read(ROOT/'configs/phase_g/best_g.json')
    if file_sha256(Path(cfg['detector']))!=cfg['detector_sha256']:raise ValueError('Frozen initialization changed')
    settings=dict(data=manifest['data_yamls'][variant],epochs=4,imgsz=640,batch=4,nbs=4,
        workers=0,device=0,optimizer='AdamW',lr0=.0003,lrf=.1,weight_decay=.0005,
        warmup_epochs=0,patience=0,seed=20261019,deterministic=True,amp=False,
        freeze=10,mosaic=0.,close_mosaic=0,fliplr=0.,flipud=0.,degrees=5.,scale=.3,
        translate=.1,shear=0.,perspective=0.,mixup=0.,copy_paste=0.,
        plots=False,cache=False,save=True,save_period=-1,val=False,pretrained=True,
        project=str(ROOT/'runs/phase_i'),name=name,exist_ok=False,verbose=False)
    config={'name':name,'variant':variant,'settings':settings,'initial_weights':cfg['detector'],
            'initial_sha256':cfg['detector_sha256'],'dataset_manifest_sha256':file_sha256(report/'dataset_manifest.json'),
            'dataset_manifest':str(report/'dataset_manifest.json'),
            'selection':'Fixed four-epoch last.pt. Crop-val diagnostics never select the candidate.',
            'OCR':'FINAL-H frozen','test_used':False}
    config_path=ROOT/'configs/phase_i'/f'{name}.json'
    atomic_json(config_path,config)
    torch.set_num_threads(4)
    torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
    model=YOLO(cfg['detector'])
    if model.task!='obb' or model.names!={0:'type1',1:'type1a',2:'type1b'}:raise ValueError('Unexpected frozen model classes/task')
    reference={k:v.detach().cpu().float().clone() for k,v in model.model.named_parameters()}
    initialization=[];epochs=[]
    def verify_initialization(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        actual=dict(unwrap_model(trainer.model).named_parameters())
        if set(actual)!=set(reference):raise ValueError('Trainer changed architecture/class heads')
        if any(not torch.equal(reference[k],v.detach().cpu().float()) for k,v in actual.items()):
            raise ValueError('Trainer reset loaded checkpoint parameters')
        initialization.append({'parameter_tensors_checked':len(reference),'all_bit_exact_before_first_batch':True,
            'first_conv_sha256':hashlib.sha256(reference['model.0.conv.weight'].numpy().tobytes()).hexdigest()})
        atomic_json(report/'training'/f'{name}_initialization.json',initialization)
    def epoch_end(trainer):
        epochs.append({'epoch':trainer.epoch+1,'peak_allocated_MiB':torch.cuda.max_memory_allocated()/2**20})
        atomic_json(report/'training'/f'{name}_progress.json',{'epochs':epochs,'training_complete':False})
    model.add_callback('on_pretrain_routine_end',verify_initialization)
    model.add_callback('on_train_epoch_end',epoch_end)
    model.train(**settings)
    if model.trainer.epoch+1!=4 or len(initialization)!=1:raise ValueError('Pilot budget/initialization verification incomplete')
    verify_selected(manifest)
    checkpoint=ROOT/'models/phase_i'/name/'last.pt'
    if checkpoint.exists():raise ValueError('Never overwrite a published pilot checkpoint')
    checkpoint.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(Path(model.trainer.last),checkpoint)
    result={**config,'created_at':now(),'config_sha256':file_sha256(config_path),'checkpoint':str(checkpoint),
            'checkpoint_sha256':file_sha256(checkpoint),'initialization':initialization,
            'epochs_completed':4,'epoch_diagnostics':epochs,'training_peak_allocated_MiB':max(e['peak_allocated_MiB'] for e in epochs),
            'training_complete':True,'promoted_to_final':False,
            'crop_val_is_partial_type1b_only':set(manifest['real_counts']['val'])=={'type1b'}}
    atomic_json(output,result)
    atomic_json(report/'training'/f'{name}_progress.json',
                {'epochs':epochs,'training_complete':True,'checkpoint_sha256':result['checkpoint_sha256']})
    print('PILOT COMPLETE',name,flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--variant',choices=['real_only','replay'],required=True)
    args=p.parse_args()
    with offline_network():train(args.variant)
