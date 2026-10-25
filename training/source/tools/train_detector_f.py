"""Config-bound bounded detector-only training in the existing CUDA environment."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from ultralytics import YOLO
from tools.phase_f_setup import ROOT,REPORT,DATA,freeze
from tools.phase_e import read,write
from dataset.schema import sha256


def main():
    config=read(Path(sys.argv[1]));name=config['name']
    assert config['train'].get('pretrained') is True,'This archived config has invalid intended initialization (pretrained=False); use verified-run configs'
    assert (REPORT/'oracle_quad.json').is_file()
    if config['task']=='pose':assert read(REPORT/'oracle_quad.json')['gate']['keypoint_experiment_justified']
    review=read(REPORT/'dataset_validation.json')
    assert review['status']=='PASS' and review['visual_review']=='PASS'
    dataset=Path(config.get('dataset_root',DATA))
    if 'dataset_manifest_sha256' in config:
        assert sha256(dataset/'manifest.json')==config['dataset_manifest_sha256']==review['manifest_sha256']
    output=REPORT/'training'/f'{name}.json'
    if output.exists():
        old=read(output);assert old['config_sha256']==sha256(Path(sys.argv[1]));print('Already completed',name);return
    torch.set_num_threads(4)
    from ultralytics.utils.torch_utils import init_seeds
    init_seeds(config['train']['seed'],deterministic=True)
    model=YOLO(config['architecture']).load(config['initial_weights'])
    reference=model.model.state_dict()['model.0.conv.weight'].detach().cpu().float().clone()
    weights_dir=ROOT/'runs/phase_f'/name/'weights'
    if weights_dir.exists():raise ValueError('Existing training run must be explicitly resumed, never silently overwritten')
    peak=[]
    initialization=[]
    def verify_initialization(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        actual=unwrap_model(trainer.model).state_dict()['model.0.conv.weight'].detach().cpu().float()
        assert torch.equal(reference,actual),'Trainer reset loaded checkpoint backbone before first optimizer step'
        import hashlib
        initialization.append({'hook':'on_pretrain_routine_end_before_first_batch','key':'model.0.conv.weight',
            'bit_exact_loaded_checkpoint_tensor':True,'sha256':hashlib.sha256(actual.numpy().tobytes()).hexdigest()})
        write(REPORT/'initialization'/f'{name}.json',{'name':name,'verification':initialization,
            'source_checkpoint_sha256':sha256(Path(config['initial_weights']))})
    model.add_callback('on_pretrain_routine_end',verify_initialization)
    def capture(trainer):peak.append(int(torch.cuda.max_memory_allocated()));torch.cuda.reset_peak_memory_stats()
    model.add_callback('on_train_epoch_end',capture)
    settings=dict(config['train']);settings['data']=config['data'];settings['exist_ok']=False
    model.train(**settings)
    best=Path(model.trainer.best)
    import shutil
    published=ROOT/'models/phase_f'/name/'best.pt';published.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(best,published)
    write(output,{'name':name,'config_sha256':sha256(Path(sys.argv[1])),
        'dataset_manifest_sha256':sha256(Path(config.get('dataset_root',DATA))/'manifest.json'),'best_checkpoint':str(published),
        'data_yaml_sha256':sha256(Path(config['data'])),
        'architecture_sha256':sha256(Path(config['architecture'])) if Path(config['architecture']).is_file() else None,
        'checkpoint_sha256':sha256(published),'initial_weights_sha256':sha256(Path(config['initial_weights'])),
        'epochs_completed':model.trainer.epoch+1,'training_peak_allocated_MB':max(peak,default=0)/2**20,
        'validation':model.trainer.metrics,'best_selection':'Ultralytics validation fitness within run; final pipeline selection uses development end-to-end FPA',
        'scope':'Mixed synthetic + exposed real development training; no untouched real test used','OCR_trained':False})
    record=read(output);record['initialization_verification']=initialization;write(output,record)
    print('Published',published,flush=True)


if __name__=='__main__':main()
