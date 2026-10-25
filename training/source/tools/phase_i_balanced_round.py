"""Explicit bounded stages for the new human-confirmed type1a/type1b round."""
from pathlib import Path
import argparse,sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_h_common import ROOT,read,now
from tools.phase_i_dataset import prepare,verify_selected
from tools.phase_i_review import atomic_json
from src.inference_io import file_sha256,offline_network

REPORT=ROOT/'reports/phase_i/balanced_20261019'
DATA=ROOT.parent/'generated_data/phase_i_balanced_20261019'
NAMES={'reference':'I_DET_0_FINAL_H_reference','real_only':'I_DET_3_real_balanced',
       'replay':'I_DET_4_real_balanced_replay'}


def prepared():
    manifest=prepare(REPORT,DATA,statement='проверил')
    if manifest['real_counts']!={'train':{'type1b':50,'type1a':40},'val':{'type1a':10,'type1b':9}}:
        raise ValueError('This bounded round requires its declared 90 train / 19 val human geometry objects')
    verify_selected(manifest)
    policy=REPORT/'selection_policy.json'
    if not policy.exists():
        old=read(ROOT/'reports/phase_i/pilot_20261019/selection_policy.json')
        old.update(created_at=now(),scope='Two fixed four-epoch pilots after 50 explicit human type1a decisions',
            candidates=list(NAMES.values()),checkpoint='Four fixed epochs; last.pt only; no crop-val selection',
            limitations='90 real train objects, no real type1 train geometry; crop scale/context bias remains.',
            dataset_manifest_sha256=file_sha256(REPORT/'dataset_manifest.json'),
            previous_policy_sha256=file_sha256(ROOT/'reports/phase_i/pilot_20261019/selection_policy.json'),
            hyperparameter_search='Only two predeclared variants; settings match previous round, no grid or threshold tuning')
        atomic_json(policy,old)
    if read(policy)['dataset_manifest_sha256']!=file_sha256(REPORT/'dataset_manifest.json'):
        raise ValueError('Round policy/dataset binding changed')
    return manifest


def main(stage):
    prepared()
    if stage=='prepare':return
    if stage.startswith('train-'):
        from tools.phase_i_train import train
        variant={'train-real':'real_only','train-replay':'replay'}[stage]
        with offline_network():train(variant,REPORT,DATA,NAMES[variant])
    elif stage.startswith('evaluate-'):
        from tools.phase_i_evaluate import evaluate
        import torch,cv2
        torch.set_num_threads(4);cv2.setNumThreads(1)
        variant={'evaluate-reference':'reference','evaluate-real':'real_only','evaluate-replay':'replay'}[stage]
        checkpoint=Path(read(ROOT/'configs/phase_g/best_g.json')['detector']) if variant=='reference' else ROOT/'models/phase_i'/NAMES[variant]/'last.pt'
        with offline_network():evaluate(NAMES[variant],checkpoint,REPORT,DATA)
    elif stage=='summary':
        from tools.phase_i_pilot_report import main as summarize
        summarize(REPORT)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',required=True,choices=['prepare','train-real','train-replay',
        'evaluate-reference','evaluate-real','evaluate-replay','summary'])
    main(p.parse_args().stage)
