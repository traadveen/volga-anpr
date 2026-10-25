"""Sequential CUDA jobs, with durable stage markers and resumable completed jobs."""
from datetime import datetime,timezone
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.phase_e import read,write
REPORT=ROOT/'reports/phase_f'


def call(stage,arguments):
    print('START',stage,flush=True)
    write(REPORT/'workflow_status.json',{'stage':stage,'state':'RUNNING','utc':datetime.now(timezone.utc).isoformat()})
    path=REPORT/'workflow_logs'/f'{stage}.log';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,*arguments],cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
    if result.returncode:
        write(REPORT/'workflow_status.json',{'stage':stage,'state':'FAILED','returncode':result.returncode,'log':str(path)})
        raise RuntimeError(stage+' failed; inspect '+str(path))
    print('DONE',stage,flush=True)


def train(name):
    if not (REPORT/'training'/f'{name}.json').exists():
        call('train_'+name,['tools/train_detector_f.py',f'configs/phase_f/{name}.json'])


def main():
    assert (REPORT/'training/F02_finetune.json').is_file(),'Complete the first fine-tune before starting this sequence'
    for size in (640,1024,1280):
        call(f'evaluate_F02_finetune_{size}', ['tools/evaluate_detector_f.py','--name',f'F02_finetune_{size}',
            '--weights','models/phase_f/F02_finetune/best.pt','--imgsz',str(size)])
    if not (REPORT/'hard_mining.json').exists():call('mine_hard',['tools/mine_hard_examples_f.py'])
    # Finish the clean/OBB/keypoint comparisons on the same original frozen dataset.
    for name,geometry in [('F02_clean','bbox'),('F03_obb','obb'),('F04_pose','quad')]:
        train(name)
        for size in (640,1024):
            call(f'evaluate_{name}_{size}',['tools/evaluate_detector_f.py','--name',f'{name}_{size}',
                '--weights',f'models/phase_f/{name}/best.pt','--imgsz',str(size),'--geometry',geometry])
    train('F02_hard')
    for size in (640,1024):
        call(f'evaluate_F02_hard_{size}',['tools/evaluate_detector_f.py','--name',f'F02_hard_{size}',
            '--weights','models/phase_f/F02_hard/best.pt','--imgsz',str(size)])
    write(REPORT/'workflow_status.json',{'stage':'initial_tournament','state':'COMPLETE',
        'remaining':['Routing comparisons','Same-weight geometry/stitch ablations','BEST selection lock',
                     'Selected detector ONNX CUDA export/parity','CPU/GPU parity','Final data/source/QA/report']})
    print('Initial tournament complete; proceed with final comparisons.',flush=True)


if __name__=='__main__':main()
