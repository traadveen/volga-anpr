"""Controlled sequential v2 tournament; each completed stage is durable."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_phase_f_models import train,call,REPORT,write

def main():
    for name,geometry in [('F02_balanced','bbox'),('F02_clean_v2','bbox'),('F03_obb_v2','obb'),('F04_pose_v2','quad')]:
        train(name)
        for size in (640,1024):
            call(f'evaluate_{name}_{size}',['tools/evaluate_detector_f.py','--name',f'{name}_{size}',
                '--weights',f'models/phase_f/{name}/best.pt','--imgsz',str(size),'--geometry',geometry])
        if name=='F02_balanced':
            if not (REPORT/'hard_mining_F02_hard_v2.json').exists():
                call('mine_balanced_hard',['tools/mine_hard_examples_f.py','--teacher',name,'--name','F02_hard_v2'])
    train('F02_hard_v2')
    for size in (640,1024):
        call(f'evaluate_F02_hard_v2_{size}',['tools/evaluate_detector_f.py','--name',f'F02_hard_v2_{size}',
            '--weights','models/phase_f/F02_hard_v2/best.pt','--imgsz',str(size)])
    write(REPORT/'workflow_status.json',{'stage':'balanced_initial_tournament','state':'COMPLETE',
        'remaining':['Hard iteration','Routing/geometry ablations','Export','Parity','Final reports']})

if __name__=='__main__':main()
