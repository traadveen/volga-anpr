"""Only verified checkpoint-transfer runs qualify for the intended tournament."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_phase_f_models import train,call,REPORT,write

def main():
    for name,geometry in [('F02_transfer','bbox'),('F02_coco','bbox'),('F03_obb_transfer','obb'),('F04_pose_transfer','quad')]:
        train(name)
        for size in (640,1024):
            call(f'evaluate_{name}_{size}',['tools/evaluate_detector_f.py','--name',f'{name}_{size}',
                '--weights',f'models/phase_f/{name}/best.pt','--imgsz',str(size),'--geometry',geometry])
        if name=='F02_transfer':
            if not (REPORT/'hard_mining_F02_hard_verified.json').exists():
                call('mine_verified_hard',['tools/mine_hard_examples_f.py','--teacher',name,'--name','F02_hard_verified'])
    train('F02_hard_verified')
    for size in (640,1024):
        call(f'evaluate_F02_hard_verified_{size}',['tools/evaluate_detector_f.py','--name',f'F02_hard_verified_{size}',
            '--weights','models/phase_f/F02_hard_verified/best.pt','--imgsz',str(size)])
    write(REPORT/'workflow_status.json',{'stage':'verified_initial_tournament','state':'COMPLETE',
        'remaining':['Routing/geometry ablations','Export','Parity','Final reports']})

if __name__=='__main__':main()
