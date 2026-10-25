"""Archive invalid initialization evidence; publish corrected independent runs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,read,write
from tools.phase_f_setup import REPORT
from dataset.schema import sha256

def main():
    write(REPORT/'initialization_audit.json',{'status':'CONFIRMED_AND_FIXED',
        'cause':'Installed Ultralytics Model.train uses weights=None when pretrained=False, even after Model.load; earlier intended transfer runs were random initialized',
        'evidence':'F02_balanced and F02_clean_v2 epoch1/2 losses/metrics identical despite distinct source checkpoints; inspected local engine/model.py training branch',
        'invalid_for_transfer_comparison':['F02_finetune','F02_balanced','F02_clean_v2'],
        'partial_clean_stopped':True,'source_sha256':sha256(ROOT/'.venv-cuda/Lib/site-packages/ultralytics/engine/model.py'),
        'fix':'pretrained=True after explicit local load; actual trainer backbone tensor checked bit-exact before first batch',
        'old_artifacts_preserved':True,'OCR_trained':False})
    for old,new in [('F02_balanced','F02_transfer'),('F02_clean_v2','F02_coco'),('F03_obb_v2','F03_obb_transfer'),('F04_pose_v2','F04_pose_transfer')]:
        c=read(ROOT/'configs/phase_f'/f'{old}.json');c.update(name=new)
        c['train'].update(name=new,pretrained=True)
        c['initialization_guard']='Actual trainer model.0.conv.weight equals explicitly loaded reference before first batch; fail immediately on reset'
        p=ROOT/'configs/phase_f'/f'{new}.json';assert not p.exists();write(p,c)
    write(REPORT/'workflow_status.json',{'stage':'initialization_repair','state':'COMPLETE','invalid_partial_run':'F02_clean_v2 stopped and retained'})

if __name__=='__main__':main()
