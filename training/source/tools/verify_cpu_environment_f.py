"""Verify final CPU inference in the preserved CPU environment itself."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,IMAGES,read,write
from tools.phase_f_setup import REPORT
from tools.run_phase_f_models import call
from tools.run_post_selection_f import strings


def main():
    config=read(ROOT/'configs/phase_f/best_f.json');name=config['selected_experiment']
    output=REPORT/'parity/cpu_predictions.csv'
    previous=read(output.with_suffix('.manifest.json'))
    write(REPORT/'parity/previous_cpu_in_cuda_environment.json',previous)
    call('preserved_cpu_environment',['tools/predict_images_f.py','--input',str(IMAGES),
        '--output',str(output),'--device','cpu','--backend','pytorch','--warmup','5','--repeat','3'])
    assert strings(output)==strings(REPORT/'experiments'/name/'predictions.csv')
    call('preserved_cpu_environment_parity',['tools/parity_pipeline_f.py','--cpu',str(output),
        '--gpu',str(REPORT/'parity/gpu_predictions.csv')])
    assert read(REPORT/'parity/cpu_gpu.json')['status']=='PASS'
    write(REPORT/'workflow_status.json',{'stage':'final_checks','state':'COMPLETE','remaining':['Final verified reports']})


if __name__=='__main__':main()
