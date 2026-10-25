"""Serial final checks after the bounded development tournament; stop for visual review."""
from pathlib import Path
import csv
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,IMAGES,read
from tools.run_phase_f_models import call,REPORT,write


def strings(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as stream:
        return sorted((r['image'],r['plate_num'],r['plate_type']) for r in csv.DictReader(stream,delimiter=';'))


def main():
    assert read(REPORT/'workflow_status.json')['state']=='COMPLETE'
    ranking=read(REPORT/'provisional_ranking.json')
    name=ranking['eligible_ranked'][0]['experiment']
    print('Selected by declared policy:',name,flush=True)
    call('audit_effective_training_bytes',['tools/audit_training_images_f.py'])
    call('selected_export',['tools/export_detector_f.py','--experiment',name])
    call('selection_lock',['tools/lock_best_f.py','--experiment',name])
    for device,backend in [('cuda','pytorch'),('cpu','pytorch')]:
        output=REPORT/'parity'/('gpu_predictions.csv' if device=='cuda' else 'cpu_predictions.csv')
        call('final_'+device+'_benchmark',['tools/predict_images_f.py','--input',str(IMAGES),
            '--output',str(output),'--device',device,'--backend',backend,'--warmup','5','--repeat','3'])
        if device=='cuda':assert strings(output)==strings(REPORT/'experiments'/name/'predictions.csv'),'Fresh selected PyTorch inference differs from tournament'
    call('final_cpu_gpu_parity',['tools/parity_pipeline_f.py','--cpu',str(REPORT/'parity/cpu_predictions.csv'),
        '--gpu',str(REPORT/'parity/gpu_predictions.csv')])
    config=read(ROOT/'configs/phase_f/best_f.json')
    if config['onnx_accepted']:
        call('accepted_onnx_cli',['tools/predict_images_f.py','--input',str(IMAGES),
            '--output',str(REPORT/'parity/onnx_cli_predictions.csv'),'--device','cuda',
            '--warmup','5','--repeat','3'])
        assert strings(REPORT/'parity/onnx_cli_predictions.csv')==strings(REPORT/'parity/gpu_predictions.csv')
    call('post_lock_synthetic_test',['tools/final_synthetic_test_f.py'])
    call('selected_attribution',['tools/attribution_f.py','--experiment',name])
    call('selected_size_analysis',['tools/size_analysis_f.py','--experiment',name])
    write(REPORT/'workflow_status.json',{'stage':'post_selection_checks','state':'COMPLETE',
        'selected_experiment':name,'remaining':['Manual actual-input review','Final verified reports']})


if __name__=='__main__':main()
