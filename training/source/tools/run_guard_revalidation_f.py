"""Revalidate an explicit exception-handling amendment without changing selection."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,IMAGES,read
from tools.phase_f_setup import REPORT
from tools.run_phase_f_models import call,write
from tools.run_post_selection_f import strings


def main():
    config=read(ROOT/'configs/phase_f/best_f.json');name=config['selected_experiment']
    assert len(read(REPORT/'selection_lock.json')['amendments'])==1
    assert not (REPORT/'final_synthetic_test.json').exists()
    for device in ('cuda','cpu'):
        output=REPORT/'parity'/('gpu_predictions.csv' if device=='cuda' else 'cpu_predictions.csv')
        call('guard_'+device+'_benchmark',['tools/predict_images_f.py','--input',str(IMAGES),
            '--output',str(output),'--device',device,'--backend','pytorch','--warmup','5','--repeat','3'])
        assert strings(output)==strings(REPORT/'experiments'/name/'predictions.csv')
    assert strings(REPORT/'export/onnx_predictions.csv')==strings(REPORT/'parity/gpu_predictions.csv')
    call('guard_cpu_gpu_parity',['tools/parity_pipeline_f.py','--cpu',str(REPORT/'parity/cpu_predictions.csv'),
        '--gpu',str(REPORT/'parity/gpu_predictions.csv')])
    assert read(REPORT/'parity/cpu_gpu.json')['status']=='PASS'
    call('completed_post_lock_synthetic_test',['tools/final_synthetic_test_f.py'])
    call('selected_size_analysis',['tools/size_analysis_f.py','--experiment',name])
    call('identical_actual_input_review',['tools/finish_input_review_f.py'])
    write(REPORT/'guard_revalidation.json',{'status':'PASS','selected_model_parameters_changed':False,
        'public_CPU_GPU_and_pre_fix_ONNX_text_type_exact':True,'synthetic_test_completed':True,
        'first_test_attempt_failed_runtime':True})
    write(REPORT/'workflow_status.json',{'stage':'guard_revalidation','state':'COMPLETE',
        'remaining':['View selected size plot','Final verified reports']})


if __name__=='__main__':main()
