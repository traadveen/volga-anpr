"""Read-only GPU diagnostics. Never installs or changes an environment."""
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch


def main():
    result={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),
            'torch_version':torch.__version__,'torch_file':torch.__file__,'torch_version_cuda':torch.version.cuda,
            'cuda_is_available':torch.cuda.is_available(),'cuda_device_name':None,'installed_cuda_related_packages':[]}
    outputs=[]
    for args in (['nvidia-smi'],['nvidia-smi','--query-gpu=name,driver_version,memory.total,compute_cap','--format=csv,noheader']):
        try:
            proc=subprocess.run(args,capture_output=True,text=True,timeout=20)
            outputs.append({'command':args,'returncode':proc.returncode,'stdout':proc.stdout,'stderr':proc.stderr})
        except (OSError,subprocess.TimeoutExpired) as exc:outputs.append({'command':args,'error':str(exc)})
    result['nvidia_smi']=outputs
    venv_config=Path(sys.prefix)/'pyvenv.cfg'
    result['venv_config']=venv_config.read_text(encoding='utf-8') if venv_config.exists() else None
    for package in importlib.metadata.distributions():
        name=package.metadata.get('Name','')
        if any(token in name.lower() for token in ('torch','cuda','cudnn','nvidia','triton')):
            result['installed_cuda_related_packages'].append({'name':name,'version':package.version,'location':str(package.locate_file(''))})
    if torch.cuda.is_available():
        from ocr.crnn import DEFAULT_ALPHABET,create_crnn_model
        result['cuda_device_name']=torch.cuda.get_device_name(0)
        result['device_capability']=list(torch.cuda.get_device_capability(0));result['compiled_architectures']=torch.cuda.get_arch_list()
        start=time.perf_counter()
        x=torch.randn(128,128,device='cuda');y=x@x.T;torch.cuda.synchronize()
        model=create_crnn_model(DEFAULT_ALPHABET).cuda()
        logits=model(torch.randn(2,1,32,160,device='cuda'));loss=logits.square().mean();loss.backward();torch.cuda.synchronize()
        result['gpu_tests']={'tensor_finite':bool(torch.isfinite(y).all()),'crnn_logits_shape':list(logits.shape),
                             'loss_finite':bool(torch.isfinite(loss)),'gradient_finite':all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None),
                             'seconds':time.perf_counter()-start,'peak_vram_bytes':torch.cuda.max_memory_allocated()}
    else:
        result['gpu_tests']={'status':'not_run: CUDA unavailable'}
    result['diagnosis']='CPU-only imported PyTorch build: torch.version.cuda is None. GPU driver is visible to nvidia-smi; this is not evidence of a broken driver.' if torch.version.cuda is None else 'CUDA runtime build installed; availability/driver/architecture require inspection.'
    result['proposal']={'status':'proposed only, not installed','preserve':'.venv and baseline',
          'candidate':'Separate .venv-cuda with Python 3.12, torch==2.6.0+cu124 (torchvision==0.21.0 only if needed)',
          'driver':'Installed566.36 exceeds CUDA12.4 Windows release driver551.61; no driver update proposed',
          'verification':'Before training: cuda.is_available, get_arch_list with Pascal support, tensor + exact CRNN forward/backward; recommendation not execution proof',
          'sources':['https://docs.pytorch.org/get-started/previous-versions/','https://download.pytorch.org/whl/cu124/torch/',
                     'https://developer.nvidia.com/cuda/gpus/legacy','https://docs.nvidia.com/cuda/archive/12.4.1/cuda-toolkit-release-notes/']}
    (ROOT/'reports').mkdir(exist_ok=True)
    text=json.dumps(result,ensure_ascii=False,indent=2)
    (ROOT/'reports/gpu_diagnostics.txt').write_text(text+'\n',encoding='utf-8')
    (ROOT/'reports/gpu_diagnostics.json').write_text(text+'\n',encoding='utf-8')
    print(json.dumps({k:result[k] for k in ('torch_version','torch_version_cuda','cuda_is_available','diagnosis','gpu_tests')},indent=2))


if __name__=='__main__':main()
