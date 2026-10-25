"""Independent-image BEST-F entry point; separate from frozen E and legacy video."""
import argparse
from pathlib import Path
import sys
import time
import os
from importlib.metadata import version
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import cv2
import torch
from tools.phase_e import read,write,load_image,summary
from src.image_anpr import FrozenCRNNOCR,ImageConfig,write_predictions
from src.image_anpr_f import DetectorF,ImagePipelineF
from dataset.schema import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'reports/phase_f/predictions.csv')
    p.add_argument('--config',type=Path,default=ROOT/'configs/phase_f/best_f.json')
    p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--backend',choices=['pytorch','onnx'],default=None)
    p.add_argument('--warmup',type=int,default=0);p.add_argument('--repeat',type=int,default=1)
    args=p.parse_args();config=read(args.config)
    args.backend=args.backend or (config.get('default_backend','pytorch') if args.device=='cuda' else 'pytorch')
    if args.warmup<0 or args.repeat<1:raise ValueError('Invalid warmup/repeat')
    output=args.output.resolve()
    if (ROOT/'reports/phase_e').resolve() in output.parents or output==ROOT/'reports/baseline_predictions.csv':
        raise ValueError('Frozen E/baseline artifacts cannot be overwritten')
    paths=sorted(path for path in args.input.rglob('*') if path.suffix.lower() in ('.jpg','.jpeg','.png','.bmp','.webp')) if args.input.is_dir() else [args.input]
    if not paths or any(not path.is_file() for path in paths):raise ValueError('No readable input files')
    if len({p.name for p in paths})!=len(paths):raise ValueError('Duplicate basenames conflict with official image field')
    weights=Path(config['detector']);ocr_weights=Path(config['ocr'])
    if sha256(weights)!=config['detector_sha256'] or sha256(ocr_weights)!=config['ocr_sha256']:
        raise ValueError('Frozen BEST-F weights changed')
    for path,digest in config.get('inference_source_sha256',{}).items():
        if sha256(Path(path))!=digest:raise ValueError('Frozen BEST-F inference source changed')
    if args.backend=='onnx':
        if args.device!='cuda':raise ValueError('The accepted ONNX backend is CUDA only; use PyTorch for CPU')
        if not config.get('onnx_accepted'):raise ValueError('ONNX did not pass the parity/speed gate')
        sys.path.insert(0,str(ROOT/'.runtime-onnx-f'));weights=Path(config['onnx_detector'])
        if sha256(weights)!=config['onnx_sha256']:raise ValueError('ONNX checksum mismatch')
        torch.cuda.init();dll_handle=os.add_dll_directory(str(Path(torch.__file__).parent/'lib'))
    torch.set_num_threads(4);cv2.setNumThreads(1)
    detector=DetectorF(weights,device=args.device,imgsz=config['imgsz'],confidence=config.get('confidence',.5))
    ocr=FrozenCRNNOCR(ocr_weights,args.device)
    pipe=ImagePipelineF(detector,ocr,geometry=config['geometry'],routing=config['routing'],image_config=ImageConfig(**config['image_config']))
    # Load images outside the compute benchmark. The CLI reports IO separately.
    start=time.perf_counter();frames=[(path.name,load_image(path)) for path in paths];io_ms=(time.perf_counter()-start)*1000
    for i in range(args.warmup):
        name,frame=frames[i%len(frames)];pipe.process(frame,image_name=name)
    records=[];predictions=[];times=[];free_memory=[]
    if args.device=='cuda':torch.cuda.reset_peak_memory_stats();free_before=torch.cuda.mem_get_info()[0]
    for repeat in range(args.repeat):
        current=[]
        for name,frame in frames:
            ocr.synchronize();start=time.perf_counter();result=pipe.process(frame,image_name=name);ocr.synchronize()
            times.append((time.perf_counter()-start)*1000);current.extend(result['predictions'])
            if args.device=='cuda':free_memory.append(torch.cuda.mem_get_info()[0])
            if repeat==0:records.append(result);predictions.extend(result['predictions'])
        if repeat:
            assert [(r['image'],r['plate_num'],r['plate_type']) for r in current]==[(r['image'],r['plate_num'],r['plate_type']) for r in predictions]
    if args.backend=='onnx':assert detector.model.predictor.model.session.get_providers()[0]=='CUDAExecutionProvider'
    write_predictions(output,predictions)
    write(output.with_suffix('.trace.json'),{'records':records})
    write(output.with_suffix('.manifest.json'),{'config_sha256':sha256(args.config),'device':args.device,'backend':args.backend,
        'runtime_versions':{'torch':torch.__version__,'ultralytics':version('ultralytics'),'opencv':cv2.__version__},
        'input_files':{str(path):sha256(path) for path in paths},'prediction_sha256':sha256(output),
        'compute_ms':summary(times),'read_all_images_ms':io_ms,'warmup':args.warmup,'repeats':args.repeat,
        'VRAM_torch_peak_allocated_MB':torch.cuda.max_memory_allocated()/2**20 if args.device=='cuda' else None,
        'VRAM_observed_device_delta_MB':max(0,free_before-min(free_memory))/2**20 if free_memory else None,
        'scope':'Preloaded independent images; model load/read/CSV outside compute, no tracking/voting/GT inputs',
        'generalization':'No test/generalization claim from arbitrary CLI inputs'})
    print({'images':len(paths),'predictions':len(predictions),'compute_ms':summary(times)},flush=True)


if __name__=='__main__':main()
