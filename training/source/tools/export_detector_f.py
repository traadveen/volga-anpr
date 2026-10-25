"""Export only the selected detector and measure CUDA EP parity, no silent fallback."""
import argparse
from pathlib import Path
import sys
import os
import shutil
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'.runtime-onnx-f'))
import cv2
import numpy as np
import torch
import onnx
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from src.image_anpr import FrozenCRNNOCR,ImageConfig,write_predictions
from src.image_anpr_f import DetectorF,ImagePipelineF
from src.geometry_f import bbox_iou
from tools.phase_e import IMAGES,BEST,read,write,load_image,official,summary
from tools.phase_f_setup import REPORT
from dataset.schema import sha256


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);args=p.parse_args()
    selected=read(REPORT/'experiments'/args.experiment/'result.json');config=selected['config']
    torch.set_num_threads(4);cv2.setNumThreads(1)
    folder=ROOT/'models/phase_f/export';folder.mkdir(parents=True,exist_ok=True)
    checkpoint=folder/'selected_detector.pt'
    shutil.copy2(config['weights'],checkpoint)
    model=YOLO(str(checkpoint))
    exported=Path(model.export(format='onnx',imgsz=config['imgsz'],dynamic=True,
        simplify=False,opset=17,device='cpu',half=False,batch=1))
    onnx.checker.check_model(onnx.load(str(exported)))
    # Torch loads its CUDA/cuDNN DLLs first; ORT1.20 uses the matching major versions.
    torch.cuda.init()
    dll=Path(torch.__file__).parent/'lib';handle=os.add_dll_directory(str(dll))
    session=ort.InferenceSession(str(exported),providers=[('CUDAExecutionProvider',{'device_id':0}), 'CPUExecutionProvider'])
    assert session.get_providers()[0]=='CUDAExecutionProvider','CUDA EP unavailable; do not report CPU as CUDA'
    model=YOLO(str(checkpoint));model.model.to('cuda').eval().fuse()
    raw=[];stride=int(model.model.stride.max())
    for path in sorted(IMAGES.glob('*.jpg'))[:12]:
        frame=load_image(path);arr=LetterBox((config['imgsz'],config['imgsz']),auto=True,stride=stride)(image=frame)
        arr=np.ascontiguousarray(arr[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255
        with torch.inference_mode():
            output=model.model(torch.from_numpy(arr).to('cuda'))
        if isinstance(output,(tuple,list)):output=output[0]
        expected=output.cpu().numpy();actual=session.run(None,{session.get_inputs()[0].name:arr})[0]
        raw.append({'image':path.name,'shape':list(expected.shape),'max_abs':float(np.max(np.abs(expected-actual))),
            'mean_abs':float(np.mean(np.abs(expected-actual))),
            'allclose':bool(np.allclose(expected,actual,rtol=1e-3,atol=.02))})
    del session,model
    import gc
    gc.collect();torch.cuda.empty_cache()
    detector=DetectorF(exported,imgsz=config['imgsz']);ocr=FrozenCRNNOCR(BEST,'cuda')
    e=read(ROOT/'configs/image_anpr_e.json')['config'];e['debug']=False
    e['stitch']=config.get('stitch',True)
    if config.get('filter_override') is not None:e['geometry_filter']=config['filter_override']
    pipe=ImagePipelineF(detector,ocr,geometry=config['geometry'],routing=config['routing'],image_config=ImageConfig(**e))
    frames=[(path.name,load_image(path)) for path in sorted(IMAGES.glob('*.jpg'))]
    device_free_before=torch.cuda.mem_get_info()[0];minimum_free=device_free_before
    for name,frame in frames[:5]:pipe.process(frame,image_name=name)
    backend=detector.model.predictor.model
    assert backend.session.get_providers()[0]=='CUDAExecutionProvider'
    times=[];traces=[];predictions=[]
    for repeat in range(3):
        current=[]
        for name,frame in frames:
            ocr.synchronize();start=time.perf_counter();out=pipe.process(frame,image_name=name);ocr.synchronize()
            times.append((time.perf_counter()-start)*1000);minimum_free=min(minimum_free,torch.cuda.mem_get_info()[0])
            current.extend(out['predictions'])
            if repeat==0:traces.append(out);predictions.extend(out['predictions'])
        if repeat:assert [(r['image'],r['plate_num'],r['plate_type']) for r in current]==[(r['image'],r['plate_num'],r['plate_type']) for r in predictions]
    folder=REPORT/'export';folder.mkdir(parents=True,exist_ok=True)
    write_predictions(folder/'onnx_predictions.csv',predictions);write(folder/'onnx_trace.json',{'records':traces})
    score,_=official(folder/'onnx_predictions.csv',folder/'onnx_score')
    original=read(REPORT/'experiments'/args.experiment/'trace.json')['records'];pairs=[];mismatches=[]
    for a,b in zip(original,traces):
        da=a['detections'];db=b['detections'];used=set()
        for d in da:
            candidates=[(bbox_iou(d['bbox'],e['bbox']),i,e) for i,e in enumerate(db) if i not in used]
            overlap,i,e=max(candidates,key=lambda v:v[0],default=(0,None,None))
            if i is not None:used.add(i)
            good=e is not None and overlap>.99 and d.get('plate_type')==e.get('plate_type') and d.get('final_ocr')==e.get('final_ocr')
            pairs.append({'image':a['image'],'IoU':overlap,'parity':good})
            if not good:mismatches.append({'image':a['image'],'pytorch':d,'onnx':e})
        if len(da)!=len(db):mismatches.append({'image':a['image'],'counts':[len(da),len(db)]})
    exact=sorted((r['image'],r['plate_num'],r['plate_type']) for r in predictions)==sorted((r['image'],r['plate_num'],r['plate_type']) for r in [p for r in original for p in r['predictions']])
    result={'selected_experiment':args.experiment,'onnx_path':str(exported),'onnx_sha256':sha256(exported),
        'checkpoint_sha256':sha256(checkpoint),'runtime_versions':{'onnx':onnx.__version__,'onnxruntime':ort.__version__,'torch':torch.__version__},
        'providers':backend.session.get_providers(),'raw_output_parity':raw,'detection_parity':pairs,
        'mismatches':mismatches,'text_and_type_exact_parity':exact,'public_development':score,
        'GPU_latency_ms':summary(times),'pytorch_GPU_latency_ms':selected['GPU_latency_ms'],
        'device_memory_delta_MB':max(0,device_free_before-minimum_free)/2**20,
        'VRAM_scope':'Observed device-wide free-memory delta; includes ORT allocator and display/other-process variation, not exact per-process peak. PyTorch allocator stats cannot measure ORT memory.',
        'latency_scope':'Same preloaded100 images, warmup5, repeats3; live detector+frozen OCR, synchronization; dynamic spatial shapes preserve native PyTorch letterbox',
        'decision':'KEEP CUDA export' if exact and not mismatches and all(r['allclose'] for r in raw) and summary(times)['mean']<selected['GPU_latency_ms']['mean'] else 'RETAIN PYTORCH; export is diagnostic',
        'generalization':'GENERALIZATION NOT YET VERIFIED',
        'reference':'https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html'}
    write(folder/'comparison.json',result);print({k:result[k] for k in ('text_and_type_exact_parity','GPU_latency_ms','decision')},flush=True)


if __name__=='__main__':main()
