"""Offline, GT-free cold/warm timing of the configured image pipeline."""
from pathlib import Path
import argparse,json,os,platform,sys,tempfile,time
from datetime import datetime,timezone
from importlib.metadata import version

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256,image_paths,load_runtime_config,offline_network
from tools.baseline_predict import summary


def measure_pass(paths,decode,process,synchronize,*,frames=None,clock=time.perf_counter):
    """Wait outside the leading boundary and inside the trailing boundary."""
    records=[]
    for path in paths:
        synchronize();start=clock()
        if frames is None:
            frame=decode(path);read_end=clock()
        else:
            frame=frames[path];read_end=start
        result=process(frame,path.name)
        synchronize();end=clock()
        if result.get('cached_detections') is not False:
            raise ValueError('Timing requires fresh detector execution for every frame')
        records.append({'image':path.name,'read_decode_hash_ms':(read_end-start)*1000,
            'pipeline_ms':(end-read_end)*1000,'total_ms':(end-start)*1000,
            'prediction_rows':len(result['predictions']),'detections':len(result['detections'])})
    return records


def main(argv=None):
    entered=time.perf_counter()
    measured_at=datetime.now(timezone.utc).isoformat()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'configs/phase_g/best_g.json')
    parser.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    parser.add_argument('--rounds',type=int,default=3)
    parser.add_argument('--warmup',type=int,default=5)
    parser.add_argument('--recognition-profile',choices=['frozen_g','russian_v1'],default='frozen_g')
    args=parser.parse_args(argv)
    try:
        if not 1<=args.rounds<=10:raise ValueError('--rounds must be1..10')
        if not 0<=args.warmup<=100:raise ValueError('--warmup must be0..100')
        if args.output.suffix.lower()!='.json' or args.output.exists():
            raise ValueError('--output must be a new JSON path; old reports are preserved')
        paths=image_paths(args.input)
        config_hash=file_sha256(args.config);config=load_runtime_config(args.config,ROOT)
        protected={args.config:config_hash,**{Path(config[k]):config[k+'_sha256'] for k in ('detector','ocr')}}
        inputs={p:file_sha256(p) for p in paths}
    except (ValueError,OSError) as error:parser.error(str(error))
    previous={k:os.environ.get(k) for k in ('YOLO_OFFLINE','YOLO_AUTOINSTALL')}
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    try:
        with offline_network():
            init_start=time.perf_counter()
            import cv2,numpy as np,torch,hashlib
            from src.image_anpr_f import DetectorF
            from src.image_anpr_g import OCRG,ImagePipelineG
            from src.multiline_g import ConfigG
            from src.plate_formats import apply_russian_profile
            old_torch_threads=torch.get_num_threads();old_cv_threads=cv2.getNumThreads()
            try:
                torch.set_num_threads(4);cv2.setNumThreads(1)
                if args.device=='cuda':
                    if not torch.cuda.is_available():raise ValueError('CUDA is unavailable')
                    torch.cuda.reset_peak_memory_stats()
                detector=DetectorF(config['detector'],args.device,config['imgsz'],config['confidence'],config['iou'])
                ocr=OCRG(config['ocr'],args.device)
                pipeline=ImagePipelineG(detector,ocr,ConfigG(**config['split']),multi=config['multi'],
                                       postprocess=config['postprocess'],expansion=config['expansion'])
                ocr.synchronize();initialization_ms=(time.perf_counter()-init_start)*1000
                preparation_ms=(time.perf_counter()-entered)*1000
                frames={}
                def decode(path):
                    encoded=np.fromfile(path,dtype=np.uint8)
                    if hashlib.sha256(encoded.tobytes()).hexdigest()!=inputs[path]:
                        raise ValueError('Input bytes changed: '+str(path))
                    frame=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                    if frame is None:raise ValueError('Unreadable image: '+str(path))
                    return frame
                def first_decode(path):
                    frames[path]=decode(path);return frames[path]
                def process(frame,name):
                    result=pipeline.process(frame,image_name=name)
                    return apply_russian_profile(result) if args.recognition_profile=='russian_v1' else result
                print('First pass',args.device,len(paths),'images',flush=True)
                first=measure_pass(paths,first_decode,process,ocr.synchronize)
                warm_start=time.perf_counter()
                for i in range(args.warmup):process(frames[paths[i%len(paths)]],paths[i%len(paths)].name)
                ocr.synchronize();warmup_ms=(time.perf_counter()-warm_start)*1000
                warm=[];with_read=[]
                for label,records,cached in [('pipeline',warm,True),('read_decode_pipeline',with_read,False)]:
                    for repeat in range(args.rounds):
                        current=measure_pass(paths,decode,process,ocr.synchronize,frames=frames if cached else None)
                        records.extend({**r,'round':repeat+1} for r in current)
                        print(args.device,label,'round',repeat+1,'mean_ms',round(summary([r['total_ms'] for r in current])['mean'],2),flush=True)
                memory={'peak_allocated_mib':torch.cuda.max_memory_allocated()/1024**2,
                        'peak_reserved_mib':torch.cuda.max_memory_reserved()/1024**2} if args.device=='cuda' else None
                device={'requested':args.device,'gpu_name':torch.cuda.get_device_name() if args.device=='cuda' else None,
                        'gpu_total_mib':torch.cuda.get_device_properties(0).total_memory/1024**2 if args.device=='cuda' else None,
                        'cpu_description':platform.processor(),'logical_cpu_count':os.cpu_count(),
                        'torch_threads':4,'opencv_threads':1}
                versions={name:version(name) for name in ('torch','torchvision','ultralytics','numpy','opencv-python')}
            finally:
                torch.set_num_threads(old_torch_threads);cv2.setNumThreads(old_cv_threads)
    finally:
        for name,value in previous.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
    if image_paths(args.input)!=paths or any(file_sha256(p)!=h for p,h in inputs.items()):
        raise ValueError('Input collection changed; benchmark not published')
    if any(file_sha256(p)!=h for p,h in protected.items()):
        raise ValueError('Config/checkpoint changed; benchmark not published')
    first_total=sum(r['total_ms'] for r in first)
    result={'measured_at_utc':measured_at,'device':device,'versions':versions,'recognition_profile':args.recognition_profile,
        'input_directory':str(args.input.resolve()),'input_images':len(paths),'rounds':args.rounds,
        'input_sha256':{str(p.resolve()):h for p,h in inputs.items()},
        'protected_sha256':{str(p.resolve()):h for p,h in protected.items()},
        'initialization_ms':initialization_ms,'preparation_including_initialization_ms':preparation_ms,
        'first_pass_read_decode_pipeline_ms':summary([r['total_ms'] for r in first]),
        'first_pass_startup_amortized_ms':(preparation_ms+first_total)/len(paths),
        'extra_warmup_frames':args.warmup,'extra_warmup_ms':warmup_ms,
        'steady_pipeline_ms':summary([r['total_ms'] for r in warm]),
        'steady_read_decode_pipeline_ms':summary([r['total_ms'] for r in with_read]),
        'vram_including_model_load':memory,'records':{'first_pass':first,'steady_pipeline':warm,'steady_with_read':with_read},
        'scopes':{'pipeline':'Uncached detector + geometry + OCR candidates + CTC + configured postprocess; GPU synchronized before/after.',
                  'read_decode_pipeline':'Repeated file read, byte hash, decode and pipeline; OS file cache already warmed by input validation/first pass.',
                  'startup_amortized':'Preparation/model imports/load plus first pass divided by input count; excludes Python interpreter startup and result serialization. Not a whole CLI wall-clock claim.',
                  'quality':'No GT loaded; procedural timing inputs do not establish independent real accuracy or reference-device SLA.'},
        'ground_truth_input':False,'training_started':False,'reference_hardware_verified':False,
        'network':'Python TCP/DNS/UDP blocked during model imports, loading and all inference'}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='benchmark-',dir=args.output.parent) as temporary:
        staged=Path(temporary)/'result.json';staged.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if args.output.exists():raise ValueError('Output appeared during benchmark; preserved')
        os.replace(staged,args.output)
    print('Benchmark saved:',args.output,flush=True)
    return result


if __name__=='__main__':main()
