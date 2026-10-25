"""GT-free selected Phase G image inference, local checkpoints only."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,hashlib,json,os,tempfile
from importlib.metadata import version
from src.inference_io import file_sha256,image_paths,load_runtime_config,offline_network
ROOT=Path(__file__).resolve().parents[1]

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--input',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'configs/phase_g/best_g.json');p.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--recognition-profile',choices=['frozen_g','russian_v1'],default='frozen_g',help='Opt-in type1b format candidate; weights and routing stay frozen')
    args=p.parse_args(argv)
    try:
        paths=image_paths(args.input)
        if args.output.suffix.lower()!='.csv':raise ValueError('--output must be a CSV file')
        config_hash=file_sha256(args.config)
        c=load_runtime_config(args.config,ROOT)
        if file_sha256(args.config)!=config_hash:raise ValueError('Config changed during validation')
    except (ValueError,OSError) as error:p.error(str(error))
    predictions=[];traces=[];environment={name:os.environ.get(name) for name in ('YOLO_OFFLINE','YOLO_AUTOINSTALL')}
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    try:
        with offline_network():
            # Imports, loading and inference all happen inside the offline guard.
            import cv2,numpy as np,torch
            from src.image_anpr_f import DetectorF
            from src.image_anpr_g import OCRG,ImagePipelineG
            from src.multiline_g import ConfigG
            from src.image_anpr import write_predictions
            torch.set_num_threads(4);cv2.setNumThreads(1)
            if args.device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA is unavailable; select --device cpu')
            detector=DetectorF(c['detector'],args.device,c['imgsz'],c['confidence'],c['iou']);ocr=OCRG(c['ocr'],args.device)
            pipeline=ImagePipelineG(detector,ocr,ConfigG(**c['split']),multi=c['multi'],postprocess=c['postprocess'],expansion=c['expansion'])
            for path in paths:
                encoded=np.fromfile(path,dtype=np.uint8)
                image=cv2.imdecode(encoded,cv2.IMREAD_COLOR)
                if image is None:raise ValueError('Unreadable image: '+str(path))
                r=pipeline.process(image,image_name=path.name)
                if args.recognition_profile=='russian_v1':
                    from src.plate_formats import apply_russian_profile
                    r=apply_russian_profile(r)
                r['input_sha256']=hashlib.sha256(encoded.tobytes()).hexdigest()
                traces.append(r);predictions.extend(r['predictions'])
    finally:
        for name,value in environment.items():
            if value is None:os.environ.pop(name,None)
            else:os.environ[name]=value
    if file_sha256(args.config)!=config_hash:raise ValueError('Config changed during inference; CSV was not published')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    manifest={'config_sha256':config_hash,'model_paths':{key:c[key] for key in ('detector','ocr')},
              'model_sha256':{key:c[key+'_sha256'] for key in ('detector','ocr')},
              'input_images':len(paths),'prediction_rows':len(predictions),'device':args.device,'recognition_profile':args.recognition_profile,
              'runtime_versions':{key:version(key) for key in ('torch','torchvision','ultralytics','numpy','opencv-python')},
              'network':'Python TCP connections, DNS lookup and UDP sendto blocked during imports, loading and inference','ground_truth_input':False}
    with tempfile.TemporaryDirectory(prefix='predict-images-',dir=args.output.parent) as temporary:
        stage=Path(temporary)
        if not stage.resolve().is_relative_to(args.output.parent.resolve()):raise ValueError('Staging directory escaped output directory')
        csv_path=stage/'result.csv';write_predictions(csv_path,predictions)
        for suffix,data in (('.trace.json',{'records':traces,'config':c,'device':args.device,'ground_truth_input':False}),('.manifest.json',manifest)):
            staged=stage/('result'+suffix);staged.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8')
            os.replace(staged,args.output.with_suffix(suffix))
        os.replace(csv_path,args.output)
    print('Predicted',len(paths),'images and',len(predictions),'plate rows',flush=True)

if __name__=='__main__':main()
