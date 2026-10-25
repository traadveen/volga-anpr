"""Score exhaustive authored scenes; never claim independent real accuracy."""
import argparse,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.evaluation import Record,evaluate
from src.plate_formats import apply_russian_profile
from src.inference_io import file_sha256


def compare(manifest_path,trace_path):
    manifest_path=Path(manifest_path);trace_path=Path(trace_path)
    manifest=json.loads(manifest_path.read_text('utf-8'));trace=json.loads(trace_path.read_text('utf-8'))
    if manifest.get('is_synthetic') is not True or manifest.get('real_images')!=0:
        raise ValueError('Authored synthetic manifest required')
    scenes=manifest['records'];names=[Path(r['image']).name for r in scenes]
    if len(names)!=len(set(names)):raise ValueError('Ambiguous image basenames')
    by_name={r['image']:r for r in trace['records']}
    if len(by_name)!=len(trace['records']) or set(by_name)!=set(names):raise ValueError('Trace coverage differs from scene manifest')
    ground_truth=[];predictions={'frozen_g':[],'russian_v1':[]}
    for scene,name in zip(scenes,names):
        frame=by_name[name]
        if frame.get('input_sha256')!=scene['sha256'] or file_sha256(manifest_path.parent/scene['image'])!=scene['sha256']:
            raise ValueError('Input image bytes differ from the scored scene')
        for obj in scene['annotations']:
            ground_truth.append(Record(name,obj['plate_num'],obj['plate_type'],bbox=tuple(obj['bbox_xyxy']),is_vehicle=True))
        for profile,result in [('frozen_g',frame),('russian_v1',apply_russian_profile(frame))]:
            detections={r['detection_index']:r for r in result['detections']}
            if len(detections)!=len(result['detections']):raise ValueError('Duplicate detection indices')
            for prediction in result['predictions']:
                detection=detections[prediction['detection_index']]
                predictions[profile].append(Record(name,prediction['plate_num'],prediction['plate_type'],
                    confidence=prediction['confidence'],bbox=tuple(detection['bbox']),raw_ocr=detection.get('raw_ocr')))
    return {'scope':'Authored exhaustive procedural development scenes; not independent real E2E or reference latency',
        'manifest_sha256':file_sha256(manifest_path),'trace_sha256':file_sha256(trace_path),
        'profiles':{profile:evaluate(ground_truth,rows,names,iou_threshold=.5) for profile,rows in predictions.items()},
        'same_detector_and_ocr_forward':True,'training_started':False,'new_real_GT_created':0,'final_split_created':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--manifest',type=Path,required=True)
    p.add_argument('--trace',type=Path,required=True);p.add_argument('--report',type=Path,required=True);args=p.parse_args()
    output=args.report.resolve()
    if output in (args.manifest.resolve(),args.trace.resolve()) or output.is_relative_to(args.manifest.resolve().parent):
        p.error('Report must stay outside the generated collection and preserve input files')
    result=compare(args.manifest,args.trace);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({k:v['metrics'] for k,v in result['profiles'].items()}))


if __name__=='__main__':main()
