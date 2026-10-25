"""PHASE I section20: vehicle-only VAL inference with cached FINAL-I plate/OCR output."""
from pathlib import Path
from collections import defaultdict
import argparse
import csv
import json
import os
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256,offline_network
from src.vehicle_filter import filter_plate_detections
from tools.build_final_i_bundle import verify_freeze

VAL=ROOT.parent/'datasets/reports/data3_2026_10_16/proposal_v4_holdout_reserved/val.csv'
CACHE=ROOT/'reports/phase_i/balanced_20261019/evaluations/I_DET_4_real_balanced_replay'
VEHICLE=ROOT/'models/phase_f/pretrained/yolo11n_coco.pt'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path,value):
    path.parent.mkdir(exist_ok=True,parents=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')


def metrics(rows,plates,traces,vehicle_cache=None):
    from src.image_anpr import Detection
    from tools.phase_h_evaluate import match_objects
    grouped=defaultdict(list)
    for row in rows:grouped[row['source_sha256']].append(row)
    by_image={r['image']:r for r in traces};records=[]
    for sha,gt in grouped.items():
        source=plates[sha]
        indices=(filter_plate_detections(source,vehicle_cache[sha])[0] if vehicle_cache is not None else list(range(len(source))))
        matches=match_objects(gt,[Detection(**source[i]) for i in indices])
        observed={d['detection_index']:d for d in by_image[sha]['detections']}
        for i,row in enumerate(gt):
            local=matches.get(i);original=indices[local] if local is not None else None
            obs=observed.get(original,{})
            detected=original is not None
            type_ok=detected and obs.get('plate_type')==row['plate_type']
            text_ok=detected and obs.get('final_ocr')==row['plate_num']
            records.append(dict(review_id=row['review_id'],source_sha256=sha,gt_type=row['plate_type'],
                detector_matched=detected,e2e_correct=bool(type_ok and text_ok),
                matched_original_detection_index=original))
    def summary(rs):
        n=len(rs);det=sum(r['detector_matched'] for r in rs);correct=sum(r['e2e_correct'] for r in rs)
        return dict(objects=n,detected=det,recall=det/n if n else None,correct=correct,FPA=correct/n if n else None)
    return dict(overall=summary(records),per_type={k:summary([r for r in records if r['gt_type']==k])
                for k in ('type1','type1a','type1b')}),records


def infer_vehicle(model,frame,classes,device):
    result=model.predict(source=frame,classes=classes,conf=.25,iou=.7,imgsz=640,device=device,
        save=False,save_txt=False,verbose=False,augment=False)[0]
    boxes=result.boxes
    if boxes is None:return []
    return [dict(bbox=b,confidence=c,class_id=int(k),class_name=model.names[int(k)])
            for b,c,k in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist())]


def run(output,device='cuda'):
    output=Path(output).resolve()
    if output.exists():raise ValueError('Existing experiment is preserved; no duplicate VAL inference')
    verify_freeze(ROOT)
    paths={'val':VAL,'plate_detections':CACHE/'val_detections.json','plate_traces':CACHE/'val_traces.json',
           'plate_metrics':CACHE/'val_metrics.json','vehicle_checkpoint':VEHICLE,'code':Path(__file__)}
    before={k:file_sha256(p) for k,p in paths.items()}
    with VAL.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    plates=read(paths['plate_detections']);traces=read(paths['plate_traces'])
    hashes={r['source_sha256'] for r in rows}
    if set(plates)!=hashes or {r['image'] for r in traces}!=hashes:raise ValueError('VAL cache membership differs')
    baseline,baseline_objects=metrics(rows,plates,traces)
    saved=read(paths['plate_metrics'])
    if baseline['overall']['detected']!=71 or baseline['overall']['correct']!=55 or any(
        baseline['overall'][k]!=saved['overall'][k] for k in ('objects','detected','correct')):
        raise ValueError('Cached data did not reproduce FINAL-I 71/93 detected and55/93 correct')
    output.mkdir(parents=True)
    policy=dict(name='I_VEHICLE_1_postfreeze_exploratory',source='User PHASE I section20',
        vehicle_weights=str(VEHICLE),vehicle_classes=['car','motorcycle','bus','truck'],
        conf=.25,iou=.7,imgsz=640,plate_center_inside=True,min_intersection_over_plate=.8,
        threshold_search=False,training=False,plate_OCR_cache_only=True,split='val',
        post_test_exploratory=True,eligible_for_FINAL_I_selection=False,runtime_enabled=False,
        source_hashes=before,source_paths={k:str(v) for k,v in paths.items()})
    write(output/'policy.json',policy)
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
    with offline_network():
        import cv2
        import numpy as np
        import torch
        from ultralytics import YOLO
        torch.set_num_threads(4);cv2.setNumThreads(1)
        if device=='cuda' and not torch.cuda.is_available():raise ValueError('CUDA unavailable')
        model=YOLO(str(VEHICLE))
        classes=[i for i,name in model.names.items() if name in policy['vehicle_classes']]
        if model.task!='detect' or len(classes)!=4:raise ValueError('Unexpected COCO model task/classes')
        synchronize=torch.cuda.synchronize if device=='cuda' else lambda:None
        def image(row):
            data=np.fromfile(row['image'],dtype=np.uint8)
            import hashlib
            if hashlib.sha256(data.tobytes()).hexdigest()!=row['source_sha256']:raise ValueError('VAL image changed')
            frame=cv2.imdecode(data,cv2.IMREAD_COLOR)
            if frame is None:raise ValueError('Unreadable VAL image')
            return frame
        first=image(rows[0])
        for _ in range(5):infer_vehicle(model,first,classes,device)
        synchronize()
        if device=='cuda':torch.cuda.reset_peak_memory_stats()
        grouped={r['source_sha256']:r for r in rows};vehicles={};timings=[];decisions={}
        for index,(sha,row) in enumerate(grouped.items()):
            frame=image(row)
            synchronize();started=time.perf_counter()
            vehicles[sha]=infer_vehicle(model,frame,classes,device)
            synchronize();timings.append((time.perf_counter()-started)*1000)
            kept,spatial=filter_plate_detections(plates[sha],vehicles[sha])
            decisions[sha]=dict(kept_original_detection_indices=kept,spatial=spatial)
            write(output/'images'/(sha+'.json'),dict(sha256=sha,vehicles=vehicles[sha],**decisions[sha]))
            if (index+1)%20==0:print('Vehicle-only VAL',index+1,'/',len(grouped),flush=True)
        performance=dict(images=len(timings),warmup=5,vehicle_only_mean_ms=float(np.mean(timings)),
            vehicle_only_p95_ms=float(np.percentile(timings,95)),
            peak_reserved_mib=torch.cuda.max_memory_reserved()/2**20 if device=='cuda' else None,
            GPU=torch.cuda.get_device_name(0) if device=='cuda' else None,
            scope='Vehicle model + preprocessing/NMS/tensor extraction; excludes disk/decode and cached plate/OCR pipeline. Not a complete end-to-end timing.')
    filtered,filtered_objects=metrics(rows,plates,traces,vehicles)
    lost=[a['review_id'] for a,b in zip(baseline_objects,filtered_objects) if a['e2e_correct'] and not b['e2e_correct']]
    gained=[a['review_id'] for a,b in zip(baseline_objects,filtered_objects) if not a['e2e_correct'] and b['e2e_correct']]
    original=sum(map(len,plates.values()));kept=sum(len(r['kept_original_detection_indices']) for r in decisions.values())
    result=dict(policy=policy,baseline=baseline,filtered=filtered,
        removed_plate_candidates=original-kept,all_plate_candidates=original,
        images_without_detected_vehicle=sum(not v for v in vehicles.values()),
        lost_correct_targets=lost,gained_correct_targets=gained,performance=performance,
        precision=None,FPR=None,non_target_detections_are_not_assumed_false=True,
        decision='DO_NOT_ENABLE_IN_FROZEN_RUNTIME',
        reason='Exploratory post-test ablation only. Partial scene GT cannot establish false-positive reduction; target losses are recorded.',
        train_started=False,plate_OCR_inference=False,new_real_test_inference=False,public_debug_read=False,
        final_I_weights_or_routing_changed=False,source_labels_or_split_changed=False)
    if any(file_sha256(paths[k])!=expected for k,expected in before.items()):raise ValueError('Experiment input changed')
    verify_freeze(ROOT)
    write(output/'result.json',result);write(output/'filtered_objects.json',filtered_objects)
    write(output/'vehicle_detections.json',vehicles)
    b,f=baseline['overall'],filtered['overall']
    report=f'''# PHASE I - отдельная проверка vehicle filtering

Пункт20 исходного задания выполнен на неизменённом REAL VAL. FINAL-I plate/OCR использованы из сохранённого кеша;
новая инференция выполнена только локальным YOLO11n COCO для транспорта. Policy записана до прогона, поиска порогов нет.

Правило: центр номера внутри vehicle bbox и пересечение не менее80% площади номера.
Классы: car/motorcycle/bus/truck; conf0.25, IoU0.7,640px. COCO веса существовали локально; сеть блокирована.

| Метрика по выбранным объектам | FINAL-I | После фильтра |
|---|---:|---:|
| Найдено /{b['objects']} |{b['detected']}|{f['detected']}|
| Правильные тип и текст |{b['correct']}|{f['correct']}|
| Full Plate Accuracy |{b['FPA']:.4f}|{f['FPA']:.4f}|

Удалено{original-kept} из{original} кандидатов. Это не число false positives: полной scene GT нет.
Утрачено{len(lost)} ранее верных ответов; получено{len(gained)} новых по сопоставлению выбранных объектов.
Среднее добавочное время vehicle-only{performance['vehicle_only_mean_ms']:.2f}мс,
p95{performance['vehicle_only_p95_ms']:.2f}мс. Полное время нового пайплайна не измерялось;
складывать этот замер с историческим средним и заявлять SLA нельзя.

Решение: не включать в замороженный runtime. Это послетестовая исследовательская ablation,
не новый выбор FINAL-I; проверка на40 тестовых изображениях не повторялась. Данные, модели,
разметка, пороги baseline, исходный split и архив кандидата не менялись.
'''
    (output/'report.md').write_text(report,encoding='utf-8')
    print(json.dumps(dict(baseline=b,filtered=f,removed=original-kept,decision=result['decision'])),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cuda')
    args=p.parse_args();run(args.output,args.device)
