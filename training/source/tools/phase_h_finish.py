"""Val-only selection, FINAL-H freeze, single final test, and measured reporting."""
from pathlib import Path
import os,sys
os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse
import csv
import gc
import platform
import time
import cv2
import numpy as np
import torch
from tools.phase_h_common import (ROOT,REPORT,CONFIG,MODELS,SPLIT,read,write,load_split,
    guard_check,image,now)
from tools.phase_h_evaluate import evaluate
from src.inference_io import offline_network,file_sha256
from src.image_anpr_f import DetectorF
from src.image_anpr_g import OCRG,ImagePipelineG
from src.multiline_g import ConfigG
from src.plate_formats import apply_russian_profile


def csv_write(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare():
    if (CONFIG/'final_h_freeze.json').exists():
        raise ValueError('FINAL-H already frozen; no new model selection')
    guard_check()
    cfg=read(ROOT/'configs/phase_g/best_g.json')
    candidates=[{'name':'H0_BEST_G','checkpoint':cfg['ocr'],'profile':'frozen_g','complexity':0}]
    for key in ('H_OCR_1_real_heavy','H_OCR_2_balanced'):
        summary=read(REPORT/'training'/key/'summary.json')
        candidates.append({'name':'H1_'+key,'checkpoint':summary['best_checkpoint'],
                           'profile':'frozen_g','complexity':2})
    # Existing typed-format handling is a distinct, paired ablation after H1;
    # the baseline attribution already proved postprocess errors on real val.
    candidates += [{**c,'name':'H4_'+c['name'],'profile':'russian_v1',
                    'complexity':c['complexity']+1} for c in list(candidates)]
    policy={'created_at':now(),'primary':'REAL VAL selected-object full text + type match',
        'close_result_policy':'Within one correct val object of maximum: prefer lower complexity; then higher FPA',
        'no_test_access':True,'candidates':candidates,
        'H2':'SKIPPED: zero reliable real orientation/quad annotations',
        'H3':'NOT APPLICABLE: no detector candidate',
        'H4':'Existing russian_v1 postprocess-only paired ablation, justified by nine val POSTPROCESS failures; no new split/rectification or threshold tuning'}
    write(CONFIG/'selection_policy.json',policy)
    comparisons=[]
    for c in candidates:
        saved=REPORT/'evaluations'/c['name']/'val_metrics.json'
        if c['name']=='H0_BEST_G':
            summary=read(saved)
        else:
            summary,_=evaluate(c['name'],c['checkpoint'],c['profile'])
        comparisons.append({**c,'metrics':summary})
    maximum=max(c['metrics']['overall']['correct'] for c in comparisons)
    close=[c for c in comparisons if c['metrics']['overall']['correct']>=maximum-1]
    selected=min(close,key=lambda c:(c['complexity'],-c['metrics']['overall']['correct']))
    selection={'created_at':now(),'selected':selected,'candidates':comparisons,
               'test_used_for_selection':False,'policy':policy}
    write(REPORT/'selection.json',selection)
    print('SELECTED',selected['name'],selected['metrics']['overall']['correct'],'/93',flush=True)
    return selection


def benchmark(name,checkpoint,profile,device):
    cfg=read(ROOT/'configs/phase_g/best_g.json')
    if device=='cuda':
        torch.cuda.empty_cache()
    detector=DetectorF(cfg['detector'],device,cfg['imgsz'],cfg['confidence'],cfg['iou'])
    ocr=OCRG(str(checkpoint),device)
    pipeline=ImagePipelineG(detector,ocr,ConfigG(**cfg['split']),multi=cfg['multi'],
        postprocess=cfg['postprocess'],expansion=cfg['expansion'])
    files={r['source_sha256']:r['image'] for r in load_split('val')}
    first=image(next(iter(files.values())))
    for _ in range(5):
        output=pipeline.process(first,image_name='warmup')
        if profile=='russian_v1':
            apply_russian_profile(output)
    ocr.synchronize()
    if device=='cuda':
        torch.cuda.reset_peak_memory_stats()
    times=[]
    outputs={}
    for sha,path in files.items():
        frame=image(path,sha)
        ocr.synchronize()
        start=time.perf_counter()
        output=pipeline.process(frame,image_name=sha)
        if profile=='russian_v1':
            output=apply_russian_profile(output)
        ocr.synchronize()
        times.append((time.perf_counter()-start)*1000)
        outputs[sha]=[(p['plate_type'],p['plate_num']) for p in output['predictions']]
    result={'model':name,'device':device,'images':len(times),'warmup_images':5,
        'mean_ms':float(np.mean(times)),'p95_ms':float(np.percentile(times,95)),
        'peak_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2 if device=='cuda' else None,
        'peak_reserved_MiB':torch.cuda.max_memory_reserved()/1024**2 if device=='cuda' else None,
        'hardware':torch.cuda.get_device_name() if device=='cuda' else platform.processor(),
        'threads':torch.get_num_threads(),'scope':'Full detector + split/stitch + OCR + postprocess, unique val images; excludes load/decode/disk/CSV',
        'timing_ms':times,'predicted_texts':outputs,
        'jury_1050Ti_measured':False,'created_at':now()}
    write(REPORT/'performance'/(name+'_'+device+'.json'),result)
    print(name,device,'mean/p95',result['mean_ms'],result['p95_ms'],flush=True)
    del pipeline,ocr,detector
    gc.collect()
    if device=='cuda':
        torch.cuda.empty_cache()
    return result


def freeze():
    path=CONFIG/'final_h_freeze.json'
    if path.exists():
        raise ValueError('FINAL-H freeze already exists')
    selection=read(REPORT/'selection.json')
    selected=selection['selected']
    cfg=read(ROOT/'configs/phase_g/best_g.json')
    performance={}
    for name,checkpoint,profile in [('BEST_G',cfg['ocr'],'frozen_g'),
            ('FINAL_H',selected['checkpoint'],selected['profile'])]:
        for device in ('cuda','cpu'):
            performance[name+'_'+device]=benchmark(name,checkpoint,profile,device)
    source=torch.load(selected['checkpoint'],map_location='cpu',weights_only=False)
    final_path=MODELS/'FINAL_H.pt'
    if final_path.exists():
        raise ValueError('FINAL_H.pt already exists')
    torch.save({k:source[k] for k in ('model','alphabet','config')},final_path)
    restored=torch.load(final_path,map_location='cpu',weights_only=False)
    if any(not torch.equal(v,restored['model'][k]) for k,v in source['model'].items()):
        raise ValueError('Exported inference weights differ')
    final_cfg={**cfg,'name':'FINAL-H','ocr':str(final_path),'ocr_sha256':file_sha256(final_path),
        'recognition_profile':selected['profile'],'selection_basis':'Frozen DATA-3 real val only',
        'real_test_available':True,'generalization':'PARTIAL-GT TEST NOT YET EVALUATED',
        'mask':'LLDDDRR/RRR for observed type1b, LDDDLLRR/RRR for white types' if selected['profile']=='russian_v1' else cfg['mask'],
        'promotion_scope':'Phase H selected-object candidate, BEST-G remains available',
        'runtime_gates':{'measured_hardware':'GTX1060 6GB; not jury GTX1050Ti',
                         'GPU_mean_ms':performance['FINAL_H_cuda']['mean_ms'],
                         'GPU_p95_ms':performance['FINAL_H_cuda']['p95_ms']}}
    write(CONFIG/'final_h.json',final_cfg)
    guard_check()
    files=[CONFIG/'final_h.json',CONFIG/'selection_policy.json',REPORT/'selection.json',
           final_path,Path(cfg['detector']),ROOT/'configs/phase_g/best_g.json']
    files += [ROOT/p for p in ('src/image_anpr_g.py','src/image_anpr_f.py','src/image_anpr.py',
        'src/multiline_g.py','src/geometry_f.py','src/ocr_preprocess.py','src/multiline.py',
        'src/plate_formats.py','src/real_data_g.py','ocr/crnn.py','ocr/utils.py',
        'tools/phase_h_common.py','tools/phase_h_evaluate.py','tools/phase_h_train.py',
        'tools/phase_h_finish.py','tools/predict_images_h.py')]
    value={'status':'FINAL-H-FROZEN','created_at':now(),'selected':selected['name'],
        'split_manifest_sha256':file_sha256(SPLIT),'hashes':{str(p):file_sha256(p) for p in files},
        'selected_source_checkpoint':selected['checkpoint'],
        'selection_uses_test':False,'test_prediction_runs_before_freeze':0,
        'changes_after_test_forbidden':True,'scope':'PARTIAL-GT INDEPENDENT TEST selected objects',
        'performance_gate':{'1060_mean_lt_100ms':performance['FINAL_H_cuda']['mean_ms']<100,
                            '1050Ti_gate':'UNVERIFIED: hardware unavailable'}}
    write(path,value)
    return value


def final_test():
    if not (CONFIG/'final_h_freeze.json').exists():
        raise ValueError('Freeze FINAL-H before test')
    receipt=REPORT/'final_test_run.json'
    if receipt.exists():
        raise ValueError('Independent test was already started; never rerun automatically')
    # All hashes/GT gate checks happen before recording the one allowed run.
    load_split('untouched_test')
    write(receipt,{'status':'STARTED','created_at':now(),
                  'freeze_sha256':file_sha256(CONFIG/'final_h_freeze.json')})
    old=read(ROOT/'configs/phase_g/best_g.json')
    new=read(CONFIG/'final_h.json')
    results={}
    for name,cfg,profile in [('BEST_G_TEST',old,'frozen_g'),('FINAL_H_TEST',new,new['recognition_profile'])]:
        result,_=evaluate(name,cfg['ocr'],profile,'untouched_test')
        results[name]=result
    value={'status':'COMPLETE','created_at':now(),'scope':'PARTIAL-GT INDEPENDENT TEST',
        'freeze_sha256':file_sha256(CONFIG/'final_h_freeze.json'),'results':results,
        'image_level_precision_computed':False,'false_positives_per_image_computed':False,
        'complete_scene_recall_computed':False,'tuning_after_test':False}
    write(REPORT/'final_test.json',value)
    write(receipt,{'status':'COMPLETE','final_test_sha256':file_sha256(REPORT/'final_test.json'),
                   'freeze_sha256':value['freeze_sha256'],'models_each_evaluated_once':True})
    return value


def report():
    selection=read(REPORT/'selection.json')
    table=[]
    waterfall=[]
    for c in selection['candidates']:
        m=c['metrics']
        table.append({'MODEL':c['name'],'REAL_VAL_FPA':m['overall']['FPA'],
            'TYPE1A_VAL':m['per_type']['type1a']['FPA'],'TYPE1B_VAL':m['per_type']['type1b']['FPA'],
            'OCR_GOOD_CROP_FPA':m['overall']['OCR_good_bbox_raw_FPA'],
            'DETECTOR_RECALL':m['overall']['selected_object_detector_recall'],
            'GPU_MEAN':None,'GPU_P95':None,'selected':c['name']==selection['selected']['name']})
        for category in ('CORRECT','DETECTOR_MISS','BAD_OBB','WRONG_TYPE','BAD_RECTIFICATION',
                         'TYPE1A_SPLIT','OCR_GOOD_INPUT','POSTPROCESS','UNKNOWN'):
            waterfall.append({'MODEL':c['name'],'CATEGORY':category,'COUNT':m['waterfall'].get(category,0),
                              'scope':'Val spatially matched selected objects; unknown geometry causes not fabricated'})
    for row in table:
        label='BEST_G' if row['MODEL']=='H0_BEST_G' else 'FINAL_H' if row['selected'] else None
        if label:
            p=read(REPORT/'performance'/(label+'_cuda.json'))
            row.update(GPU_MEAN=p['mean_ms'],GPU_P95=p['p95_ms'])
    csv_write(REPORT/'val_comparison.csv',table)
    csv_write(REPORT/'error_attribution.csv',waterfall)
    training={k:read(REPORT/'training'/k/'summary.json') for k in ('H_OCR_1_real_heavy','H_OCR_2_balanced')}
    freeze_value=read(CONFIG/'final_h_freeze.json')
    test=read(REPORT/'final_test.json')
    summary={'status':'PHASE_H_COMPLETE_WITH_GEOMETRY_AND_EXPORT_LIMITATIONS','created_at':now(),
        'training_started':True,'experiments':training,'selected':selection['selected']['name'],
        'H2_detector':'SKIPPED_NO_RELIABLE_QUAD_OBB','H3':'NOT_APPLICABLE',
        'H4':'existing typed yellow postprocess ablation; no new preprocessing',
        'freeze_sha256':file_sha256(CONFIG/'final_h_freeze.json'),
        'frozen_best_g_unchanged':True,'split_membership_GT_groups_unchanged':True,
        'test_complete':test['status']=='COMPLETE','test_tuning':False,
        'limitations':['Selected-object partial GT only','No reliable orientation GT for detector training',
                       'Type1 real val has only two objects','Jury GTX1050Ti unavailable',
                       'Source-disjointness and unknown external pretraining not established']}
    write(REPORT/'training_summary.json',summary)
    guard_check()
    lines=['# PHASE H — real data adaptation','',
        'Завершена ограниченная OCR-адаптация. BEST-G, исходная разметка, membership и группы DATA-3 сохранены.',
        '', '## Данные и протокол','',
        'Train 370: type1 5, type1a 121, type1b 244. Val 93: 2/30/61. Подтверждённых quad/OBB 0: H2 и H3 пропущены. Обычные bbox в OBB не превращались.',
        'Два CRNN fine-tuning от одного BEST Phase D checkpoint: .75 и .5 real, replay 400 каждого типа, независимый от replay synthetic control 30 каждого типа. Умеренные аугментации; shared BEST-G preprocessing. При обучении type1a используются только автоматические split/stitch кандидаты, без выбора по GT. Оригиналы сохранены.',
        'OCR checkpoint выбирался по real val raw OCR FPA; итоговая конфигурация — по val E2E FPA. В пределах одного совпадения от максимума выбран более простой вариант. Профиль russian_v1 — отдельная postprocess ablation по подтверждённым val ошибкам, пороги и detector не изменены.',
        '', '## Real val','',
        '| MODEL | FPA | type1a | type1b | OCR good bbox | detector recall | GPU mean/p95 ms |',
        '|---|---:|---:|---:|---:|---:|---:|']
    pct=lambda v:f'{v*100:.1f}%' if v is not None else '—'
    for r in table:
        speed=f"{r['GPU_MEAN']:.1f}/{r['GPU_P95']:.1f}" if r['GPU_MEAN'] is not None else '—'
        lines.append('| '+r['MODEL']+' | '+' | '.join(pct(r[k]) for k in
            ('REAL_VAL_FPA','TYPE1A_VAL','TYPE1B_VAL','OCR_GOOD_CROP_FPA','DETECTOR_RECALL'))+' | '+speed+' |')
    lines += ['', '**FINAL-H:** '+selection['selected']['name']+'. Freeze: configs/phase_h/final_h_freeze.json.',
        '', 'Good-bbox OCR использует подтверждённый bbox и GT type только как отдельную OCR диагностику. Quad GT отсутствует. Split success — структурный proxy, полного эталона строк нет. Waterfall не назначает BAD_RECTIFICATION только по несовпадению текста; неподтверждённые причины остаются UNKNOWN.',
        '', '## Обучение','']
    for key,t in training.items():
        lines.append(f"- {key}: {len(t['history'])} эпох, best raw val {t['best_correct']}/93, stop={t['stop_reason']}; best.pt/last.pt, optimizer/RNG и history сохранены.")
    lines += ['', '## PARTIAL-GT INDEPENDENT TEST','',
        'Один прогон BEST-G и FINAL-H после freeze. Все 40 GT targets подтверждены человеком. Результаты не использовались для изменений модели.',
        '', '| MODEL | selected-object FPA | type1a | type1b | raw OCR good bbox | selected detector recall |',
        '|---|---:|---:|---:|---:|---:|']
    for key,m in test['results'].items():
        lines.append('| '+key+' | '+' | '.join([pct(m['overall']['FPA']),pct(m['per_type']['type1a']['FPA']),
            pct(m['per_type']['type1b']['FPA']),pct(m['overall']['OCR_good_bbox_raw_FPA']),
            pct(m['overall']['selected_object_detector_recall'])])+' |')
    lines += ['', 'Полные image-level precision, FP/image, complete-scene recall и multiple-plate correctness не вычислялись: дополнительные номера сцен не размечены. 20 объектов на тип дают только небольшой целевой пилот. Не гарантируется отсутствие публичных фото в неизвестном pretraining.',
        '', '## Производительность','']
    for label in ('BEST_G','FINAL_H'):
        for device in ('cuda','cpu'):
            p=read(REPORT/'performance'/(label+'_'+device+'.json'))
            lines.append(f"- {label} {device}: mean {p['mean_ms']:.2f} ms, p95 {p['p95_ms']:.2f} ms, {p['images']} unique val images, 5 warmups; peak allocated/reserved {p['peak_allocated_MiB']}/{p['peak_reserved_MiB']} MiB.")
    lines += ['', 'Измерено на GTX1060 6GB и локальном CPU, 4 threads; full pipeline без загрузки весов, чтения/декодирования и CSV. Это не замер на GTX1050Ti 4GB. Соответствие лимиту жюри 100ms на его устройстве не подтверждено.',
        '', '## Запуск и ограничения','',
        'CUDA: `.venv-cuda/Scripts/python.exe tools/predict_images_h.py --input <folder> --output <result.csv> --device cuda`.',
        'CPU: тот же скрипт с `--device cpu`. Инференс не получает GT и проверяет freeze hashes, использует локальные веса и offline guard.',
        'Следующий отдельный пункт — полная человеческая разметка всех номеров 40 test scenes; detector DATA-QUAD, vehicle-check experiment и organizer-ready dataset export остаются отдельными работами. Полный пакет сдачи этим этапом не объявляется готовым.']
    (ROOT/'docs/phase_h_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('PHASE H report complete',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=['compare','freeze','test','report'])
    args=parser.parse_args()
    cv2.setNumThreads(1)
    torch.set_num_threads(4)
    with offline_network():
        {'compare':compare,'freeze':freeze,'test':final_test,'report':report}[args.stage]()
