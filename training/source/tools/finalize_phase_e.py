"""Assemble completed classical inference experiments, QA and scoped decisions."""
from collections import Counter
from datetime import datetime,timezone
from dataclasses import asdict
import csv
import json
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset.schema import sha256
from tools.phase_e import ROOT,REPORT,PUBLIC,BEST,read,write,official,freeze


def main():
    frozen=freeze()
    experiments=read(REPORT/'experiments.json');best=experiments['BEST_CLASSICAL_E']
    assert best['official']['full_match_total']==21
    assert best['official']['per_type']['type1']['full_match']==8
    assert best['official']['per_type']['type1a']['full_match']==13
    assert sha256(REPORT/'legacy_A_cpu.csv')==sha256(ROOT/'reports/baseline_predictions.csv')
    assert sha256(REPORT/'legacy_B_cpu.csv')==sha256(ROOT/'reports/phase_d/real_best_capture.csv')
    cpu=read(REPORT/'benchmark_cpu_final.manifest.json')
    gpu=read(REPORT/'benchmark_gpu.manifest.json')
    for name in ('benchmark_cpu_final','benchmark_gpu'):
        metrics,_=official(REPORT/f'{name}.csv',REPORT/name)
        assert metrics['full_match_total']==21
        assert metrics['per_type']['type1']['full_match']==8
        assert metrics['per_type']['type1a']['full_match']==13
    def texts(path):
        with path.open(encoding='utf-8',newline='') as f:
            return sorted((r['image'],r['plate_num'],r['plate_type']) for r in csv.DictReader(f,delimiter=';'))
    assert texts(REPORT/'benchmark_cpu_final.csv')==texts(REPORT/'benchmark_gpu.csv')==texts(Path(best['predictions']))
    suite=ET.parse(REPORT/'final_cpu_tests.xml').getroot()
    suites=suite.findall('.//testsuite')
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    tests['passed']=tests['tests']-tests['skipped']
    assert tests['passed']==282 and tests['skipped']==4 and tests['errors']==tests['failures']==0
    reviewed=read(REPORT/'input_review/manifest.json');assert reviewed['review_status']=='PASS' and reviewed['reviewed_count']==83
    split=read(REPORT/'type1a_split/diagnostics.json');assert split['review_status']=='PASS_WITH_DOCUMENTED_FAILURES'
    oracle=read(REPORT/'oracle_expanded_comparison.json');spatial=read(REPORT/'spatial_oracle_analysis.json')
    assert oracle['count']==31
    assert sha256(REPORT/'oracle_expansion/manifest.json')==oracle['manifest_sha256']
    assert read(REPORT/'oracle_expansion/manifest.json')['review_status']=='PASS'
    attribution=read(REPORT/'error_attribution_summary.json');counts=attribution['categories']
    assert sum(counts.values())==115 and counts['CORRECT']==21
    post=read(REPORT/'postprocess_analysis.json');deskew=read(REPORT/'deskew_quality.json')
    table=[]
    def row(label,official_result,latency,scope,decision,name):
        m=official_result
        return {'configuration':label,'correct':m['full_match_total'],'total':m['gt_target_plates'],
            'full_plate_accuracy':m['full_match_total']/m['gt_target_plates'],
            'type1':{'correct':m['per_type']['type1']['full_match'],'total':m['per_type']['type1']['gt_plates']},
            'type1a':{'correct':m['per_type']['type1a']['full_match'],'total':m['per_type']['type1a']['gt_plates']},
            'official_score':m['score'],'latency_ms':latency,'latency_scope':scope,'decision':decision,'experiment':name}
    for key,label in (('legacy_A_cpu','OLD BASELINE'),('legacy_B_cpu','BEST CRNN + OLD GEOMETRY')):
        manifest=read(REPORT/f'{key}.manifest.json');score=read(REPORT/f'{key}.official.json')
        table.append(row(label,score,manifest['image_timing_ms'],'Full CPU, read/hash/CSV included, first image cold',
                         'KEEP frozen reference',key))
    timing=read(REPORT/'diagnostic_timings.json')
    for name,label in (('filter_none','NO GEOMETRY FILTER'),('deskew_none','NO DESKEW'),
                       ('padding_00','BEST PADDING'),('type1a_stitch','TYPE1A STITCH'),
                       ('oracle_type_routing','ORACLE TYPE ROUTING'),('raw_ocr','RAW OCR'),('postprocess','POSTPROCESS')):
        result=experiments[name];lat=(timing.get(name) or result)['cached_pipeline_ms']
        scope='Cached CPU detections + GPU OCR; source image IO/detector/debug writes excluded, one pass; not E2E latency'
        decision='DIAGNOSTIC ONLY' if name=='oracle_type_routing' else 'KEEP component in joint configuration' if name in ('padding_00','type1a_stitch','postprocess','deskew_none') else 'REJECT alone; assess jointly'
        table.append(row(label,result['official'],lat,scope,decision,result['name']))
    # Oracle crop has a different denominator and supplies manual geometry/type.
    om=oracle['metrics']['oracle_known_type'];total=om['all']['count']
    oracle_precision=om['all']['correct']/total
    oracle_score=.15*om['type1']['full_plate_accuracy']+.35*om['type1a']['full_plate_accuracy']+.5*oracle_precision
    table.insert(7,{'configuration':'ORACLE CROP','correct':om['all']['correct'],'total':total,
        'full_plate_accuracy':om['all']['full_plate_accuracy'],
        'type1':{'correct':om['type1']['correct'],'total':om['type1']['count']},
        'type1a':{'correct':om['type1a']['correct'],'total':om['type1a']['count']},
        'official_score':None,'diagnostic_subset_score_formula':oracle_score,
        'official_score_note':'Not an official full-image submission: manually located crops and known types. Score deliberately N/A.',
        'latency_ms':oracle['latency']['oracle_known_type'],'latency_scope':oracle['latency_scope'],
        'decision':'DIAGNOSTIC ONLY; never production routing/rectification','experiment':'expanded31_oracle'})
    table.append(row('BEST-CLASSICAL-E',best['official'],cpu['read_plus_pipeline_ms'],
        'Full CPU, live YOLO+read, warmup5 excluded,3 repeats/300 images; CSV excluded',
        'KEEP independent image configuration; no hidden-test claim',best['name']))
    initial=read(REPORT/'error_attribution_baseline_summary.json')
    decision={'primary':'CASE A: detector coverage / detector boxes / geometry',
        'evidence':{'detector_miss':counts['DETECTOR_MISS'],'detector_bad_box':counts['DETECTOR_BAD_BOX'],
            'type1a_bad_split':counts['TYPE1A_BAD_SPLIT'],'verified_OCR_failure':counts['OCR_FAILURE'],
            'oracle_type_routing_correct':experiments['oracle_type_routing']['official']['full_match_total'],
            'auto_type_routing_correct':21,'oracle_crop_raw':oracle['metrics']},
        'secondary':'Type1a row evidence and real OCR domain gap are also measurable; full-input OCR-only oracle FPA remains58.06%, type1a45.45% including5 failed splits.',
        'next_scope':'Separately authorized detector data/coverage and actual crop geometry, then robust type1a conversion; re-evaluate remaining correctly prepared real inputs before OCR model tournament.',
        'not_started':['OCR training','FastPlateOCR','SVTR','PARSeq','YOLO OBB','YOLO Pose','new classifier training']}
    result={'status':'COMPLETE','phase':'E0-E20','date':'2026-10-03','generated_utc':datetime.now(timezone.utc).isoformat(),
        'development_set_scope':frozen['scope'],'baseline_freeze_sha256':sha256(REPORT/'baseline_freeze.json'),
        'BEST_CLASSICAL_E':{'config':best['config'],'detector':'models/yolo11_plate.pt',
            'OCR_checkpoint':str(BEST),'OCR_sha256':sha256(BEST),'official':best['official'],
            'entry_point':'tools/predict_images.py','config_file':'configs/image_anpr_e.json',
            'legacy_video_unchanged':True,'GT_in_production':False,'new_training':False},
        'baseline_cascade':initial,'final_cascade':attribution,'table':table,
        'all_ablations':{k:{'config':v['config'],'official':v['official'],'retained':v['retained'],
            'paired_target_proposals_retained':v['manually_paired_target_proposals_retained'],
            'unpaired_proposals_retained':v['unpaired_proposals_retained'],'false_detections_retained':None,
            'stitch_successes':v['type1a_stitched']} for k,v in experiments.items()},
        'crop_success':{'ocr_calls':best['ocr_calls'],'retained':best['retained'],
            'type1a_routed':split['attempted'],'type1a_stitched':split['successful'],
            'square_type_not_rejected_by_old_assumptions':True,'perspective_correction':False,
            'note':'Current detector outputs axis-aligned bbox only, no quad. 05_rectified debug images show the actual unchanged/deskew stage, not a fabricated homography.'},
        'router':read(REPORT/'router_analysis.json'),'postprocess':post,'deskew':deskew,
        'oracle':oracle,'spatial_oracle':spatial,
        'performance':{'CPU':cpu,'GPU':gpu,
            'preliminary_CPU':{'path':'reports/phase_e/benchmark_cpu.manifest.json','used_for_final':False,
                'reason':'Preliminary run overlapped diagnostic artifact activity; repeated in isolation after all other tasks completed.'},
            'hardware':'Intel i5-6500, GTX1060 6GB; not reference i5-7600/GTX1050Ti',
            'stage_other_overhead':'BBox extraction and routing measured separately. Resize/upscale, metadata and tensor hash overhead are included in total and residual overhead, not falsely assigned to detector.'},
        'tests':tests,'decision_gate':decision,
        'integrity':{'frozen_sources_weights_public_hashes_pass':True,'baseline_A_CSV_bit_identical':True,
            'baseline_B_CSV_bit_identical':True,'CPU_GPU_all_prediction_text_types_identical':True,
            'Phase_C_D_preprocessing_unchanged':True},
        'artifacts':{'report':'docs/phase_e_report.md','experiments':'docs/experiments.md',
            'attribution':'reports/phase_e/error_attribution.csv','baseline_attribution':'reports/phase_e/error_attribution_baseline.csv',
            'split_diagnostics':'reports/phase_e/type1a_split/diagnostics.json',
            'oracle_annotations':'reports/phase_e/oracle_expansion/manifest.json',
            'public_predictions':'reports/phase_e/benchmark_cpu_final.csv','test_report':'reports/phase_e/final_cpu_tests.xml'}}
    if (REPORT/'artifact_validation.json').exists():
        result['final_artifact_QA']=read(REPORT/'artifact_validation.json')
        result['frozen_stitch_label_check']=read(REPORT/'stitch_A123BC777_validation.json')
        result['artifacts']['artifact_QA']='reports/phase_e/artifact_validation.json'
        result['artifacts']['frozen_stitch_label_check']='reports/phase_e/stitch_A123BC777_validation.json'
    write(REPORT/'final_comparison.json',result)
    # Stage sums exclude explicit resize/metadata overhead; record that residual.
    for name,measure in (('CPU',cpu),('GPU',gpu)):
        result['performance'][name]['mean_other_overhead_ms']=measure['pipeline_ms']['mean']-sum(v['mean'] for v in measure['stage_ms_per_image'].values())
    write(REPORT/'final_comparison.json',result)
    lines=['# PHASE E — geometry и type1a routing','',
        'Дата: 2026-10-03. E0–E20 завершены. Новый OCR/classifier/detector training не запускался; архитектуры и исходные weights сохранены.','',
        '**BEST-CLASSICAL-E: 21/115 (18.26%) против 8/115 (6.96%) baseline.** Type1: 8/15 до/после; type1a: 0/100 → 13/100. Official score: 18.81 OLD / 19.43 BEST CRNN+old geometry → 25.35.','',
        'Public debug здесь является набором для разработки: на нём по заданию сравнивались параметры geometry/padding/deskew. Это не независимая оценка обобщения. Ни public, ни manual oracle crops не использовались для training или выбора OCR weights.','',
        '## E0–E3: заморозка и независимый image path','',
        'BASELINE-A и B воспроизведены. Оба CSV побайтно совпали с сохранёнными Phase D результатами. `baseline_freeze.json` содержит SHA256 исходных sources, YOLO/старого и BEST CRNN, baseline CSV, common preprocessing, всех100 фото, GT и scorer. Все hashes повторно проверены.','',
        'Новый `src/image_anpr.py::ImageANPRPipeline` обрабатывает каждую картинку независимо. У него нет tracker, IoU tracking, temporal votes, frame accumulation, target_fps, OCR every N frames или200ms limiter. Legacy `run.py`/video не изменён. Отдельный entry point — `tools/predict_images.py`. Debug OFF по умолчанию.','',
        'Debug для всех83 proposals содержит доступные стадии01_original/02_bbox/03_padded_crop/04_deskew/05_rectified/06_top/07_bottom/08_stitched/09_ocr_input и metadata. Не выполненные стадии явно указаны. Quad у старого YOLO отсутствует: perspective correction в обычном inference не делалась. Имя05_rectified означает текущий geometry input, а не выдуманную homography.','',
        '## E1: количественное разложение115 целей','',
        '| Причина | Baseline-B | BEST-CLASSICAL-E |','|---|---:|---:|',
        f"| Правильный номер | 8 | {counts['CORRECT']} |",
        f"| DETECTOR_MISS | 36 | {counts['DETECTOR_MISS']} |",
        f"| DETECTOR_BAD_BOX | 20 | {counts['DETECTOR_BAD_BOX']} |",
        f"| GEOMETRY_FILTER_REJECTION | 33 | {counts['GEOMETRY_FILTER_REJECTION']} |",
        f"| TYPE_NOT_ROUTED | 15 | {counts['TYPE_NOT_ROUTED']} |",
        f"| TYPE1A_BAD_SPLIT | 0, split не вызывался | {counts['TYPE1A_BAD_SPLIT']} |",
        f"| OCR_FAILURE на проверенном входе | 2 | {counts['OCR_FAILURE']} |",
        f"| UNKNOWN | 1 | {counts['UNKNOWN']} |",'| Итого |115|115|','',
        'Это primary причина каждого scored target, а не сумма139 literal FN/FP records. Три ignored GT исключены, wildcard/unknown-region semantics применяет неизменённый scorer. Все83 proposals просмотрены; spatial pairing ручное и не заменяет официальные GT bbox. В cropped/failed-split случаях ошибка не назначается OCR. BAD_DESKEW и POSTPROCESS_FAILURE как primary причина в итоговом no-deskew pipeline отсутствуют; это не означает отсутствия любых ошибок на других стадиях.','',
        'Исходный E1 CSV сохранён как `error_attribution_baseline.csv`; окончательное разложение — `error_attribution.csv`. UNKNOWN сохраняет две частично читаемые цели без надёжной пространственной пары.','',
        '## E4–E6: filter, deskew, padding','',
        '| Routed geometry filter | Retained proposals | Manually paired target proposals | FPA |','|---|---:|---:|---:|']
    for mode in ('old','none','relaxed','type_aware'):
        v=experiments['routed_filter_'+mode]
        lines.append(f"| {mode} |{v['retained']}/83|{v['manually_paired_target_proposals_retained']}|{v['official']['full_match_total']}/115|")
    lines += ['',
        'OLD FILTER REJECT: он отбрасывал33 вручную сопоставленные type1a proposals до OCR. Без routing/stitch снятие фильтра оставляет8/115; вместе со stitch результат21/115. Выбран relaxed filter:0.8≤aspect≤8, width≥20,height≥10. Он оставляет82/83,77 paired proposals; NONE оставляет83/83,78 paired. Разница — сильно перспективный B095CO122, а не доказанная ложная детекция. Relaxed даёт тот же FPA и немного больший official score; остаётся1 известный false rejection.','',
        'Число unpaired partial proposals равно5; оно не объявляется числом false detections. Глобальная detector IoU recall по этому набору без полной spatial GT недоступна.','',
        'После type1a integration, при одинаковом padding0/filter/router: NO DESKEW21/115, OLD DESKEW18/115, corrected-sign21/115. Контрольные вращения ±5/10° показали удвоение угла старым знаком. При ±15° остаток достигает примерно ±30°, выходит за legacy search window ±20° и прежний estimator возвращает0; это не успешное выравнивание. Отдельный diagnostic Hough measurement с окном ±45° подтверждает этот остаток. Corrected исправляет знак, но не улучшает real FPA относительно NO DESKEW; выбран NO DESKEW. Сложный production deskew не создавался.','',
        'Grid padding0/3/5/6/8/12% проверен как отдельно, так и совместно со stitch и тремя deskew вариантами. В финальной relaxed ветке NO DESKEW:21/21/18/16/14/10 правильных соответственно; при3% type1 падает до7/15. Выбран один общий default0% по заданному критерию FPA/no type1 regression/score. Для отдельных фото padding не менялся.','',
        '## E7–E11: router, общий stitch и rectification','',
        'Router — отдельный заменяемый `classify(crop,detection_metadata)`. Baseline использует только размеры: aspect≤3→type1a, иначе type1; надёжный explicit detector class имеет приоритет. Старый detector имеет один license-plate class, поэтому type1b геометрически не отличим от type1 и использует тот же monoline OCR path. Other explicit class отклоняется. Confusion/error examples сохранены с разделением complete и clipped boxes.','',
        f"Phase C horizontal_projection/stitch и Phase D `prepare_ocr_input` переиспользованы без изменения кода/config: height64,padding4,gap10, normalized32×160. Regression A123BC777 проверяет порядок top/bottom, transforms и точное равенство normalized tensor на одном resolved input. Из{split['attempted']} routed type1a inputs успешно stitched{split['successful']}; остальные документированно fallback whole crop, а не успешный split.",'',
        'Все83 padded crops/actual inputs и все routed type1a diagnostics просмотрены. Для28 успешных split обе строки, region и порядок видимы; structural evidence и отсутствие разреза видимых glyphs проверены. Это не доказывает наличие символов, которые уже отсутствовали в detector bbox. RUS/flag остаются видимыми; они могут влиять на OCR, поэтому отдельная metadata отметка не объявляет их безвредными для recognizer.','',
        'Обычная rectification — bbox/padding baseline. Не создавались fake quad или neural rectifier. Настоящие ручные quad используются исключительно ниже как диагностика.','',
        '## E12–E13, E17: oracle и область применимости','',
        'Automatic vs ORACLE TYPE ROUTING:21/115 в обоих случаях. GT type принудительно задаётся только вручную сопоставленным proposals. Unpaired proposals сохраняют automatic routing и не исчезают из scoring по подсказке GT. Это условный oracle на сопоставленных областях, не полная spatial oracle верхняя граница. Router не выглядит главным новым ограничением.','',
        'Oracle crops расширены до31:9 type1 и22 type1a (8 прежних +23 additions). Quad размечены по видимым borders в исходных координатах, crops/input просмотрены до expanded OCR evaluation; не использовались для обучения. Для AUTO и ORACLE одинаковы известный type, resize policy и общий preprocessing.','',
        '| Известный type, RAW OCR | AUTO | ORACLE |','|---|---:|---:|',
        '| Все31 |13/31 (41.94%)|18/31 (58.06%)|','| type1 |7/9|8/9|','| type1a |6/22|10/22|',
        '| CER, все31 |27.24%|16.79%|','',
        'Среди22 oracle type1a пять остаются whole-crop fallback. Среди всех26 oracle inputs с корректным monoline/успешным stitch RAW FPA18/26 (69.23%); качество OCR на реальном домене также не идеально. Подборка удобная/видимая, содержит трудные случаи, не репрезентативна для всех115. Её18/31 нельзя смешивать с E2E21/115.','',
        f"На этих31 manually annotated plates recall по enclosing GT bbox: IoU0.3={spatial['recall_at_IoU']['0.3']*100:.2f}%, IoU0.5={spatial['recall_at_IoU']['0.5']*100:.2f}%, IoU0.75={spatial['recall_at_IoU']['0.75']*100:.2f}%. Это selected detectable subset; никаких заявлений о global detector recall не делается.",'',
        '## E14: RAW / current postprocess','',
        'На одинаковых82 OCR calls postprocessing исправил2 full matches, испортил0, не изменил29, изменил без full-match gain51. Общая FPA RAW19/115→POST21/115. На37 вручную проверенных complete и правильно подготовленных inputs CER9.75%→14.15%; для type1a12.08%→18.33%. Current postprocessing оставлен по full-plate gain, но его увеличение CER на ошибочных строках задокументировано. Само изменение строки не считается улучшением.','',
        '## E15, E18, E20: итоговая таблица','',
        'Столбец latency явно задаёт scope: E2E CPU с чтением для baseline/final, cached detections+GPU OCR для exploratory ablations; oracle crop — только supplied crop+OCR. Эти разные latency нельзя сравнивать как ускорение полного pipeline.','',
        '| Configuration | Correct/total | FPA | type1 | type1a | Score | Mean ms / scope | KEEP/REJECT |','|---|---:|---:|---:|---:|---:|---|---|']
    for v in table:
        a,b=v['type1'],v['type1a'];score=f"{v['official_score']*100:.2f}" if v['official_score'] is not None else 'N/A subset'
        latency=v['latency_ms']['mean']
        scope='CPU+read' if v['configuration'] in ('OLD BASELINE','BEST CRNN + OLD GEOMETRY','BEST-CLASSICAL-E') else 'crop+GPU OCR' if v['configuration']=='ORACLE CROP' else 'cached+GPU OCR'
        lines.append(f"|{v['configuration']}|{v['correct']}/{v['total']}|{v['full_plate_accuracy']*100:.2f}%|{a['correct']}/{a['total']}|{b['correct']}/{b['total']}|{score}|{latency:.2f} {scope}|{v['decision']}|")
    lines += ['',
        '**BEST-CLASSICAL-E:** frozen YOLO11n→relaxed filter→bbox padding0→NO DESKEW→geometry router→Phase C/D type1a stitch→EXP-D-01 BEST CRNN→current postprocess. В image path нет GT/temporal logic. Новый default находится в `configs/image_anpr_e.json`; старый production/video entry point не заменён.','',
        'Изолированный финальный benchmark:5 warmup images,3 repeats×100, debug OFF, CUDA synchronized. CPU/GPU тексты и types всех82 predictions совпали с cached experiment. Hardware: i5-6500/GTX1060 6GB, не reference i5-7600/GTX1050Ti. Первый CPU замер выполнялся рядом с записью diagnostics и исключён; итоговый повтор измерен после завершения остальных задач.','',
        '| Backend / latency | Mean | Median | p90 | p95 |','|---|---:|---:|---:|---:|']
    for name,measurement in (('CPU pipeline без чтения',cpu['pipeline_ms']),('CPU read+pipeline',cpu['read_plus_pipeline_ms']),('GPU pipeline без чтения',gpu['pipeline_ms']),('GPU read+pipeline',gpu['read_plus_pipeline_ms'])):
        lines.append('|'+name+'|'+ '|'.join(f"{measurement[k]:.2f}" for k in ('mean','median','p90','p95'))+'|')
    lines += ['', '| Stage, ms/image mean | CPU | GPU |','|---|---:|---:|']
    for stage in cpu['stage_ms_per_image']:
        lines.append(f"|{stage}|{cpu['stage_ms_per_image'][stage]['mean']:.3f}|{gpu['stage_ms_per_image'][stage]['mean']:.3f}|")
    lines += ['',
        'Resize/upscale, metadata/tensor hash и прочая малая overhead включены в total; stage crop измеряет bbox extraction. Детальные mean/median/p90/p95 каждого этапа находятся в JSON. Скорость за счёт ухудшения accuracy не оптимизировалась.','',
        '## E16, E19: решение и проверки','',
        'Основной следующий gate — CASE A:36 missed+20 bad-box targets, ещё18 failed real type1a splits. Старый filter/deskew и отсутствие stitch исправлены; новая архитектура пока не запускалась. Нужны detector data/coverage и настоящая crop geometry, затем устойчивость split. CASE C тоже подтверждается на части правильно подготовленных реальных inputs, поэтому OCR не объявлен решённым; его дальнейший выбор следует делать после отделения crop/split/domain ошибок. Oracle type gap отсутствует, а postprocess даёт2 gains/0 breaks, поэтому learned router/decoder redesign не выбран главным следующим шагом.','',
        'Финальная CPU suite:282 passed,4 documented legacy xfail,0 errors/failures. Добавлены26 regression cases: image independence; old/new geometry; square type1a; replaceable router/explicit classes/other; failed splits; A123BC777 stitch order и preprocessing equality; no deskew; padding/bounds; multiple/zero plates; debug OFF; real quad provenance; exact output CSV; sign controls. TEMP/TMP и test checkpoints находятся на D:, исходная CPU environment сохранена.','',
        'Дополнительно настоящий frozen BEST CRNN распознал независимо отрендеренный type1a A123BC777 без ошибки. Финальная проверка1115 ссылок на intermediate artifacts не нашла отсутствующих файлов; никаких substitute изображений/предсказаний не создано.','',
        'Никаких новых OCR weights, OBB/Pose, OCR tournament или classifier training не создано. PHASE E останавливается на измеренном BEST-CLASSICAL-E и decision gate.','',
        '## Запуск','',
        'Из `D:\\volga\\volga-anpr`:','',
        '```powershell',
        '.\\.venv\\Scripts\\python.exe tools/predict_images.py --input "D:\\path\\to\\images" --output reports/image_predictions.csv',
        '.\\.venv-cuda\\Scripts\\python.exe tools/predict_images.py --input "D:\\path\\to\\images" --output reports/image_predictions_gpu.csv --device cuda --warmup 5',
        '# Debug по явному запросу:',
        '.\\.venv\\Scripts\\python.exe tools/predict_images.py --input "D:\\path\\to\\image.jpg" --output reports/image_debug.csv --debug',
        '```','',
        'Исчерпывающие configs/results, per-detection metadata, contact sheets, split diagnostics, oracle annotation provenance, attribution CSV и JUnit находятся в `reports/phase_e/`. Машиночитаемый итог — `final_comparison.json`.']
    (ROOT/'docs/phase_e_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    path=ROOT/'docs/experiments.md';text=path.read_text(encoding='utf-8')
    marker='## EXP-E — Independent image geometry and type1a routing'
    if marker not in text:
        addition=f"""\n\n{marker}

Date: 2026-10-03. Public debug is explicitly the development set for classical inference parameters; no independent generalization claim and no new model training. Frozen legacy A/B CSV reproduced byte-for-byte. Independent ImageANPRPipeline has no tracking/voting/time deduplication, uses shared Phase C/D preprocessing and replaceable dimension-only router.

Best relaxed filter, padding0, no deskew, ratio≤3→type1a, Phase C stitch, fixed EXP-D-01, current postprocess: **21/115 (18.26%)** vs8/115. Type1 stays8/15; type1a0→13/100. Official score18.81/19.43→25.35. Old filter retains47/83 vs82/83 relaxed; one remaining known false rejection. With correct routing, old deskew18 vs no/corrected21; padding increases do not improve the accepted configuration. Oracle type routing21/115, no significant classifier ceiling gap on paired proposals.

Expanded manually verified oracle31, evaluation-only: raw auto13/31→oracle18/31; type1 7/9→8/9, type1a6/22→10/22. Five oracle type1a fail structural split; among26 properly prepared oracle inputs18 correct. Domain gap remains alongside dominant detection/geometry losses. Selected bbox IoU recall is scoped to visible manually annotated targets, never global.

Final target cascade: correct21,miss36,bad_box20,filter_reject1,wrong_route1,bad_split18,verified_OCR16,unknown2; total115. Postprocess fixed2/broken0/unchanged29/changed-no-full-gain51, RAW19→POST21; paired good-input CER increases9.75→14.15%. Whole CPU/GPU warm benchmarks and component percentiles in final JSON. All82 CPU/GPU texts/types agree. Tests282 passed+4 legacy xfail. Old sources/models/video remain intact; no OBB/Pose/OCR tournament started. Next gate CASE A with measurable split/domain secondary issues.

Artifacts: `docs/phase_e_report.md`, `reports/phase_e/final_comparison.json`, `error_attribution.csv`, `type1a_split/diagnostics.json`, `oracle_expansion/manifest.json`, `final_cpu_tests.xml`. New image entry point `tools/predict_images.py`, config `configs/image_anpr_e.json`.
"""
        path.write_text(text+addition,encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','FPA':'21/115','type1':'8/15','type1a':'13/100',
        'score':best['official']['score'],'tests':tests,'CPU_mean':cpu['pipeline_ms']['mean'],'GPU_mean':gpu['pipeline_ms']['mean']}))


if __name__=='__main__':main()
