"""Assemble Phase D evidence after completed training/evaluation; no selection by test."""
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset.schema import sha256


ROOT=Path(__file__).resolve().parents[1]
REPORT=ROOT/'reports/phase_d'


def read(path):return json.loads((ROOT/path).read_text(encoding='utf-8-sig'))
def write(path,value):(ROOT/path).write_text(json.dumps(value,indent=2,ensure_ascii=False),encoding='utf-8')


def main():
    pre=read('reports/phase_d/pretrain_comparison.json')
    final=read('reports/phase_d/final_synthetic_comparison.json')
    real=read('reports/phase_d/real_benchmark/comparison.json')
    assert len(real['trials'])==6 and pre['dataset_fingerprint']==final['dataset_fingerprint']
    selection=read('reports/phase_d/selection.json')
    train={k:read(f'models/phase_d/exp_d_0{i}_{k}/training_report.json')
        for i,k in ((1,'finetune'),(2,'random'))}
    assert train['finetune']['data_hashes']==train['random']['data_hashes']
    selected=Path(selection['checkpoint']);assert sha256(selected)==selection['checkpoint_sha256']
    assert selection['validation']['full_plate_accuracy']==1 and selection['validation']['cer']==0
    reference=read('reports/baseline_predictions.manifest.json')
    for path,digest in reference['source_hashes'].items():assert sha256(ROOT/path)==digest
    for path,info in reference['models'].items():assert sha256(ROOT/path)==info['sha256']
    assert sha256(Path(real['trials']['old']['predictions']))==sha256(ROOT/'reports/baseline_predictions.csv')
    assert sha256(Path(real['trials']['finetune']['predictions']))==sha256(ROOT/'reports/phase_d/real_best_capture.csv')
    phase_b=read('reports/phase_b/accepted_dataset.json');broot=Path(phase_b['accepted_dataset'])
    for filename,key in (('meta.csv','meta_sha256'),('generator/manifest.json','generator_manifest_sha256'),('splits/manifest.json','split_manifest_sha256')):
        assert sha256(broot/filename)==phase_b[key]
    dataset=read('reports/phase_d/dataset_validation.json');assert dataset['status']==dataset['visual_gate']=='PASS'
    splits=read('dataset/splits/ocr_manifest.json');droot=Path(splits['dataset_root'])
    for name,info in splits['splits'].items():
        assert sha256(Path(info['export']))==info['sha256']==sha256(droot/'splits'/f'{name}.csv')
    assert dataset['cache_sha256']==sha256(droot/'generator/ocr_inputs.npy')
    contract=json.loads((droot/'generator/ocr_inputs.json').read_text())
    for path,digest in contract['preprocessing_hashes'].items():assert sha256(Path(path))==digest
    suites=ET.parse(REPORT/'final_cpu_tests.xml').getroot().findall('.//testsuite')
    tests={k:sum(int(s.get(k,0)) for s in suites) for k in ('tests','failures','errors','skipped')}
    assert not tests['failures'] and not tests['errors'];tests['passed']=tests['tests']-tests['skipped']
    paths={'old':'models/ocr_crnn_nomeroff_clean.pt','smoke':'models/phase_d/smoke/best.pt',
        'finetune':'models/phase_d/exp_d_01_finetune/best.pt','random':'models/phase_d/exp_d_02_random/best.pt'}
    synt={k:(pre if k in ('old','smoke') else final)['results'][p] for k,p in paths.items()}
    assert synt['finetune']['metrics']['count']==synt['random']['metrics']['count']==7000
    labels={'old':'Old CRNN','smoke':'Smoke CRNN','finetune':'Fine-tuned CRNN','random':'Random-init CRNN'}
    decisions={'old':'KEEP production baseline','smoke':'REJECT production; mechanical checkpoint',
        'finetune':'KEEP best OCR candidate; REJECT automatic production promotion',
        'random':'REJECT; validation tie, poor real transfer'}
    table=[]
    for k in paths:
        trial=real['trials'][k];m=synt[k]['metrics'];official=trial['official']
        table.append({'model':labels[k],'checkpoint':paths[k],'checkpoint_sha256':trial['checkpoint_sha256'],
            'synthetic_test':m,'synthetic_by_difficulty':synt[k]['by_difficulty'],
            'real_correct':official['full_match_total'],'real_total':official['gt_target_plates'],
            'real_accuracy':official['full_match_total']/official['gt_target_plates'],
            'real_type1':official['per_type']['type1']['accuracy'],'real_type1a':official['per_type']['type1a']['accuracy'],
            'real_type1b':None,'official_score':official['primary_score'],'real_literal':trial['literal_metrics'],
            'latency_cpu_ms':trial['latency'],'latency_cuda_ocr_only_ms':synt[k]['latency_ms'],'decision':decisions[k]})
    table.append({**table[2],'model':'Best CRNN (validation-locked EXP-D-01)'})
    error_summary=read('reports/phase_d/real_best_analysis/summary.json')
    post=read('reports/phase_d/real_best_analysis/postprocess_ablation.json')
    oracle=read('reports/phase_d/oracle_comparison.json')
    best_oracle=oracle['results'][paths['finetune']]
    promotion=table[2]['real_accuracy']>table[0]['real_accuracy'] and table[2]['real_type1']>=table[0]['real_type1']
    assert not promotion
    production={'checkpoint':paths['old'],'checkpoint_sha256':table[0]['checkpoint_sha256'],
        'pipeline':'Frozen YOLO11n + original CRNN greedy + old deskew + constant type1 + current postprocess',
        'reason':'Predeclared promotion gate requires an E2E FPA gain and no type1 regression. New FPA ties baseline; preserve production. New evaluated OCR variant has slightly higher precision/official score.',
        'automatic_promotion':False}
    deskew={'checkpoint_fixed':paths['finetune'],'checkpoint_sha256':table[2]['checkpoint_sha256'],
        'old_deskew':real['trials']['finetune']['official'],'no_deskew':real['trials']['finetune_no_deskew']['official'],
        'old_weights_old_deskew':real['trials']['old']['official'],'old_weights_no_deskew':real['trials']['old_no_deskew']['official'],
        'decision':'E2E FPA ties8/115 in all four variants; no measured FPA harm/gain. Keep old deskew for locked candidate and preserved production.'}
    write('reports/phase_d/deskew_ablation.json',deskew)
    cuda=read('reports/phase_d/cuda_validation.json')
    cuda['large_training_measurements']={k:{'batch_size':r['config']['batch_size'],'GPU_memory_usage':r['GPU_memory_usage'],
        'images_per_sec_including_validation_and_checkpoints':r['images_per_sec_including_validation_and_checkpoints']} for k,r in train.items()}
    write('reports/phase_d/cuda_validation.json',cuda)
    selection_final={'status':'CONFIRMED','frozen_selection_sha256':sha256(REPORT/'selection.json'),
        'frozen_selection_time_utc':datetime.fromtimestamp((REPORT/'selection.json').stat().st_mtime,timezone.utc).isoformat(),
        'checkpoint':selection['checkpoint'],'checkpoint_sha256':selection['checkpoint_sha256'],
        'candidate_validation':{k:r['best_metric'] for k,r in train.items()},'test_public_used_for_selection':False,
        'tie_policy':selection['criterion'],'production':production}
    write('reports/phase_d/selection_final.json',selection_final)
    report={'status':'COMPLETE','phase':'D5-D21','generated_utc':datetime.now(timezone.utc).isoformat(),
        'CUDA':{k:cuda[k] for k in ('status','torch_version','torch_cuda_version','GPU','VRAM_bytes','driver','training_device','checks')},
        'CPU_environment':{'torch_actual_import':'2.14.0+cpu','cuda':None,'modified':False},
        'hardware':'Intel Core i5-6500 @3.20GHz; GTX1060 6GB. Not i5-7600/GTX1050Ti reference hardware.',
        'dataset':{'root':str(droot),'version':splits['version'],'size':70000,'types':dataset['counts']['by_type'],
            'difficulty':{'clean':14000,'medium':35000,'hard':21000},'difficulty_ratios':{'clean':.2,'medium':.5,'hard':.3},
            'splits':splits,'validation':dataset,'visual_review':str(REPORT/'dataset_preview/visual_review.json'),
            'prior_smoke_identities_excluded':True,'complete_public_GT_identity_overlap':0,
            'mask':'LDDDLLRR/RRR for all three types per user; three-digit region leading1/2/7',
            'limitations':'Authored approximate glyph bank, crop-only backgrounds, known generator homography/oracle type/stitch. Does not prove full-frame real accuracy.'},
        'selection':selection_final,'experiments':train,'adaptive_experiments':selection['adaptive_experiments'],
        'table':table,'primary_real_comparison':{'old':real['trials']['old']['official'],'new':real['trials']['finetune']['official'],
            'old_and_new_47_actual_OCR_inputs_identical':True,'old_predictions_bit_identical_to_saved_baseline':True,
            'new_uncaptured_predictions_bit_identical_to_captured':True,
            'global_OCR_CER':None,'CER_reason':'Full public GT lacks spatial boxes; report only manually paired complete-input and fixed oracle-subset CER.'},
        'errors':error_summary,'real_error_artifacts':str(REPORT/'real_best_analysis/errors'),
        'postprocess_ablation':{k:post[k] for k in ('scope','counts','raw_exact_count','final_exact_count','paired_metrics','paired_distance_effects')},
        'oracle_comparison':{'scope':oracle['scope'],'known_type_and_C_stitch_for_both':True,
            'auto_crop_bypasses_legacy_geometry_filter':True,'subset_size':8,
            'results':{p:{mode:v[mode]['metrics'] for mode in ('auto_crop','oracle_crop')} for p,v in oracle['results'].items()},
            'limitation':'Eight preselected manually reviewed type1a plates; not representative of all115 targets or detector recall.'},
        'deskew_ablation':deskew,'best_checkpoint':selection['checkpoint'],'best_pipeline':production,
        'best_evaluated_OCR_variant':{'checkpoint':paths['finetune'],'deskew':'old','official_score':table[2]['official_score'],
            'real_correct':8,'production_promoted':False},
        'bottleneck':{'primary':['detector coverage','legacy geometry filter/partial row crops','absent type1a stitch and type routing'],
            'evidence':'53 geometry failures,36 manually unmatched targets,15 absent type1a conversion among literal missed-GT cases; only2 verified complete monoline input OCR failures. Fixed type1 policy guarantees type1a E2E0. Oracle best6/8 vsauto4/8.',
            'secondary':'Residual real-domain/font errors exist (2/8 oracle plates); perfect synthetic metrics overstate transfer. Random-init0/8 oracle despite perfect synthetic test.',
            'next_action':'Stop CRNN tuning; next separately authorized phase should address detector/geometry/type1a integration. No other architectures started.'},
        'tests':tests,'integrity':{'original_sources_models_and_B_unchanged':True,'same_experiment_train_val_fingerprints':True,
            'same_four_model_synthetic_test_fingerprint':True,'fixed_split_exports_unchanged':True,'cache_and_preprocessing_hashes_match':True},
        'artifacts':{'cuda_validation':'reports/phase_d/cuda_validation.json','fixed_split_manifest':'dataset/splits/ocr_manifest.json',
            'pretrain':'reports/phase_d/pretrain_comparison.json','final_synthetic':'reports/phase_d/final_synthetic_comparison.json',
            'real_benchmark':'reports/phase_d/real_benchmark/comparison.json','oracle':'reports/phase_d/oracle_comparison.json',
            'postprocess':'reports/phase_d/real_best_analysis/postprocess_ablation.json','deskew':'reports/phase_d/deskew_ablation.json',
            'tests':'reports/phase_d/final_cpu_tests.xml'},
        'execution_notes':['Preliminary tests hit full C TEMP, then missing D TEMP parent. Own test checkpoints preserved by file moves to D; final tests on D pass. CPU packages unchanged.',
            'Initial redundant all-image preprocessing interrupted before training; final single-pass all70k validator passed with exact shared-preprocessing regression test.',
            'Benchmark scorer console needed explicit UTF8 in child Python; valid completed old trial reused with hash validation. Final six scored trials complete.']}
    write('reports/phase_d/final_comparison.json',report)
    write('reports/phase_d/best_checkpoint.json',{'status':'FINAL_VALIDATION_SELECTED','synthetic_best':selection_final,
        'production':production,'registry':{k:{'path':p,'sha256':sha256(ROOT/p)} for k,p in paths.items()}})
    write('reports/phase_d/summary.json',{'status':'COMPLETE','phase':'D5-D21','best_checkpoint':selection['checkpoint'],
        'production':production,'synthetic_best_FPA':1.,'real_old':8,'real_new':8,'tests':tests,
        'final_report':'docs/phase_d_report.md','final_comparison':'reports/phase_d/final_comparison.json'})
    rows=[]
    for r in table:
        m=r['synthetic_test'];t=m['by_type'];rows.append(f"| {r['model']} | {m['full_plate_accuracy']*100:.2f}% | {m['cer']*100:.2f}% | {t['type1']*100:.2f}% | {t['type1a']*100:.2f}% | {t['type1b']*100:.2f}% | {r['real_correct']}/115 ({r['real_accuracy']*100:.2f}%) | {r['latency_cpu_ms']['mean']:.2f} | {r['decision']} |")
    rows='\n'.join(rows);p=post['paired_metrics'];o=best_oracle
    old=table[0];best=table[2]
    document=f'''# PHASE D — завершённые D5–D21

BEST CRNN по synthetic validation — EXP-D-01, epoch1: FPA100%, CER0. На fixed synthetic test7000 он также получил100%, CER0. Это OCR-only оценка с известной геометрией генератора и известным типом; она не сравнивается напрямую с whole-image real debug.

Замена только OCR-весов в неизменённом real pipeline сохранила8/115 (6.96%). Type1:8/15 (53.33%) до/после; type1a:0/100 до/после. Official score18.81→19.43, precision21.62%→22.86%. Literal exact7/118 до/после: internal literal не применяет все wildcard/ignore правила организатора. Реальных type1b в public нет.

BEST synthetic checkpoint: `models/phase_d/exp_d_01_finetune/best.pt`, SHA256 `{selection['checkpoint_sha256']}`. Production остаётся исходный YOLO11n+original CRNN greedy+old deskew+constant type1+current postprocess: заранее записанный gate требует роста E2E FPA, который не получен. Fine-tuned вариант сохранён как лучший OCR-кандидат для следующей геометрической фазы; старые файлы не заменены.

## CUDA и проверки

Отдельная `.venv-cuda`: torch2.6.0+cu124, runtime CUDA12.4, GTX1060 6GB, driver566.36. Forward/backward/AdamW, DataLoader, save/load/resume и реальный mixed smoke прошли. Рабочая `.venv` сохраняет torch2.14.0+cpu. Resume сохраняет правильное размещение AdamW scalar step и moments; непрерывный и прерванный run проверены.

Большое обучение: batch64, CUDA FP32, CTC loss на CPU с обратным переносом градиента. EXP-D-01: peak allocated {train['finetune']['GPU_memory_usage']['peak_allocated_bytes']/1024**3:.2f} GiB, reserved {train['finetune']['GPU_memory_usage']['peak_reserved_bytes']/1024**3:.2f} GiB; {train['finetune']['images_per_sec_including_validation_and_checkpoints']:.2f} train images/sec включая validation/checkpoint overhead. Подробные измерения обоих runs — `cuda_validation.json` и `training_report.json`.

Финальная CPU regression: {tests['passed']} passed, {tests['skipped']} legacy expected failures,0 failures/errors. CUDA training/cache checks, pixel equality shared preprocessing, invalid hash/label controls также прошли. Предварительные ошибки TEMP/кодировки исправлены; финальные результаты успешны.

## D6–D9: данные и исходное сравнение

Версия `synthetic_d_large_v1`, путь `D:/volga/generated_data/synthetic_d_large_v1`:70000 уникальных plate_num; type1=20000, type1a=30000, type1b=20000. Clean14000 (20%), medium35000 (50%), hard21000 (30%). Маска всех типов LDDDLLRR/RRR по решению пользователя. Hard bounded/readable;134 попытки отвергнуты raster gate и заменены, без снижения порога.

Все70000 final images прошли raster-to-GT проверку; labels/alphabet/type/CTC/image hashes/decoding/common preprocessing/finite tensors проверены. Все30000 type1a успешно stitched. Logits39 timesteps, минимум26 spare timesteps сверх CTC target с adjacent repeats. Нормализованные32×160 tensors заморожены; byte hash кеша и hashes preprocessing проверяются. Кеш не добавляет новых augmentations и не передаёт текст labels в preprocessing.

Просмотрены все16 random contact sheets:100/type и те же100 type1a stitched. Нет систематических обрезаний, неверных captions, перестановки строк или нечитаемых hard samples. Previews: `reports/phase_d/dataset_preview/`. Шрифт авторский приближённый, не certified GOST; изображения являются crops, не полноценной detector training выборкой.

Fixed split: train56000 (16000/24000/16000), validation7000 (2000/3000/2000), test7000 (2000/3000/2000). Seed20261002, independent population seeds20261002/27183/31415. Нет пересечений plate_num;300 идентичностей старого smoke набора исключены. Совпадений с полными public GT plate_num нет. Экспорты `dataset/splits/ocr_{{train,val,test}}.csv`, manifest `ocr_manifest.json`; hashes между runs сохранены.

Meta SHA256 `{dataset['meta_sha256']}`. OCR cache SHA256 `{dataset['cache_sha256']}`. До обучения OLD и smoke оценены на том же fixed test по D9: OLD19.23% FPA/CER20.15%/1−CER79.85%; smoke87.16%/1.65%/98.35%. Это заменяет механическое сравнение на другой90-image validation.

## D10–D13, D19–D20: эксперименты и выбор

EXP-D-01: production weights→synthetic fine-tune. EXP-D-02: random initialization→та же выборка. Архитектура CRNN8,718,999 params,32×160,alphabet22+blank22 не изменена. Seed42, batch64, AdamW lr0.001/wd0.0001, clip5, StepLR5/gamma0.5, max20 epochs, patience3, min_delta0.0001, CUDA FP32, no online photometric augmentation. Отличаются initialization/output paths; RNG/splits/normalized pixels/optimizer conditions одинаковы. В EXP-D-02 исключено повторное чтение уже проверенных raw bytes внутри одного запуска; fingerprints и численные входы совпадают.

Оба runs достигли validation100%/CER0 в epoch1 и завершились early stopping в epoch4. Best выбирался по validation FPA, CER только tie-break. Последующие более низкие train/validation losses не заменяли epoch1 при равной FPA/CER. До trained-model test/public зафиксирован tie-break: первый зарегистрированный кандидат, EXP-D-01. Validation ceiling100%/CER0 не мог быть превзойдён; real/test метрики выбора не меняли.

D12 не потребовался: у winning mixed checkpoint все типы100%, spread0. D13 не потребовался: mixed validation уже на абсолютном FPA/CER потолке; заранее записанное условие plateau ниже98% не наступило. Curriculum/targeted увеличение всей выборки не запускались.

| Model / configuration | Synthetic test FPA | CER | type1 | type1a | type1b | Real correct / % | CPU mean ms | KEEP/REJECT |
|---|---:|---:|---:|---:|---:|---:|---:|---|
{rows}

Все synthetic строки относятся к одному test7000 и одному fingerprint. Type columns — synthetic FPA. Real correct — официальный whole-image result115 targets. Latency — последовательные uncaptured100-image CPU runs после training, image read/hash/CSV включены, model load отдельно, первый image cold. Hardware i5-6500/GTX1060, не reference i5-7600/GTX1050Ti. OLD mean/p95 {old['latency_cpu_ms']['mean']:.2f}/{old['latency_cpu_ms']['p95']:.2f} ms; BEST {best['latency_cpu_ms']['mean']:.2f}/{best['latency_cpu_ms']['p95']:.2f} ms. Небольшая разница mean не доказывает ускорение архитектуры. Старые307ms и capture timings под нагрузкой не используются как paired benchmark.

Random-init идеален на synthetic test, но real0/115 и oracle0/8. Fine-tune real8/115 и oracle6/8: старые веса полезны для переноса на реальный домен, хотя финальная synthetic FPA не выявила разницы. Public не использовался для epochs/hyperparameters/checkpoint selection.

## D14–D16: real inputs, все ошибки и oracle

Unchanged YOLO, filters, old deskew,32×160 normalization, greedy decode, thresholds, postprocess/tracker и constant type1. Все47 actual OCR inputs/bboxes побайтно одинаковы до/после; output старого trial побайтно совпадает с сохранённым baseline. Новый uncaptured CSV побайтно совпадает с capture CSV. Нет подстановки GT/type в основной real pipeline. Type1a0% ожидаем при constant type1 и отсутствии Phase C stitch в этой ветке.

Сохранены139 literal error records:111 missed GT и28 unmatched outputs, включая пять partial GT. Каждому приложены original image, все detector boxes, доступные detected crop/legacy geometry result/OCR input, GT/raw/final/confidence/type/tokens. До OCR заблокированным случаям не выдумываются rectified/OCR images. Legacy geometry result — padding/upscale/rotation; perspective homography в этом pipeline отсутствует. Каталог `reports/phase_d/real_best_analysis/errors/`, index `errors.json`.

Вручную просмотрены47 actual inputs и36 rejected proposals. Из47 inputs24 complete/correct,21 clipped,2 partial/unpaired. Из36 rejected proposals33 сопоставлены target type1a. Missed-GT primary categories: detection36, geometry53, absent type1a conversion15, verified OCR failures2, unresolved partial GT5. Всего это111 missed-GT records; общие counts139 включают также FP и не являются независимыми plate counts. BAD_RECTIFICATION/POSTPROCESS_FAILURE как первичная причина не подтверждены; их отсутствие в count не означает, что геометрия идеальна. OCR_FAILURE не назначался неверному/обрезанному crop.

Oracle subset выбран по видимости до оценки новых weights:8 type1a, вручную заданные quad из исходных фото, сохранены источник/hashes и crop/stitch review. Для auto и oracle используется одинаковый известный type1a/C preprocessing. Auto diagnostic обходит legacy filter; это не E2E pipeline. BEST auto4/8 (50%),CER22.06%; oracle6/8 (75%),CER4.41%,1−CER95.59%. OLD oracle0/8,CER29.41%; smoke1/8,CER20.59%; random0/8,CER55.88%. Восемь readable plates не дают репрезентативной оценки всех115 целей; full GT без пространственных boxes не позволяет честно вычислить глобальную OCR CER/detector IoU recall.

## D17–D18: ablations

На одинаковых47 OCR calls postprocessing: исправил full text1, испортил ранее точный full text0, не изменил27, изменил без полного exact gain19. Raw exact6→post exact7. На24 вручную paired complete crops CER {p['all']['raw']['cer']*100:.2f}%→{p['all']['postprocessed']['cer']*100:.2f}%; distance effects {post['paired_distance_effects']}. На9 monoline crops full6/9→7/9,CER5.13%→3.85%; на15 legacy whole type1a full0→0,CER83.72%→87.60%. FPA postprocess не ухудшил, но изменение строки само по себе не польза; type1a-specific обработку следует пересмотреть после правильного stitch в отдельной фазе.

OLD DESKEW vs NO DESKEW с одним фиксированным BEST CRNN: обе8/115, type1 8/15,type1a0/100,score19.43. Дополнительный контроль OLD weights:8/115 в обоих случаях; score18.81→19.43. Ухудшения FPA от old deskew в этом наборе не обнаружено; в locked candidate сохранён old deskew. Выбор weight checkpoint по этой ablation не менялся.

## D21: остановка и следующий bottleneck

Главное ограничение — detector coverage, геометрический фильтр/однострочные обрезанные proposals и отсутствие type1a stitch/type routing в real branch. Conditional oracle6/8 существенно лучше strict type1a E2E0/100; дальнейшее наращивание CRNN synthetic training не исправляет отсутствующие символы и неверный type. Остаются2/8 oracle errors и ограниченность шрифта, поэтому OCR не объявлен решённым для всех реальных сцен. Следующая отдельно разрешённая фаза должна заниматься detector/geometry/type1a integration; CRNN tuning здесь остановлено. FastPlateOCR/SVTR/RepSVTR/PARSeq/YOLO OBB/Pose не запускались.

Машиночитаемый итог `reports/phase_d/final_comparison.json`; protocol/selection snapshots, CUDA/dataset validation, fixed split manifest, training reports, pretrain/final synthetic predictions, sequential real benchmark, all real error artifacts, oracle/postprocess/deskew ablations и final tests сохранены. D1–D4 исторический отчёт сохранён как `docs/phase_d_report_d1_d4.md`.
'''
    previous=ROOT/'docs/phase_d_report.md';backup=ROOT/'docs/phase_d_report_d1_d4.md'
    if not backup.exists():shutil.copyfile(previous,backup)
    previous.write_text(document,encoding='utf-8')
    experiments=ROOT/'docs/experiments.md';history=experiments.read_text(encoding='utf-8')
    marker='\n## EXP-D-01 / EXP-D-02 — Large CRNN training and real transfer'
    if marker in history:history=history.split(marker)[0]
    history+=f'''{marker}

Date:2026-10-02. На70000 authored synthetic crops compare original pretrained→fine-tune vs random→same data. Fixed56000/7000/7000 identity split,20/50/30 clean/medium/hard; all raster/CTC/stitch/visual gates прошли. Отдельная torch2.6/cu124 CUDA среда; CPU env и baseline сохранены.

Same CRNN32×160/22+blank, seed42,batch64,AdamW0.001,StepLR5/0.5,patience3. Best обоих epoch1 validation100%/CER0; early stop epoch4. EXP-D-01 runtime {train['finetune']['training_time_seconds']:.2f}s; EXP-D-02 {train['random']['training_time_seconds']:.2f}s, включая validation/checkpoints, без preflight. First registered candidate tie-break зафиксирован до trained TEST/public. Test7000: оба100%/CER0; OLD19.23%/20.15%,smoke87.16%/1.65%. D12/D13 не требуются при validation ceiling и spread0.

| Configuration | Real official | Score | Oracle type1a | CPU mean/p95 ms | Decision |
|---|---:|---:|---:|---:|---|
| OLD |8/115|18.81|0/8|{old['latency_cpu_ms']['mean']:.2f}/{old['latency_cpu_ms']['p95']:.2f}|KEEP production |
| Fine-tuned BEST |8/115|19.43|6/8|{best['latency_cpu_ms']['mean']:.2f}/{best['latency_cpu_ms']['p95']:.2f}|KEEP OCR candidate; promotion gate not met |
| Random-init |0/115|0|0/8|{table[3]['latency_cpu_ms']['mean']:.2f}/{table[3]['latency_cpu_ms']['p95']:.2f}|REJECT real transfer |

Type1 до/после BEST8/15, type1a0/100; real pipeline constanttype1/no stitch. Oracle subset8 known-type plates не равен global real OCR accuracy. Primary missed-GT causes geometry53,detection36,absent conversion15,OCR2,partial unknown5. All139 error records сохранены. Postprocess fixed1/damaged0/unchanged27/changed-no-gain19; paired CER ухудшился на type1a whole inputs. Deskew/no-deskew FPA одинаков8/115. BEST CRNN `models/phase_d/exp_d_01_finetune/best.pt`; BEST production сохраняет OLD согласно predeclared E2E FPA promotion gate. Следующий bottleneck — detector/geometry/type1a routing; другие архитектуры не запускались.

Evidence: `docs/phase_d_report.md`, `reports/phase_d/final_comparison.json`, `selection.json`, `selection_final.json`, `experiment_protocol.json`, `real_benchmark/comparison.json`, `oracle_comparison.json`, `real_best_analysis/`, final tests {tests['passed']}passed/{tests['skipped']}legacy expected failures.
'''
    experiments.write_text(history,encoding='utf-8')
    print(json.dumps({'status':'COMPLETE','tests':tests,'synthetic_best':1.,'real_old':8,'real_new':8,
        'best_checkpoint':str(selected),'reports':['docs/phase_d_report.md','docs/experiments.md','reports/phase_d/final_comparison.json']}))


if __name__=='__main__':main()
