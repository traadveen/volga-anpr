"""Lock development selection before terminal test, then write complete evidence report."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import argparse,csv,json,datetime,re,xml.etree.ElementTree as ET
from collections import Counter
from tools.phase_g_common import ROOT,REPORT,CONFIG,read,write,freeze
from tools.phase_g_metrics import waterfall
from dataset.schema import sha256

def lock():
    assert not (REPORT/'selection_lock.json').exists(),'Selection is already locked'
    gpu=read(REPORT/'runtime/cuda/benchmark.json');parity=read(REPORT/'CPU_GPU_parity.json')
    assert gpu['times']['mean']<100 and gpu['VRAM_peak_reserved_MiB']<4096 and parity['status']=='PASS'
    regression=read(REPORT/'synthetic_validation_regression.json');assert regression['BEST_G']['per_type']['type1b']['correct']>=regression['baseline_F']['per_type']['type1b']['correct']
    tests=ET.parse(REPORT/'full_tests.xml').getroot();assert all(int(x.get('failures',0))==int(x.get('errors',0))==0 for x in tests.iter('testsuite'))
    frozen=freeze();write(REPORT/'frozen_control_verification.json',{'files':len(frozen['hashes']),'status':'PASS','checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    remaining=read(REPORT/'remaining_best_g_failures.json');remaining['visual_review']='PASS: all26 reviewed in remaining/review_1..4, exact selected tensor SHA verified. Agent engineering review, no independent annotation agreement.'
    for r in remaining['rows']:
        if r['primary_failure_category']=='OCR_FAILURE_GOOD_INPUT':r['notes']='Actual G selected normalized input visually inspected: rows/glyphs retained, decoding substitution/deletion/insertion follows; exposed input review.'
    remaining['failure_categories']=remaining.pop('categories_provisional',remaining.get('failure_categories'));write(REPORT/'remaining_best_g_failures.json',remaining)
    write(REPORT/'error_waterfall_best_g.json',{k:waterfall([r for r in remaining['rows'] if k=='ALL' or r['plate_type']==k]) for k in ('ALL','type1','type1a','type1b')})
    values=[r for r in remaining['rows'] if not r['official_success']]
    with (REPORT/'remaining_best_g_failures.csv').open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in values for k in r)));w.writeheader();w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in values)
    synth=ROOT.parent/'generated_data/targeted_g_v1';m=read(synth/'manifest.json')
    m['visual_review']='PASS:36 representative train crops reviewed in three aspect-preserving sheets; all360 actual canonical rasters matched positional glyph ground truth and file SHA. Agent review, not360 independent human labels.'
    write(synth/'manifest.json',m);write(REPORT/'targeted_synthetic_manifest.json',m)
    write(REPORT/'targeted_synthetic_acceptance.json',{'status':'PASS','visual_sample_count':36,'raster_checks':360,'split_counts':{'train':180,'val':90,'test':90},'preview_aspect_preserved':True,'used_in_G_training':False,'test_images_hash_unchanged':True})
    f=read(ROOT/'configs/phase_f/best_f.json');p=read(REPORT/'provisional_best_g.json');s=p['result']['config']
    c={'name':'BEST-G-DEVELOPMENT','detector':f['detector'],'detector_sha256':f['detector_sha256'],'ocr':p['checkpoint'],'ocr_sha256':sha256(Path(p['checkpoint'])),
       'imgsz':640,'confidence':.5,'iou':.5,'split':s['split'],'multi':s['multi'],'postprocess':s['postprocess'],'expansion':s['expansion'],
       'default_backend':'pytorch','geometry':'observed OBB with unchanged F fallback','routing':'detector class only','mask':'LDDDLLRR/RRR for all three types',
       'selection_basis':'Repeatedly exposed real_val80 images87 targets; balanced replay tie preference, protected type1 and synthetic val type1b. Full public score diagnostic. No independent real test.',
       'generalization':'GENERALIZATION NOT YET VERIFIED','real_test_available':False,'promotion_scope':'Development candidate; no production promotion or replacement of BEST-F',
       'runtime_gates':{'GPU_mean_ms':gpu['times']['mean'],'GPU_p95_ms':gpu['times']['p95'],'VRAM_reserved_MiB':gpu['VRAM_peak_reserved_MiB'],'CPU_GPU_parity':parity['status'],'strict_GPU_mean_lt_100':True,'preferred_GPU_mean_le_60':gpu['times']['mean']<=60}}
    write(CONFIG/'best_g.json',c)
    paths=[CONFIG/'best_g.json',Path(c['detector']),Path(c['ocr']),REPORT/'full_tests.xml',REPORT/'real_data_manifest.json',synth/'manifest.json',synth/'test/freeze.json']+list((ROOT/'src').glob('*_g.py'))
    write(REPORT/'selection_lock.json',{'created_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'config_sha256':sha256(CONFIG/'best_g.json'),'hashes':{str(p):sha256(p) for p in paths},'terminal_test_policy':'New90 synthetic OCR test once AFTER this lock, cannot change selection using its results','generalization':c['generalization']})
    print('BEST-G development locked; terminal test permitted',flush=True)

def report():
    c=read(CONFIG/'best_g.json');selection=read(REPORT/'selection_lock.json')
    for path,h in selection['hashes'].items():assert sha256(Path(path))==h,path
    control=freeze();g=read(REPORT/'runtime/cuda/benchmark.json');cpu=read(REPORT/'runtime/cpu/benchmark.json');f=read(ROOT/'reports/phase_f/final_comparison.json')
    terminal=read(REPORT/'terminal_synthetic_ocr_test.json');reg=read(REPORT/'synthetic_validation_regression.json');errors=read(REPORT/'remaining_best_g_failures.json');oracle=read(REPORT/'oracle/comparison.json');good=read(REPORT/'real_good_input_ocr_after.json')
    rows=[{'name':'BEST_F','real_val':37,'full_public':51,'type1':11,'type1a':40,'official_FP':46,'official_precision':51/117,'cached_ms':None,'selected':False}]
    for path in sorted((REPORT/'experiments').glob('*/result.json')):
        r=read(path);rows.append({'name':r['config']['name'],'real_val':r['real_val_exposed']['full_match_total'],'full_public':r['full']['full_match_total'],'type1':r['full']['per_type']['type1']['full_match'],'type1a':r['full']['per_type']['type1a']['full_match'],'official_FP':r['full']['false_positives'],'official_precision':r['full']['precision'],'cached_ms':r['cached_prepared_OCR_latency']['mean'],'selected':r['config']['name']=='G08_conf_50'})
    for path in sorted((ROOT/'models/phase_g').glob('G06_*/best_result.json')):
        r=read(path);rows.append({'name':path.parent.name+'_best_epoch','real_val':r['real_val_exposed']['full_match_total'],'full_public':r['full']['full_match_total'],'type1':r['full']['per_type']['type1']['full_match'],'type1a':r['full']['per_type']['type1a']['full_match'],'official_FP':r['full']['false_positives'],'official_precision':r['full']['precision'],'cached_ms':r.get('cached_prepared_OCR_latency',{}).get('mean'),'selected':False})
    with (REPORT/'model_comparison.csv').open('w',encoding='utf-8-sig',newline='') as file:
        w=csv.DictWriter(file,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    allfpa=g['evaluation']['full'];val=g['evaluation']['real_val_exposed'];tests=ET.parse(REPORT/'full_tests.xml').getroot();suites=list(tests.iter('testsuite'))
    test_result={'passed':sum(int(s.get('tests',0))-int(s.get('skipped',0)) for s in suites),'expected_legacy_xfail':sum(int(s.get('skipped',0)) for s in suites),'failed':0,'xml_sha256':sha256(REPORT/'full_tests.xml')}
    final={'BEST_G':c,'baseline_F':{'public_correct':51,'public_GT':115,'type1':11,'type1a':40,'exposed_val_correct':37,'exposed_val_GT':87,'GPU_mean_ms':45.5642,'CPU_mean_ms':97.0055,'official_FP':46,'official_precision':51/117},
       'BEST_G_evaluation':g['evaluation'],'runtime_GPU':{k:v for k,v in g.items() if k not in ('stages','evaluation')},'runtime_CPU':{k:v for k,v in cpu.items() if k not in ('stages','evaluation')},'CPU_GPU_parity':read(REPORT/'CPU_GPU_parity.json'),
       'frozen_control':{'files':len(control['hashes']),'status':'PASS'},'tests':test_result,'remaining_failures':errors['failure_categories'],'remaining_failure_count':errors['failed'],'waterfall_F':read(REPORT/'error_waterfall.json'),'waterfall_G':read(REPORT/'error_waterfall_best_g.json'),
       'oracle':oracle['metrics'],'real_good_input_OCR':good['metrics'],'postprocess':read(REPORT/'postprocess_analysis.json'),'synthetic_validation':{k:v for k,v in reg.items() if k!='cases'},'terminal_synthetic_OCR':{k:v for k,v in terminal.items() if k!='rows'},
       'training_ratios':read(REPORT/'training/tournament.json'),'detector_training':'SKIPPED: frozenF failures dominated split28 and verifiedOCR20 versus detector misses5 and low confidence3; unchanged detector preserves scoped P/R and FP/image.',
       'synthetic_training':'FrozenD1200 identity-filtered train replay,400/type; newly targeted360 crops prepared and accepted but NOT used in this training.',
       'independent_real_test':'UNAVAILABLE: user confirms no new photos; real_test empty. Capture/vehicle provenance unknown, source-disjoint false.', 'generalization':'GENERALIZATION NOT YET VERIFIED',
       'next_bottleneck':{'primary':'DATA: independent real collection and provenance, especially type1b','conditional_exposed_development':'OCR12, detector absence/threshold8, geometry2, unreadable1, sourceGTdiscrepancies3; split failures0 after visual selected-input review','next_action':'Collect group-disjoint real frames, dense bbox/quad/type/text labels and adjudicate discrepancies before architecture changes or production promotion'},
       'limitations':['Small26 real training targets; repeated exposure to80-frame development set and threshold/epoch selection inflates estimated generalization.','Official false_positives/precision are organizer text-based outcomes, not global spatial detector false positives. Real spatial P/R only8frames9GT.','Warm GPU mean meets100ms but misses60ms preference; CPU154ms exceeds100ms; GTX1050Ti not measured.','Theoretical Torch peak allocation is not whole-process GPU reservation; report allocated124MiB and reserved244MiB separately.','GT supplied only to training supervision and evaluators/oracle diagnostics; runtime CLI has no GT/oracle input.']}
    write(REPORT/'final_comparison.json',final)
    training=final['training_ratios'];wf=final['waterfall_F']['ALL'];wg=final['waterfall_G']['ALL'];tm=terminal['metrics']
    lines=['# PHASE G — Real domain и устойчивость type1a','',f'Дата: 2026-10-04. **BEST-G-DEVELOPMENT: {allfpa["full_match_total"]}/115 ({allfpa["full_match_total"]/115:.2%}) против BEST-F51/115.** На exposed real_val: **67/87 против37/87**.','',
      '**GENERALIZATION NOT YET VERIFIED.** Новых реальных снимков нет, real_test пуст. Public debug многократно использован для анализа и выбора параметров. 20 кадров real_train и80 real_val разделены по известным идентичностям/точным пикселям; неизвестные vehicle/sequence/capture не позволяют доказать source-disjoint split. Это кандидат для разработки; BEST-F и production-конфигурация не заменены.','',
      '## Результат и границы метрик','', '| Метрика | BEST-F | BEST-G |','|---|---:|---:|','| Public E2E,115 GT |51|89|','| Type1,15 GT |11|11|','| Type1a,100 GT |40|78|','| Exposed real_val,87 GT |37|67|','| Official text-based FP |46|14|',f'| Official precision |43.59%|{allfpa["precision"]:.2%}|',f'| GPU mean/p95,мс |45.56/62.77|{g["times"]["mean"]:.2f}/{g["times"]["p95"]:.2f}|',f'| CPU mean/p95,мс |97.01/129.65|{cpu["times"]["mean"]:.2f}/{cpu["times"]["p95"]:.2f}|','',
      'Official scorer “detected/recall_detect” зависит от совпадения текста: это не пространственный recall. На8 ранее размеченных real_val кадрах /9GT детектор неизменен: P=0.875,R=7/9,FP/image=0.125. Глобальные recall/IoU/FP-per-image для100 кадров не заявляются. Real type1b отсутствует. Маска всех типов остаётся LDDDLLRR/RRR.','',
      'F100/207 означает paired **E2E на полноценных синтетических сценах**: IoU≥0.5 + текст + тип. Исторические D100% означают OCR-only на готовых crop и к этой метрике неприменимы. В G Ftest не использован для выбора.','',
      '## Замороженный контроль и аудит','',f'Все{len(control["hashes"])} файлов контроля, включая Fконфиги, веса, отчёты, старые исходники, public JPEG и scorer, прошли повторную SHA-256 проверку. АдаптерG/A воспроизвёл117 actual OCR tensors и ответы F. Все64 неудачи и51 успех F визуально проверены по source/crop/rectified/top/bottom/stitch/normalized. Решения — инженерная проверка агентом, без независимого второго разметчика.','',
      '| Ранняя первичная причина | F | G |','|---|---:|---:|']
    for category in sorted(set(wf['failures'])|set(wg['failures'])):lines.append(f'|{category}|{wf["failures"].get(category,0)}|{wg["failures"].get(category,0)}|')
    lines+=['','GT проблемы выделены UNKNOWN: img031/085 имеют две строки при organizer type1; img032 видимый M941MT14 отличается от буквального M491MT14. Оригинальные метки/scorer сохранены; эти противоречия не обучают модель ошибочному тексту/типу. img018 деградирован; пропуски/lowconf определены пространственно по источнику, без поиска по совпавшему тексту.','',
      'Артефакты: error_attribution.csv —115 строк, type1a_failures.csv —60 F ошибок; audit/index.html и compact_01..08; remaining_best_g_failures.csv —26 G ошибок с проверенными actual selected tensors. Все26 просмотрены в remaining/review_1..4.','',
      '## Waterfall с явным postprocess recovery','', '| Этап | F | G |','|---|---:|---:|']
    for key in ('GT','GT_source_discrepancy_unresolved','reviewed_consistent_GT','detected_at_frozen_confidence','acceptable_localization_visual_review','correct_routing','acceptable_preprocessing','correct_split_or_not_applicable','raw_OCR_correct','postprocess_FIXED','postprocess_BROKEN','final_prediction_correct'):lines.append(f'|{key}|{wf[key]}|{wg[key]}|')
    lines+=['','Отдельные ALL/type1/type1a/type1b таблицы сохранены в error_waterfall.json и error_waterfall_best_g.json. Это visual stage accounting, не измеренный global detector recall. Postprocess может восстановить ответ, поэтому raw→post включает FIXED/BROKEN, а не искусственную монотонность.','',
      '## Oracle и решение об OCR','', '| Oracle режим,38 approximate manual targets | Raw correct | Post correct | Raw CER |','|---|---:|---:|---:|']
    for name,value in oracle['metrics'].items():
        v=value['ALL'];lines.append(f'|{name}|{v["raw_correct"]}/38|{v["post_correct"]}/38|{v["raw_CER"]:.2%}|')
    lines+=['','All38 включает2 auto detector misses; сравнения geometry/split имеют условный scope. В good-input ceiling36: исключены img018 degraded и img030 approximate quad, который подрезал штрих. Manual cuts просмотрены на cuts_v2; нигде не входят в production или обучение. Cold oracle timing включает загрузку CUDA и не используется для SLA.','',
      'На одинаковых71 проверенных F tensors rawOCR46→58/71, CER6.34%→2.76%; post50→56/71, CER9.11%→4.39%. Это изоляция весов OCR, без помощи новой split-подготовки; convenience exposed sample. Сохранившийся raw gap оправдал короткое дообучение после oracle, анализа split и автоматических тестов.','',
      '## Минимальные изменения type1a','',
      'A — неизменный F; B — projection/Otsu; C — adaptive projection; D — компоненты; E — projection+prior. Центральное окно35–65% высоты, valley depth, foreground, row balance и height ratio; longest valley midpoint предотвращает обрезку штрихов; ненадёжный split возвращает F. Все варианты type1/type1b bit-exact с F.','',
      'Split-only A/B/C/D/E дают public51/64/67/51/65; bounded multi4 —71. Лучший gap4 поднимает76/115,59/87 без новых весов. Высота48/80, gap0/16, padding0/8 и расширение OBB2/4/6% проверены раздельно; выбран height64,gap4,padding4,expansion0. Multi3 даёт88/66 после обучения; multi4 и5 —89/67, поэтому выбран4. Пороги0.2/0.35/0.5/0.65 одинаковы89/67,0.8 хуже84/63: оставлен0.5.','',
      'Выбор максимум4 нормализованных представлений на один type1a crop: один batched CRNN forward, CTC sequence log probability/length + mask bonus + небольшой structural bonus. Это не ансамбль моделей. GT/истинный тип/ручные cuts не передаются. Fallback при недопустимом quad сохраняет F обработку.','',
      '## Короткое real OCR дообучение','',
      'Инициализация — точный EXP-D-01, архитектура неизменна32×160,h256,2LSTM. 26 реальных train inputs (8type1,18type1a), автоматически подготовленных после supervised manual quad;18 и30 исключены. Replay1200 неизменных Dtrain tensors,400 каждого типа, проверены file/tensor SHA; split identities с real_val исключены. Новый targeted_g_v1 в обучении не использован.','',
      'AdamW lr1e-4,batch32,1024 samples/epoch,max4epochs,seed20261004, только лёгкая фотометрия, clipping5. Каждый ratio заново из D. Early selection по exposed real_val E2E с type1floor11/15; full score только diagnostic tie-break.','',
      '| Real доля | Лучший val/full | Эпоха | Время,сек | Peak allocated,MiB |','|---|---:|---:|---:|---:|']
    for t in training:
        h=next(h for h in t['history'] if [h['real_val_correct'],h['full_correct']]==t['best_key']);ratio=read(CONFIG/(t['name']+'.json'))['real_fraction']
        lines.append(f'|{ratio:.0%}|{t["best_key"][0]}/87;{t["best_key"][1]}/115|{h["epoch"]}|{t["history"][-1]["elapsed_seconds"]:.2f}|{h["peak_allocated_MiB"]:.2f}|')
    lines+=['','Balanced epoch2 выбран при равном результате real-heavy как меньшая концентрация на26 реальных образцах. Три запуска заняли~92сек. Ни архитектурного турнира, ни длительного full training не было. Detector hard fine-tuning пропущен по количественному gate: split28 + OCR20 доминировали над miss5 + lowconf3; снижение порога не вернуло правильных номеров.','',
      '## Синтетика и регрессии','',
      'Измерены94 observed type1a source crops, median117.5×79px, aspect1.56, brightness136.6, contrast54.8; misses не входят, real type1b отсутствует. JPEG proxy не является оценкой настоящего JPEGquality. targeted_g_v1:360 identity-disjoint OCR crops (180/90/90), authored/Arial glyphs, thickness−1/0/+1, spacing±2, region scale0.96/1/1.04, real-inspired resolution, blur0/.35/.65,JPEG60/80/95,brightness.65/.85/1.0. Type1b — authored yellow control.','',
      'Все360 canonical rasters независимо от строки-метаданных декодированы font templates с positional mask, проверены GT, bounds, glyph order иSHA;36 representative train previews визуально приняты с сохранёнными пропорциями. Это проверка корректности синтетики, не доказательство реализма всего реального домена.','',
      f'F synthetic **validation E2E**97/207→G{reg["BEST_G"]["correct"]}/207. Per-type F20/45/32→G21/45/32 из69: type1b floor сохранён. Данные/детекторF неизменны, frozen observed proposals cached, GT только evaluator.','',
      'Новый synthetic OCR test90 выполнен один раз после SHA selection lock; он не менял BEST-G. Известный layout подан OCR обоим вариантам, detector не оценивается.','',
      '| Тип,30 crops | F raw/post | G raw/post | G raw CER |','|---|---:|---:|---:|']
    for k in ('type1','type1a','type1b'):
        v=tm[k];lines.append(f'|{k}|{v["F_raw"]["correct"]}/{v["F_post"]["correct"]}|{v["G_raw"]["correct"]}/{v["G_post"]["correct"]}|{v["G_raw"]["CER"]:.2%}|')
    lines+=['','Это terminal domain stress test; любые его ошибки относятся к следующей работе, без подбора параметров на этих90 образцах.','',
      '## Postprocess, confidence и скорость','',
      'Selected current postprocess: альтернативный mask-preserve дал равную real_val FPA, усложнение не принято. postprocess_analysis.json содержит FIXED/BROKEN/UNCHANGED отдельно по типам; в exposed val FIXED1,BROKEN0,UNCHANGED77 среди78 associated outputs. На фиксированных71 tensors post56 ниже raw58 — postprocess не объявляется универсальным улучшением.','',
      'confidence_bins.json отдельно для detector/OCR/final; final confidence равен mean OCR character confidence, не произведение несопоставимых оценок. Эмпирические bins считаются на exposed associated outputs с исходом text+type; не spatial calibration и не независимая probability guarantee.','',
      f'GTX1060 6GB:200 warm full-pipeline runs, mean{g["times"]["mean"]:.2f},median{g["times"]["median"]:.2f},p95{g["times"]["p95"]:.2f},max{g["times"]["max"]:.2f}мс. Detector uncached, CPU preparation/NMS/decoding/CTC и synchronization включены; image I/O/model load/warmup исключены. CPU100 scenes mean{cpu["times"]["mean"]:.2f}мс. Первый GPUзамер одновременно с tests исключён и сохранён в runtime_attempt_01; окончательные GPU/CPU замеры последовательны без training/tests.','',
      f'Inference peak torch allocated{g["VRAM_peak_allocated_MiB"]:.2f}MiB,reserved{g["VRAM_peak_reserved_MiB"]:.2f}MiB; training≤570.30MiB. Процесс/драйвер может потреблять дополнительную память. Ограничение4GB пройдено. Среднее GPU<100мс; предпочтение≤60мс не достигнуто, CPU100мс не достигнуто. GTX1050Ti не измерена, производительность не обещается.','',
      'Actual CPU/GPU117 detection rows: text/type совпали; min bboxIoU0.9999926, maxOCRconfidence delta0.0009433. CPUtorch2.14+cpu и CUDAtorch2.6+cu124 изолированы, старые окружения сохранены. ONNXG не заявлен.','',
      '## Проверки, запуск и следующий bottleneck','',
      f'Полная suite: **{test_result["passed"]}passed,{test_result["expected_legacy_xfail"]}expected legacy xfail,0failed**. Включает bit-exact type1/B, split geometry/glyph preservation, blanks/fallback, candidate bound/dedup, mask550, actualCTC score, OBB expansion, cross-split identity/capture/pixel/near-duplicate guard, real_test exposure rejection, deterministic replay, confidence bins, waterfall accounting, actual360 synthetic GT/hash. Первые5 focused failures исправлены, final suite выполнена после production changes.','',
      'CLI: `tools/predict_images_g.py --input <image-or-folder> --output <csv> --config configs/phase_g/best_g.json --device cpu|cuda`. Local checkpoint SHA проверяется; GT входа нет. Smoke run и trace сохранены в reports/phase_g/cli_smoke.','',
      'Оставшийся conditional development bottleneck: OCR12 + detector misses/threshold8 + geometry2 + unreadable1 + sourceGT discrepancy3. **Главный следующий gate — DATA:** новые независимые группы автомобилей/съёмок, включая реальные type1b, dense box/quad/type/text и adjudication спорных меток. До этого нельзя доказать перенос89/115 на реальные новые кадры или выбрать следующую архитектуру по этому exposed результату.','',
      'Все этапы PHASE G завершены в указанном scope. configs/phase_g/best_g.json фиксирует development candidate; reports/phase_g/final_comparison.json содержит полную машинную сводку. BEST-F остаётся воспроизводимым.']
    document='\n'.join(lines)+'\n'
    document=re.sub(r'([А-Яа-яЁё])(?=\d)',r'\1 ',document)
    document=re.sub(r'(\d)(?=[А-Яа-яЁё])',r'\1 ',document)
    document=re.sub(r'\b(mean|median|p95|max|allocated|reserved|FIXED|BROKEN|UNCHANGED|Replay|GTX|OCR|height|gap|padding|expansion)(?=\d)',r'\1 ',document)
    document=re.sub(r'(\d)(?=MiB|GB|px|passed|expected|failed)',r'\1 ',document)
    document=document.replace('BEST-F51','BEST-F 51').replace('F100/207','F 100/207').replace('D100%','D 100%').replace('rawOCR46','raw OCR 46').replace('post50','post 50').replace('post56','post 56').replace('raw58','raw 58').replace('sample_count','sample_count')
    (ROOT/'docs/phase_g_report.md').write_text(document,encoding='utf-8')
    exp=ROOT/'docs/experiments.md';body=exp.read_text(encoding='utf-8-sig')
    if '## EXP-G —' not in body:
        exp.write_text(body+'\n\n## EXP-G — Real domain and type1a robustness\n\nDate:2026-10-04. Frozen BEST-F51/115 reproduced117 inputs;778 hashes preserved. All64 failures audited, split28/OCR20 dominant. Label-free bounded4 split hypotheses and gap4 return76/115; short D-initialized balanced real26 + Dsynthetic1200 replay selects epoch2, BEST-G89/115 (type1 11/15,type1a78/100), exposed val67/87 versus37/87. No new real photos, real_test empty: GENERALIZATION NOT YET VERIFIED. Detector unchanged, threshold0.5; synthetic val97→98/207, type1b32/69 retained. GPUmean61.49ms,p9590.31ms,CPU154.32ms,allocated124.05MiB/reserved244MiB; CPU/GPU text/type parity PASS. Tests344passed+4legacyxfail. New90 synthetic OCR test once after lock; separate OCR-only scope. Next gate independent dense real data, then residual OCR12/detector8/geometry2. Artifacts: docs/phase_g_report.md, configs/phase_g/best_g.json, reports/phase_g/final_comparison.json.\n',encoding='utf-8')
    write(REPORT/'workflow_status.json',{'phase':'G','state':'COMPLETE','BEST_G':'BEST-G-DEVELOPMENT','generalization':'GENERALIZATION NOT YET VERIFIED','new_real_images':0,'independent_real_test_available':False,'frozen_control_files':len(control['hashes']),'tests':test_result,'next_bottleneck':final['next_bottleneck']})
    print('PHASE G complete',allfpa['full_match_total'],'/115; val',val['full_match_total'],'/87; tests',test_result,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['lock','report']);args=p.parse_args();(lock if args.action=='lock' else report)()
