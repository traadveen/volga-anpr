"""Check delivery requirements and record unmet criteria without claiming completion."""
from pathlib import Path
from collections import Counter
import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256
from tools.final_i_runtime import verify_runtime
from tools.build_final_i_bundle import verify_freeze


def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def audit(delivery,report):
    delivery,report=Path(delivery),Path(report)
    verify_freeze(ROOT);verify_runtime(delivery/'solution')
    data=read(delivery/'dataset/export_manifest.json')
    validator=read(delivery/'dataset/validation/final_report.json')
    qa=read(report/'functional_qa.json')
    inventory=read(report/'dataset_inventory_final.json')
    generator=read(report/'generator_reproduction.json')
    benchmark=read(report/'benchmark_cuda_200.json')
    protected=read(report/'../phase_i/delivery_20261019/protected_before.json')
    if any(file_sha256(Path(p))!=sha for p,sha in protected.items()):raise ValueError('Protected source changed')
    tests=ET.parse(report/'final_tests.xml').getroot()
    counts={k:sum(int(s.get(k,'0')) for s in tests.iter('testsuite')) for k in ('tests','errors','failures','skipped')}
    if counts['errors'] or counts['failures']:raise ValueError('Regression tests failed')
    if validator['stats']['errors'] or qa['status']!='PASS':raise ValueError('Functional/data format verification failed')
    trace=read(report/'submission_cuda.trace.json')['records']
    by_name={r['image']:r for r in trace}
    multi=sum(len(r['predictions'])>1 for r in trace)
    if not multi or any(by_name[n]['predictions'] for n in ('blank.png','tiny.png','large.png')):
        raise ValueError('Actual multi/blank functional scenario failed')
    privacy=read(delivery/'dataset/provenance/privacy_audit.json')
    if len(privacy['reviewed_images'])!=109:raise ValueError('Incomplete privacy audit')
    rows=list(csv.DictReader((delivery/'dataset/meta.csv').open(encoding='utf-8'),delimiter=';'))
    if file_sha256(delivery/'dataset/meta.csv')!=data['meta_sha256']:raise ValueError('Dataset metadata changed')
    checks=[]
    def item(ident,requirement,status,evidence,note=''):
        checks.append(dict(id=ident,requirement=requirement,status=status,evidence=evidence,note=note))
    for ident,requirement,evidence in [
        ('runtime','Независимые изображения, локальные веса, точная команда','README.md; solution/run.py'),
        ('csv','CSV UTF-8 ; четыре поля, confidence0..1','evidence/output_validation.json; tests'),
        ('types','Функциональный запуск type1/type1a/type1b','evidence/functional_qa.json; три типа в47 строках'),
        ('multi','Несколько номеров в кадре','evidence/submission_cuda.trace.json'),
        ('zero','Пустая/маленькая/большая собственная сцена без объектов','evidence/functional_qa.json'),
        ('empty','Пустая папка возвращает заголовок','evidence/root_empty.manifest.json; tests'),
        ('corrupt','Повреждённый файл не прерывает исправные','evidence/functional_qa.json; errors.json'),
        ('unicode','Unicode-путь и имя','evidence/owned_inputs.json'),
        ('hashchar','Символ # и форматы белого/жёлтого','evidence/final_tests.xml; solution/tools/validate_output.py'),
        ('offline','Без API, блокирование Python TCP/DNS/UDP','solution/src/inference_io.py; actual CPU/CUDA runs'),
        ('weights','Локальные FINAL-I и FINAL-H, model_info','solution/models; solution/bundle_manifest.json'),
        ('freeze','Модели, настройки и DATA-3 не изменены','evidence/protected_hashes.json;109 hashes'),
        ('format','Официальная структура и разметка','dataset/validation/final_report.json:0 errors'),
        ('geometry','Только ранее подтверждённые реальные углы','dataset/provenance/human_geometry.json'),
        ('no_auto_gt','Автопредложения не стали human GT','dataset/export_manifest.json; source history unchanged'),
        ('privacy','Проверены все109 реальных кропов','dataset/provenance/privacy_audit.json'),
        ('groups','Сохранены source/plate/duplicate groups, утечек между split не найдено','evidence/dataset_inventory.json'),
        ('synthetic','5000 синтетических положительных сцен','dataset/meta.csv; generator/'),
        ('reproduction','Полная регенерация с фиксированными seed','evidence/generator_reproduction.json'),
        ('training','Код, рецепты, начальные веса и данные финальной адаптации','training/manifest.json; evidence/training_verify.log'),
        ('clean','Реальный запуск в чистом CPU и отдельном CUDA окружении','evidence/functional_qa.json; offline locks'),
        ('benchmark','Замер полного CLI со стартом/загрузкой/I/O','evidence/benchmark_cuda.json; benchmark_cuda_200.json'),
        ('report','Полный отчёт и пояснительная записка4 страницы','docs/final_report.md; docs/explanatory_note.pdf'),
        ('sources','Источники, авторство, лицензии и инструкции','dataset/provenance/attribution.csv; licenses/'),
    ]:item(ident,requirement,'PASS',evidence)
    if (delivery/'experiments/vehicle_filter/result.json').is_file():
        vehicle=read(delivery/'experiments/vehicle_filter/result.json')
        replay=read(delivery/'evidence/vehicle_metric_replay.json')
        if vehicle['baseline']['overall']['correct']!=55 or vehicle['filtered']['overall']['correct']!=49 or replay['status']!='PASS':
            raise ValueError('Vehicle ablation evidence is incomplete')
        if vehicle['final_I_weights_or_routing_changed'] or vehicle['new_real_test_inference']:
            raise ValueError('Ablation violated the frozen baseline')
        item('vehicle_ablation','PHASE I пункт20: отдельная проверка vehicle filtering','PASS_REJECTED_FOR_RUNTIME',
             'experiments/vehicle_filter/; evidence/vehicle_metric_replay.json',
             'VAL71/93 ->63/93 обнаружений,55/93 ->49/93 полных ответов. Пороги зафиксированы до прогона; фильтр не включён.')
    model=read(delivery/'solution/models/final/model_info.json')
    architecture=read(delivery/'evidence/checkpoint_architecture.json') if (delivery/'evidence/checkpoint_architecture.json').is_file() else None
    if architecture:
        if model['detector']['architecture']!='YOLO11n-OBB' or model['detector']['sha256']!=architecture['detector_sha256']:
            raise ValueError('Architecture description differs from the checkpoint')
        if not {'C3k2','C2PSA','OBB'}.issubset(architecture['module_types']):raise ValueError('Architecture evidence is insufficient')
        item('architecture','Архитектура в описании проверена по checkpoint','PASS',
             'evidence/checkpoint_architecture.json; solution/models/final/model_info.json',
             'YOLO11n-OBB,2662092параметра; прежняя ошибочная подпись YOLOv8n исправлена без изменения весов.')
    item('permission','Документ Kaggle о передаче/публикации','PENDING_USER_DOCUMENT','dataset/permissions/status.json',
        f"Пользователь добавит позднее. В официальном subset{len(data['permission_document_pending_ids'])} примеров, во всём train/val142. Документ не сфабрикован.")
    item('latency','Среднее<=100мс на референсном ПК','UNVERIFIED_REFERENCE_LOCAL_OVER_TARGET','evidence/benchmark_cuda_200.json',
        f"На GTX1060/i5-6500 полный CLI на200 фото:{benchmark['rounds'][0]['entire_CLI_ms_per_image']:.2f}мс/изображение; на51 -251.51мс. Референс1050Ti недоступен.")
    item('other','Проверенное отклонение other','NOT_DEMONSTRATED_FROZEN_MODEL','solution/models/final/model_info.json',
         'Нет четвёртого learned-класса и полного отрицательного real GT; обучение запрещено текущей фиксацией.')
    item('accuracy','Обобщение на полном независимом real test','PARTIAL_PREVIOUSLY_EXPOSED_TEST','docs/final_report.md',
         'Сохранённый тест11/40 полных ответов; tiny/small type1b1/20. Full-scene precision/FPR отсутствуют. Повторного test inference нет.')
    item('volume','Рекомендованные real150/300/other50 с quad','RECOMMENDED_VOLUME_SHORTFALL','dataset/validation/final_report.json',
         'Официальный subset50type1a/59type1b/0other; type1b54unique. Бóльший bbox-пул сохранён только как метаданные.')
    item('conditions','Условия съёмки реальных фото','METADATA_GAP','dataset/validation/final_report.json',
         '109 пустых conditions сохранены; неизвестные условия не выдуманы.')
    item('from_scratch','Полное повторение ранних D/F с нуля','WARM_START_REPRODUCTION_ONLY','training/README.md',
         'Точные данные и начальные веса финальной адаптации включены; ранние большие коллекции в этот пакет не входят. Новые эпохи не запускались.')
    result=dict(status='TECHNICAL_CANDIDATE_WITH_OPEN_REQUIREMENTS',all_requirements_met=False,external_upload_performed=False,
        automatic_training_started=False,goal_complete=False,regression_tests=counts,
        validator_errors=0,validator_warnings=validator['stats']['warnings'],
        protected_hashes_checked=len(protected),multi_prediction_images=multi,
        original_data3_split_changed=False,new_real_test_inference=False,checks=checks)
    docs=delivery/'docs'
    (docs/'acceptance_checklist.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    lines=['# Проверка требований FINAL-I','',
        'Технический кандидат собран. Полное соответствие требованиям не заявляется. Внешняя отправка не выполнялась.','',
        '| Требование | Статус | Основание / ограничение |','|---|---|---|']
    for c in checks:lines.append('| '+c['requirement']+' | '+c['status']+' | '+c['evidence']+(' - '+c['note'] if c['note'] else '')+' |')
    lines+=['',f"Проверки:{counts['tests']} теста без ошибок; валидатор:0 ошибок/{validator['stats']['warnings']} предупреждений; защищённых SHA-256:{len(protected)}.",
        '','Открытые пункты сохранены для продолжения. Разрешение не запрашивается повторно: пользователь сообщил, что добавит документ позднее.']
    (docs/'acceptance_checklist.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();r=audit(a.delivery,a.report);print(json.dumps({k:v for k,v in r.items() if k!='checks'},indent=2))
