"""Assemble a new submission snapshot; keep all published and frozen sources intact."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256
from tools.build_final_i_bundle import build_final_i_bundle,verify_freeze
from tools.final_i_runtime import verify_runtime


def inventory(root):
    return {p.relative_to(root).as_posix():dict(sha256=file_sha256(p),bytes=p.stat().st_size)
            for p in sorted(root.rglob('*')) if p.is_file() and '__pycache__' not in p.parts
            and p.name!='bundle_manifest.json'}


def build(output,root=ROOT):
    root,output=Path(root).resolve(),Path(output).resolve()
    build_final_i_bundle(output,root)
    manifest=json.loads((output/'bundle_manifest.json').read_text(encoding='utf-8'))
    for source,destination in [('tools/run_final_submission.py','run.py'),
                               ('tools/validate_output.py','tools/validate_output.py'),
                               ('tools/benchmark.py','tools/benchmark.py')]:
        target=output/destination;target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(root/source,target)
    config=json.loads((output/'configs/phase_i/final_i.json').read_text(encoding='utf-8'))
    model_info=dict(name='FINAL-I',detector=dict(architecture='YOLO11n-OBB',path=config['detector'],
        sha256=config['detector_sha256'],initialization='F03_obb_transfer',epochs=4,seed=20261019),
        ocr=dict(architecture='CRNN CNN + bidirectional LSTM + CTC',path=config['ocr'],
        sha256=config['ocr_sha256'],initialization='Phase D exp_d_01_finetune',
        selection='H_OCR_1_real_heavy; validation-selected epoch',seed=20261018),
        routing='Detector class only; color and OCR are not ground truth',
        types=['type1','type1a','type1b','other'],learned_detector_classes=['type1','type1a','type1b'],
        other_limitation='No independently validated learned other-class detector',
        profile='russian_v1',imgsz=640,confidence=.5,nms_iou=.5,
        dataset_version='DATA-3 frozen_v1, Phase-I balanced_20261019 human geometry and replay',
        training_config='training/recipes/detector.json; training/recipes/ocr.json',
        metrics=dict(real_val_objects=93,real_val_correct=55,saved_test_objects=40,saved_test_correct=11,
                     saved_test_is_fresh_untouched=False,full_scene_precision=None),
        benchmark='docs/final_report.md and evidence/benchmark_cuda.json; reference hardware unverified',
        latency_reference_hardware_verified=False,real_test_inference_repeated=False,
        license_notes='Ultralytics AGPL-3.0 conditions apply; photograph rights are separate in dataset/permissions.')
    (output/'models/final').mkdir(parents=True)
    (output/'models/final/model_info.json').write_text(json.dumps(model_info,ensure_ascii=False,indent=2),encoding='utf-8')
    (output/'README.md').write_text('''# Volga ANPR — FINAL-I

Python 3.12, Windows x64. Офлайн-зависимости находятся в соседней папке offline/.
Установка CPU: `python ../offline/install_offline_runtime.py --variant cpu --environment NEW_ENV`.
Точные параметры установщика: `python ../offline/install_offline_runtime.py --help`.
Для NVIDIA используйте вариант cu124. Установленные версии фиксированы в lock-файлах.

Запуск: `python run.py --input INPUT_DIR --output OUTPUT.csv --device cpu`.
Для NVIDIA: `--device cuda`. По умолчанию CPU. CSV UTF-8: image;plate_num;plate_type;confidence.
Отдельный файл и папка поддерживаются. Нечитаемый файл пропускается и записывается в OUTPUT.errors.json.
При полностью повреждённом входе код завершения 2, прежний CSV сохраняется. Пустая папка даёт заголовок.
Локальные веса и неизменяемый профиль russian_v1 проверяются по SHA-256. Сетевые обращения блокируются.

Проверка CSV: `python tools/validate_output.py --input OUTPUT.csv --images INPUT_DIR`.
Скорость всего запуска: `python tools/benchmark.py --input INPUT_DIR --output NEW_BENCHMARK.json --device cuda`.
Бенчмарк не читает GT. Его результат включает запуск Python, загрузку моделей, чтение и сохранение файлов.

Состав моделей: models/final/model_info.json. Результаты и ограничения: соседние docs/final_report.md и explanatory_note.pdf.
type1 — белый длинный A000AA77; type1a — белый двухстрочный того же формата;
type1b — жёлтый длинный AA00077 (2–3 цифры региона). Ошибка цвета фотографии сама по себе не меняет тип.
Модель other не обучена как четвёртый класс: это известное ограничение.
Низкое качество на сохранённой реальной тестовой партии указано в отчёте; производственная точность не заявляется.

Программа не запускает обучение. Материалы воспроизведения адаптации находятся в ../training/.
Условия Ultralytics AGPL-3.0 и авторство исходных библиотек сохраняются в ../licenses/.
''',encoding='utf-8')
    manifest.update(scope='FINAL-I submission adapter with per-file input errors; frozen recognition unchanged',
                    submission_entry='run.py',files=inventory(output))
    (output/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    verify_runtime(output);verify_freeze(root)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    result=build(parser.parse_args().output)
    print('Submission files:',len(result['files']))
