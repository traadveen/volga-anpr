"""Package fixed adaptation inputs, original source and warm starts without training."""
from pathlib import Path
import argparse
import json
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256
from tools.build_final_i_bundle import verify_freeze


def build(output,root=ROOT):
    root,output=Path(root).resolve(),Path(output).resolve()
    if output.exists():raise ValueError('Existing training snapshot is preserved')
    verify_freeze(root)
    output.mkdir(parents=True)
    def copy(source,destination):
        target=output/destination;target.parent.mkdir(exist_ok=True,parents=True)
        shutil.copy2(source,target)
    for directory in ('src','ocr','tools','dataset','training'):
        for path in sorted((root/directory).rglob('*')):
            if path.is_file() and path.suffix in ('.py','.yaml','.json','.md') and '__pycache__' not in path.parts:
                copy(path,'source/'+path.relative_to(root).as_posix())
    for directory in ('phase_d','phase_f','phase_g','phase_h','phase_i'):
        folder=root/'configs'/directory
        if folder.exists():shutil.copytree(folder,output/'source/configs'/directory)
    copy(root/'tools/reproduce_final_adaptation.py','tools/reproduce.py')
    detector=json.loads((root/'configs/phase_i/I_DET_4_real_balanced_replay.json').read_text(encoding='utf-8'))
    ocr=json.loads((root/'reports/phase_h/training/H_OCR_1_real_heavy/summary.json').read_text(encoding='utf-8'))
    for name,path,expected in [('detector',detector['initial_weights'],detector['initial_sha256']),
                              ('ocr',ocr['spec']['initial_checkpoint'],ocr['spec']['initial_sha256'])]:
        if file_sha256(Path(path))!=expected:raise ValueError('Warm start changed: '+name)
        copy(path,'initial/'+name+'.pt')
    copy(root/'configs/phase_i/I_DET_4_real_balanced_replay.json','recipes/detector.json')
    (output/'recipes/ocr.json').write_text(json.dumps(ocr['spec'],indent=2),encoding='utf-8')
    for name in ('prepared.json','input_review.json','arrays.npz'):
        copy(root/'reports/phase_h/training'/name,'ocr_data/'+name)
    if file_sha256(output/'ocr_data/prepared.json')!=ocr['spec']['data_sha256']:
        raise ValueError('OCR tensor manifest changed')
    if file_sha256(output/'ocr_data/arrays.npz')!=json.loads((output/'ocr_data/prepared.json').read_text(encoding='utf-8'))['tensor_sha256']:
        raise ValueError('OCR tensor data changed')
    split=root.parent/'datasets/reports/data3_2026_10_18/frozen_v1/manifest.json'
    if file_sha256(split)!=ocr['spec']['split_sha256']:raise ValueError('Frozen split manifest changed')
    copy(split,'evidence/split_manifest.json')
    for name in ('dataset_manifest.json','selection_policy.json','training/I_DET_4_real_balanced_replay.json',
                 'training/I_DET_4_real_balanced_replay_initialization.json'):
        copy(root/'reports/phase_i/balanced_20261019'/name,'evidence/detector/'+name)
    for name in ('history.json','summary.json'):
        copy(root/'reports/phase_h/training/H_OCR_1_real_heavy'/name,'evidence/ocr/'+name)
    # Byte-identical source regions and replay images, not a new split or labels.
    data=Path(detector['settings']['data']).parent
    shutil.copytree(data/'images',output/'detector_data/images')
    shutil.copytree(data/'labels',output/'detector_data/labels')
    selected=json.loads((output/'evidence/detector/dataset_manifest.json').read_text(encoding='utf-8'))
    for record in selected['records']:
        if record['variant']!='replay':continue
        for key,sha in [('export_image','export_sha256'),('export_label','label_sha256')]:
            source=Path(record[key]);relative=source.relative_to(data)
            if file_sha256(output/'detector_data'/relative)!=record[sha]:
                raise ValueError('Training replay changed: '+str(relative))
    (output/'README.md').write_text('''# Воспроизведение финальной адаптации

Обучение в ходе упаковки не запускалось. Архив содержит точные исходные тензоры OCR,
изображения/labels детектора, исходные функции обучения, два начальных checkpoint и рецепты.
Исходный split не меняется; test не участвует в оптимизации или выборе.

Проверка без обучения: `python tools/reproduce.py --stage verify`.
Явный запуск четырёх эпох детектора: `python tools/reproduce.py --stage detector --output NEW_DIR`.
Явный запуск OCR: `python tools/reproduce.py --stage ocr --output NEW_DIR`.
Нужны зависимости cu124 из ../offline/, NVIDIA GPU и Python 3.12.

Детектор: F03 -> четыре фиксированные эпохи, seed20261019, 90 реальных объектов train,
19 val, синтетический replay; выбран last.pt. OCR: checkpoint Phase D -> H_OCR_1_real_heavy,
seed20261018, real_fraction0.75, до12 эпох, patience3, лимит1200с, выбор по real val raw OCR.
Полные параметры и фактически выполненные эпохи: recipes/ и evidence/.

Ограничение воспроизведения: гарантируется идентичность входных файлов и рецепта.
Битовая идентичность новых весов на другой GPU не обещается; в OCR действует лимит времени.
Переносимый запуск обучения проверен до стадии загрузки/целостности; новые эпохи специально не запускались.
Исходники старых этапов сохранены в source/ как архив истории, они могут содержать исходные локальные пути.
Полное повторение Phase D/F с нуля требует исходных больших коллекций; финальная адаптация использует
включённые warm-start checkpoint. Генератор сдаваемых5000 сцен находится отдельно в ../dataset/generator/.

Правовой статус фотографий и производных OCR-тензоров наследуется от исходных данных.
Перед передачей добавьте обещанный документ Kaggle в ../dataset/permissions/.
''',encoding='utf-8')
    files={p.relative_to(output).as_posix():file_sha256(p) for p in sorted(output.rglob('*')) if p.is_file()}
    manifest=dict(scope='Exact inputs and historical warm starts for final adaptation; not a new training run',
        files=files,training_started=False,test_inference_started=False,source_freeze_unchanged=True,
        resulting_weight_bit_identity_guaranteed=False)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    verify_freeze(root)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True)
    result=build(p.parse_args().output);print('Training package files:',len(result['files']))
