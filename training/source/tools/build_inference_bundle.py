"""Create a portable, hash-bound runtime snapshot without data or training."""
from pathlib import Path
import argparse
import json
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256,load_runtime_config

RUNTIME_FILES=(
    'src/__init__.py','src/inference_io.py','src/image_anpr.py','src/image_anpr_f.py',
    'src/image_anpr_g.py','src/geometry_f.py','src/multiline.py','src/multiline_g.py',
    'src/ocr_preprocess.py','src/real_data_g.py','src/plate_formats.py',
    'ocr/__init__.py','ocr/crnn.py','ocr/utils.py',
    'tools/predict_images_g.py','tools/benchmark_inference_g.py','tools/baseline_predict.py','requirements-inference.txt',
    'dataset/ATTRIBUTION.md','docs/type1b_format_review.md',
)


def build_bundle(output,root=ROOT,config=None):
    root=Path(root).resolve();output=Path(output).resolve()
    config=Path(config) if config else root/'configs/phase_g/best_g.json'
    original_config_hash=file_sha256(config)
    settings=load_runtime_config(config,root)
    sources=[root/name for name in RUNTIME_FILES]+[config,Path(settings['detector']),Path(settings['ocr'])]
    if any(p.is_relative_to(output) for p in sources):
        raise ValueError('Bundle output overlaps source files')
    if output.exists():
        raise ValueError('Bundle output already exists; snapshots are preserved')
    if not all(p.is_file() for p in sources):
        raise ValueError('Runtime source file missing')
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='bundle-stage-',dir=output.parent) as temporary:
        temporary=Path(temporary).resolve()
        if not temporary.is_relative_to(output.parent):
            raise ValueError('Staging directory escaped bundle parent')
        stage=temporary/'runtime';stage.mkdir();source_hashes={}
        for name in RUNTIME_FILES:
            source=root/name;target=stage/name;target.parent.mkdir(parents=True,exist_ok=True)
            source_hashes[name]=file_sha256(source);shutil.copy2(source,target)
            if file_sha256(target)!=source_hashes[name]:raise ValueError('Source changed while copying: '+name)
        config_copy=dict(settings)
        for kind in ('detector','ocr'):
            source=Path(settings[kind]);name=f'models/{kind}/best.pt'
            target=stage/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
            if file_sha256(target)!=settings[kind+'_sha256']:raise ValueError('Checkpoint changed while copying')
            source_hashes[name]=settings[kind+'_sha256'];config_copy[kind]=name
        target=stage/'configs/phase_g/best_g.json';target.parent.mkdir(parents=True,exist_ok=True)
        target.write_text(json.dumps(config_copy,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        if file_sha256(config)!=original_config_hash:raise ValueError('Original config changed during packaging')
        relocated=load_runtime_config(target,stage)
        if not all(Path(relocated[k]).is_relative_to(stage) for k in ('detector','ocr')):
            raise ValueError('Portable config escaped bundle')
        (stage/'README.md').write_text(
            '# Volga ANPR — runtime snapshot\n\n'
            'Контроль BEST-G-DEVELOPMENT. Независимое качество и время на GTX1050Ti пока не подтверждены.\n'
            'Этот комплект содержит код и веса инференса; датасеты и Python-библиотеки устанавливаются отдельно.\n\n'
            'Python 3.12. Создать окружение: `python -m venv .venv-inference`.\n'
            'Для CPU: `.venv-inference/Scripts/python.exe -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu`.\n'
            'Затем: `.venv-inference/Scripts/python.exe -m pip install -r requirements-inference.txt`.\n\n'
            'Запуск: `.venv-inference/Scripts/python.exe tools/predict_images_g.py --input INPUT_DIR --output OUTPUT.csv --device cpu`.\n'
            'На NVIDIA можно установить torch/torchvision из cu124 index и выбрать `--device cuda`.\n'
            'Инференс использует только локальные веса и блокирует Python TCP/DNS/UDP. Установка зависимостей требует предварительной загрузки.\n'
            'Для опытной маски AA00077 добавить `--recognition-profile russian_v1`; по умолчанию остаётся frozen_g.\n'
            'Скорость: `python tools/benchmark_inference_g.py --input INPUT_DIR --output NEW_REPORT.json --device cpu`. Отчёт разделяет первый и прогретые проходы, чтение файлов и загрузку моделей; GT не требуется.\n'
            'Сведения об авторстве: dataset/ATTRIBUTION.md. Фотографии в комплект не включены.\n'
            'Весовые байты совпадают с контролем; в копии конфига изменены только пути к весам.\n',encoding='utf-8')
        manifest={'scope':'Runtime snapshot; no dataset, training or final split',
                  'source_config_sha256':original_config_hash,'source_hashes':source_hashes,
                  'config_path_fields_relocated':True,'default_profile':'frozen_g',
                  'files':{p.relative_to(stage).as_posix():{'sha256':file_sha256(p),'bytes':p.stat().st_size}
                           for p in sorted(stage.rglob('*')) if p.is_file()}}
        (stage/'bundle_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        stage.rename(output)
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args(argv)
    result=build_bundle(args.output)
    print(f"Prepared {len(result['files'])} files under {args.output}")


if __name__=='__main__':main()
