"""Package predownloaded, pinned wheels and frozen local inference assets."""
from __future__ import annotations

import argparse
from email.parser import BytesParser
from email.policy import default as email_policy
import json
from pathlib import Path
import re
import shutil
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_inference_bundle import build_bundle
from tools.install_offline_runtime import sha256, verify_distribution


def canonical(name):
    return re.sub(r'[-_.]+', '-', name).lower()


def read_pins(path):
    pins = {}
    for line in Path(path).read_text('utf-8-sig').splitlines():
        if not line.strip():
            continue
        match = re.fullmatch(r'([A-Za-z0-9_.-]+)==([A-Za-z0-9_.+!-]+)', line)
        if not match:
            raise ValueError('Only exact package pins are supported')
        name, version = canonical(match[1]), match[2]
        if name in pins:
            raise ValueError('Duplicate pinned package: ' + name)
        pins[name] = version
    if pins.get('torch') != '2.6.0+cpu' or pins.get('torchvision') != '0.21.0+cpu':
        raise ValueError('Expected the tested CPU 2.6.0/0.21.0 freeze')
    return pins


def inspect_wheel(path):
    with ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names if re.fullmatch(r'[^/]+\.dist-info/METADATA', name)]
        if len(metadata_names) != 1:
            raise ValueError('Wheel must contain one METADATA file')
        metadata = BytesParser(policy=email_policy).parsebytes(archive.read(metadata_names[0]))
        if not metadata['Name'] or not metadata['Version']:
            raise ValueError('Wheel package identity is missing')
        dependencies = [str(value) for value in metadata.get_all('Requires-Dist', [])]
        if any('@' in dependency or '://' in dependency for dependency in dependencies):
            raise ValueError('Direct URL dependency is not allowed in offline wheels')
        wheel = BytesParser(policy=email_policy).parsebytes(archive.read(metadata_names[0].rsplit('/', 1)[0] + '/WHEEL'))
        tags = [str(value) for value in wheel.get_all('Tag', [])]
        if not tags or any(not tag.endswith(('-win_amd64', '-any')) for tag in tags):
            raise ValueError('Expected Windows x64 or platform independent wheel')
        return {
            'name': canonical(str(metadata['Name'])), 'version': str(metadata['Version']), 'tags': tags,
            'requires_dist': dependencies,
            'license_expression': str(metadata['License-Expression']) if metadata['License-Expression'] is not None else None,
            'license_field': str(metadata['License']) if metadata['License'] is not None else None,
            'license_files_in_wheel': [name for name in names if '.dist-info/' in name and re.search(r'(?i)(licen[sc]e|copying|notice)', name)],
        }


def wheel_inventory(downloads, pins):
    result = {}
    for variant in ('common', 'cpu', 'cu124'):
        expected = {name: version for name, version in pins.items() if name not in ('torch', 'torchvision')}
        if variant != 'common':
            expected = {'torch': '2.6.0+' + variant, 'torchvision': '0.21.0+' + variant}
        entries = {}
        directory = Path(downloads) / 'wheels' / variant
        for path in sorted(directory.glob('*.whl')):
            entry = inspect_wheel(path)
            name = entry['name']
            if name in entries or expected.get(name) != entry['version']:
                raise ValueError('Unexpected or duplicate wheel: ' + path.name)
            entries[name] = {**entry, 'filename': path.name, 'sha256': sha256(path), 'bytes': path.stat().st_size}
        if set(entries) != set(expected):
            raise ValueError('Missing wheels in ' + variant + ': ' + ', '.join(sorted(set(expected) - set(entries))))
        result[variant] = entries
    return result


def build_distribution(downloads, pins_path, output, *, runtime_builder=None, readme=None, default_profile='frozen_g'):
    downloads, output = Path(downloads).resolve(), Path(output).resolve()
    if output.exists() or output.is_relative_to(downloads) or downloads.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the download staging area')
    pins = read_pins(pins_path)
    inventory = wheel_inventory(downloads, pins)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='offline-package-', dir=output.parent) as temporary:
        if not Path(temporary).resolve().is_relative_to(output.parent):
            raise ValueError('Staging directory escaped output parent')
        stage = Path(temporary) / 'distribution'
        stage.mkdir()
        (runtime_builder or build_bundle)(stage / 'runtime')
        shutil.copy2(ROOT / 'tools/install_offline_runtime.py', stage / 'install_offline_runtime.py')
        for variant, entries in inventory.items():
            directory = stage / 'wheels' / variant
            directory.mkdir(parents=True)
            for entry in entries.values():
                shutil.copy2(downloads / 'wheels' / variant / entry['filename'], directory / entry['filename'])
                if sha256(directory / entry['filename']) != entry['sha256']:
                    raise ValueError('Wheel changed while copying')
        for variant in ('cpu', 'cu124'):
            entries = {**inventory['common'], **inventory[variant]}
            lines = [f"{name}=={entry['version']} --hash=sha256:{entry['sha256']}" for name, entry in sorted(entries.items())]
            (stage / f'requirements-{variant}.lock').write_text('\n'.join(lines) + '\n', 'utf-8')
        (stage / 'package_inventory.json').write_text(json.dumps(inventory, ensure_ascii=False, indent=2) + '\n', 'utf-8')
        (stage / 'README.md').write_text(
            '# Volga ANPR — автономный комплект\n\n'
            'Windows x64, заранее установленный CPython 3.12 с venv/ensurepip. Python и драйвер NVIDIA в комплект не входят.\n'
            'Перед переносом скопируйте весь каталог. ZIP/wheels сохраняют исходные license/notice files; инвентарь находится в package_inventory.json.\n\n'
            'CPU: `python -I install_offline_runtime.py --variant cpu --environment D:/volga-runtime-cpu`.\n'
            'NVIDIA: `python -I install_offline_runtime.py --variant cu124 --environment D:/volga-runtime-cuda`.\n'
            'Нужен совместимый драйвер CUDA 12.4; CUDA Toolkit отдельно не требуется.\n'
            'Выберите НОВУЮ папку окружения вне комплекта, желательно на диске с достаточным местом. Существующее окружение не изменяется.\n'
            'Установщик проверяет SHA-256, создаёт venv без system-site-packages, устанавливает только локальные wheels с --no-index/--require-hashes. Во время pip install Python TCP/DNS/UDP блокируются.\n'
            'При сбое новая частичная среда сохраняется для диагностики; повторите в другой новой папке.\n\n'
            'Из каталога runtime: `D:/volga-runtime-cpu/Scripts/python.exe -I tools/predict_images_g.py --input D:/photos --output D:/result.csv --device cpu`.\n'
            'Для CUDA замените путь к среде и устройство на cuda. По умолчанию frozen_g; опытный профиль: --recognition-profile russian_v1.\n'
            'Веса BEST-G сохранены. Комплект не содержит реальных датасетов, независимого test или окончательного split. Независимое качество и норматив времени на референсном ПК ещё не подтверждены.\n'
            'Документация pip: https://pip.pypa.io/en/stable/cli/pip_download/ и https://pip.pypa.io/en/stable/topics/secure-installs/.\n'
            'Источники wheels: https://pypi.org/simple; https://download.pytorch.org/whl/cpu; https://download.pytorch.org/whl/cu124.\n', 'utf-8')
        if readme is not None:
            (stage / 'README.md').write_text(readme, 'utf-8')
        manifest = {
            'platform': 'Windows CPython 3.12 x64', 'python_installer_included': False,
            'default_profile': default_profile, 'packages_per_variant': len(pins),
            'files': {p.relative_to(stage).as_posix(): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                      for p in sorted(stage.rglob('*')) if p.is_file()},
        }
        (stage / 'distribution_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', 'utf-8')
        verify_distribution(stage)
        stage.rename(output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--downloads', required=True, type=Path)
    parser.add_argument('--pins', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    manifest = build_distribution(args.downloads, args.pins, args.output)
    print(f"Prepared {len(manifest['files'])} files, {manifest['packages_per_variant']} packages per variant")


if __name__ == '__main__':
    main()
