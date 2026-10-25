"""Portable, byte-bound input validation for offline image inference."""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from pathlib import Path, PureWindowsPath
import re

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}


@contextmanager
def offline_network():
    """Reuse TCP protection and also block Python DNS and UDP during inference."""
    import socket
    from tools.baseline_predict import deny_network
    names=('getaddrinfo','gethostbyname','gethostbyname_ex','gethostbyaddr')
    originals={name:getattr(socket,name) for name in names}
    original_sendto=socket.socket.sendto
    def blocked(*args,**kwargs):raise RuntimeError('Network access is disabled for image inference')
    for name in names:setattr(socket,name,blocked)
    socket.socket.sendto=blocked
    try:
        with deny_network():yield
    finally:
        for name,value in originals.items():setattr(socket,name,value)
        socket.socket.sendto=original_sendto


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def image_paths(source: Path) -> list[Path]:
    source = Path(source)
    if source.is_file():
        if source.suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError('Input must be a JPG, JPEG, PNG or BMP image')
        return [source]
    if not source.is_dir():
        raise ValueError(f'Input does not exist: {source}')
    paths = sorted(p for p in source.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not paths:
        raise ValueError('Input directory contains no supported image files')
    return paths


def checkpoint_path(value: str, project_root: Path) -> Path:
    """Relocate old absolute configs using their bundled models/... suffix."""
    if not isinstance(value, str) or not value:
        raise ValueError('Checkpoint path must be a nonempty string')
    path = Path(value)
    # Bundled relative paths belong to the project, independent of shell CWD.
    if not path.is_absolute() and not PureWindowsPath(value).drive:
        relative = project_root / path
        if relative.is_file():
            return relative.resolve()
    if path.is_file():
        return path.resolve()
    parts = PureWindowsPath(value).parts if '\\' in value or PureWindowsPath(value).drive else path.parts
    if 'models' in parts:
        suffix = parts[parts.index('models'):]
        if any(part in ('.', '..') for part in suffix):
            raise ValueError('Checkpoint relocation cannot traverse directories')
        relocated = project_root.joinpath(*suffix)
        if relocated.is_file():
            return relocated.resolve()
    raise ValueError(f'Local checkpoint missing: {value}; automatic download is disabled')


def load_runtime_config(path: Path, project_root: Path) -> dict:
    config = json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if not isinstance(config, dict):
        raise ValueError('Inference config must be a JSON object')
    result = dict(config)
    keys = ('detector', 'ocr') + (('ordinary_detector',) if 'ordinary_detector' in config else ())
    for key in keys:
        expected = config.get(key + '_sha256')
        if not isinstance(expected, str) or not re.fullmatch('[0-9a-fA-F]{64}', expected):
            raise ValueError(f'{key}_sha256 must bind the local checkpoint bytes')
        resolved = checkpoint_path(config.get(key), Path(project_root))
        with resolved.open('rb') as stream:
            if stream.read(128).startswith(b'version https://git-lfs.github.com/spec/v1'):
                raise ValueError(f'{key}: Git LFS pointer is not a model checkpoint')
        if file_sha256(resolved) != expected.lower():
            raise ValueError(f'{key}: checkpoint SHA-256 mismatch; inference was not started')
        result[key] = str(resolved)
    return result
