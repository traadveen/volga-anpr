"""Install a prepared Windows CPython 3.12 runtime into a NEW environment."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import runpy
import socket
import subprocess
import sys
import tempfile
import venv


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify_distribution(root):
    root = Path(root).resolve()
    manifest = json.loads((root / 'distribution_manifest.json').read_text('utf-8'))
    if manifest.get('platform') != 'Windows CPython 3.12 x64':
        raise ValueError('Unexpected distribution platform')
    files = manifest.get('files')
    if not isinstance(files, dict) or not files:
        raise ValueError('Distribution manifest has no files')
    for name, entry in files.items():
        relative = PurePosixPath(name)
        path = (root / name).resolve()
        if relative.is_absolute() or '..' in relative.parts or not path.is_relative_to(root):
            raise ValueError('Manifest path escapes distribution')
        if not path.is_file() or path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError('Distribution file missing or modified: ' + name)
    for variant in ('cpu', 'cu124'):
        if f'requirements-{variant}.lock' not in files:
            raise ValueError('Distribution lock missing: ' + variant)
    return manifest


def validate_target(root, environment):
    root, environment = Path(root).resolve(), Path(environment).resolve()
    if environment.exists():
        raise ValueError('Environment already exists; choose a new directory')
    if environment.is_relative_to(root) or root.is_relative_to(environment):
        raise ValueError('Environment must be outside the distribution')
    return environment


@contextmanager
def deny_network():
    def blocked(*args, **kwargs):
        raise RuntimeError('Offline installation: Python network access is disabled')
    names = ('create_connection', 'getaddrinfo', 'gethostbyname', 'gethostbyname_ex', 'gethostbyaddr')
    functions = {name: getattr(socket, name) for name in names}
    methods = {name: getattr(socket.socket, name) for name in ('connect', 'connect_ex', 'sendto')}
    try:
        for name in functions:
            setattr(socket, name, blocked)
        for name in methods:
            setattr(socket.socket, name, blocked)
        yield
    finally:
        for name, function in functions.items():
            setattr(socket, name, function)
        for name, function in methods.items():
            setattr(socket.socket, name, function)


def pip_arguments(root, variant):
    root = Path(root).resolve()
    if variant not in ('cpu', 'cu124'):
        raise ValueError('Unknown variant')
    return [
        '--isolated', '--disable-pip-version-check', '--no-cache-dir', 'install',
        '--no-index', '--only-binary=:all:', '--require-hashes',
        '--find-links', str(root / 'wheels/common'),
        '--find-links', str(root / 'wheels' / variant),
        '-r', str(root / f'requirements-{variant}.lock'),
    ]


def install_worker(root, environment, variant):
    if Path(sys.prefix).resolve() != environment or sys.prefix == sys.base_prefix:
        raise ValueError('Worker must run inside its new virtual environment')
    verify_distribution(root)
    with deny_network():
        original_argv = sys.argv
        try:
            sys.argv = ['pip', *pip_arguments(root, variant)]
            runpy.run_module('pip', run_name='__main__')
        finally:
            sys.argv = original_argv
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', type=Path, required=True)
    parser.add_argument('--variant', choices=('cpu', 'cu124'), default='cpu')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent
    environment = args.environment.resolve()
    if args.worker:
        return install_worker(root, environment, args.variant)
    if sys.version_info[:2] != (3, 12) or sys.implementation.name != 'cpython' or platform.system() != 'Windows' or sys.maxsize <= 2**32:
        raise ValueError('Use Windows x64 CPython 3.12 to create this environment')
    verify_distribution(root)
    environment = validate_target(root, environment)
    environment.parent.mkdir(parents=True, exist_ok=True)
    previous = {key: os.environ.get(key) for key in ('TEMP', 'TMP')}
    # Keep wheel extraction off a possibly nearly full system drive.
    with tempfile.TemporaryDirectory(prefix='offline-install-', dir=environment.parent) as temporary:
        if not Path(temporary).resolve().is_relative_to(environment.parent):
            raise ValueError('Temporary directory escaped environment parent')
        try:
            os.environ['TEMP'] = os.environ['TMP'] = temporary
            venv.EnvBuilder(with_pip=True, system_site_packages=False).create(environment)
            python = environment / 'Scripts/python.exe'
            subprocess.run([str(python), '-I', '-X', 'utf8', str(Path(__file__).resolve()),
                            '--environment', str(environment), '--variant', args.variant, '--worker'], check=True)
            subprocess.run([str(python), '-I', '-m', 'pip', '--isolated', '--disable-pip-version-check', 'check'], check=True)
        finally:
            for key, value in previous.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
    print('Installed into:', environment)
    print('Runtime:', root / 'runtime')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
