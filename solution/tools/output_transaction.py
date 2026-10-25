"""Publish a recoverable CSV/JSON set; detect interrupted or mixed generations.

The manifest is the commit record. Readers must use verify_publication() before
consuming a set. A hard crash leaves .publishing.json and the recovery directory;
recover() restores the previous set, never guesses which generation succeeded.
"""
from contextlib import contextmanager
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import shutil
import tempfile
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.validate_output import COLUMNS, validate_rows


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def paths(output):
    output = Path(output)
    return [output, *(output.with_suffix(s) for s in
                     ('.trace.json', '.errors.json', '.manifest.json'))]


def write_json(path, value):
    with Path(path).open('w', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


@contextmanager
def exclusive(output):
    lock = Path(output).with_suffix('.publication.lock')
    # OS advisory locking is released even when a process dies. Keep the lock
    # inode: unlinking it would let concurrent writers lock different files.
    with lock.open('a+b') as stream:
        if stream.tell() == 0:
            stream.write(b'0'); stream.flush()
        stream.seek(0)
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def _restore(output, stage, journal):
    for target in paths(output):
        old = journal['previous'][target.name]
        if old is None:
            target.unlink(missing_ok=True)
        else:
            if target.exists() and digest(target) == old:
                continue
            backup = stage / ('old-' + target.name)
            if digest(backup) != old:
                raise ValueError('Recovery backup checksum mismatch: ' + target.name)
            temporary = stage / ('restore-' + target.name)
            shutil.copyfile(backup, temporary)
            os.replace(temporary, target)


def recover(output):
    """Explicit crash recovery. Refuse while another writer owns the lock."""
    output = Path(output).resolve()
    with exclusive(output):
        pending = output.with_suffix('.publishing.json')
        journal = json.loads(pending.read_text('utf-8'))
        stage = (output.parent / journal['stage']).resolve()
        if stage.parent != output.parent or not stage.name.startswith('publication-'):
            raise ValueError('Unsafe recovery directory')
        if journal['output'] != output.name:
            raise ValueError('Recovery output mismatch')
        _restore(output, stage, journal)
        pending.unlink()
        shutil.rmtree(stage)


def verify_publication(output):
    output = Path(output).resolve()
    with exclusive(output):
        if output.with_suffix('.publishing.json').exists():
            raise ValueError('Interrupted publication; recover before reading')
        manifest = json.loads(output.with_suffix('.manifest.json').read_text('utf-8'))
        files = manifest['publication']['files']
        expected = paths(output)[:3]
        if set(files) != {p.name for p in expected}:
            raise ValueError('Publication file set mismatch')
        for path in expected:
            if digest(path) != files[path.name]:
                raise ValueError('Publication checksum mismatch: ' + path.name)
        return manifest


def publish(output, predictions, traces, errors, manifest):
    output = Path(output).resolve()
    validation = validate_rows([{k: row[k] for k in COLUMNS} for row in predictions],
                               {r['image'] for r in traces})
    if not validation['valid']:
        raise ValueError('Model output failed CSV validation: ' + '; '.join(validation['errors'][:3]))
    output.parent.mkdir(parents=True, exist_ok=True)
    with exclusive(output):
        pending = output.with_suffix('.publishing.json')
        if pending.exists():
            raise ValueError('Interrupted publication; explicit recovery required')
        stage = Path(tempfile.mkdtemp(prefix='publication-', dir=output.parent))
        try:
            targets = paths(output)
            staged = [stage / p.name for p in targets]
            with staged[0].open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=COLUMNS, delimiter=';', lineterminator='\n')
                writer.writeheader()
                writer.writerows({k: row[k] for k in COLUMNS} for row in predictions)
                stream.flush(); os.fsync(stream.fileno())
            generation = uuid.uuid4().hex
            write_json(staged[1], dict(records=traces, ground_truth_input=False, generation=generation))
            write_json(staged[2], dict(errors=errors, skipped_images=len(errors), generation=generation))
            write_json(staged[3], dict(manifest, output_validation=validation,
                publication=dict(version=1, generation=generation,
                                 files={p.name: digest(p) for p in staged[:3]})))
            previous = {}
            for target in targets:
                if target.exists():
                    backup = stage / ('old-' + target.name)
                    shutil.copyfile(target, backup)
                    previous[target.name] = digest(backup)
                else:
                    previous[target.name] = None
            journal = dict(output=output.name, stage=stage.name, previous=previous)
            write_json(stage / 'journal.json', journal)
            os.replace(stage / 'journal.json', pending)
            try:
                # Commit manifest last, and keep the interruption marker until
                # all files are replaced. Exceptions restore the entire old set.
                for source, target in zip(staged, targets):
                    os.replace(source, target)
            except BaseException:
                _restore(output, stage, journal)
                pending.unlink()
                raise
            pending.unlink()
        finally:
            # Preserve journal/backups if rollback failed or the process died.
            if not pending.exists():
                shutil.rmtree(stage)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()
    if args.recover:
        recover(args.output)
    else:
        print(json.dumps(verify_publication(args.output), ensure_ascii=False, indent=2))
