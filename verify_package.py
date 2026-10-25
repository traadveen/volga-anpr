"""Verify all immutable delivery files using only the Python standard library."""
from pathlib import Path,PurePosixPath
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'package_manifest.json').read_text('utf-8'))
for name,expected in manifest['files'].items():
    rel=PurePosixPath(name);path=(root/name).resolve()
    if rel.is_absolute() or '..' in rel.parts or not path.is_relative_to(root):raise ValueError('Unsafe path: '+name)
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    if h.hexdigest()!=expected['sha256'] or path.stat().st_size!=expected['bytes']:raise ValueError('Changed file: '+name)
print('Verified files:',len(manifest['files']))
print('Delivery status: docs/DELIVERY_STATUS.md')
