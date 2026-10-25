"""Freeze a local delivery candidate and verify every archived byte, without publishing."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import sys
import zipfile

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256
from tools.build_final_i_bundle import verify_freeze
from tools.final_i_runtime import verify_runtime

VERIFY_SOURCE='''from pathlib import Path,PurePosixPath
import hashlib,json
root=Path(__file__).resolve().parent
manifest=json.loads((root/'package_manifest.json').read_text(encoding='utf-8'))
total=0
for name,expected in manifest['files'].items():
    relative=PurePosixPath(name);p=(root/name).resolve()
    if relative.is_absolute() or '..' in relative.parts or not p.is_relative_to(root):raise ValueError('Invalid path: '+name)
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for b in iter(lambda:stream.read(4*1024*1024),b''):h.update(b)
    if h.hexdigest()!=expected['sha256'] or p.stat().st_size!=expected['bytes']:raise ValueError('Changed: '+name)
    total+=1
print('Verified files:',total,'; status:',manifest['status'])
print('Rights document and unresolved technical criteria: docs/acceptance_checklist.md')
'''


def package(delivery,archive):
    delivery,archive=Path(delivery).resolve(),Path(archive).resolve()
    if archive.exists():raise ValueError('Existing archive is preserved')
    verify_freeze(ROOT);verify_runtime(delivery/'solution')
    (delivery/'verify_package.py').write_text(VERIFY_SOURCE,encoding='utf-8')
    checklist=json.loads((delivery/'docs/acceptance_checklist.json').read_text(encoding='utf-8'))
    if checklist['all_requirements_met']:raise ValueError('Open requirements must not be hidden')
    files={p.relative_to(delivery).as_posix():dict(sha256=file_sha256(p),bytes=p.stat().st_size)
           for p in sorted(delivery.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p!=delivery/'package_manifest.json'}
    manifest=dict(status='LOCAL_CANDIDATE_WITH_OPEN_REQUIREMENTS',files=files,
        all_requirements_met=False,external_upload_performed=False,
        permission_document_status='USER_ATTESTED_DOCUMENT_DEFERRED',
        official_validator_errors=0,official_validator_warnings=113,
        source_models_and_split_unchanged=True,automatic_training_started=False,
        explanatory_note_pages=4,full_real_test_inference_repeated=False)
    (delivery/'package_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
    archive.parent.mkdir(exist_ok=True,parents=True)
    prefix=archive.stem
    with zipfile.ZipFile(archive,'w',allowZip64=True,compresslevel=6) as z:
        for name in [*files,'package_manifest.json']:
            kind=zipfile.ZIP_DEFLATED if Path(name).suffix.lower() in ('.py','.json','.csv','.txt','.md','.yaml','.yml','.xml','.log','.lock') else zipfile.ZIP_STORED
            z.write(delivery/name,prefix+'/'+name,compress_type=kind)
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist())!=len(files)+1 or len(set(z.namelist()))!=len(files)+1:raise ValueError('Archive entries changed')
        for name,expected in [*files.items(),('package_manifest.json',dict(sha256=file_sha256(delivery/'package_manifest.json'),bytes=(delivery/'package_manifest.json').stat().st_size))]:
            h=hashlib.sha256();size=0
            with z.open(prefix+'/'+name) as f:
                for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block);size+=len(block)
            if size!=expected['bytes'] or h.hexdigest()!=expected['sha256']:raise ValueError('Archive data changed: '+name)
    sha=file_sha256(archive)
    archive.with_suffix('.zip.sha256').write_text(sha+'  '+archive.name+'\n',encoding='ascii')
    result=dict(status='PASS',archive=str(archive),sha256=sha,bytes=archive.stat().st_size,
        entries=len(files)+1,all_archived_SHA256_and_CRC_verified=True,external_upload_performed=False,
        all_requirements_met=False,checklist='docs/acceptance_checklist.md')
    verify_freeze(ROOT)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--delivery',type=Path,required=True);p.add_argument('--archive',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    a=p.parse_args();result=package(a.delivery,a.archive)
    a.report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
