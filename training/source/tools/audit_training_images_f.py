"""Document exact, deterministic loader JPEG repairs without rewriting the freeze."""
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import hashlib
import cv2
import numpy as np
from PIL import Image,ImageOps
from tools.phase_e import ROOT,IMAGES,read,write
from tools.phase_f_setup import REPORT
from dataset.schema import sha256


def verify_dataset_integrity():
    frozen=read(REPORT/'dataset_freeze.json')
    audit=read(REPORT/'training_image_repair_audit.json')
    expected={**frozen['hashes'],**audit['effective_hashes']}
    mismatches=[p for p,digest in expected.items() if not Path(p).is_file() or sha256(Path(p))!=digest]
    assert not mismatches,mismatches[:10]
    return {'frozen_files':len(expected),'byte_unchanged_files':len(expected)-len(audit['effective_hashes']),
            'deterministic_loader_reencoded_images':len(audit['effective_hashes']),
            'original_freeze_sha256':sha256(REPORT/'dataset_freeze.json'),
            'repair_audit_sha256':sha256(REPORT/'training_image_repair_audit.json')}


def main():
    frozen=read(REPORT/'dataset_freeze.json');changed=[]
    for p,digest in frozen['hashes'].items():
        if sha256(Path(p))!=digest:changed.append((Path(p),digest))
    assert len(changed)==18 and all(p.name.startswith('real_img_092_') for p,_ in changed)
    source=IMAGES/'img_092.jpg';original=source.read_bytes()
    with ZipFile(IMAGES.parents[1]/'source_download.zip') as archive:
        members={Path(name).name:name for name in archive.namelist() if '/debug set/' in name and name.endswith('.jpg')}
        public_files=list(IMAGES.glob('*.jpg'))
        assert len(public_files)==100
        assert all(sha256(p)==hashlib.sha256(archive.read(members[p.name])).hexdigest() for p in public_files)
    assert all(digest==sha256(source) for _,digest in changed)
    buffer=BytesIO()
    with Image.open(source) as im:ImageOps.exif_transpose(im).save(buffer,'JPEG',subsampling=0,quality=100)
    repaired=buffer.getvalue();repair_sha=hashlib.sha256(repaired).hexdigest()
    assert all(sha256(p)==repair_sha for p,_ in changed),'All task/replica bytes must equal the exact installed loader operation'
    before=cv2.imdecode(np.frombuffer(original,np.uint8),cv2.IMREAD_COLOR)
    after=cv2.imdecode(np.frombuffer(repaired,np.uint8),cv2.IMREAD_COLOR)
    assert before.shape==after.shape
    evidence={}
    for p in (REPORT/'workflow_logs').glob('train_*.log'):
        lines=[s for s in p.read_text(encoding='utf-8',errors='replace').replace('\r','\n').splitlines() if 'corrupt JPEG restored' in s]
        if lines:evidence[p.name]=lines
    result={'status':'EXPLAINED_DETERMINISTIC_REPAIR','original_source_sha256':sha256(source),
        'source_unchanged_verified_against_download_zip':True,'all_public_source_images_zip_verified':len(public_files),'source_terminal_bytes':original[-2:].hex(),
        'cause':'Installed Ultralytics verifier sees trailing bytes instead of final FFD9; atomically replaces dataset copies with EXIF-transposed Pillow JPEG quality100/subsampling0. Original source hard link is preserved.',
        'effective_hashes':{str(p):repair_sha for p,_ in changed},'original_hashes':{str(p):h for p,h in changed},
        'reconstruction_bit_exact':True,'shape':list(before.shape),
        'pixel_mean_abs_difference_0_255':float(np.mean(np.abs(before.astype(float)-after.astype(float)))),
        'pixel_max_abs_difference':int(np.max(np.abs(before.astype(int)-after.astype(int)))),
        'installed_verifier_sha256':sha256(ROOT/'.venv-cuda/Lib/site-packages/ultralytics/data/utils.py'),
        'training_log_evidence':evidence,
        'scope':'Retrospective byte audit. Same effective repaired bytes across detect/OBB/pose and all six train repeats. Only train scene affected; validation/test and official evaluation source unchanged. Original manifest/freeze preserved, with explicit effective-byte amendment.',
        'hashes_are_not_pixel_identical':True}
    write(REPORT/'training_image_repair_audit.json',result)
    print(verify_dataset_integrity(),flush=True)


if __name__=='__main__':main()
