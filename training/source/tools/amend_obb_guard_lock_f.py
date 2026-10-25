"""One explicit runtime-safety amendment; preserve model selection before test metrics."""
from datetime import datetime,timezone
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,read,write
from tools.phase_f_setup import REPORT
from dataset.schema import sha256


def main():
    assert not (REPORT/'final_synthetic_test.json').exists()
    assert read(REPORT/'workflow_status.json')['stage']=='post_lock_synthetic_test'
    config_path=ROOT/'configs/phase_f/best_f.json';lock_path=REPORT/'selection_lock.json'
    c=read(config_path);old=read(lock_path)
    assert old['config_sha256']==sha256(config_path) and not old.get('amendments')
    source=ROOT/'src/image_anpr_f.py'
    assert all(sha256(Path(p))==h for p,h in c['inference_source_sha256'].items() if Path(p)!=source)
    assert sha256(Path(c['detector']))==c['detector_sha256'] and sha256(Path(c['ocr']))==c['ocr_sha256']
    history=REPORT/'selection_history/before_obb_guard';history.mkdir(parents=True,exist_ok=True)
    for source_path,target in [(config_path,history/'best_f.json'),(lock_path,history/'selection_lock.json')]:
        if target.exists():assert sha256(target)==sha256(source_path)
        else:shutil.copy2(source_path,target)
    # Recover and hash-check the exact pre-fix source, not merely a prose claim.
    text=source.read_text(encoding='utf-8')
    start=text.index('                raw=b.xyxyxyxy[0].cpu().numpy()')
    end=text.index('        elif result.boxes is not None:',start)
    original=text[:start]+'''                q=order_obb(b.xyxyxyxy[0].cpu().numpy())
                box=np.r_[q.min(axis=0),q.max(axis=0)]
                c=int(b.cls[0]);detections.append(Detection(tuple(map(float,box)),float(b.conf[0]),c,
                    result.names[c],tuple(map(tuple,q.tolist())),'model_OBB_rectangle_not_perspective_corners'))
'''+text[end:]
    previous=history/'image_anpr_f.py';previous.write_bytes(original.encode('utf-8'))
    assert sha256(previous)==c['inference_source_sha256'][str(source)]
    original_config_sha=sha256(config_path);c['inference_source_sha256'][str(source)]=sha256(source)
    write(config_path,c)
    amendment={'utc':datetime.now(timezone.utc).isoformat(),'reason':'Degenerate low-confidence OBB raised before guarded fallback during first synthetic test attempt',
        'change':'Adapter preserves finite bbox and sets quad=None for invalid OBB; nonfinite/zero-area observations are skipped. No learned weight, threshold, router, image size or preprocessing parameter changed.',
        'old_config_sha256':original_config_sha,'new_config_sha256':sha256(config_path),
        'old_source_sha256':sha256(previous),'new_source_sha256':sha256(source),
        'first_test_attempt':'FAILED_RUNTIME; no aggregate test metrics saved, inspected or used for model/parameter choice',
        'required_revalidation':'Focused and full CPU tests, unchanged public GPU predictions, CPU/GPU parity before final completed synthetic evaluation'}
    old['config_sha256']=sha256(config_path);old['original_lock_sha256']=sha256(history/'selection_lock.json');old['amendments']=[amendment]
    write(lock_path,old);write(REPORT/'synthetic_test_runtime_fix.json',amendment)
    print('Explicit safety amendment saved; original selection preserved',flush=True)


if __name__=='__main__':main()
