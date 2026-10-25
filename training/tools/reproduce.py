"""Verify or explicitly reproduce FINAL-I detector / FINAL-H OCR adaptation, offline."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'source'))


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):value.update(block)
    return value.hexdigest()


def verify():
    manifest=json.loads((ROOT/'manifest.json').read_text(encoding='utf-8'))
    for name,expected in manifest['files'].items():
        p=(ROOT/name).resolve()
        if not p.is_relative_to(ROOT) or digest(p)!=expected:raise ValueError('Training input changed: '+name)
    return manifest


def detector(workspace):
    import torch
    from ultralytics import YOLO
    recipe=json.loads((ROOT/'recipes/detector.json').read_text(encoding='utf-8'))
    data=workspace/'detector_data.yaml'
    data.write_text('path: '+(ROOT/'detector_data').as_posix()+'\ntrain: images/train\nval: images/val\nnames:\n  0: type1\n  1: type1a\n  2: type1b\n',encoding='utf-8')
    settings=dict(recipe['settings'],data=str(data),project=str(workspace/'runs'),name='detector_reproduction')
    torch.set_num_threads(4)
    model=YOLO(str(ROOT/'initial/detector.pt'))
    reference={k:v.detach().cpu().float().clone() for k,v in model.model.named_parameters()}
    def before_first_batch(trainer):
        from ultralytics.utils.torch_utils import unwrap_model
        actual=dict(unwrap_model(trainer.model).named_parameters())
        if set(actual)!=set(reference) or any(not torch.equal(reference[k],v.detach().cpu().float()) for k,v in actual.items()):
            raise ValueError('Trainer reset the initialization')
    model.add_callback('on_pretrain_routine_end',before_first_batch)
    model.train(**settings)
    if model.trainer.epoch+1!=4:raise ValueError('Incomplete four-epoch reproduction')
    return Path(model.trainer.last)


def ocr(workspace):
    # Reuse the original loop and its exact input arrays; only locations and
    # integrity guard relocate. Historical test inference is never invoked.
    from tools import phase_h_train as original
    original.ROOT=workspace
    original.REPORT=workspace/'ocr_report'
    original.CONFIG=workspace/'configs'
    original.MODELS=workspace/'models'
    original.D_WEIGHTS=ROOT/'initial/ocr.pt'
    original.SPLIT=ROOT/'evidence/split_manifest.json'
    original.guard_check=verify
    original.REPORT.mkdir(parents=True)
    (original.REPORT/'training').mkdir(exist_ok=True)
    for name in ('prepared.json','input_review.json','arrays.npz'):
        shutil.copy2(ROOT/'ocr_data'/name,original.REPORT/'training'/name)
    original.torch.set_num_threads(4);original.cv2.setNumThreads(1)
    result=original.train('H_OCR_1_real_heavy',.75)
    return Path(result['best_checkpoint'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=('verify','detector','ocr'),default='verify')
    p.add_argument('--output',type=Path)
    args=p.parse_args(argv)
    verify()
    if args.stage=='verify':print('Training inputs and sources verified; training was not started');return 0
    if args.output is None or args.output.exists():p.error('Training requires a new --output directory')
    output=args.output.resolve()
    if output.is_relative_to(ROOT) or ROOT.is_relative_to(output):p.error('Output overlaps immutable training package')
    os.environ.update(YOLO_OFFLINE='true',YOLO_AUTOINSTALL='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    from src.inference_io import offline_network
    output.mkdir(parents=True)
    with offline_network():checkpoint=detector(output) if args.stage=='detector' else ocr(output)
    verify()
    (output/'result.json').write_text(json.dumps(dict(checkpoint=str(checkpoint),sha256=digest(checkpoint),
        historical_weights_bit_identity_guaranteed=False,
        note='Pinned inputs and recipe; hardware, kernels and OCR wall-time early stopping can change final bytes. No test evaluation.'),indent=2),encoding='utf-8')
    return 0


if __name__=='__main__':raise SystemExit(main())
