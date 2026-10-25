"""Recompute the vehicle-ablation metrics from saved observations, without image inference."""
from pathlib import Path
import argparse
import csv
import json
import sys
from collections import defaultdict

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.inference_io import file_sha256,offline_network
from src.vehicle_filter import filter_plate_detections


def target_box(row):
    value=row.get('bbox_xyxy') or row['bbox']
    return json.loads(value) if isinstance(value,str) else value


def iou(a,b):
    x=max(0.,min(a[2],b[2])-max(a[0],b[0]));y=max(0.,min(a[3],b[3])-max(a[1],b[1]))
    intersection=x*y
    union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union>0 else 0.


def portable_match(rows,detections):
    """Exact cardinality-first assignment for the frozen selected-target VAL.

    This snapshot has at most2 targets/frame; a guard prevents exponential
    work on an unrelated full-scene annotation. No SciPy dependency is needed.
    """
    if len(rows)>3:raise ValueError('Portable cache replay supports at most3 selected targets per frame')
    edges=[[(j,iou(target_box(row),d['bbox'])) for j,d in enumerate(detections)
            if iou(target_box(row),d['bbox'])>=.5] for row in rows]
    best_score=(-1,-1.);best={}
    def visit(index,used,matches,overlap):
        nonlocal best,best_score
        if index==len(rows):
            score=(len(matches),overlap)
            if score>best_score:best_score=score;best=dict(matches)
            return
        for j,value in edges[index]:
            if j not in used:
                matches[index]=j;visit(index+1,used|{j},matches,overlap+value);del matches[index]
        visit(index+1,used,matches,overlap)
    visit(0,set(),{},0.)
    return best


def portable_metrics(rows,plates,traces,vehicles=None):
    grouped=defaultdict(list)
    for r in rows:grouped[r['source_sha256']].append(r)
    traces={r['image']:r for r in traces};records=[]
    for sha,gt in grouped.items():
        source=plates[sha]
        indices=filter_plate_detections(source,vehicles[sha])[0] if vehicles is not None else list(range(len(source)))
        matched=portable_match(gt,[source[i] for i in indices])
        observed={r['detection_index']:r for r in traces[sha]['detections']}
        for i,row in enumerate(gt):
            local=matched.get(i);original=indices[local] if local is not None else None
            obs=observed.get(original,{})
            records.append(dict(gt_type=row['plate_type'],matched=original is not None,
                correct=original is not None and obs.get('plate_type')==row['plate_type'] and obs.get('final_ocr')==row['plate_num']))
    def summary(rs):
        n=len(rs);det=sum(r['matched'] for r in rs);correct=sum(r['correct'] for r in rs)
        return dict(objects=n,detected=det,recall=det/n if n else None,correct=correct,FPA=correct/n if n else None)
    return dict(overall=summary(records),per_type={k:summary([r for r in records if r['gt_type']==k])
                for k in ('type1','type1a','type1b')})


def replay(packet):
    packet=Path(packet).resolve()
    read=lambda p:json.loads(p.read_text(encoding='utf-8'))
    manifest=read(packet/'input_manifest.json')
    for name,sha in manifest['files'].items():
        path=(packet/name).resolve()
        if not path.is_relative_to(packet) or file_sha256(path)!=sha:
            raise ValueError('Ablation replay input changed: '+name)
    with offline_network():
        with (packet/'inputs/val.csv').open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
        plates=read(packet/'inputs/val_detections.json');traces=read(packet/'inputs/val_traces.json')
        vehicles=read(packet/'vehicle_detections.json')
        baseline=portable_metrics(rows,plates,traces)
        filtered=portable_metrics(rows,plates,traces,vehicles)
    expected=read(packet/'result.json')
    if baseline!=expected['baseline'] or filtered!=expected['filtered']:
        raise ValueError('Replayed metrics differ from the recorded ablation')
    return dict(status='PASS',baseline=baseline,filtered=filtered,
        metric_values_identical=True,image_inference=False,training=False,real_test_access=False,
        plate_and_vehicle_observations_from_cache=True,GT_created_or_changed=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--packet',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Existing output is preserved')
    result=replay(a.packet);a.output.parent.mkdir(exist_ok=True,parents=True)
    a.output.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result,indent=2))
