import math
import csv
import json
import pytest
from src.vehicle_filter import vehicle_support,filter_plate_detections
from tools.vehicle_filter_ablation import metrics


def test_spatial_support_and_fraction_boundary():
    assert vehicle_support([0,0,10,10],[{'bbox':[-10,-10,8,20]}])['keep']
    assert not vehicle_support([0,0,10,10],[{'bbox':[-10,-10,6,20]}])['keep']
    assert not vehicle_support([0,0,10,10],[{'bbox':[20,20,30,30]}])['keep']
    assert vehicle_support([0,0,10,10],[])['reason']=='NO_VEHICLE_DETECTED'


@pytest.mark.parametrize('box',[[0,0,0,10],[0,0,10,math.nan],[0,0,10]])
def test_invalid_observations_are_not_accepted(box):
    with pytest.raises(ValueError):vehicle_support(box,[])


def test_kept_indices_retain_original_cache_identity():
    plates=[dict(bbox=[0,0,10,10]),dict(bbox=[30,30,40,40])]
    indices,decisions=filter_plate_detections(plates,[dict(bbox=[25,25,45,45])])
    assert indices==[1] and decisions[1]['detection_index']==1


def test_cached_metric_reproduces_original_indices_after_filter():
    plates={'img':[dict(bbox=[0,0,10,10],confidence=.9),dict(bbox=[30,30,40,40],confidence=.8)]}
    rows=[dict(review_id='r',source_sha256='img',plate_type='type1b',plate_num='AA12377',bbox='[30,30,40,40]')]
    trace=[dict(image='img',detections=[dict(detection_index=0,plate_type='type1',final_ocr='A123AA77'),
                                      dict(detection_index=1,plate_type='type1b',final_ocr='AA12377')])]
    result,objects=metrics(rows,plates,trace,{'img':[dict(bbox=[25,25,45,45])]})
    assert result['overall']['correct']==1 and objects[0]['matched_original_detection_index']==1
    miss,_=metrics(rows,plates,trace,{'img':[]})
    assert miss['overall']['detected']==0


def test_saved_packet_replays_without_loading_images(tmp_path):
    from tools.replay_vehicle_ablation import replay
    from src.inference_io import file_sha256
    (tmp_path/'inputs').mkdir()
    row=dict(review_id='fixture',source_sha256='img',plate_type='type1a',plate_num='A123AA77',
             bbox='[30,30,40,40]',image='nonexistent-image-must-never-be-loaded.png')
    with (tmp_path/'inputs/val.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    plates={'img':[dict(bbox=[30,30,40,40],confidence=.8)]}
    traces=[dict(image='img',detections=[dict(detection_index=0,plate_type='type1a',final_ocr='A123AA77')])]
    baseline,_=metrics([row],plates,traces)
    filtered,_=metrics([row],plates,traces,{'img':[]})
    payload={'inputs/val_detections.json':plates,'inputs/val_traces.json':traces,
             'vehicle_detections.json':{'img':[]},'result.json':dict(baseline=baseline,filtered=filtered)}
    for name,value in payload.items():(tmp_path/name).write_text(json.dumps(value),encoding='utf-8')
    files={p.relative_to(tmp_path).as_posix():file_sha256(p) for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path/'input_manifest.json').write_text(json.dumps(dict(files=files)))
    result=replay(tmp_path)
    assert result['metric_values_identical'] and not result['image_inference']
    (tmp_path/'vehicle_detections.json').write_text('{}')
    with pytest.raises(ValueError,match='changed'):replay(tmp_path)


def test_portable_assignment_maximizes_matches_before_iou():
    from tools.replay_vehicle_ablation import portable_match
    rows=[dict(bbox=[0,0,10,10]),dict(bbox=[2,0,12,10])]
    detections=[dict(bbox=[1,0,11,10]),dict(bbox=[-2,0,8,10])]
    assert portable_match(rows,detections)=={0:1,1:0}
    assert portable_match(rows,[])=={}
