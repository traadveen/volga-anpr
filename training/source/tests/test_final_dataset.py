import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.inference_io import file_sha256
from tools.export_final_dataset import export_dataset, real_annotation, license_text
from tools.phase_i_review import sha_json


@pytest.fixture
def human(tmp_path):
    image=tmp_path/'source.png'
    pixels=np.arange(80*160*3,dtype=np.uint8).reshape(80,160,3)
    cv2.imencode('.png',pixels)[1].tofile(image)
    crop=tmp_path/'crop.png'
    cv2.imencode('.png',pixels[10:70,20:140])[1].tofile(crop)
    label=tmp_path/'label.txt';label.write_text('source label')
    record=dict(domain='human_real',variant='real_only',repeat=0,review_id='human_1',
                plate_num='AA12377',plate_type='type1b',split='train',source_sha256=file_sha256(image),
                source_image=str(image),training_region=[20,10,140,70],export_image=str(crop),
                export_sha256=file_sha256(crop),export_label=str(label),label_sha256=file_sha256(label),
                duplicate_group_id='dup_1',original_image_group_id='photo_1',source_family='fixture_family')
    decision=dict(record,status='accepted',human_confirmed=True,quad_verified=True,
                  training_region_complete=True,has_other_plates=False,quad=[[30,20],[130,20],[130,60],[30,60]])
    source=dict(review_id='human_1',image=str(image),plate_num='AA12377',plate_type='type1b',
                source_sha256=file_sha256(image),source='kaggle_russian_five_classes',
                license='UNVERIFIED',source_license_verified='False')
    return record,decision,source


def test_human_geometry_is_translated_not_created(human):
    record,decision,source=human
    row,box,quad=real_annotation(record,source,decision)
    assert box==[10,10,100,40]
    assert quad==[[10,10],[110,10],[110,50],[10,50]]
    assert row['license']=='UNVERIFIED'
    assert license_text('Public Domain')=='Public Domain'
    with pytest.raises(ValueError,match='human-confirmed'):
        real_annotation(record,source,dict(decision,human_confirmed=False))
    with pytest.raises(ValueError,match='incomplete'):
        real_annotation(record,source,dict(decision,has_other_plates=True))


def test_export_keeps_rights_pending_and_sources_unchanged(tmp_path,human):
    record,decision,source=human
    synth=tmp_path/'synthetic'
    for d in ('provenance','labels','generator','images/synthetic'):(synth/d).mkdir(parents=True)
    fields=('image','plate_num','plate_type','bbox','quad','is_vehicle','is_synthetic','source','license','conditions')
    with (synth/'meta.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,delimiter=';');w.writeheader()
        w.writerow(dict(image='images/synthetic/a.png',plate_num='A123AA77',plate_type='type1',
                        bbox='0,0,5,5',quad='0,0,5,0,5,5,0,5',is_vehicle=1,is_synthetic=1,source='own',license='CC-BY-4.0'))
    (synth/'export_manifest.json').write_text(json.dumps(dict(meta_sha256=file_sha256(synth/'meta.csv'),official_synthetic_images=1)))
    index=tmp_path/'index.json';index.write_text('{}')
    state=tmp_path/'state.json';state.write_text(json.dumps(dict(decisions={'human_1':decision})))
    manifest=tmp_path/'manifest.json';manifest.write_text(json.dumps(dict(index_sha256=file_sha256(index),
        selected_decision_sha256={'human_1':sha_json(decision)},records=[record])))
    split=tmp_path/'train.csv'
    with split.open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=source);w.writeheader();w.writerow(source)
    before={p:file_sha256(p) for p in (index,state,manifest,split,Path(record['source_image']))}
    target=tmp_path/'export'
    info=export_dataset(target,synth,manifest,state,index,{'train':split})
    assert info['official_real_images']==1 and info['plate_objects']==2
    assert info['permission_document_pending_ids']==['human_1'] and not info['publication_ready']
    permission=json.loads((target/'permissions/status.json').read_text())
    assert permission['document'] is None and permission['document_verified'] is False
    proof=json.loads((target/'provenance/human_geometry.json').read_text())[0]
    assert not proof['human_confirmation_created'] and not proof['source_license_verified']
    assert all(file_sha256(p)==digest for p,digest in before.items())
    with pytest.raises(ValueError,match='new directory'):
        export_dataset(target,synth,manifest,state,index,{'train':split})


def test_changed_decision_is_rejected_before_publication(tmp_path,human):
    record,decision,source=human
    with pytest.raises(ValueError,match='bind'):
        real_annotation(record,source,dict(decision,plate_num='AA12477'))
    with pytest.raises(ValueError,match='image'):
        real_annotation(record,dict(source,source_sha256='changed'),decision)


def test_combined_manifest_reproduces_only_synthetic_recipe(tmp_path,monkeypatch):
    from tools import reproduce_organizer_synthetic as tool
    dataset=tmp_path/'packet';(dataset/'provenance').mkdir(parents=True)
    recipe=dict(parts=[{}],meta_sha256='same',official_synthetic_images=3)
    (dataset/'export_manifest.json').write_text(json.dumps(dict(synthetic_export=recipe,official_real_images=109)))
    (dataset/'provenance/part00.manifest.json').write_text(json.dumps(dict(count=3,seed=123,profile='mixed')))
    workspace=tmp_path/'rebuilt';commands=[]
    def run(command,**kwargs):
        commands.append(command)
        if any(str(x).endswith('export_organizer_synthetic.py') for x in command):
            target=workspace/'dataset';target.mkdir()
            (target/'export_manifest.json').write_text(json.dumps(recipe))
    monkeypatch.setattr(tool.subprocess,'run',run)
    result=tool.reproduce(dataset,workspace)
    assert result['official_synthetic_images']==3 and len(commands)==3
    assert '109' not in commands[0] and '123' in commands[0]
