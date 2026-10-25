"""Phase G evidence namespace and immutable Phase F control."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json,importlib.util
ROOT=Path(__file__).resolve().parents[1]
IMAGES=ROOT.parent/'external_data/organizer_debug/Полуфинал - public/debug set'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def write(path,data):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')

def scorer():
    spec=importlib.util.spec_from_file_location('g_organizer_frozen_scorer',IMAGES.parent/'evaluate.py')
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module
from dataset.schema import sha256

REPORT=ROOT/'reports/phase_g'
CONFIG=ROOT/'configs/phase_g'
DATA=ROOT.parent/'generated_data/real_g_v1'
FCONFIG=ROOT/'configs/phase_f/best_f.json'
SEED=20261004

def spatial_references():
    records={}
    for r in read(ROOT/'reports/phase_e/oracle_expansion/manifest.json')['records']:
        records[(r['image'],r['ground_truth'])]={'image':r['image'],'ground_truth':r['ground_truth'],'plate_type':r['plate_type'],'quad':r['manual_quad']}
    for r in read(ROOT/'reports/phase_f/real_annotation_v2/manual_annotations.json')['records']:
        for a in r['annotations']:
            key=(r['original_image'],a.get('organizer_literal_plate_num',a['plate_num']))
            records[key]={'image':key[0],'organizer_ground_truth':key[1],'ground_truth':a['plate_num'],'plate_type':a['plate_type'],'quad':a['quad']}
    return list(records.values())


def freeze():
    output=REPORT/'freeze_best_f.json'
    if output.exists():
        result=read(output)
        bad=[p for p,h in result['hashes'].items() if not Path(p).is_file() or sha256(Path(p))!=h]
        assert not bad,('Frozen BEST-F changed',bad[:5])
        return result
    c=read(FCONFIG)
    paths=[p for p in (ROOT/'reports/phase_f').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'configs/phase_f').rglob('*') if p.is_file()]
    paths += [p for p in (ROOT/'models/phase_f').rglob('*') if p.is_file()]
    paths += [Path(c['ocr']),ROOT/'docs/phase_f_report.md']
    paths += [p for p in (ROOT/'src').rglob('*.py')]
    paths += list((ROOT/'tools').glob('*_f.py'))
    paths += [ROOT/'tools/phase_f_setup.py',ROOT/'tools/phase_e.py']
    paths += list(IMAGES.glob('*.jpg'))+[IMAGES.parent/'debug_labels.csv',IMAGES.parent/'evaluate.py']
    paths=sorted(set(paths))
    result={'hashes':{str(p):sha256(p) for p in paths},'baseline':c,
        'control':{'FPA':51,'GT':115,'type1':11,'type1a':40,'common_untrained':37,'common_GT':87},
        'policy':'No F config/report/source/checkpoint/prediction edits. G corrections and metric clarifications are sidecar evidence. Public data is exposed development, not independent test.'}
    CONFIG.mkdir(parents=True,exist_ok=True);write(output,result)
    return result


def main():
    f=freeze();test=read(ROOT/'reports/phase_f/final_synthetic_test.json')
    terms={
        'F_100_of_207':{'level':'END_TO_END_SYNTHETIC_PAIRED_FPA','correct':test['full_match'],'total':test['GT_targets'],
            'definition':'Held-out procedural full scenes after F selection lock; conf>=0.5, predicted bbox IoU>=0.5 AND exact text AND correct type. Includes detection, geometry, split/stitch, frozen OCR and postprocess.',
            'per_type':test['per_type'],'not_comparable_to_D_OCR_only':True},
        'F_detector_only_synthetic':{k:test['detector'][k] for k in ('Precision','Recall','F1','FP_per_image')},
        'D_OCR_only_synthetic':{'definition':'Exact text on pre-cropped authored OCR inputs from D dataset; no scene detection stage. Historical reported100% is specific to that crop distribution.'},
        'required_levels':['OCR_ONLY_SYNTHETIC_FPA','DETECTOR_ONLY_SYNTHETIC_RECALL','DETECTOR_ONLY_SYNTHETIC_PRECISION',
            'END_TO_END_SYNTHETIC_FPA','OCR_ONLY_REAL_GOOD_INPUT_FPA','DETECTOR_ONLY_REAL_RECALL','END_TO_END_REAL_FPA'],
        'real_scope':'Exposed public development until independently sourced group-disjoint real data exists; never rename public to independent real test',
        'F_unchanged':'Terminology clarified here because F artifacts are frozen'}
    write(REPORT/'metric_definitions.json',terms)
    write(REPORT/'workflow_status.json',{'stage':'freeze_and_metric_definitions','state':'COMPLETE','frozen_files':len(f['hashes']),
        'next':'All64 failure audit, waterfall, type1a oracle and real good-input OCR; no training before diagnostic gates'})
    print('BEST-F frozen',len(f['hashes']),'files; F100/207 is paired scene end-to-end FPA',flush=True)


if __name__=='__main__':main()
