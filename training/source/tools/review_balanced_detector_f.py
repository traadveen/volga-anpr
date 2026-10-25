"""Record completed human visual review and freeze the v2 training inputs."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,read,write
from tools.phase_f_setup import DATA,REPORT
from dataset.schema import sha256

def main():
    m=read(DATA/'manifest.json');v=read(REPORT/'dataset_validation.json');assert v['status']=='PASS'
    assert v['manifest_sha256']==sha256(DATA/'manifest.json')
    manual=REPORT/'real_annotation_v2/manual_annotations.json';a=read(manual)
    a.update(status='PASS',visual_review='All seven original full scenes and all fifteen rectified plate inputs viewed; both rows/region inside quad; source pixel grids reviewed. img032 literal GT transposition recorded; official source unchanged.')
    write(manual,a)
    v.update(visual_review='PASS',visual_evidence=[str(REPORT/'real_annotation_v2/rectified_review.jpg'),str(REPORT/'dataset_review/manifest.json')],
        scope='V1 synthetic scenes unchanged and reviewed in v1; same real validation; all additional v2 real train scenes exhaustively reviewed')
    write(REPORT/'dataset_validation.json',v)
    source=[ROOT/'dataset/generator/scenes_f.py',ROOT/'tools/prepare_balanced_detector_f.py',ROOT/'tools/validate_detector_data_f.py',manual]
    hashes={str(DATA/'manifest.json'):sha256(DATA/'manifest.json'),**v['label_hashes'],**{str(p):sha256(p) for p in source}}
    for r in m['records']:
        for task in ('detect','obb','pose'):
            hashes[str(DATA/task/'images'/r['split']/r['image'])]=r['sha256']
    for task in ('detect','obb','pose'):hashes[str(DATA/f'{task}.yaml')]=sha256(DATA/f'{task}.yaml')
    write(REPORT/'dataset_freeze.json',{'dataset':str(DATA),'manifest_sha256':sha256(DATA/'manifest.json'),'hashes':hashes,
        'status':'PASS','visual_review':'PASS','scope':'Immutable version2; version1 inputs/artifacts preserved separately'})
    write(REPORT/'dataset_composition.json',{'dataset':str(DATA),'manifest_sha256':sha256(DATA/'manifest.json'),
        'composition':m['composition'],'real_import':m['real_import'],'limitations':m['limitations'],'statistics':v['statistics']})
    for name in ('F02_balanced','F02_clean_v2','F03_obb_v2','F04_pose_v2'):
        p=ROOT/'configs/phase_f'/f'{name}.json';c=read(p)
        c['dataset_manifest_sha256']=sha256(DATA/'manifest.json')
        c['train'].update(epochs=12,patience=4)
        c['budget']='Controlled 12-epoch v2 comparison, same seed/data/optimizer/resolution. Inspect convergence; no claim of exhaustive architecture optimum.'
        write(p,c)
    print('V2 review/freeze PASS',v['statistics'],flush=True)

if __name__=='__main__':main()
