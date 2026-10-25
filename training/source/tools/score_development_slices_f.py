"""Separate seen training scenes from exposed but untrained public development."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,IMAGES,read,write,scorer
from tools.phase_f_setup import DATA,REPORT


def score(path):
    m=scorer();gt,known=m.load_gt(str(IMAGES.parent/'debug_labels.csv'),[])
    known|={p.name for p in IMAGES.glob('*.jpg')}
    preds=m.load_pred(str(path),known,[])
    data=read(DATA/'manifest.json');train=set(data['real_import']['original_train_images'])
    seen=[g for g in gt if g.image in train and g.is_target]
    excluded=train|{g.image for g in gt if g.is_target and any(g.ptype==s.ptype and m.compare_plates(g.plate,s.plate,g.ptype)[1] for s in seen)}
    result={}
    for label,images in [('training_seen_development',excluded),('untrained_exposed_development',known-excluded)]:
        raw=m.evaluate([g for g in gt if g.image in images],images,[p for p in preds if p.image in images],
            {'type1':.15,'type1a':.35,'type1b':.35,'precision':.15})
        result[label]={k:v for k,v in raw.items() if not k.startswith('_')}
        result[label]['images']=sorted(images)
    result['scope']='Previously exposed public development only. Exclude training scenes and conservatively any readable-GT match to training identities; not an independent holdout.'
    return result


def update():
    for path in (REPORT/'experiments').glob('*/result.json'):
        result=read(path);result['development_slices']=score(path.parent/'predictions.csv');write(path,result)
    baseline=score(ROOT/'reports/phase_e/benchmark_cpu_final.csv')
    write(REPORT/'baseline_development_slices.json',baseline)
    write(REPORT/'selection_policy.json',{'primary':'Full-plate accuracy on common untrained-exposed public DEVELOPMENT targets excluding all v2 training scenes/identities. Overall115 includes training targets and is diagnostic.',
        'eligibility':['Full development type1 >=8/15','Untrained-exposed type1 cannot fall below frozen E baseline','Untrained-exposed development FPA cannot fall below frozen E baseline',
            'GPU compute mean <100ms; strongly prefer <=60ms on GTX1060'],
        'ties':'Prefer simpler geometry, then faster pipeline; check rare-type support on synthetic validation explicitly',
        'not_a_holdout':'Both public slices were exposed in E. No generalization claim or untouched real-test tuning.',
        'baseline_untrained':baseline['untrained_exposed_development']['full_match_total'],
        'baseline_untrained_total':baseline['untrained_exposed_development']['gt_target_plates'],
        'dataset_manifest_sha256':__import__('dataset.schema',fromlist=['sha256']).sha256(DATA/'manifest.json'),
        'amendment':'Common slice updated before v2 training because seven new scenes were designated TRAIN; preserve original v1 policy in dataset_versions/v1',
        'declared_before_any_retrained_detector_public_evaluation':False})
    print('E baseline development slices:',[(k,v['full_match_total'],v['gt_target_plates']) for k,v in baseline.items() if isinstance(v,dict)])


if __name__=='__main__':update()
