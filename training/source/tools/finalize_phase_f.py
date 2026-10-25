"""Assemble final evidence only after training, lock, export, parity, and review."""
from pathlib import Path
import sys
import xml.etree.ElementTree as ET
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import ROOT,read,write
from tools.phase_f_setup import REPORT,freeze
from dataset.schema import sha256

def main():
    config=read(ROOT/'configs/phase_f/best_f.json');name=config['selected_experiment'];best=read(REPORT/'experiments'/name/'result.json')
    assert not read(REPORT/'workflow_status.json').get('state')=='RUNNING','Finish CUDA training before final benchmark/report'
    lock=read(REPORT/'selection_lock.json');assert lock['config_sha256']==sha256(ROOT/'configs/phase_f/best_f.json')
    export=read(REPORT/'export/comparison.json');parity=read(REPORT/'parity/cpu_gpu.json');test=read(REPORT/'final_synthetic_test.json')
    attribution=read(REPORT/'attribution/remaining_failures.json');assert attribution['review_status']=='PASS'
    ranking=read(REPORT/'provisional_ranking.json');dataset=read(REPORT/'dataset_composition.json');oracle=read(REPORT/'oracle_quad.json')
    training={p.stem:read(p) for p in (REPORT/'training').glob('*.json')}
    suite=ET.parse(REPORT/'full_cpu_tests.xml').getroot().find('testsuite');assert suite.get('errors')=='0' and suite.get('failures')=='0'
    focused=ET.parse(REPORT/'focused_tests.xml').getroot().find('testsuite');assert focused.get('errors')=='0' and focused.get('failures')=='0'
    frozen=freeze();dsfreeze=read(REPORT/'dataset_freeze.json')
    from tools.audit_training_images_f import verify_dataset_integrity
    dataset_integrity=verify_dataset_integrity()
    cpu=parity['CPU_benchmark'];gpu=parity['GPU_benchmark'];assert gpu['compute_ms']['mean']<100
    effective_gpu=export['GPU_latency_ms'] if config['default_backend']=='onnx' else gpu['compute_ms']
    assert effective_gpu['mean']<100
    family=Path(config['detector']).parent.name;base=f'{family}_{config["imgsz"]}'
    ablation=[{'experiment':'BEST-CLASSICAL-E','full_match':21,'type1':8,'type1a':13,'geometry':'bbox','routing':'geometry','stitch':True}]
    names=['F02_transfer_1024',base+'_bbox',base+'_bbox_none',base+'_no_stitch',base,name]
    for experiment in dict.fromkeys(names):
        path=REPORT/'experiments'/experiment/'result.json'
        if path.is_file():
            item=read(path);s=item['public_development'];cfg=item['config']
            ablation.append({'experiment':experiment,'full_match':s['full_match_total'],
                'type1':s['per_type']['type1']['full_match'],'type1a':s['per_type']['type1a']['full_match'],
                'geometry':cfg['geometry'],'routing':cfg['routing'],'stitch':cfg.get('stitch',True),
                'filter_override':cfg.get('filter_override'),'weights_sha256':cfg['weights_sha256']})
    result={'BEST_F':config,'selection_lock':lock,'baseline_E':{'correct':21,'total':115,'type1_correct':8,'type1a_correct':13,
            'historical_GPU_ms':38,'historical_CPU_ms':156,'note':'Historical E timings are not substituted for this serial F protocol'},
        'selected_development':best['public_development'],'common_development_slices':best['development_slices'],
        'dataset':dataset,'detector_baseline':read(REPORT/'detector_baseline.json'),'oracle_quad':oracle,
        'tournament':ranking['rows'],'end_to_end_ablation':ablation,'training':training,'remaining_failures':attribution,
        'size_analysis':read(REPORT/'size_analysis.json'),'export':export,'CPU_GPU_parity':parity,
        'initialization_audit':read(REPORT/'initialization_audit.json'),
        'guard_revalidation':read(REPORT/'guard_revalidation.json'),
        'final_synthetic_test':test,'tests':{'full_passed':int(suite.get('tests'))-int(suite.get('skipped')),
            'legacy_xfail':int(suite.get('skipped')),'focused_passed':int(focused.get('tests'))},
        'training_image_repair_audit':read(REPORT/'training_image_repair_audit.json'),
        'integrity':{'E_artifacts_unchanged':len(frozen['hashes']),'v2_dataset_files_unchanged':dataset_integrity['byte_unchanged_files'],
            'dataset_integrity':dataset_integrity,
            'fixed_OCR_sha256':config['ocr_sha256'],'detector_sha256':config['detector_sha256']},
        'independent_real_test':'UNAVAILABLE','generalization':'GENERALIZATION NOT YET VERIFIED',
        'YOLO11s':'NOT RUN: high synthetic validation fit and scarce real labels do not establish n-capacity bottleneck; no unmotivated size sweep',
        'complete_pipeline_GPU_ms':effective_gpu,'complete_pipeline_backend':config['default_backend'],
        'next_bottleneck':attribution['next_bottleneck'],'OCR_trained':False}
    write(REPORT/'final_comparison.json',result)
    s=best['public_development'];u=best['development_slices']['untrained_exposed_development']
    from tools.report_phase_f import build_report
    lines=build_report(result,best,ranking)
    (ROOT/'docs/phase_f_report.md').write_text('\n'.join(lines),encoding='utf-8')
    experiments=ROOT/'docs/experiments.md';text=experiments.read_text(encoding='utf-8');heading='## EXP-F — Detector and quad geometry'
    if heading not in text:
        text+=f'\n\n{heading}\n\nDate: 2026-10-03. Frozen E/D; real + procedural scene detection. Compared old/fine-tuned/clean YOLO11n, OBB, four-corner pose and hard iteration; no OCR training. Oracle: raw 13→18/31 justified neural geometry. Final {name}: development {s["full_match_total"]}/115, type1 {s["per_type"]["type1"]["full_match"]}/15, type1a {s["per_type"]["type1a"]["full_match"]}/100; common untrained-exposed {u["full_match_total"]}/87 vs E 11/87. Real holdout unavailable: GENERALIZATION NOT YET VERIFIED. Backend {config["default_backend"]}; complete GPU mean/p95 {effective_gpu["mean"]:.2f}/{effective_gpu["p95"]:.2f} ms, CPU {cpu["compute_ms"]["mean"]:.2f} ms. ONNX: {export["decision"]}; CPU/GPU: {parity["status"]}. Evidence: docs/phase_f_report.md, reports/phase_f/final_comparison.json, model_tournament.csv, selection_lock.json.\n'
        experiments.write_text(text,encoding='utf-8')
    write(REPORT/'workflow_status.json',{'stage':'phase_f','state':'COMPLETE','independent_real_test':'UNAVAILABLE','generalization':'GENERALIZATION NOT YET VERIFIED'})
    print('Final report published',name,s['full_match_total'],'/',115,flush=True)

if __name__=='__main__':main()
