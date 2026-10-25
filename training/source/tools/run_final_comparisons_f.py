"""A small, prespecified routing/geometry comparison after verified training."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_phase_f_models import call,REPORT,write
from tools.phase_e import read

def result(name):return read(REPORT/'experiments'/name/'result.json')

def evaluate(name,weights,size,geometry='bbox',routing='geometry',extras=()):
    call('evaluate_'+name,['tools/evaluate_detector_f.py','--name',name,'--weights',weights,
        '--imgsz',str(size),'--geometry',geometry,'--routing',routing,*extras])

def best_size(family):
    choices=[result(f'{family}_{n}') for n in (640,1024)]
    return max(choices,key=lambda r:(r['development_slices']['untrained_exposed_development']['full_match_total'],
        -r['GPU_latency_ms']['mean']))['config']['imgsz']

def main():
    # Genuine transfer resolution control, unlike retained invalid-init v1 probes.
    evaluate('F02_transfer_1280','models/phase_f/F02_transfer/best.pt',1280)
    specifications=[('F02_transfer','bbox'),('F02_coco','bbox'),('F03_obb_transfer','obb'),
                    ('F04_pose_transfer','quad'),('F02_hard_verified','bbox')]
    for family,geometry in specifications:
        size=best_size(family)
        if family=='F02_transfer':
            sizes=[result(f'{family}_{s}') for s in (640,1024,1280)]
            size=max(sizes,key=lambda r:r['development_slices']['untrained_exposed_development']['full_match_total'])['config']['imgsz']
        for routing in ('detector','combined'):
            evaluate(f'{family}_{size}_{routing}',f'models/phase_f/{family}/best.pt',size,geometry,routing)
    # Hold the model and observed type signal fixed to isolate the image transform.
    for family,geometry in [('F03_obb_transfer','obb'),('F04_pose_transfer','quad')]:
        size=best_size(family);weight=f'models/phase_f/{family}/best.pt'
        evaluate(f'{family}_{size}_bbox',weight,size,'bbox')
        evaluate(f'{family}_{size}_no_stitch',weight,size,geometry,extras=('--no-stitch',))
        evaluate(f'{family}_{size}_bbox_none',weight,size,'bbox',extras=('--filter','none'))
    call('final_ranking',['tools/summarize_tournament_f.py'])
    write(REPORT/'workflow_status.json',{'stage':'routing_and_geometry_comparisons','state':'COMPLETE',
        'remaining':['Selected detector serial benchmark/export','Lock','CPU/GPU parity','Synthetic post-lock test','Manual input review','Final reports']})

if __name__=='__main__':main()
