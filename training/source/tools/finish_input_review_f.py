"""Reuse human review only when the selected model produced identical actual inputs."""
from collections import Counter
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.phase_e import read,write
from tools.phase_f_setup import REPORT


def main():
    candidate=read(REPORT/'attribution/candidate_review_F03_obb_transfer_640.json')
    selected=read(REPORT/'attribution/remaining_failures.json')
    assert candidate['review_status']=='PASS'
    prior={(r['image'],r['ground_truth']):r for r in candidate['rows']}
    differences=[]
    for row in selected['rows']:
        if not row['spatial_GT_available']:continue
        old=prior[(row['image'],row['ground_truth'])]
        if row['official_success']!=old['official_success']:
            differences.append((row['image'],'official result changed'));continue
        if row.get('input_review')!=old.get('input_review'):
            differences.append((row['image'],'input hash/path changed'));continue
        a=row.get('chosen');b=old.get('chosen')
        if bool(a)!=bool(b) or (a and any(a.get(k)!=b.get(k) for k in ('plate_type','status','final_ocr'))):
            differences.append((row['image'],'processed type/status/text changed'));continue
        row['category']=old['category'];row['manual_review']=old['manual_review']
    assert not differences,('New direct visual review required',differences)
    selected['categories']=dict(Counter(r['category'] for r in selected['rows']))
    selected['review_status']='PASS';selected['next_bottleneck']=candidate['next_bottleneck']
    selected['review_reuse_gate']='Every spatial-GT crop/normalized tensor SHA and processed type/status/text is identical to the visually reviewed candidate; no extrapolation to changed inputs.'
    write(REPORT/'attribution/remaining_failures.json',selected)
    print('Actual-input visual review PASS',selected['categories'],flush=True)


if __name__=='__main__':main()
