"""Read-only DATA-3 prerequisites and development exposure, without a split."""
from collections import Counter,defaultdict
import argparse,csv,json,math,re
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset.schema import normalize_plate,quad as parse_quad,current_plate_error
from src.inference_io import file_sha256

ROOT=Path(__file__).resolve().parents[1]
TARGETS={'type1a':(150,50),'type1b':(300,100)}
SHA=re.compile('[0-9a-f]{64}')


def yes(value):return value is True or str(value).lower() in ('true','1')


def normalized_number(value):
    return ''.join(c for c in normalize_plate(value or '') if not c.isspace() and c not in '-._')


def manual(row):
    source=row.get('ground_truth_source','').lower()
    return (yes(row.get('verified')) and row.get('review_status') in ('accepted','corrected')
            and bool(row.get('reviewer')) and bool(row.get('reviewed_at'))
            and ('manual' in source or 'human' in source) and not yes(row.get('is_synthetic')))


def components(tokens_by_row):
    parents=list(range(len(tokens_by_row)));keys={}
    def find(i):
        while parents[i]!=i:parents[i]=parents[parents[i]];i=parents[i]
        return i
    for index,tokens in enumerate(tokens_by_row):
        for token in tokens:
            if token in keys:parents[find(index)]=find(keys[token])
            else:keys[token]=index
    return [find(i) for i in range(len(tokens_by_row))]


def has_quad(row):
    try:
        value=row.get('quad');points=parse_quad(value if isinstance(value,str) else json.dumps(value))
        if not points:return False
        bbox=row.get('bbox')
        if bbox:
            bbox=json.loads(bbox) if isinstance(bbox,str) else bbox
            if not isinstance(bbox,(list,tuple)) or len(bbox)!=4 or not all(
                    not isinstance(v,bool) and isinstance(v,(int,float)) and math.isfinite(v) for v in bbox):return False
            if not (0<=bbox[0]<bbox[2] and 0<=bbox[1]<bbox[3]):return False
            if any(not (bbox[0]<=x<=bbox[2] and bbox[1]<=y<=bbox[3]) for x,y in points):return False
        return True
    except (ValueError,TypeError):return False


def audit(rows,exposures,blocked_hashes=()):
    blocked=set(blocked_hashes);issues=[];accepted=[];eligible=[]
    for row in rows:
        if not manual(row):continue
        row={**row,'normalized_plate_num':normalized_number(row.get('plate_num'))}
        if row.get('source_sha256') in blocked or row.get('crop_sha256') in blocked:
            issues.append({'code':'protected_evaluation_source','id':row.get('review_id')});continue
        accepted.append(row)
        for key in ('review_id','duplicate_group_id','original_image_group_id','source_family'):
            if not row.get(key):issues.append({'code':'missing_group_provenance','field':key,'id':row.get('review_id')})
        for key in ('source_sha256','crop_sha256'):
            if not SHA.fullmatch(row.get(key,'')):issues.append({'code':'invalid_byte_hash','field':key,'id':row.get('review_id')})
        if (row.get('plate_type') in ('type1','type1a','type1b') and yes(row.get('ocr_usable'))
                and yes(row.get('geometry_GT_verified')) and yes(row.get('is_vehicle'))):
            if current_plate_error(row['normalized_plate_num'],row.get('plate_type'),partial=False) is None:
                if all(SHA.fullmatch(row.get(k,'')) for k in ('source_sha256','crop_sha256')):eligible.append(row)
            else:issues.append({'code':'invalid_full_OCR_label','id':row.get('review_id')})
    labels=defaultdict(set);seen_ids=Counter(r.get('review_id') for r in accepted)
    for row in accepted:
        label=(row['normalized_plate_num'],row.get('plate_type'))
        labels['review:'+str(row.get('review_id'))].add(label)
        if row.get('source_sha256') and row.get('crop_sha256'):
            labels['object:'+row['source_sha256']+':'+row['crop_sha256']].add(label)
    conflicts={key:sorted(values) for key,values in labels.items() if len(values)>1}
    issues.extend({'code':'duplicate_current_review_id','id':key} for key,count in seen_ids.items() if count>1)
    eligible=[r for r in eligible if 'review:'+r['review_id'] not in conflicts
              and 'object:'+r['source_sha256']+':'+r['crop_sha256'] not in conflicts]
    nodes=[]
    for row in accepted:nodes.append((row,False))
    for row in exposures:
        nodes.append(({**row,'review_id':row.get('id',row.get('review_id')),
                       'normalized_plate_num':normalized_number(row.get('plate_num'))},True))
    tokens_by_row=[]
    for index,(row,_) in enumerate(nodes):
        tokens=[key+':'+row[key] for key in ('review_id','source_sha256','crop_sha256','duplicate_group_id','original_image_group_id') if row.get(key)]
        text=row.get('normalized_plate_num','')
        if row.get('plate_type') in ('type1','type1a','type1b') and any(current_plate_error(text,kind,partial=False) is None for kind in ('type1','type1a','type1b')):tokens.append('plate:'+text)
        tokens_by_row.append(tokens)
    roots=components(tokens_by_row)
    exposed={roots[index] for index,(_,is_exposed) in enumerate(nodes) if is_exposed}
    eligible_ids={r['review_id'] for r in eligible};groups=defaultdict(list)
    for index,(row,source) in enumerate(nodes):
        if not source:groups[roots[index]].append(row)
    group_rows=[]
    for index,(group_id,component) in enumerate(sorted(groups.items(),key=lambda item:sorted(r.get('review_id','') for r in item[1]))):
        held=group_id in exposed
        group_rows.append({'group':index,'hold_from_future_untouched_test':held,
            'ids':sorted(r.get('review_id','') for r in component),
            'complete_plates':sorted({r['normalized_plate_num'] for r in component if r.get('review_id') in eligible_ids}),
            'source_families':sorted({r.get('source_family','') for r in component}),
            'eligible_records':sum(r.get('review_id') in eligible_ids for r in component)})
    targets={}
    for kind,(images,unique) in TARGETS.items():
        subset=[r for r in eligible if r['plate_type']==kind]
        image_keys=[['sha:'+r['source_sha256']]+(['original:'+r['original_image_group_id']] if r.get('original_image_group_id') else []) for r in subset]
        count=len(set(components(image_keys)));plates=len({r['normalized_plate_num'] for r in subset})
        targets[kind]={'verified_real_images':count,'unique_plate_numbers':plates,'target_real_images':images,
            'target_unique_plates':unique,'image_deficit':max(0,images-count),'unique_plate_deficit':max(0,unique-plates),
            'sources':sorted({r.get('source','') for r in subset}),
            'source_OCR_rows':dict(Counter(r.get('source','') for r in subset))}
    met=all(v['image_deficit']==v['unique_plate_deficit']==0 for v in targets.values())
    missing_quad=[r.get('review_id') for r in accepted if not has_quad(r)]
    invalid_quad=[r.get('review_id') for r in accepted if r.get('quad') and not has_quad(r)]
    unconfirmed_quad=[r.get('review_id') for r in accepted if has_quad(r) and not (
        yes(r.get('quad_GT_verified')) and r.get('quad_status') in ('manual','source'))]
    issues.extend({'code':'invalid_stored_quad','id':ident} for ident in invalid_quad)
    return {'scope':'Read-only prerequisites; candidates are not an assigned or validated independent test',
        'targets':targets,'manual_records':len(accepted),'eligible_complete_OCR_records':len(eligible),
        'groups':group_rows,'held_groups':sum(g['hold_from_future_untouched_test'] for g in group_rows),
        'unexposed_eligible_groups':sum(not g['hold_from_future_untouched_test'] and g['eligible_records']>0 for g in group_rows),
        'issues':issues,'plate_type_conflicts':conflicts,'targets_met':met,
        'data3_planning_ready':met and not issues and not conflicts,'independent_test_available':False,
        'privacy_review':'pending: byte/group accounting does not establish face or person clearance',
        'quad_coverage':{'manual_records':len(accepted),'with_stored_quad':len(accepted)-len(missing_quad),
                         'with_confirmed_quad':len(accepted)-len(missing_quad)-len(unconfirmed_quad),
                         'invalid_review_ids':invalid_quad,'unconfirmed_review_ids':unconfirmed_quad,
                         'missing_quad':len(missing_quad),'missing_review_ids':missing_quad,
                         'policy':'Stored corners must pass convex clockwise TL-first geometry; corner GT also requires explicit manual/source confirmation'},
        'organizer_real_export_ready':False,
        'source_family_policy':'recorded for future source holdouts; not merged merely because files share a dataset publisher',
        'mining_exposure_policy':'Frozen proposal mining alone is not counted as a GT comparison; sampling and annotation bias still require review',
        'license_policy':'Facts and attribution retained; uncertainties are warnings, not preflight blockers',
        'source_licenses':dict(Counter(r.get('license','') for r in accepted)),
        'final_split_created':False,'training_started':False,'new_GT_created':0}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--index',type=Path,default=ROOT.parent/'datasets/real_verified/index.csv')
    parser.add_argument('--exposure',type=Path,action='append',required=True)
    parser.add_argument('--report',type=Path,required=True);args=parser.parse_args()
    output=args.report.resolve()
    if not output.is_relative_to((ROOT/'reports').resolve()):parser.error('Preflight report must stay in project reports')
    if output in {args.index.resolve(),*(p.resolve() for p in args.exposure)}:parser.error('Report cannot overwrite inputs')
    before={str(p):file_sha256(p) for p in [args.index,*args.exposure]}
    with args.index.open(encoding='utf-8-sig',newline='') as stream:rows=list(csv.DictReader(stream))
    exposures=[r for path in args.exposure for r in json.loads(path.read_text('utf-8'))['records']]
    from dataset.validation import excluded_hashes
    protected=ROOT.parent/'external_data/organizer_debug/source_manifest.json'
    blocked=excluded_hashes([protected]) if protected.exists() else set()
    report=audit(rows,exposures,blocked)
    for row in rows:
        if manual(row) and row.get('source_sha256') not in blocked and row.get('crop_sha256') not in blocked:
            for path_key,hash_key in [('image','source_sha256'),('crop','crop_sha256')]:
                if file_sha256(Path(row[path_key]))!=row[hash_key]:raise ValueError('Manual record bytes changed: '+row['review_id'])
    if any(file_sha256(Path(p))!=value for p,value in before.items()):raise ValueError('Input metadata changed during preflight')
    report.update(input_sha256=before,manual_image_and_crop_bytes_checked=True,public_debug_images_read=False)
    output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('targets','held_groups','unexposed_eligible_groups','data3_planning_ready','issues')}))


if __name__=='__main__':main()
