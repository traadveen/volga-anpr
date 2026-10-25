"""Verify copied bytes and exact-to-integer provenance in a portable export."""
import argparse,csv,json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dataset.schema import FIELDS,resolve_image,sha256


def check(root):
    root=Path(root).resolve();info=json.loads((root/'export_manifest.json').read_text('utf-8'))
    if info.get('real_images')!=0:raise ValueError('Synthetic-only export required')
    for name,digest in info.get('exporter_source_sha256',{}).items():
        if sha256(resolve_image(root,name))!=digest:raise ValueError('Packaged exporter source changed')
    if sha256(root/'meta.csv')!=info['meta_sha256']:raise ValueError('Exported metadata bytes changed')
    with (root/'meta.csv').open(encoding='utf-8',newline='') as stream:
        reader=csv.DictReader(stream,delimiter=';')
        if tuple(reader.fieldnames)!=FIELDS:raise ValueError('Wrong organizer header')
        rows=list(reader)
    if any(row['is_synthetic']!='1' or row['is_vehicle']!='1' for row in rows):raise ValueError('Unexpected non-synthetic or non-vehicle object')
    proof_rows=[json.loads(line) for line in (root/'provenance/objects.jsonl').read_text('utf-8').splitlines()]
    if len(rows)!=len(proof_rows):raise ValueError('Object provenance count differs from metadata')
    parts=[]
    for index,part in enumerate(info['parts']):
        path=root/f'provenance/part{index:02d}.manifest.json'
        if sha256(path)!=part['manifest_sha256']:raise ValueError('Source manifest snapshot changed')
        value=json.loads(path.read_text('utf-8'));parts.append({r['image']:r for r in value['records']})
        generator=root/f'generator/part{index:02d}'
        for name,digest in value['source_sha256'].items():
            if sha256(resolve_image(generator/'source',name))!=digest:raise ValueError('Packaged generator or glyph changed')
    seen=set();max_error=0.
    for row,proof in zip(rows,proof_rows):
        if row['image']!=proof['image']:raise ValueError('Object row order or image binding changed')
        source=parts[proof['source_part']][proof['source_image']]
        obj=source['annotations'][proof['object_index']]
        if row['plate_num']!=obj['plate_num'] or row['plate_type']!=obj['plate_type']:
            raise ValueError('Exported text/type differs from actual authored source')
        box=list(map(int,row['bbox'].split(',')));flat=list(map(int,row['quad'].split(',')))
        quad=[flat[i:i+2] for i in range(0,8,2)]
        if box!=proof['integer_bbox'] or quad!=proof['integer_quad']:raise ValueError('Integer geometry differs from provenance')
        if proof['exact_float_quad']!=obj['quad']:raise ValueError('Exact source corners changed')
        error=max(abs(float(before)-after) for old,new in zip(obj['quad'],quad) for before,after in zip(old,new))
        if error>.50000001:raise ValueError('Corner rounding error exceeds0.5px')
        max_error=max(max_error,error)
        x,y,w,h=box
        if any(not x<=px<=x+w or not y<=py<=y+h for px,py in obj['quad']):raise ValueError('Integer box does not enclose source surface')
        if row['image'] not in seen:
            if sha256(resolve_image(root,row['image']))!=source['sha256']:raise ValueError('Copied image bytes differ from source')
            seen.add(row['image'])
        canonical=root/f"generator/part{proof['source_part']:02d}"/Path(obj['canonical']).relative_to('generator')
        if sha256(canonical)!=obj['canonical_sha256']:raise ValueError('Canonical raster proof changed')
    for name in ('negative_manifest.json','invalid_region_manifest.json'):
        for record in json.loads((root/'auxiliary'/name).read_text('utf-8'))['records']:
            source=parts[record['source_part']][record['source_image']]
            if sha256(resolve_image(root,record['image']))!=source['sha256']:raise ValueError('Auxiliary image bytes changed')
    if len(seen)!=info['official_synthetic_images'] or len(rows)!=info['plate_objects']:raise ValueError('Exported counts differ from manifest')
    return {'verified_copied_images':len(seen),'verified_objects':len(rows),'max_corner_coordinate_error_px':max_error,
            'source_text_and_type_preserved':True,'exact_float_geometry_preserved':True,'source_generator_and_canonical_hashes_verified':True,
            'real_images':0,'final_split_created':False,'training_started':False}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
    args=p.parse_args()
    if args.report.resolve().is_relative_to(args.dataset.resolve()):p.error('Check report must stay outside the dataset')
    result=check(args.dataset);args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(result,indent=2),encoding='utf-8');print(json.dumps(result))


if __name__=='__main__':main()
