"""Export authored scenes to the organizer CSV without fabricating negatives."""
import argparse,csv,json,math,shutil
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from PIL import Image
from dataset.schema import FIELDS,read_metadata,resolve_image,sha256
from src.geometry_f import validate_quad

TYPES=('type1','type1a','type1b')


def package_exporter(stage):
    project=Path(__file__).resolve().parents[1];hashes={}
    for name in ('tools/export_organizer_synthetic.py','tools/check_organizer_synthetic.py','tools/reproduce_organizer_synthetic.py',
                 'dataset/__init__.py','dataset/schema.py','src/__init__.py','src/geometry_f.py','src/plate_formats.py'):
        target=stage/'generator/exporter_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(project/name,target);hashes[target.relative_to(stage).as_posix()]=sha256(target)
    target=stage/'generator/reproduce.py';shutil.copy2(project/'tools/reproduce_organizer_synthetic.py',target)
    hashes[target.relative_to(stage).as_posix()]=sha256(target)
    (stage/'generator/requirements.txt').write_text('numpy==2.2.3\nopencv-python==5.0.0.93\nPillow==12.0.0\n',encoding='utf-8')
    return hashes


def integer_geometry(annotation,width,height):
    if annotation.quad is None or annotation.bbox is None:raise ValueError('Actual authored quad required')
    original=validate_quad(annotation.quad,(height,width))
    corners=[[round(float(x)),round(float(y))] for x,y in original]
    validate_quad(corners,(height,width))
    x,y,w,h=annotation.bbox
    left=math.floor(min(x,min(p[0] for p in corners)));top=math.floor(min(y,min(p[1] for p in corners)))
    right=math.ceil(max(x+w,max(p[0] for p in corners)));bottom=math.ceil(max(y+h,max(p[1] for p in corners)))
    if not 0<=left<right<=width or not 0<=top<bottom<=height:raise ValueError('Quantized geometry leaves the image')
    return [left,top,right-left,bottom-top],corners


def export(collections,output,*,exclude_invalid_regions=False):
    roots=[Path(p).resolve() for p in collections];output=Path(output).resolve()
    if output.exists():raise ValueError('Output exists; published exports are preserved')
    if not roots or len(roots)!=len(set(roots)):raise ValueError('Distinct source collections required')
    if any(output.is_relative_to(root) or root.is_relative_to(output) for root in roots):raise ValueError('Output overlaps input collection')
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='organizer-export-',dir=output.parent) as temporary:
        temporary=Path(temporary).resolve()
        if not temporary.is_relative_to(output.parent):raise ValueError('Staging escaped export parent')
        stage=temporary/'dataset';(stage/'images/synthetic').mkdir(parents=True);(stage/'images/real').mkdir()
        for directory in ('labels','provenance','generator','auxiliary/negative_images','auxiliary/negative_labels','auxiliary/invalid_region_images'):(stage/directory).mkdir(parents=True)
        rows=[];provenance=[];negatives=[];invalid_regions=[];parts=[];max_error=0.
        for part,root in enumerate(roots):
            manifest_path=root/'manifest.json';manifest_hash=sha256(manifest_path)
            manifest=json.loads(manifest_path.read_text('utf-8'))
            if manifest.get('is_synthetic') is not True or manifest.get('real_images')!=0:raise ValueError('Only authored synthetic scene collections are supported')
            if sha256(root/'meta.csv')!=manifest['metadata_sha256']:raise ValueError('Canonical metadata changed')
            for name,value in manifest['source_sha256'].items():
                if sha256(resolve_image(root/'generator/source',name))!=value:raise ValueError('Packaged generator source changed')
            # Legacy zero-region rows must reach the mandatory reject/quarantine
            # branch below without changing their original pixels or labels.
            annotations,issues=read_metadata(root/'meta.csv',format_profile='russian_v1',allow_legacy_zero_region=True)
            if issues:raise ValueError('Invalid canonical metadata: '+str(issues[:2]))
            by_image={r['image']:r for r in manifest['records']}
            if len(by_image)!=len(manifest['records']):raise ValueError('Duplicate source image record')
            objects={}
            for annotation in annotations:objects.setdefault(annotation.image,[]).append(annotation)
            if set(objects)!=set(by_image):raise ValueError('Canonical metadata and image inventory disagree')
            prefix=f'p{part:02d}_';renamed={}
            for image,record in by_image.items():
                source=resolve_image(root,image)
                if sha256(source)!=record['sha256']:raise ValueError('Source scene bytes changed')
                values=objects[image];name=prefix+Path(image).name
                if all(a.annotation_kind=='negative' for a in values):
                    if record['annotations']:raise ValueError('Contradictory negative record')
                    target=stage/'auxiliary/negative_images'/name;shutil.copy2(source,target)
                    (stage/'auxiliary/negative_labels'/Path(name).with_suffix('.txt')).write_text('',encoding='utf-8')
                    negatives.append({'image':target.relative_to(stage).as_posix(),'source_part':part,'source_image':image,
                                      'sha256':record['sha256'],'negative_kind':record['negative_kind'],'reason':'No plate object exists; no bbox or quad fabricated'})
                    continue
                if any(a.annotation_kind!='full' or not a.is_synthetic or a.is_vehicle is not True or a.plate_type not in TYPES for a in values):
                    raise ValueError('Full authored vehicle plate annotations required')
                invalid=[a for a in values if set(a.plate_num[5 if a.plate_type=='type1b' else 6:])=={'0'}]
                if invalid:
                    if not exclude_invalid_regions:raise ValueError('Region00/000 rejected by organizer; source text must not be repaired')
                    target=stage/'auxiliary/invalid_region_images'/name;shutil.copy2(source,target)
                    invalid_regions.append({'image':target.relative_to(stage).as_posix(),'source_part':part,'source_image':image,
                        'sha256':record['sha256'],'original_annotations':record['annotations'],'reason':'Region00/000; entire frame excluded without relabeling'})
                    continue
                if len(values)!=len(record['annotations']):raise ValueError('Canonical object count and raster proof disagree')
                for annotation,proof in zip(values,record['annotations']):
                    if annotation.plate_num!=proof['plate_num'] or annotation.plate_type!=proof['plate_type'] or list(map(list,annotation.quad or ()))!=proof['quad']:
                        raise ValueError('Canonical metadata disagrees with authored raster proof')
                target=stage/'images/synthetic'/name;shutil.copy2(source,target);renamed[image]=target.relative_to(stage).as_posix()
                with Image.open(source) as decoded:width,height=decoded.size
                labels=[]
                for index,annotation in enumerate(values):
                    box,quad=integer_geometry(annotation,width,height);x,y,w,h=box
                    conditions='angle' if abs(float(record['annotations'][index]['angle']))>1 else ''
                    rows.append({'image':renamed[image],'plate_num':annotation.plate_num,'plate_type':annotation.plate_type,
                        'bbox':','.join(map(str,box)),'quad':','.join(str(v) for point in quad for v in point),
                        'is_vehicle':'1','is_synthetic':'1','source':annotation.source,'license':annotation.license,'conditions':conditions})
                    normalized=[(x+w/2)/width,(y+h/2)/height,w/width,h/height]+[v/d for point in quad for v,d in zip(point,(width,height))]
                    labels.append(str(TYPES.index(annotation.plate_type))+' '+' '.join(f'{v:.10f}' for v in normalized))
                    error=max(abs(float(old)-new) for before,after in zip(annotation.quad,quad) for old,new in zip(before,after))
                    max_error=max(max_error,error)
                    provenance.append({'image':renamed[image],'object_index':index,'source_part':part,'source_image':image,
                        'source_image_sha256':record['sha256'],'source_group_id':annotation.group_id,
                        'exact_float_bbox':list(annotation.bbox),'exact_float_quad':[list(p) for p in annotation.quad],
                        'integer_bbox':box,'integer_quad':quad,'max_corner_coordinate_error_px':error,
                        'original_conditions':list(annotation.conditions),'split':'unassigned'})
                (stage/'labels'/Path(name).with_suffix('.txt')).write_text('\n'.join(labels)+'\n',encoding='utf-8')
            (stage/'provenance'/f'part{part:02d}.manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
            shutil.copy2(root/'meta.csv',stage/'provenance'/f'part{part:02d}.canonical.csv')
            shutil.copytree(root/'generator',stage/'generator'/f'part{part:02d}',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
            if sha256(manifest_path)!=manifest_hash or sha256(root/'meta.csv')!=manifest['metadata_sha256']:
                raise ValueError('Source state changed during export')
            parts.append({'root':str(root),'manifest_sha256':manifest_hash,'seed':manifest['seed'],'count':manifest['count'],
                'positive_images':len(renamed),'negative_auxiliary_images':sum(r['source_part']==part for r in negatives)})
        if not rows:raise ValueError('No plate objects to export')
        with (stage/'meta.csv').open('w',encoding='utf-8',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=FIELDS,delimiter=';',lineterminator='\n');writer.writeheader();writer.writerows(rows)
        (stage/'provenance/objects.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in provenance),encoding='utf-8')
        (stage/'auxiliary/negative_manifest.json').write_text(json.dumps({'records':negatives,'is_synthetic':True,'not_counted_as_official_object_images':True},ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'auxiliary/invalid_region_manifest.json').write_text(json.dumps({'records':invalid_regions,'is_synthetic':True,'original_GT_preserved':True},ensure_ascii=False,indent=2),encoding='utf-8')
        info={'scope':'Synthetic-only organizer-format export, not final combined dataset','parts':parts,
              'official_synthetic_images':len({r['image'] for r in rows}),'plate_objects':len(rows),
              'auxiliary_objectless_negatives':len(negatives),'max_corner_coordinate_error_px':max_error,
              'excluded_invalid_region_frames':len(invalid_regions),
              'quantization':'Nearest integer corners; enclosing integer bbox; exact floats preserved in provenance',
              'source_backgrounds':'Authored procedural graphics only','split':'unassigned','final_split_created':False,'training_started':False,
              'real_images':0,'meta_sha256':sha256(stage/'meta.csv'),'exporter_sha256':sha256(Path(__file__))}
        info['exporter_source_sha256']=package_exporter(stage)
        (stage/'export_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'README.md').write_text('# Синтетический набор: экспорт для валидатора организаторов\n\n'
            'Источники: собственные процедурные сцены Volga ANPR. Лицензия: CC BY 4.0; авторство глифов - Volga ANPR procedural generator. '
            'Чужие фотографии, реальные снимки людей и public debug не включены. Это стилизованный домен, реальное качество не доказано.\n\n'
            f"В images/synthetic: {info['official_synthetic_images']} кадров с объектами, {len(rows)} номеров. "
            f'Отдельно auxiliary: {len(negatives)} кадров без номера с пустыми labels; они не размечены фиктивной рамкой other и не увеличивают официальный счётчик объектов.\n'
            f'Кадры с неподдерживаемым регионом00/000: {len(invalid_regions)}, сохранены в auxiliary без исправления нарисованного текста; из основной разметки исключён весь кадр.\n'
            'CSV содержит ровно десять обязательных полей; целые bbox и quad записаны через запятую без JSON-скобок. '
            'Labels: YOLO class cx cy w h плюс восемь нормированных координат quad; классы0/1/2 соответствуют type1/type1a/type1b. '
            'Точная исходная геометрия и все дополнительные условия сохранены в provenance. Погрешность округления каждой координаты угла не более0.5px.\n\n'
            'Generator содержит исходники и зависимости каждой части с seed и оригинальным README; воспроизведение выполняется в новый каталог, затем тем же exporter. '
            'Полный переносимый сценарий: python -I generator/reproduce.py --workspace NEW_WORKSPACE; он включает обе стадии и проверяет совпадение метаданных. '
            'Окончательный train/val/test split не сделан, обучение не запускалось. Реальные минимумы остаются невыполненными.\n',encoding='utf-8')
        (stage/'LICENSE').write_text('Authored procedural images and glyphs: Volga ANPR procedural generator. CC BY 4.0.\nhttps://creativecommons.org/licenses/by/4.0/\n',encoding='utf-8')
        stage.rename(output)
    return info


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--collection',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--exclude-invalid-regions',action='store_true');args=p.parse_args()
    result=export(args.collection,args.output,exclude_invalid_regions=args.exclude_invalid_regions)
    print(json.dumps({k:result[k] for k in ('official_synthetic_images','plate_objects','auxiliary_objectless_negatives','max_corner_coordinate_error_px')}))


if __name__=='__main__':main()
