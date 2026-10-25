"""Export authored synthetic data and existing human geometry, never fill missing GT."""
from pathlib import Path
import argparse
from collections import Counter
import csv
import json
import shutil
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cv2
import numpy as np
from src.inference_io import file_sha256
from src.geometry_f import validate_quad
from tools.phase_i_review import sha_json

FIELDS = ('image','plate_num','plate_type','bbox','quad','is_vehicle','is_synthetic','source','license','conditions')
CLASSES = {'type1':0,'type1a':1,'type1b':2,'other':3}
SPLITS = {
    'train': ROOT.parent/'datasets/reports/data3_2026_10_16/proposal_v4_holdout_reserved/train.csv',
    'val': ROOT.parent/'datasets/reports/data3_2026_10_16/proposal_v4_holdout_reserved/val.csv',
    'test': ROOT.parent/'datasets/reports/data3_2026_10_18/frozen_v1/untouched_test.csv',
}


def read_csv(path, delimiter=','):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream, delimiter=delimiter))


def write_csv(path, fields, rows, delimiter=';'):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=delimiter, lineterminator='\n', extrasaction='ignore')
        writer.writeheader(); writer.writerows(rows)


def license_text(value):
    return {'CC BY 4.0':'CC-BY-4.0','CC0':'CC0-1.0'}.get(value, value)


def dataset_readme(info):
    real=info['real_images_by_type']
    unique=info.get('real_unique_plates_by_type',{})
    return f'''# Volga ANPR - датасет для сдачи

## Источники и лицензии

Собственная процедурная синтетика и реальные фотографии из Wikimedia Commons,
Roboflow (rf_sergai_rare_v1, rf_artem_russian, roboflow_two_line) и Kaggle russian_five_classes.
Авторы, исходные URL, лицензии CC BY 4.0 / CC0 / Public Domain и статус разрешения Kaggle
сохранены в provenance/attribution.csv и meta.csv. Авторское разрешение Kaggle заявлено
пользователем; документ будет добавлен позднее в permissions/. Проверенным он пока не считается.
Условия исходных фотографий не заменяются автоматически лицензией собственных аннотаций.

## Методика и статистика

{info['official_synthetic_images']} синтетических положительных сцен и{info['official_real_images']} реальных кропов.
Всего{info['plate_objects']} объектов. Реальные: type1a {real.get('type1a',0)}, type1b {real.get('type1b',0)}.
Уникальных номеров по типам: {json.dumps(unique,ensure_ascii=False)}.
Сохраняется ранее подтверждённое членство train/val. Углы берутся из существующего ручного решения,
переводятся в координаты точного исходного кропа и округляются до целых пикселей.
Визуальная проверка приватности кропов: provenance/privacy_audit.json.
Исходные фото, решения и членство DATA-3 не изменялись; новые human GT подтверждения не создавались.

Полный bbox-пул:{info['label_only_pool_records']} целевых записей, метаданные
в supplemental/bbox_only_pool.csv. В нём отсутствующие quad не заполнены. Исходные большие
фотографии этого дополнительного пула в пакет не копировались. Рекомендованные минимумы150/300
нужно оценивать раздельно для bbox-пула и официального subset с подтверждёнными углами.
Недостающий объём реальных other нельзя заменить синтетическими отрицательными кадрами.

834 синтетических кадра без объектов находятся в auxiliary/, с пустыми labels;
они не считаются реальными other.34 исторических кадра с недопустимым регионом хранятся
в quarantine auxiliary без исправления номера. Официальная разметка meta.csv/labels совпадает.

## Генератор и проверка

`python -I generator/reproduce.py --workspace NEW_DIR` повторяет синтетическую часть.
Seeds исходных партий:20261015,20261016,20261017. Источники генератора и зависимости включены.
Реальные чужие фотографии не генерируются. Исходный валидатор и отчёт: validation/.
Модель и новое обучение этим экспортом не запускаются. Группы для защиты от leakage: splits/image_groups.csv.

Перед внешней передачей требуется добавить обещанный документ Kaggle и проверить его содержание.
Это локальный кандидат; автоматической публикации не было.
'''


def real_annotation(record, source, decision):
    if not all(decision.get(k) is True for k in ('human_confirmed','quad_verified','training_region_complete')):
        raise ValueError('Real geometry is not explicitly human-confirmed')
    if decision.get('status') != 'accepted' or decision.get('has_other_plates') is not False:
        raise ValueError('Real region is rejected or incomplete')
    if any(record[k] != decision[k] for k in ('review_id','plate_num','plate_type','split','source_sha256')):
        raise ValueError('Human decision does not bind the selected source label')
    if source['source_sha256'] != record['source_sha256']:
        raise ValueError('Original image identity changed')
    x,y,right,bottom = record['training_region']
    if decision['training_region'] != [x,y,right,bottom]:
        raise ValueError('Confirmed region changed')
    q = np.asarray(decision['quad'], dtype=float) - [x,y]
    validate_quad(q, (bottom-y,right-x))
    points = [[round(float(a)),round(float(b))] for a,b in q]
    validate_quad(points, (bottom-y,right-x))
    left,top = min(p[0] for p in points),min(p[1] for p in points)
    end_x,end_y = max(p[0] for p in points),max(p[1] for p in points)
    if not 0 <= left < end_x < right-x or not 0 <= top < end_y < bottom-y:
        raise ValueError('Rounded human geometry leaves the crop')
    box = [left,top,end_x-left,end_y-top]
    return dict(image='images/real/r_'+record['review_id']+'.png',plate_num=record['plate_num'],
        plate_type=record['plate_type'],bbox=','.join(map(str,box)),
        quad=','.join(str(v) for p in points for v in p),is_vehicle='1',is_synthetic='0',
        source=source.get('source_page') or source.get('original_url/path') or source['source'],
        license=license_text(source['license']),conditions=source.get('conditions','')), box, points


def export_dataset(output, synthetic, manifest_path, state_path, index_path, split_paths=None):
    output, synthetic = Path(output).resolve(), Path(synthetic).resolve()
    manifest_path,state_path,index_path = map(Path,(manifest_path,state_path,index_path))
    if output.exists() or output.is_relative_to(synthetic) or synthetic.is_relative_to(output):
        raise ValueError('Output must be a new directory outside the source dataset')
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    state = json.loads(state_path.read_text(encoding='utf-8'))
    if file_sha256(index_path) != manifest['index_sha256']:
        raise ValueError('Immutable annotation index changed')
    all_rows=[]
    for split,path in (split_paths or SPLITS).items():
        all_rows.extend(dict(r, frozen_split=split) for r in read_csv(path))
    sources={r['review_id']:r for r in all_rows}
    selected=[r for r in manifest['records'] if r['domain']=='human_real' and r['variant']=='real_only' and r.get('repeat',0)==0]
    synthetic_info=json.loads((synthetic/'export_manifest.json').read_text(encoding='utf-8'))
    if file_sha256(synthetic/'meta.csv') != synthetic_info['meta_sha256']:
        raise ValueError('Synthetic metadata changed')
    synthetic_rows=read_csv(synthetic/'meta.csv',';')
    by_split={k:{r['plate_num'] for r in all_rows if r['frozen_split']==k} for k in ('val','test')}
    if any(r['plate_num'] in by_split['val']|by_split['test'] for r in synthetic_rows):
        raise ValueError('Synthetic replay collides with a reserved real validation/test plate')
    output.parent.mkdir(parents=True,exist_ok=True)
    with TemporaryDirectory(prefix='final-data-',dir=output.parent) as temporary:
        stage=Path(temporary)/'dataset';shutil.copytree(synthetic,stage)
        provenance=stage/'provenance'
        shutil.copy2(stage/'meta.csv',provenance/'synthetic_meta.csv')
        shutil.copy2(stage/'export_manifest.json',provenance/'synthetic_export_manifest.json')
        (stage/'images/real').mkdir(exist_ok=True)
        rows=list(synthetic_rows);records=[];attributions=[];pending=[];groups=[]
        for r in synthetic_rows:
            groups.append(dict(image=r['image'],split='train',domain='authored_synthetic',
                               plate_num=r['plate_num'],duplicate_group_id=r['image'],original_image_group_id=r['image'],source_family='authored_procedural'))
        for r in selected:
            ident=r['review_id'];decision=state['decisions'][ident];source=sources[ident]
            if sha_json(decision) != manifest['selected_decision_sha256'][ident]:
                raise ValueError('A selected human decision changed: '+ident)
            for path,key in [('export_image','export_sha256'),('export_label','label_sha256')]:
                if file_sha256(Path(r[path])) != r[key]:raise ValueError('Selected crop/label changed')
            if file_sha256(Path(r['source_image'])) != r['source_sha256']:
                raise ValueError('Source photograph changed')
            annotation,box,points=real_annotation(r,source,decision)
            image=stage/annotation['image'];shutil.copy2(r['export_image'],image)
            raw=cv2.imdecode(np.fromfile(r['source_image'],dtype=np.uint8),cv2.IMREAD_COLOR)
            crop=cv2.imdecode(np.fromfile(image,dtype=np.uint8),cv2.IMREAD_COLOR)
            x,y,x2,y2=r['training_region']
            if raw is None or crop is None or not np.array_equal(raw[y:y2,x:x2],crop):
                raise ValueError('Exported real crop is not the approved source region')
            h,w=crop.shape[:2];left,top,bw,bh=box
            label=[CLASSES[r['plate_type']],(left+bw/2)/w,(top+bh/2)/h,bw/w,bh/h,
                   *[v/(w if i%2==0 else h) for i,v in enumerate(v for p in points for v in p)]]
            (stage/'labels'/image.with_suffix('.txt').name).write_text(' '.join(map(str,label))+'\n',encoding='ascii')
            rows.append(annotation)
            evidence_pending=source.get('source')=='kaggle_russian_five_classes'
            if evidence_pending:pending.append(ident)
            record=dict(review_id=ident,image=annotation['image'],sha256=file_sha256(image),source_sha256=r['source_sha256'],
                source_region=r['training_region'],human_decision_sha256=sha_json(decision),
                quad_source='existing explicit human decision; translated into source crop',quad_verified=True,
                human_confirmation_created=False,source_license_verified=source.get('source_license_verified')=='True',
                permission_evidence_status='USER_ATTESTED_DOCUMENT_DEFERRED' if evidence_pending else 'SOURCE_LICENSE_RECORDED',
                source_plate_num=r['plate_num'],source_plate_type=r['plate_type'],split=r['split'])
            records.append(record)
            groups.append(dict(image=annotation['image'],split=r['split'],domain='human_real',plate_num=r['plate_num'],
                duplicate_group_id=r['duplicate_group_id'],original_image_group_id=r['original_image_group_id'],source_family=r['source_family']))
            attributions.append(dict(image=annotation['image'],source=annotation['source'],author=source.get('author',''),
                license=source['license'],license_url=source.get('license_url',''),
                permission_evidence_status=record['permission_evidence_status'],modification='Crop to existing human-confirmed region; original pixels retained'))
        write_csv(stage/'meta.csv',FIELDS,rows)
        write_csv(provenance/'attribution.csv',('image','source','author','license','license_url','permission_evidence_status','modification'),attributions)
        write_csv(stage/'splits/image_groups.csv',('image','split','domain','plate_num','duplicate_group_id','original_image_group_id','source_family'),groups)
        supplemental=[]
        selected_ids={r['review_id'] for r in selected}
        for r in all_rows:
            supplemental.append(dict(r,quad_verified='False',official_geometry_ready='True' if r['review_id'] in selected_ids else 'False',
                original_image_path=r['image'],note='Original target-level record; missing quad is not filled; not additional human confirmation'))
        fields=tuple(dict.fromkeys(k for r in supplemental for k in r))
        write_csv(stage/'supplemental/bbox_only_pool.csv',fields,supplemental)
        (provenance/'human_geometry.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
        shutil.copy2(ROOT/'tools/reproduce_organizer_synthetic.py',stage/'generator/reproduce.py')
        info=dict(scope='Combined organizer dataset using existing human geometry and authored synthetic scenes',
            synthetic_export=synthetic_info,official_synthetic_images=synthetic_info['official_synthetic_images'],
            plate_objects=len(rows),official_real_images=len(selected),
            real_images_by_type=dict(Counter(r['plate_type'] for r in selected)),
            real_unique_plates_by_type={kind:len({r['plate_num'] for r in selected if r['plate_type']==kind})
                                       for kind in sorted({r['plate_type'] for r in selected})},
            label_only_pool_records=len(all_rows),label_only_pool_types=dict(Counter(r['plate_type'] for r in all_rows)),
            supplemental_is_not_official_quad_GT=True,missing_geometry_not_filled=True,
            meta_sha256=file_sha256(stage/'meta.csv'),source_geometry_manifest_sha256=file_sha256(manifest_path),
            permission_document_pending_ids=pending,publication_ready=not pending,
            training_started=False,model_or_split_changed=False,new_human_GT_created=False,
            privacy='Only pre-existing cropped plate regions; original full frames excluded; separate visual privacy audit required')
        (stage/'export_manifest.json').write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding='utf-8')
        (stage/'permissions').mkdir()
        (stage/'permissions/status.json').write_text(json.dumps(dict(status='USER_ATTESTED_DOCUMENT_DEFERRED',
            document=None,document_verified=False,affected_review_ids=pending,
            required_scope='Transmission to organizers and CC BY4.0-compatible publication',
            user_statement='User says permission exists and will add the document later; no source license is rewritten'),indent=2),encoding='utf-8')
        (stage/'permissions/README.md').write_text('Permission evidence is pending. Add the author permission and bind its SHA-256 in status.json. Do not mark document_verified without inspecting the actual terms. No external publication has been performed.\n',encoding='utf-8')
        (stage/'LICENSE').write_text('Authored synthetic images, generator assets and our annotations: Creative Commons Attribution 4.0 International, https://creativecommons.org/licenses/by/4.0/ . Original real photographs retain the conditions recorded in meta.csv and provenance/attribution.csv. Some permissions are user-attested with a deferred document; see permissions/status.json. This private draft does not assert that those rights have been independently verified.\n',encoding='utf-8')
        (stage/'README.md').write_text(dataset_readme(info),encoding='utf-8')
        stage.rename(output)
    return info


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--synthetic',type=Path,default=ROOT.parent/'generated_data/organizer_synthetic_5000_v3_20261015')
    args=parser.parse_args(argv)
    info=export_dataset(args.output,args.synthetic,ROOT/'reports/phase_i/balanced_20261019/dataset_manifest.json',
        ROOT/'reports/phase_i/geometry_review_state.json',ROOT/'reports/phase_i/annotation_index.json')
    print(json.dumps({k:v for k,v in info.items() if k!='synthetic_export'},ensure_ascii=True,indent=2))


if __name__=='__main__':main()
