"""Rebuild an official export from its packaged generator sources only."""
import argparse,json,subprocess,sys
from pathlib import Path


def reproduce(dataset,workspace):
    dataset=Path(dataset).resolve();workspace=Path(workspace).resolve()
    if workspace.exists():raise ValueError('Workspace exists; existing results are preserved')
    if workspace.is_relative_to(dataset) or dataset.is_relative_to(workspace):raise ValueError('Workspace overlaps source export')
    exported=json.loads((dataset/'export_manifest.json').read_text('utf-8'))
    # A combined delivery keeps the original synthetic recipe as provenance.
    info=exported.get('synthetic_export',exported);workspace.mkdir(parents=True)
    parts=[]
    def run(script,*args):
        subprocess.run([sys.executable,'-I','-B','-X','utf8',str(script),*map(str,args)],check=True)
    for index,_ in enumerate(info['parts']):
        source=json.loads((dataset/f'provenance/part{index:02d}.manifest.json').read_text('utf-8'))
        target=workspace/f'part{index:02d}';parts.append(target)
        args=['--output',target,'--count',source['count'],'--seed',source['seed'],'--profile',source['profile']]
        if 'negative_every' in source:args+=['--negative-every',source['negative_every']]
        run(dataset/f'generator/part{index:02d}/source/tools/generate_current_scenes.py',*args)
    exporter=dataset/'generator/exporter_source/tools/export_organizer_synthetic.py';target=workspace/'dataset'
    args=[]
    for part in parts:args+=['--collection',part]
    if info.get('excluded_invalid_region_frames'):args+=['--exclude-invalid-regions']
    run(exporter,*args,'--output',target)
    run(dataset/'generator/exporter_source/tools/check_organizer_synthetic.py','--dataset',target,'--report',workspace/'integrity.json')
    rebuilt=json.loads((target/'export_manifest.json').read_text('utf-8'))
    if rebuilt['meta_sha256']!=info['meta_sha256'] or rebuilt['official_synthetic_images']!=info['official_synthetic_images']:
        raise ValueError('Reproduced metadata differs; check the pinned library versions')
    return {'metadata_bit_identical':True,'official_synthetic_images':rebuilt['official_synthetic_images'],'dataset':str(target)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--dataset',type=Path);p.add_argument('--workspace',type=Path,required=True)
    args=p.parse_args();dataset=args.dataset or Path(__file__).resolve().parents[1]
    if not (dataset/'export_manifest.json').is_file():p.error('--dataset is required when running outside an exported collection')
    print(json.dumps(reproduce(dataset,args.workspace)))


if __name__=='__main__':main()
