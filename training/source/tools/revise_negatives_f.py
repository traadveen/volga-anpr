"""One pre-training visual-review revision, preserving previous authored pixels."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import shutil
import cv2
import numpy as np
from dataset.generator.scenes_f import compose
from dataset.schema import sha256
from tools.phase_f_setup import DATA,REPORT
from tools.phase_e import read,write

def main():
    manifest=read(DATA/'manifest.json')
    if manifest.get('negative_visual_revision'):raise ValueError('Already revised')
    assert not list((REPORT/'training').glob('*.json'))
    archive=DATA.parent/'detector_f_negative_v1_archive';archive.mkdir(exist_ok=False)
    for i,r in enumerate(manifest['records']):
        if not r.get('negative_kind'):continue
        path=DATA/'detect/images'/r['split']/r['image']
        shutil.copy2(path,archive/r['image'])
        rng=np.random.default_rng(np.random.SeedSequence([manifest['seed'],i,820]))
        rgb,targets=compose(rng,[],[],r['negative_kind']);assert not targets
        # All three task images are hardlinks to this authored file. No real source is modified.
        cv2.imencode('.jpg',cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR),[cv2.IMWRITE_JPEG_QUALITY,90])[1].tofile(path)
        r['sha256']=sha256(path)
    manifest['negative_visual_revision']={'reason':'Initial visual review found collapsed road/shop/billboard visual domains',
        'archive':str(archive),'before_training':True,'categories':'Distinct authored street sign/shopfront/billboard/screen/paper plate/text panels; still procedural, not photographic negatives'}
    for source in manifest['source_hashes']:manifest['source_hashes'][source]=sha256(Path(source))
    manifest['composition']['real_scenes']=manifest['real_import']['unique_frames']
    write(DATA/'manifest.json',manifest)
    composition=read(REPORT/'dataset_composition.json');composition['manifest_sha256']=sha256(DATA/'manifest.json')
    composition['composition']=manifest['composition'];composition['negative_visual_revision']=manifest['negative_visual_revision']
    write(REPORT/'dataset_composition.json',composition)

if __name__=='__main__':main()
