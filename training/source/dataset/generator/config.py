"""Strict bounded configuration, so broken augmentations cannot silently enter data."""
import json
import math
from pathlib import Path

DEFAULT=Path(__file__).resolve().parents[2]/'configs/synthetic_v2.json'
EFFECTS=('brightness','contrast','gamma','motion_blur','defocus_blur','gaussian_blur','noise','jpeg','low_resolution','shadow','glare','rain','snow','dirt','partial_occlusion','overexposure','underexposure','color_temperature')


def load_config(path=DEFAULT):
    value=json.loads(Path(path).read_text(encoding='utf-8'))
    if set(value)!={'version','supersample','clean_fraction','output_sizes','geometry','appearance','augmentation'} or value['version']!='2.0.0':
        raise ValueError('Unknown config fields/version')
    if isinstance(value['supersample'],bool) or not isinstance(value['supersample'],int) or not 1<=value['supersample']<=4:
        raise ValueError('supersample must be integer 1..4')
    if not isinstance(value['clean_fraction'],(float,int)) or isinstance(value['clean_fraction'],bool) or not 0<=value['clean_fraction']<=1:
        raise ValueError('clean_fraction must be 0..1')
    if set(value['output_sizes'])!={'type1','type1a','type1b'}:
        raise ValueError('Three output sizes are required')
    for dimensions in value['output_sizes'].values():
        if not isinstance(dimensions,list) or len(dimensions)!=2 or any(isinstance(n,bool) or not isinstance(n,int) or not 160<=n<=2048 for n in dimensions):
            raise ValueError('Output dimensions must be two integers 160..2048')
    def ranges(section,bounds):
        obj=value[section]
        allowed=set(bounds)|({'max_effects','probability','effects'} if section=='augmentation' else set())
        if set(obj)!=allowed:raise ValueError(f'Unknown/missing {section} fields')
        for key,(lo,hi) in bounds.items():
            pair=obj[key]
            if not isinstance(pair,list) or len(pair)!=2 or any(isinstance(n,bool) or not isinstance(n,(float,int)) or not math.isfinite(n) for n in pair) or not lo<=pair[0]<=pair[1]<=hi:
                raise ValueError(f'Invalid or excessive range {section}.{key}')
    ranges('geometry',{'rotation_deg':(-9,9),'scale':(.6,.95),'translation_fraction':(-.05,.05),'perspective_fraction':(0,.08)})
    ranges('appearance',{'plate_texture_sigma':(0,2),'background_gray':(0,150)})
    ranges('augmentation',{'brightness':(.7,1.3),'contrast':(.8,1.2),'gamma':(.75,1.3),'motion_blur_px':(1,5),'defocus_blur_px':(0,1),'gaussian_blur_px':(0,1),'noise_sigma':(0,4),'jpeg_quality':(60,100),'low_resolution_scale':(.35,1),'shadow_strength':(0,.25),'glare_strength':(0,.2),'rain_density':(0,.001),'snow_density':(0,.001),'dirt_density':(0,.001),'occlusion_fraction':(0,.05),'exposure':(.7,1.3),'color_temperature':(.85,1.15)})
    a=value['augmentation']
    if not isinstance(a['effects'],list) or not a['effects'] or any(e not in EFFECTS for e in a['effects']) or len(a['effects'])!=len(set(a['effects'])):
        raise ValueError('Effects must be distinct known names')
    if isinstance(a['max_effects'],bool) or not isinstance(a['max_effects'],int) or not 1<=a['max_effects']<=min(3,len(a['effects'])):
        raise ValueError('max_effects must be integer 1..3, no greater than effects count')
    if isinstance(a['probability'],bool) or not isinstance(a['probability'],(int,float)) or not 0<=a['probability']<=1:
        raise ValueError('augmentation probability must be 0..1')
    if not a['exposure'][0]<=1<=a['exposure'][1]:raise ValueError('exposure range must straddle 1')
    return value
