"""Record visual review decisions; model quadrilaterals never become oracle GT."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import csv,json
from collections import Counter
from tools.phase_g_common import ROOT,REPORT,IMAGES,freeze,read,write
from dataset.schema import sha256

# Decisions from source context, actual rectified crop AND exact OCR tensor.
# These are engineering annotations by the reviewing agent, not independently
# adjudicated human labels. Existing E/F manually localized GT stays separate.
GEOMETRY={'072','095','098'}
POST={'008'}
GT_ISSUE={'031','032','085'}
MISS={'009','012','065','090'}
LOW={'019','044_partial','052'}
SELECT={'085':1,'031':None,'044_partial':None,'058_partial':None,'012':None,'035_partial':1}
NOTES={
 '031':'Source crop visibly has two rows (C983 / HK...); organizer target is type1. Partial plate is clipped at image right; low-confidence proposal exists. GT class discrepancy; not an OCR training label.',
 '032':'Visible identity M941MT14 differs from organizer M491MT14. Frozen scorer literal retained; corrected visible identity separate.',
 '085':'Source visibly has two rows E152 / MH14; organizer target type1. F class type1a is visually appropriate; annotation discrepancy, not wrong model routing.',
 '009':'Fence wires occlude a physically visible plate; no proposal at conf>=0.01.',
 '012':'Existing manual quad proves target absent; other output belongs to a different plate.',
 '019':'Spatially associated type1a proposal at conf=0.218, below frozen 0.5 gate.',
 '044_partial':'Background partially occluded monoline 15KC75; spatially associated type1 proposal conf=0.478.',
 '052':'Tiny truck two-row plate; best associated proposal conf=0.335 predicts type1. Low confidence is first gate; wrong class is secondary.',
 '058_partial':'Visible P338T prefix lies at image right, remainder outside/occluded; low-confidence proposal elsewhere is not this plate.',
 '065':'Strong perspective on front-mounted square plate; no spatially associated proposal at conf>=0.01.',
 '090':'Blurred snowy scene; visible source identity cannot be independently verified character by character. No spatially associated proposal.',
 '018':'Visible plate localized, but source motion/texture degradation persists in normalized input; do not attribute to good-input OCR.',
 '072':'Observed rectangle leaves strong perspective skew and bumper/background edges in stitch; geometry distortion precedes decoding error.',
 '095':'Observed OBB cannot remove perspective shear; row baselines remain strongly slanted in actual OCR input.',
 '098':'Perspective shear remains after OBB warp; glyph distortion is visible in actual stitched input.',
 '008':'Raw A641TK172 is exact, current postprocess changes T to #.',
}

def key(r):
    n=r['image'][4:7]
    if r['plate_type']=='type1' and n in ('035','044','058'):n+='_partial'
    return n

def main():
    freeze();index=read(REPORT/'audit/artifact_index.json')['records'];rows=[]
    for image in index:
        for g in image['GT']:
            n=key(g);ds=image['detections'];chosen=g['chosen']
            if chosen is None:
                chosen=max((d for d in ds if d.get('plate_type')==g['plate_type']),key=lambda d:d['detector_confidence'],default=None)
            if n in SELECT and (not g['official_success'] or n=='035_partial'):
                chosen=next((d for d in ds if d['detection_index']==SELECT[n]),None)
            if chosen:chosen=next(d for d in ds if d['detection_index']==chosen['detection_index'])
            success=g['official_success'];category='SUCCESS';secondary=[]
            if not success:
                if n in GT_ISSUE:category='UNKNOWN'
                elif n in MISS or n=='058_partial':category='DETECTOR_MISS'
                elif n in LOW:category='LOW_DETECTOR_CONFIDENCE'
                elif n=='018':category='UNREADABLE'
                elif n in GEOMETRY:category='BAD_RECTIFICATION'
                elif n in POST:category='POSTPROCESS_FAILURE'
                elif chosen and chosen['preprocessing']['effective_mode']=='whole_plate':category='TYPE1A_SPLIT_FAILURE'
                elif chosen:category='OCR_FAILURE_GOOD_INPUT'
                else:category='UNKNOWN'
                if n=='052':secondary=['WRONG_DETECTOR_CLASS']
                if n in ('009','090','058_partial'):secondary.append('UNREADABLE_OR_PARTIALLY_OCCLUDED_SOURCE')
            row={'image':g['image'],'ground_truth':g['ground_truth'],'plate_type':g['plate_type'],
                'official_success':success,'primary_failure_category':category,'secondary_failure_categories':secondary,
                'notes':NOTES.get(n,'Both rows and characters retained in actual input; failure follows decoding.' if category=='OCR_FAILURE_GOOD_INPUT' else 'Two rows remain stacked in actual 32x160 OCR input; split fell back to whole crop.' if category=='TYPE1A_SPLIT_FAILURE' else ''),
                'source':str(IMAGES/g['image']),'source_sha256':sha256(IMAGES/g['image']),
                'detected':False if (n=='031' and not success) or category in ('DETECTOR_MISS','LOW_DETECTOR_CONFIDENCE') else True if chosen else None,
                'spatial_GT_available':g['spatial_GT_available'],'spatial_presence_review':'VISUALLY_REVIEWED',
                'geometry_GT_provenance':'Existing independent-of-F manual E/F borders' if g['spatial_GT_available'] else 'NONE; model proposal is not GT',
                'good_input_verified':category in ('OCR_FAILURE_GOOD_INPUT','POSTPROCESS_FAILURE','SUCCESS') and '#' not in g['ground_truth'],
                'detector_index':chosen['detection_index'] if chosen else None,
                'detector_class':chosen.get('detector_class') if chosen else None,
                'detector_confidence':chosen['detector_confidence'] if chosen else None,
                'bbox':chosen['bbox'] if chosen else None,'OBB':chosen.get('observed_quad') if chosen else None,
                'crop':chosen.get('artifacts',{}).get('crop') if chosen else None,
                'rectified_crop':chosen.get('artifacts',{}).get('rectified') if chosen else None,
                'top':chosen.get('artifacts',{}).get('top') if chosen else None,
                'bottom':chosen.get('artifacts',{}).get('bottom') if chosen else None,
                'stitched':chosen.get('artifacts',{}).get('stitched') if chosen else None,
                'normalized_input':chosen.get('artifacts',{}).get('normalized') if chosen else None,
                'raw_OCR':chosen.get('raw_ocr') if chosen else None,'postprocessed_OCR':chosen.get('postprocessed_ocr') if chosen else None,
                'final_OCR':chosen.get('final_ocr') if chosen else None,
                'split_status':chosen.get('preprocessing',{}).get('effective_mode') if chosen else None}
            rows.append(row)
    def export(path,values):
        with path.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader()
            w.writerows({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()} for r in values)
    export(REPORT/'error_attribution.csv',rows)
    export(REPORT/'type1a_failures.csv',[r for r in rows if not r['official_success'] and r['plate_type']=='type1a'])
    categories=dict(Counter(r['primary_failure_category'] for r in rows if not r['official_success']))
    assert len(rows)==115 and sum(categories.values())==64
    write(REPORT/'error_summary.json',{'rows':rows,'failure_categories':categories,'GT':115,'correct':51,'failed':64,
        'failure_visual_review':'64/64 reviewed in compact_01..08 and source/missing/ambiguous sheets; reviewer is agent, no independent annotator agreement',
        'success_visual_review':'51/51 reviewed in success_01..07; masked Y364OH#### additionally reviewed in img_035 source and actual OCR input',
        'policy':'One earliest attributable primary cause per failed target; secondary categories excluded from sums. Unknown source GT discrepancies never counted as OCR failures. Presence review is not accurate localization IoU GT.'})
    print(categories,flush=True)

if __name__=='__main__':main()
