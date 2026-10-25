"""Capture the exact legacy OCR inputs without editing frozen pipeline sources."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.baseline_predict import deny_network


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True)
    parser.add_argument('--input',required=True);parser.add_argument('--ocr',required=True)
    parser.add_argument('--no-deskew',action='store_true')
    args=parser.parse_args();output=Path(args.output);capture=output.with_suffix('')
    capture=capture.with_name(capture.name+'_captures');capture.mkdir(parents=True,exist_ok=False)
    os.environ['YOLO_OFFLINE']='true';os.environ['YOLO_AUTOINSTALL']='false'
    os.environ['YOLO_CONFIG_DIR']=str(Path(__file__).resolve().parents[1]/'.baseline_yolo')
    with deny_network():
        import cv2
        import numpy as np
        from PIL import Image
        import run as legacy
        from tools import baseline_predict
        image_names={}
        for path in sorted(Path(args.input).glob('*.jpg')):
            frame=cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_COLOR)
            image_names[hashlib.sha256(frame.tobytes()).hexdigest()]=path.name
        original_recognizer=legacy.CRNNRecognizer
        class CapturingRecognizer(original_recognizer):
            def __init__(self,*a,**kw):
                super().__init__(*a,**kw)
                self.raw_tokens=None
                self.model.register_forward_hook(self.capture_tokens)
            def capture_tokens(self,module,inputs,logits):
                self.raw_tokens=logits.argmax(2).transpose(0,1)[0].cpu().tolist()
            def readtext(self,image,*a,**kw):
                self.captured_input=image.copy()
                return super().readtext(image,*a,**kw)
        legacy.CRNNRecognizer=CapturingRecognizer
        original_ocr=legacy.ocr_plate;records=[]
        def observed(reader,frame,bbox,*a,**kw):
            reader.captured_input=None
            result=original_ocr(reader,frame,bbox,*a,**kw)
            text,confidence,meta=result
            index=len(records);record={'image':image_names[hashlib.sha256(frame.tobytes()).hexdigest()],
                'bbox_xyxy':[float(v) for v in bbox[:4]],'det_conf':float(bbox[4]),
                'ocr_stage_prediction':text,'confidence':confidence,
                'raw_ctc_tokens':reader.raw_tokens,
                'geometry_stage':'legacy padded bbox -> upscale -> '+('no deskew' if args.no_deskew else 'old deskew'),
                'perspective_rectification':False}
            if meta.get('crop_bgr') is not None:
                path=capture/f'{index:04}_crop.png';Image.fromarray(meta['crop_bgr'][:,:,::-1]).save(path)
                record['detected_crop']=str(path);record['crop_bbox_xyxy']=list(meta['crop_bbox'])
            if reader.captured_input is not None:
                path=capture/f'{index:04}_ocr_input.png';Image.fromarray(reader.captured_input).save(path)
                record['OCR_input']=str(path)
                record['rectified_crop']=str(path)
                record['rectification_note']='Actual legacy geometry result; padded bbox/upscale/optional rotation, no perspective homography'
            records.append(record)
            return result
        legacy.ocr_plate=observed
        if args.no_deskew:legacy._deskew_estimate=lambda gray:0.
        baseline_predict.main(['--input',args.input,'--output',args.output,'--ocr',args.ocr,'--device','cpu'])
        (capture/'manifest.json').write_text(json.dumps({'observations':records,'no_deskew':args.no_deskew,
            'checkpoint':args.ocr,'scope':'Original detector, geometry filter, OCR preprocessing and postprocess; only specified weights/deskew differ'},indent=2),encoding='utf-8')


if __name__=='__main__':main()
