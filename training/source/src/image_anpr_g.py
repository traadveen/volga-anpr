"""Phase G inference adapters: observed pixels/class only, no oracle arguments."""
from dataclasses import asdict,replace
import hashlib,time
import cv2,numpy as np,torch
from src.image_anpr import FrozenCRNNOCR,padded_crop,current_postprocess
from src.image_anpr_f import ImagePipelineF,RouterF
from src.geometry_f import canonical_rectify,validate_quad
from src.multiline_g import ConfigG,prepare_g,candidates,candidate_score
from src.real_data_g import valid_complete_plate

def expanded_quad(quad,fraction,image_shape):
    if fraction not in (0,.02,.04,.06):raise ValueError('Predeclared total size expansions only')
    q=validate_quad(quad);center=q.mean(axis=0);proposed=center+(q-center)*(1+fraction)
    # No per-corner clipping/shear: leave original if expansion exits frame.
    h,w=image_shape[:2]
    if (proposed<0).any() or (proposed[:,0]>w-1).any() or (proposed[:,1]>h-1).any():return q,'original_out_of_bounds_expansion'
    return validate_quad(proposed),'expanded' if fraction else 'unchanged'

def gray_for_ocr(crop):
    h,w=crop.shape[:2];scale=max(480/w,160/h)
    if scale>1:crop=cv2.resize(crop,(int(w*scale),int(h*scale)),interpolation=cv2.INTER_CUBIC)
    return cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)

class OCRG(FrozenCRNNOCR):
    def recognize_candidates(self,arrays):
        if not 1<=len(arrays)<=5:raise ValueError('Bounded1..5 inputs required')
        with torch.inference_mode():
            tensor=torch.from_numpy(np.stack(arrays)).unsqueeze(1).to(self.device)
            logits=self.model(tensor);texts,confs,probs=self.converter.decode_logits(logits)
            log_probs=logits.log_softmax(2).cpu()
            lengths=torch.full((1,),logits.shape[0],dtype=torch.long)
            loss=torch.nn.CTCLoss(blank=self.converter.blank_idx,reduction='sum',zero_infinity=False)
            output=[]
            for i,text in enumerate(texts):
                targets,target_lengths=self.converter.encode([text])
                logp=-float(loss(log_probs[:,i:i+1],targets,lengths,target_lengths))
                output.append((text,float(confs[i]),list(map(float,probs[i])),logp))
        return output

class ImagePipelineG:
    def __init__(self,detector,ocr,config=None,*,multi=False,postprocess='current',expansion=None):
        if postprocess not in ('current','mask_preserve','raw'):raise ValueError('Unknown postprocess')
        self.detector=detector;self.ocr=ocr;self.config=config or ConfigG();self.multi=multi
        self.postprocess=postprocess;self.expansion=expansion or {};self.router=RouterF('detector')
        self.fallback=ImagePipelineF(detector,ocr,geometry='obb',routing='detector')

    def process(self,frame,*,image_name='',detections=None):
        start=time.perf_counter();cached=detections is not None
        if detections is None:detections=self.detector.detect(frame);self.ocr.synchronize()
        det_ms=(time.perf_counter()-start)*1000;records=[];predictions=[]
        for index,d in enumerate(detections):
            t=time.perf_counter()
            try:tight,_=padded_crop(frame,d.bbox,0)
            except ValueError:
                records.append({'image':image_name,'detection_index':index,'bbox':list(d.bbox),'status':'rejected','reason':'empty source crop'});continue
            route=self.router.classify(tight,d)
            try:
                q,state=expanded_quad(d.quad,self.expansion.get(route.plate_type,0),frame.shape)
                crop,_=canonical_rectify(frame,q,route.plate_type,d.quad_source);gray=gray_for_ocr(crop)
            except (ValueError,cv2.error,TypeError):
                out=self.fallback.process(frame,image_name=image_name,detections=[d])
                for r in out['detections']:r['detection_index']=index;r['G_fallback']='frozen_F';records.append(r)
                for p in out['predictions']:p['detection_index']=index;predictions.append(p)
                continue
            prep_start=time.perf_counter()
            cs=candidates(gray,route.plate_type,self.config) if self.multi else [prepare_g(gray,route.plate_type,self.config)]
            prep_ms=(time.perf_counter()-prep_start)*1000
            ocr_start=time.perf_counter();decoded=self.ocr.recognize_candidates([c[0] for c in cs])
            scores=[candidate_score(v[0],v[3],c[2]['structural_confidence'],valid_complete_plate(v[0])) for c,v in zip(cs,decoded)]
            selected=max(range(len(scores)),key=lambda i:scores[i]);raw,confidence,probs,logp=decoded[selected]
            self.ocr.synchronize();ocr_ms=(time.perf_counter()-ocr_start)*1000
            post=current_postprocess(raw,probs)
            if self.postprocess=='mask_preserve' and valid_complete_plate(raw):post=raw
            final=raw if self.postprocess=='raw' else post
            r={'image':image_name,'detection_index':index,'bbox':list(d.bbox),'observed_quad':q.tolist(),
                'detector_class':d.class_name,'detector_confidence':d.confidence,'plate_type':route.plate_type,
                'status':'ocr','raw_ocr':raw,'postprocessed_ocr':post,'final_ocr':final,'ocr_confidence':confidence,
                'char_probs':probs,'sequence_log_probability':logp,'preprocessing':cs[selected][2],
                'normalized_sha256':hashlib.sha256(cs[selected][0].tobytes()).hexdigest(),'expansion_status':state,
                'candidates':[{'method':c[2]['method'],'raw':v[0],'CTC_log_probability':v[3],'selection_score':s,
                    'structural_confidence':c[2]['structural_confidence'],'mask_valid':valid_complete_plate(v[0])} for c,v,s in zip(cs,decoded,scores)],
                'selected_candidate':selected,'candidate_count':len(cs),'stage_ms':{'split_stitch':prep_ms,'OCR_and_candidate_scoring':ocr_ms},
                'detection_total_ms':(time.perf_counter()-t)*1000}
            records.append(r)
            if final and set(final)!={'#'}:predictions.append({'image':image_name,'plate_num':final,'plate_type':route.plate_type,'confidence':confidence,'detection_index':index})
        return {'image':image_name,'predictions':predictions,'detections':records,'cached_detections':cached,'detector_ms':det_ms,'total_ms':(time.perf_counter()-start)*1000}
