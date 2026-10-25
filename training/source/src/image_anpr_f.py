"""Phase F task adapters. No GT enters prediction, OCR remains Phase D/E frozen."""
from dataclasses import replace
import time
from pathlib import Path
import cv2
import numpy as np
from src.geometry_f import canonical_rectify, order_obb, validate_quad
from src.image_anpr import (Detection, Route, GeometryPlateRouter, ImageConfig,
                           ImageANPRPipeline)


class HybridDetectorF:
    """Experimental complementary adapter: old monoline + new square/yellow.

    Input shape/class/confidence only; no GT/text/identity decision. A square
    candidate suppresses a contained old single-row fragment of that plate.
    """
    def __init__(self, primary, monoline):
        self.primary=primary;self.monoline=monoline

    @property
    def confidence(self):return self.primary.confidence

    @confidence.setter
    def confidence(self,value):self.primary.confidence=value;self.monoline.confidence=value

    def detect(self,frame):
        from src.geometry_f import bbox_iou
        router=RouterF('geometry');h,w=frame.shape[:2]
        def shape(d):
            a,b,c,e=map(int,d.bbox);crop=frame[max(0,b):min(h,e),max(0,a):min(w,c)]
            return router.classify(crop,d).plate_type if crop.size else 'other'
        primary=[d for d in self.primary.detect(frame) if shape(d)=='type1a' or (shape(d)=='type1' and d.class_name=='type1b')]
        old=[replace(d,class_name='type1',class_id=0) for d in self.monoline.detect(frame) if shape(d)=='type1']
        output=list(primary)
        for d in old:
            area=max(1,(d.bbox[2]-d.bbox[0])*(d.bbox[3]-d.bbox[1]))
            duplicate=False
            for p in primary:
                a=d.bbox;b=p.bbox
                inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                if bbox_iou(a,b)>=.5 or inter/area>=.8:
                    duplicate=True;break
            if not duplicate:output.append(d)
        return output


class DetectorF:
    def __init__(self, weights, device='cuda', imgsz=640, confidence=.5, iou=.5):
        from ultralytics import YOLO
        if not Path(weights).is_file():raise ValueError('Local checkpoint required')
        self.model=YOLO(str(weights));self.device=device;self.imgsz=imgsz
        self.confidence=confidence;self.iou=iou

    def detect(self, frame):
        result=self.model.predict(frame,imgsz=self.imgsz,conf=self.confidence,iou=self.iou,
            device=0 if self.device=='cuda' else 'cpu',verbose=False)[0]
        detections=[]
        if result.obb is not None:
            for b in result.obb:
                raw=b.xyxyxyxy[0].cpu().numpy()
                if raw.shape!=(4,2) or not np.isfinite(raw).all():continue
                box=np.r_[raw.min(axis=0),raw.max(axis=0)]
                if box[2]<=box[0] or box[3]<=box[1]:continue
                try:
                    q=tuple(map(tuple,order_obb(raw).tolist()))
                    source='model_OBB_rectangle_not_perspective_corners'
                except ValueError as error:
                    # Low-confidence thin rectangles also reach AP evaluation.
                    # Preserve their finite bbox and let the guarded pipeline
                    # fall back, rather than aborting the entire image batch.
                    q=None;source='model_OBB_invalid_bbox_fallback: '+str(error)
                c=int(b.cls[0]);detections.append(Detection(tuple(map(float,box)),float(b.conf[0]),c,
                    result.names[c],q,source))
        elif result.boxes is not None:
            for i,b in enumerate(result.boxes):
                c=int(b.cls[0]);q=None;source=None
                if result.keypoints is not None:
                    points=result.keypoints.xy[i].cpu().numpy()
                    visibility=result.keypoints.conf[i].cpu().numpy() if result.keypoints.conf is not None else np.ones(4)
                    if points.shape==(4,2) and np.all(visibility>=.5):
                        # Invalid order is left to the guarded pipeline, not repaired using GT.
                        q=tuple(map(tuple,points.tolist()));source='model_pose_TL_TR_BR_BL'
                detections.append(Detection(tuple(map(float,b.xyxy[0].cpu())),float(b.conf[0]),c,result.names[c],q,source))
        return detections


class RouterF:
    def __init__(self, mode='geometry', square_max_ratio=3.):
        if mode not in ('geometry','detector','combined'):raise ValueError(mode)
        self.mode=mode;self.geometry=GeometryPlateRouter(square_max_ratio)

    def classify(self,crop,detection):
        geometric=self.geometry.classify(crop,replace(detection,class_name='license-plate'))
        if detection.quad is not None:
            try:
                q=validate_quad(detection.quad)
                width=(np.linalg.norm(q[1]-q[0])+np.linalg.norm(q[2]-q[3]))/2
                height=(np.linalg.norm(q[3]-q[0])+np.linalg.norm(q[2]-q[1]))/2
                ratio=float(width/height)
                geometric=Route('type1a' if ratio<=self.geometry.square_max_ratio else 'type1',
                                'observed_quad_edge_ratio',ratio)
            except ValueError:
                pass
        name=detection.class_name
        if self.mode=='geometry' or name not in ('type1','type1a','type1b'):
            return geometric
        if self.mode=='detector':return Route(name,'detector_class_only',geometric.aspect_ratio)
        # Context of yellow one-line type1b only accepted when geometry is one-line.
        if name=='type1b' and geometric.plate_type=='type1':
            return Route(name,'geometry_one_line_and_detector_type1b',geometric.aspect_ratio)
        if name in ('type1','type1a') and 2.4<=geometric.aspect_ratio<=3.6:
            return Route(name,'detector_class_when_shape_within_20percent_of_ratio3_boundary',geometric.aspect_ratio)
        return Route(geometric.plate_type,'geometry_shape_with_type1b_confirmation',geometric.aspect_ratio)


class ImagePipelineF:
    def __init__(self, detector, ocr, *, geometry='bbox', routing='geometry', image_config=None):
        if geometry not in ('bbox','obb','quad'):raise ValueError(geometry)
        self.detector=detector;self.ocr=ocr;self.geometry=geometry
        self.router=RouterF(routing)
        self.config=image_config or ImageConfig(geometry_filter='relaxed',padding=0,deskew='none')
        self.base=ImageANPRPipeline(detector,ocr,self.config,self.router)

    def process(self, frame, *, image_name='', detections=None):
        start=time.perf_counter();cached=detections is not None
        if detections is None:
            detections=self.detector.detect(frame);self.ocr.synchronize()
        detector_ms=(time.perf_counter()-start)*1000
        if self.geometry=='bbox':
            out=self.base.process(frame,image_name=image_name,detections=detections)
            for r in out['detections']:
                d=detections[r['detection_index']]
                r.update(detector_class=d.class_name,detector_class_id=d.class_id,
                    observed_quad=d.quad,quad_source=d.quad_source,geometry_method='bbox')
            out.update(detector_ms=detector_ms,total_ms=(time.perf_counter()-start)*1000,cached_detections=cached)
            return out
        records=[];predictions=[]
        for index,d in enumerate(detections):
            # Routing uses observed source bbox/class, never annotation text/type.
            x1,y1,x2,y2=map(int,d.bbox);h,w=frame.shape[:2]
            tight=frame[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
            if tight.size==0:continue
            route=self.router.classify(tight,d);quad_ms=0.;fallback=None
            try:
                if d.quad is None:raise ValueError('No sufficiently confident four corners')
                t=time.perf_counter();crop,matrix=canonical_rectify(frame,d.quad,route.plate_type,d.quad_source)
                quad_ms=(time.perf_counter()-t)*1000
                ch,cw=crop.shape[:2]
                # The inner pipeline sees only the observed, rectified crop. Fixed type
                # is the already computed prediction route, not supplied GT.
                canonical_detection=replace(d,bbox=(0,0,cw,ch),class_name=route.plate_type,quad=None,quad_source=None)
                # E's legacy clipping uses an exclusive W-1/H-1 bound. Add a
                # hidden one-pixel border so its crop contains the full canvas.
                padded_canvas=cv2.copyMakeBorder(crop,0,1,0,1,cv2.BORDER_REPLICATE)
                # Valid quad geometry supplies the gate here; the legacy bbox
                # size/aspect filter is disabled on the canonical canvas.
                # Compare bbox with filter=none to isolate this from the warp.
                inner=ImageANPRPipeline(self.detector,self.ocr,replace(self.config,geometry_filter='none',rectify=False),
                    RouterF('detector'))
                out=inner.process(padded_canvas,image_name=image_name,detections=[canonical_detection])
            except (ValueError,cv2.error) as error:
                fallback=str(error);matrix=None
                out=self.base.process(frame,image_name=image_name,detections=[d])
            for r in out['detections']:
                r.update(detection_index=index,bbox=list(d.bbox),observed_quad=d.quad,
                    detector_class=d.class_name,detector_class_id=d.class_id,
                    quad_source=d.quad_source,geometry_method=self.geometry,
                    perspective_rectification=fallback is None,quad_fallback=fallback,
                    routing_decision={'plate_type':route.plate_type,'reason':route.reason,'aspect_ratio':route.aspect_ratio})
                r['stage_ms']['rectification']=quad_ms
                if matrix is not None:r['canonical_homography']=matrix.tolist()
                records.append(r)
            for p in out['predictions']:
                p['detection_index']=index;predictions.append(p)
        return {'image':image_name,'predictions':predictions,'detections':records,
            'detector_ms':detector_ms,'total_ms':(time.perf_counter()-start)*1000,'cached_detections':cached}
