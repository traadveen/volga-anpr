"""Independent image ANPR: no tracking, voting, clocks, or label inputs.

Geometry routing is deliberately replaceable. The trained Phase C/D stitch and
normalization are reused verbatim; failed row evidence is reported explicitly.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import csv
import hashlib
import json
from pathlib import Path
import time

import cv2
import numpy as np

from ocr.utils import PLATE_ALPHABET, matches_plate_pattern, pattern_for_len
from src.multiline import MultilineConfig, prepare_multiline
from src.ocr_preprocess import prepare_ocr_input


@dataclass(frozen=True)
class Detection:
    bbox: tuple[float, float, float, float]
    confidence: float
    class_id: int = 0
    class_name: str = 'license-plate'
    quad: tuple | None = None
    quad_source: str | None = None


@dataclass(frozen=True)
class Route:
    plate_type: str
    reason: str
    aspect_ratio: float


class GeometryPlateRouter:
    """Objective dimensions only; no OCR text or GT access.

    A single-class detector cannot distinguish type1b from type1. Explicit
    detector classes are respected, otherwise long plates use the type1 path.
    """
    def __init__(self, square_max_ratio=3.0):
        if not 1 < square_max_ratio < 4:
            raise ValueError('Invalid square ratio')
        self.square_max_ratio = square_max_ratio

    def classify(self, crop, detection_metadata):
        h, w = crop.shape[:2]
        ratio = w / max(1, h)
        name = detection_metadata.class_name
        if name in ('type1', 'type1a', 'type1b', 'other'):
            return Route(name, 'explicit_detector_class', ratio)
        return Route('type1a' if ratio <= self.square_max_ratio else 'type1',
                     'bbox_dimensions_only', ratio)


@dataclass(frozen=True)
class ImageConfig:
    geometry_filter: str = 'none'
    padding: float = .06
    deskew: str = 'none'
    routing: str = 'geometry'
    stitch: bool = True
    postprocess: str = 'current'
    square_max_ratio: float = 3.0
    rectify: bool = False
    upscale: bool = True
    debug: bool = False

    def __post_init__(self):
        if self.geometry_filter not in ('old', 'none', 'relaxed', 'type_aware'):
            raise ValueError('Unknown geometry filter')
        if self.deskew not in ('none', 'old', 'corrected'):
            raise ValueError('Unknown deskew')
        if self.routing not in ('constant', 'geometry'):
            raise ValueError('GT routing is not a production configuration')
        if self.postprocess not in ('raw', 'current') or not 0 <= self.padding <= .3:
            raise ValueError('Invalid postprocess/padding')


def geometry_filter(detection, mode, route_type):
    x1, y1, x2, y2 = detection.bbox
    w, h = x2-x1, y2-y1
    if not np.isfinite([x1, y1, x2, y2]).all() or min(w, h) <= 0:
        return False, 'invalid_box'
    ar = w/h
    if mode == 'none':
        return True, 'nonempty_only'
    if mode == 'old':
        ok = 1.6 <= ar <= 6.5 and w >= 50 and h >= 14
    elif mode == 'relaxed':
        ok = .8 <= ar <= 8 and w >= 20 and h >= 10
    elif mode == 'type_aware':
        low, high = (.8, 3.5) if route_type == 'type1a' else (2., 8.)
        ok = low <= ar <= high and w >= 20 and h >= 10
    else:
        raise ValueError('Unknown filter')
    return bool(ok), 'accepted' if ok else 'dimensions_or_aspect_ratio'


def padded_crop(frame, bbox, padding):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)
    if x2 <= 0 or y2 <= 0 or x1 >= w or y1 >= h or x2 <= x1 or y2 <= y1:
        raise ValueError('BBox does not intersect the image')
    dx, dy = int(max(1, x2-x1)*padding), int(max(1, y2-y1)*padding)
    # Identical bounds to legacy, including its exclusive W-1/H-1 endpoint.
    box = max(0, x1-dx), max(0, y1-dy), min(w-1, x2+dx), min(h-1, y2+dy)
    a, b, c, d = box
    if c <= a or d <= b:
        raise ValueError('Empty crop after bounds clipping')
    return frame[b:d, a:c].copy(), box


def deskew_estimate(gray):
    lines = cv2.HoughLines(cv2.Canny(gray, 50, 120), 1, np.pi/180, 60)
    if lines is None:
        return 0.
    angles = [(theta*180/np.pi)-90 for _, theta in lines[:, 0]]
    angles = [float(a) for a in angles if -20 <= a <= 20]
    return float(np.median(angles)) if angles else 0.


def rotate(gray, angle):
    if abs(angle) < .5:
        return gray
    h, w = gray.shape[:2]
    matrix = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.)
    return cv2.warpAffine(gray, matrix, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def rectify_from_quad(frame, detection):
    if detection.quad is None or not detection.quad_source:
        raise ValueError('Real observed quad and its provenance are required')
    quad = np.asarray(detection.quad, dtype=np.float32)
    if quad.shape != (4, 2) or not np.isfinite(quad).all() or not cv2.isContourConvex(quad):
        raise ValueError('Invalid quad')
    w = round(max(np.linalg.norm(quad[1]-quad[0]), np.linalg.norm(quad[2]-quad[3])))
    h = round(max(np.linalg.norm(quad[3]-quad[0]), np.linalg.norm(quad[2]-quad[1])))
    if min(w, h) < 8:
        raise ValueError('Quad too small')
    destination = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    matrix = cv2.getPerspectiveTransform(quad, destination)
    return cv2.warpPerspective(frame, matrix, (w, h)), matrix


DIG_FROM_LET = {'O':'0','D':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6','T':'7','A':'4'}
LET_FROM_DIG = {'0':'O','3':'E','4':'A','6':'G','7':'T','8':'B','1':'H'}


def current_postprocess(raw, probabilities):
    """The existing confidence masks/padding/position coercion, without votes.

    This is a single read operation. No pattern beam search, model change, or
    per-image threshold selection is introduced.
    """
    text = ''.join(ch if p >= (.5 if ch.isdigit() else .7) else '#'
                   for ch, p in zip(raw, probabilities))
    length = 9 if len(text) >= 9 else 8
    text = text[:length].ljust(length, '#')
    result = []
    for ch, kind in zip(text, pattern_for_len(length)):
        if ch == '#':
            result.append(ch)
        elif kind == 'D':
            result.append(ch if ch.isdigit() else DIG_FROM_LET.get(ch, '#'))
        else:
            result.append(ch if ch in PLATE_ALPHABET else LET_FROM_DIG.get(ch, '#'))
    return ''.join(result)


class FrozenCRNNOCR:
    def __init__(self, checkpoint, device='cpu'):
        import torch
        from ocr.crnn import load_crnn_checkpoint
        self.torch, self.device = torch, device
        self.model, self.converter, h, w = load_crnn_checkpoint(str(checkpoint), device)
        if (h, w) != (32, 160):
            raise ValueError('Phase D preprocessing contract requires 32x160')

    def synchronize(self):
        if self.device == 'cuda':
            self.torch.cuda.synchronize()

    def recognize(self, normalized):
        with self.torch.inference_mode():
            tensor = self.torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0).to(self.device)
            logits = self.model(tensor)
            texts, confidence, probs = self.converter.decode_logits(logits)
        return texts[0], float(confidence[0]), list(map(float, probs[0]))


class YOLOImageDetector:
    def __init__(self, weights, device='cpu', confidence=.5, iou=.5):
        from ultralytics import YOLO
        if not Path(weights).is_file():
            raise ValueError('Local detector weights required')
        self.model, self.device = YOLO(str(weights)), device
        self.confidence, self.iou = confidence, iou

    def detect(self, frame):
        result = self.model.predict(frame, conf=self.confidence, iou=self.iou,
                                    verbose=False, device='cuda:0' if self.device == 'cuda' else 'cpu')[0]
        boxes = result.boxes
        if boxes is None:
            return []
        return [Detection(tuple(map(float, b.xyxy[0].cpu().tolist())),
                          float(b.conf[0].cpu()), int(b.cls[0].cpu()),
                          self.model.names[int(b.cls[0].cpu())]) for b in boxes]


def save_image(path, image):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    extension = path.suffix
    ok, data = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f'Cannot encode {path}')
    data.tofile(path)


class ImageANPRPipeline:
    def __init__(self, detector, ocr, config=None, router=None):
        self.detector, self.ocr = detector, ocr
        self.config = config or ImageConfig()
        self.router = router or GeometryPlateRouter(self.config.square_max_ratio)
        self.multiline_config = MultilineConfig()

    def process(self, frame, *, image_name='', detections=None, debug_directory=None):
        if not isinstance(frame, np.ndarray) or frame.dtype != np.uint8 or frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError('Expected nonempty BGR uint8 image')
        if min(frame.shape[:2]) < 2:
            raise ValueError('Empty image')
        if self.config.debug and debug_directory is None:
            raise ValueError('Debug directory required when debug is enabled')
        start = time.perf_counter()
        cached_detections = detections is not None
        if detections is None:
            detections = self.detector.detect(frame)
            self.ocr.synchronize()
        detector_ms = (time.perf_counter()-start)*1000
        records, predictions = [], []
        for index, detection in enumerate(detections):
            times = {}
            record = {'image': image_name, 'detection_index': index, 'bbox': list(detection.bbox),
                      'detector_confidence': detection.confidence, 'status': 'rejected', 'stage_ms': times}
            artifacts = {}
            t0 = time.perf_counter()
            try:
                tight, _ = padded_crop(frame, detection.bbox, 0)
                crop, crop_bbox = padded_crop(frame, detection.bbox, self.config.padding)
            except ValueError as error:
                record['rejection_reason'] = str(error)
                records.append(record)
                continue
            times['crop'] = (time.perf_counter()-t0)*1000
            t0 = time.perf_counter()
            route = self.router.classify(tight, detection)
            geometry_route = route.plate_type
            if self.config.routing == 'constant':
                route = Route('type1', 'constant_type1_ablation', route.aspect_ratio)
            times['routing'] = (time.perf_counter()-t0)*1000
            passed, reason = geometry_filter(detection, self.config.geometry_filter, geometry_route)
            record.update({'crop_bbox': crop_bbox, 'crop_dimensions': list(crop.shape),
                           'aspect_ratio': route.aspect_ratio, 'geometry_filter_pass': passed,
                           'geometry_filter_reason': reason, 'plate_type': route.plate_type,
                           'routing_decision': asdict(route), 'perspective_rectification': False})
            if self.config.debug:
                marked = frame.copy()
                x1, y1, x2, y2 = map(int, detection.bbox)
                cv2.rectangle(marked, (x1,y1), (x2,y2), (0,255,0), 2)
                artifacts.update({'01_original': frame, '02_bbox': marked, '03_padded_crop': crop})
            if passed and route.plate_type != 'other':
                t0 = time.perf_counter()
                rectified = crop
                if self.config.rectify and detection.quad is not None:
                    rectified, matrix = rectify_from_quad(frame, detection)
                    record.update({'perspective_rectification': True, 'quad_source': detection.quad_source,
                                   'homography': matrix.tolist()})
                record['rectification_status'] = 'observed_quad' if record['perspective_rectification'] else 'NOT_AVAILABLE_NO_QUAD'
                times['rectification'] = (time.perf_counter()-t0)*1000
                gray = cv2.cvtColor(rectified, cv2.COLOR_BGR2GRAY)
                if self.config.upscale:
                    h, w = gray.shape
                    scale = max(480/max(1,w), 160/max(1,h))
                    if scale > 1:
                        # Resize BGR first exactly as legacy, then convert to gray.
                        gray = cv2.cvtColor(cv2.resize(rectified, (int(w*scale),int(h*scale)),
                                                     interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2GRAY)
                t0 = time.perf_counter()
                angle = deskew_estimate(gray) if self.config.deskew != 'none' else 0.
                applied = -angle if self.config.deskew == 'old' else angle
                transformed = rotate(gray, applied)
                times['deskew'] = (time.perf_counter()-t0)*1000
                record.update({'deskew_estimate': angle, 'deskew_angle': applied,
                               'deskew_applied': abs(applied) >= .5})
                if self.config.debug:
                    artifacts.update({'04_deskew': transformed, '05_rectified': transformed})
                t0 = time.perf_counter()
                kind = route.plate_type if self.config.stitch else 'type1'
                prepared = prepare_ocr_input(transformed, kind, multiline_config=self.multiline_config)
                times['split_stitch'] = (time.perf_counter()-t0)*1000
                record['preprocessing'] = prepared.diagnostics
                record['normalized_sha256'] = hashlib.sha256(prepared.normalized.tobytes()).hexdigest()
                if self.config.debug:
                    if kind == 'type1a' and prepared.diagnostics['row_boxes']:
                        for name, box in zip(('06_type1a_top','07_type1a_bottom'), prepared.diagnostics['row_boxes']):
                            a,b,c,d = box
                            artifacts[name] = transformed[b:d,a:c]
                        artifacts['08_type1a_stitched'] = prepared.crop
                    artifacts['09_ocr_input'] = np.clip((prepared.normalized+1)*127.5,0,255).astype(np.uint8)
                t0 = time.perf_counter()
                raw, confidence, probabilities = self.ocr.recognize(prepared.normalized)
                self.ocr.synchronize()
                times['OCR'] = (time.perf_counter()-t0)*1000
                t0 = time.perf_counter()
                post = current_postprocess(raw, probabilities)
                final = post if self.config.postprocess == 'current' else raw
                times['postprocess'] = (time.perf_counter()-t0)*1000
                record.update({'status':'ocr', 'raw_ocr':raw, 'postprocessed_ocr':post, 'final_ocr':final,
                               'ocr_confidence':confidence, 'char_probs':probabilities})
                if final and not (set(final) == {'#'}):
                    predictions.append({'image':image_name, 'plate_num':final,
                                        'plate_type':route.plate_type, 'confidence':confidence,
                                        'detection_index':index})
            else:
                record['rejection_reason'] = 'other' if route.plate_type == 'other' else reason
            if self.config.debug:
                folder = Path(debug_directory)/f'detection_{index:03}'
                folder.mkdir(parents=True,exist_ok=True)
                for name, array in artifacts.items():
                    path = folder/f'{name}.png'
                    save_image(path, array)
                    record.setdefault('artifacts', {})[name] = str(path)
                record['unavailable_stages'] = {name:'not attempted or unavailable; no synthetic quad/fake rows'
                    for name in ('04_deskew','05_rectified','06_type1a_top','07_type1a_bottom','08_type1a_stitched','09_ocr_input')
                    if name not in artifacts}
                (folder/'metadata.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
            records.append(record)
        return {'image':image_name, 'predictions':predictions, 'detections':records,
                'detector_ms':detector_ms, 'total_ms':(time.perf_counter()-start)*1000,
                'cached_detections': cached_detections}


def write_predictions(path, predictions):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    columns = ('image','plate_num','plate_type','confidence')
    with path.open('w',encoding='utf-8',newline='') as stream:
        writer = csv.DictWriter(stream,fieldnames=columns,delimiter=';',lineterminator='\n')
        writer.writeheader()
        writer.writerows({key:row[key] for key in columns} for row in predictions)
