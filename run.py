import argparse
import csv
import glob
import os
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import cv2
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO

from ocr import CRNNRecognizer

DEFAULT_PLATE_MODEL = os.path.join('models', 'yolo11_plate.pt')
FALLBACK_PLATE_MODEL = 'yolo11n.pt'
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
VIDEO_EXTS = {'.mp4', '.mov', '.avi', '.mkv', '.m4v', '.ts', '.webm'}

# Фиксированные параметры постобработки для треков.
TRACK_VOTE_MIN = 2.5
TRACK_LOCK_CONF = 12.0
# Разрешаем выпускать номера с неполным распознаванием (символы '#').
REQUIRE_FULL_PLATE = False


class PaddleEasyAdapter:
    def __init__(self, rec_model_dir: str | None, use_gpu: bool, char_dict: str | None = None):
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as e:
            raise RuntimeError("paddleocr is not installed. Try: pip install paddleocr") from e
        kwargs = dict(use_gpu=use_gpu, det=False, rec=True, cls=False)
        if rec_model_dir:
            kwargs['rec_model_dir'] = rec_model_dir
        if char_dict and os.path.exists(char_dict):
            kwargs['rec_char_dict_path'] = char_dict
        self.ocr = PaddleOCR(**kwargs)

    # Emulate EasyOCR's readtext signature/return
    def readtext(self, img, detail=1, allowlist=None, paragraph=False, **kwargs):
        res = self.ocr.ocr(img, det=False, rec=True, cls=False)
        out = []
        if not res:
            return out
        # res can be nested; flatten to list of (bbox,text,conf)
        seqs = res[0] if isinstance(res, list) and len(res) and isinstance(res[0], list) else res
        for it in seqs:
            # it expected as (bbox, (text, conf)) or ((text, conf)) when det=False
            if isinstance(it, (list, tuple)) and len(it) >= 2 and isinstance(it[1], (list, tuple)):
                text, conf = it[1][0], float(it[1][1])
                out.append(((0, 0, 0, 0), text, conf))
            elif isinstance(it, (list, tuple)) and len(it) == 2 and isinstance(it[0], str):
                # (text, conf)
                text, conf = it[0], float(it[1])
                out.append(((0, 0, 0, 0), text, conf))
        return out

# ========================
# Input helpers
# ========================


def infer_input_type(input_path: str, forced: str = 'auto') -> str:
    if forced != 'auto':
        return forced
    path = Path(input_path)
    if path.is_dir():
        return 'images'
    if glob.has_magic(input_path):
        matches = [Path(p) for p in glob.glob(input_path)]
        if matches and all(m.suffix.lower() in IMAGE_EXTS for m in matches):
            return 'images'
    suffix = path.suffix.lower()
    if suffix in IMAGE_EXTS:
        return 'images'
    if suffix in VIDEO_EXTS:
        return 'video'
    return 'video'


def collect_image_paths(input_path: str) -> List[Path]:
    path = Path(input_path)
    if path.is_dir():
        candidates = sorted(
            (p for p in path.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_EXTS),
            key=lambda p: str(p).lower(),
        )
        return candidates
    if glob.has_magic(input_path):
        matches = [
            Path(p) for p in glob.glob(input_path)
            if Path(p).is_file() and Path(p).suffix.lower() in IMAGE_EXTS
        ]
        return sorted(matches, key=lambda p: str(p).lower())
    if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
        return [path]
    return []


# ========================
# Plate alphabet helpers
# ========================
ALPH = "ABEKMHOPCTYX"  # допустимые латинские литеры для RU номерных знаков
PLATE_RX = re.compile(rf"^[{ALPH}]\d{{3}}[{ALPH}]{{2}}\d{{2,3}}$")

# Кириллица -> латиница для визуально одинаковых букв
CYR_EQ = "\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0423\u0425"  # АВЕКМНОРСТУХ
_kir_map = {c: l for c, l in zip(CYR_EQ, ALPH)}
_kir_map.update({c.lower(): l for c, l in zip(CYR_EQ, ALPH)})
KIR2LAT = str.maketrans(_kir_map)

SIMILAR_FIX = {
    'O':'0','o':'0','I':'1','l':'1','S':'5','Z':'2'
}


def mmssms_from_ms(ts_ms: int) -> str:
    total_centis = int(round(ts_ms / 10.0))
    mm, rem_centis = divmod(total_centis, 6000)
    ss, cs = divmod(rem_centis, 100)
    return f"{mm:02d}:{ss:02d}.{cs:02d}"


def sanitize_text(t: str) -> str:
    if not t:
        return ''
    t = t.translate(KIR2LAT).upper()
    return ''.join(ch for ch in t if ch.isdigit() or ch in ALPH)


def sanitize_text_with_probs(t: str, probs: List[float] | None):
    if not t:
        return '', None
    t_norm = t.translate(KIR2LAT).upper()
    if probs is None:
        return ''.join(ch for ch in t_norm if ch.isdigit() or ch in ALPH), None
    sanitized_chars: List[str] = []
    sanitized_probs: List[float] = []
    for ch, prob in zip(t_norm, probs):
        if ch.isdigit() or ch in ALPH:
            sanitized_chars.append(ch)
            sanitized_probs.append(float(prob))
    return ''.join(sanitized_chars), sanitized_probs


def matches_plate_pattern(t: str, allow_missing: bool = False) -> bool:
    if not t:
        return False
    if allow_missing:
        pattern = pattern_for_len(len(t))
        if not pattern:
            return False
        for ch, kind in zip(t, pattern):
            if ch == '#':
                continue
            if kind == 'D' and not ch.isdigit():
                return False
            if kind == 'L' and ch not in ALPH:
                return False
        return True
    return bool(PLATE_RX.match(t))


def is_valid_plate(t: str) -> bool:
    return matches_plate_pattern(t, allow_missing=False)


def normalize_plate_pattern(text: str) -> str:
    if not text:
        return ''
    L = len(text)
    pattern = pattern_for_len(L)
    if not pattern:
        return text
    out_chars: List[str] = []
    for ch, need in zip(text[:L], pattern):
        if ch == '#':
            out_chars.append('#')
            continue
        if need == 'D':
            if ch.isdigit():
                out_chars.append(ch)
            else:
                out_chars.append(DIG_FROM_LET.get(ch, '#'))
        else:
            if ch in ALPH:
                out_chars.append(ch)
            else:
                out_chars.append(LET_FROM_DIG.get(ch, '#'))
    return ''.join(out_chars)

# Позиционные приведения символов к нужной категории (буква/цифра)
DIG_FROM_LET = {'O':'0','D':'0','Q':'0','I':'1','L':'1','Z':'2','S':'5','B':'8','G':'6','T':'7','A':'4'}
LET_FROM_DIG = {'0':'O','3':'E','4':'A','6':'G','7':'T','8':'B','1':'H'}


def coerce_char_to_kind(ch: str, kind: str) -> str:
    """Жёстко приводим символ к цифре ('D') или букве ('L') набора ALPH, иначе возвращаем '#'."""
    if ch == '#':
        return '#'
    if kind == 'D':
        if ch.isdigit():
            return ch
        return DIG_FROM_LET.get(ch, '#')
    else:  # 'L'
        if ch in ALPH:
            return ch
        return LET_FROM_DIG.get(ch, '#')


def pattern_for_len(L: int):
    # LDDDLLDD + опциональный регион D
    return (['L','D','D','D','L','L','D','D'] + (['D'] if L == 9 else []))


# ========================
# Models
# ========================

def load_models(
    plate_model_path: str,
    device: str,
    ocr_lang: str = 'en',
    ocr_engine: str = 'easy',
    paddle_rec_dir: str | None = None,
    paddle_char_dict: str | None = None,
    crnn_checkpoint: str | None = None,
):
    plate_model = YOLO(plate_model_path)
    if ocr_engine == 'paddle':
        reader = PaddleEasyAdapter(paddle_rec_dir, use_gpu=(device == 'cuda'), char_dict=paddle_char_dict)
    elif ocr_engine == 'crnn':
        if not crnn_checkpoint or not os.path.exists(crnn_checkpoint):
            raise FileNotFoundError(f"CRNN checkpoint not found: {crnn_checkpoint}")
        reader = CRNNRecognizer(crnn_checkpoint, device=device)
    else:
        langs = [s.strip() for s in ocr_lang.split(',') if s.strip()]
        reader = easyocr.Reader(langs or ['en'], gpu=(device == 'cuda'))
    return plate_model, reader


def detect_plate_bboxes(yolo_model: YOLO, frame_bgr, conf_thres: float, iou_thres: float, device: str):
    """Возвращает список (x1,y1,x2,y2,conf). Применяет простой фильтр геометрии для номерной таблички."""
    device_arg = 'cuda:0' if device == 'cuda' else 'cpu'
    results = yolo_model.predict(
        frame_bgr, conf=conf_thres, iou=iou_thres, verbose=False,
        device=device_arg
    )

    bboxes = []  # (x1,y1,x2,y2, conf)
    if len(results):
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy()
            for (x1,y1,x2,y2), cf in zip(xyxy, conf):
                w = max(1.0, x2 - x1); h = max(1.0, y2 - y1)
                ar = w / h
                if 1.6 <= ar <= 6.5 and w >= 50 and h >= 14:
                    bboxes.append((int(x1), int(y1), int(x2), int(y2), float(cf)))
    bboxes.sort(key=lambda x: x[4], reverse=True)
    return bboxes


# ========================
# Frame iteration helpers
# ========================


def iter_video_frames(cap: cv2.VideoCapture, src_fps: float, stride: int) -> Iterator[Tuple[int, int, np.ndarray, Dict[str, object] | None]]:
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        ts_ms = int((frame_idx / max(src_fps, 1e-6)) * 1000.0)
        if (frame_idx % stride) != 0:
            frame_idx += 1
            continue
        yield frame_idx, ts_ms, frame, None
        frame_idx += 1


def iter_image_frames(image_paths: List[Path], frame_interval_ms: float) -> Iterator[Tuple[int, int, np.ndarray, Dict[str, object]]]:
    interval = max(1.0, frame_interval_ms)
    for idx, path in enumerate(image_paths):
        frame = cv2.imread(str(path))
        if frame is None:
            print(f"[WARN] �� 㤠���� ������ �������� �ਬ��: {path}")
            continue
        ts_ms = int(idx * interval)
        yield idx, ts_ms, frame, {'source_path': str(path)}


# ========================
# OCR (с TTA и deskew)
# ========================

def _deskew_estimate(gray):
    edges = cv2.Canny(gray, 50, 120)
    lines = cv2.HoughLines(edges, 1, np.pi/180, 60)
    if lines is None:
        return 0.0
    angs = []
    for rho, theta in lines[:,0]:
        a = (theta*180/np.pi) - 90
        if -20 <= a <= 20:
            angs.append(a)
    return float(np.median(angs)) if angs else 0.0


def _rotate(gray, angle):
    if abs(angle) < 0.5:
        return gray
    h, w = gray.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def ocr_plate(
    reader,
    frame_bgr: np.ndarray,
    bbox,
    max_w=480,
    max_h=160,
    margin: float = 0.06,
    char_conf_thr: float = 0.0,
    char_conf_thr_digits: float | None = None,
    char_conf_thr_letters: float | None = None,
    return_crop: bool = False,
):
    x1, y1, x2, y2, det_conf = bbox
    H, W = frame_bgr.shape[:2]
    # expand bbox a bit to avoid tight crops cutting characters
    mw = max(1, x2 - x1)
    mh = max(1, y2 - y1)
    dx = int(mw * margin)
    dy = int(mh * margin)
    x1 = max(0, x1 - dx); y1 = max(0, y1 - dy); x2 = min(W-1, x2 + dx); y2 = min(H-1, y2 + dy)
    if x2 <= x1 or y2 <= y1:
        return '', 0.0, {}

    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return '', 0.0, {}

    crop_meta: Dict[str, object] = {}
    if return_crop:
        crop_meta = {'crop_bgr': crop, 'crop_bbox': (x1, y1, x2, y2)}

    # апскейл
    ch, cw = crop.shape[:2]
    scale = max(max_w / max(1, cw), max_h / max(1, ch))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(cw*scale), int(ch*scale)), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    # deskew
    angle = _deskew_estimate(gray)
    gray = _rotate(gray, -angle)

    use_crnn = isinstance(reader, CRNNRecognizer)

    if use_crnn:
        variants = [gray]
    else:
        # TTA препроцессы
        def tta_imgs(g):
            outs = [g]
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            outs.append(clahe.apply(g))
            outs.append(cv2.adaptiveThreshold(g,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,25,9))
            k = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)
            outs.append(cv2.filter2D(g, -1, k))
            return outs

        variants = []
        base = tta_imgs(gray)
        inv = [255 - im for im in base]
        variants.extend(base + inv)

    best_text, best_conf = '', 0.0
    best_meta: Dict[str, object] = {}
    best_score = (0, 0, 0.0)
    L_votes = None
    for img in variants:
        try:
            res = reader.readtext(
                img,
                detail=1,
                allowlist=ALPH + '0123456789',
                paragraph=False,
                decoder='beamsearch',
                beamWidth=5,
            )
        except TypeError:
            # older EasyOCR without decoder/beamWidth params
            res = reader.readtext(img, detail=1, allowlist=ALPH + '0123456789', paragraph=False)
        if not res:
            continue
        candidate = None
        for item in res:
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                _, raw_text, conf = item[:3]
                extras = item[3] if len(item) >= 4 and isinstance(item[3], dict) else {}
                char_probs = extras.get('char_probs')
                sanitized_text, sanitized_probs = sanitize_text_with_probs(raw_text, char_probs)
                if sanitized_probs:
                    adjusted_chars = []
                    adjusted_probs = []
                    for ch, prob in zip(sanitized_text, sanitized_probs):
                        thr = char_conf_thr
                        if ch.isdigit() and char_conf_thr_digits is not None:
                            thr = char_conf_thr_digits
                        elif ch in ALPH and char_conf_thr_letters is not None:
                            thr = char_conf_thr_letters
                        if prob < thr:
                            adjusted_chars.append('#')
                            adjusted_probs.append(prob)
                        else:
                            adjusted_chars.append(ch)
                            adjusted_probs.append(prob)
                    sanitized_text = ''.join(adjusted_chars)
                    sanitized_probs = adjusted_probs
                else:
                    sanitized_text = sanitize_text(raw_text)
                    sanitized_probs = None
                candidate = (sanitized_text, float(conf), sanitized_probs)
                break
        if not candidate:
            continue
        text, conf, sanitized_probs = candidate
        if not text:
            continue
        full_ok = matches_plate_pattern(text, allow_missing=False)
        partial_ok = matches_plate_pattern(text, allow_missing=True)
        cand_score = (int(full_ok), int(partial_ok), conf)
        if cand_score > best_score:
            best_score = cand_score
            best_text, best_conf = text, conf
            best_meta = {'char_probs': sanitized_probs} if sanitized_probs else {}
            if crop_meta:
                best_meta = {**crop_meta, **best_meta} if best_meta else dict(crop_meta)
        L = 9 if len(text) >= 9 else 8
        if L_votes is None or len(L_votes) != L:
            L_votes = [Counter() for _ in range(L)]
        norm = (text[:L]).ljust(L, '#')
        if sanitized_probs:
            trimmed = sanitized_probs[:L]
            prob_series = trimmed + [None] * (L - len(trimmed))
        else:
            prob_series = [None] * len(norm)
        for i, ch in enumerate(norm):
            if ch == '#':
                continue
            weight = conf
            if i < len(prob_series) and prob_series[i] is not None:
                weight *= float(prob_series[i])
            L_votes[i][ch] += weight

    if L_votes:
        agg = ''.join((c.most_common(1)[0][0] if c else '#') for c in L_votes)
        if matches_plate_pattern(agg, allow_missing=True):
            denom = max(1, sum(1 for c in L_votes if c))
            agg_conf = max(
                best_conf,
                sum((c.most_common(1)[0][1] if c else 0.0) for c in L_votes if c) / denom,
            )
            meta_out = dict(crop_meta) if crop_meta else {}
            if best_meta:
                meta_out = {**meta_out, **best_meta} if meta_out else dict(best_meta)
            return agg, agg_conf, meta_out
    if crop_meta:
        if best_meta:
            meta_out = {**crop_meta, **best_meta}
        else:
            meta_out = dict(crop_meta)
        return best_text, best_conf, meta_out
    return best_text, best_conf, best_meta


# ========================
# Tracking
# ========================
@dataclass
class Track:
    id: int
    bbox: tuple
    last_seen_frame: int
    start_ms: int
    last_seen_ms: int
    best_plate: str = ''
    best_conf: float = 0.0
    best_ms: int = 0
    votes: list | None = None
    first_ideal_ms: int = -1
    emitted: bool = False
    best_crop: np.ndarray | None = field(default=None, repr=False)
    best_source: str | None = None
    locked: bool = False


class SimpleTracker:
    def __init__(self, iou_match: float = 0.3, max_misses: int = 20, lock_conf_thresh: float = 12.0):
        self.iou_match = iou_match
        self.max_misses = max_misses
        self.lock_conf_thresh = lock_conf_thresh
        self.tracks: Dict[int, Track] = {}
        self._next_id = 1

    @staticmethod
    def iou(a, b) -> float:
        ax1, ay1, ax2, ay2 = a[:4]
        bx1, by1, bx2, by2 = b[:4]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        iw, ih = max(0, ix2-ix1), max(0, iy2-iy1)
        inter = iw*ih
        area_a = max(0, (ax2-ax1)*(ay2-ay1))
        area_b = max(0, (bx2-bx1)*(by2-by1))
        union = area_a + area_b - inter + 1e-6
        return inter/union

    def update(
        self,
        detections: List[Tuple[int, int, int, int, float]],
        frame_idx: int,
        ts_ms: int,
        ocr_fn,
        reader,
        frame_bgr,
        ocr_every_n: int = 1,
        frame_meta: Dict[str, object] | None = None,
    ) -> List[Track]:
        assigned_det = set()
        use_kind_coercion = not isinstance(reader, CRNNRecognizer)

        def save_best_artifacts(tr: Track, meta_dict: Dict[str, object] | None):
            if meta_dict and 'crop_bgr' in meta_dict:
                crop_obj = meta_dict['crop_bgr']
                if isinstance(crop_obj, np.ndarray):
                    tr.best_crop = np.ascontiguousarray(crop_obj)
            if isinstance(frame_meta, dict):
                source = frame_meta.get('source_path')
                if source:
                    tr.best_source = str(source)

        def update_votes(tr: Track, text: str, ocr_conf: float, det_conf: float, char_probs: List[float] | None):
            if not text or tr.locked:
                return False
            L = 9 if len(text) >= 9 else 8
            if tr.votes is None or len(tr.votes) != L:
                tr.votes = [Counter() for _ in range(L)]
            norm = (text[:L]).ljust(L, '#')
            probs = []
            if char_probs:
                trimmed = char_probs[:L]
                probs = trimmed + [None] * (L - len(trimmed))
            else:
                probs = [None] * len(norm)
            w_base = max(0.1, det_conf * float(ocr_conf))
            for i, ch in enumerate(norm):
                if ch == '#':
                    continue
                ch2 = coerce_char_to_kind(ch, 'D') if (use_kind_coercion and ch.isdigit()) else ch
                if ch2 == '#':
                    continue
                weight = w_base
                if i < len(probs) and probs[i] is not None:
                    weight *= float(probs[i])
                tr.votes[i][ch2] += weight
            agg = ''.join((c.most_common(1)[0][0] if c else '#') for c in tr.votes)
            agg_norm = normalize_plate_pattern(agg)
            if agg_norm and agg_norm.count('#') <= agg.count('#'):
                agg = agg_norm
            conf_sum = sum((c.most_common(1)[0][1] if c else 0.0) for c in tr.votes)
            full_now = matches_plate_pattern(agg, allow_missing=False)
            partial_now = matches_plate_pattern(agg, allow_missing=True)
            if full_now and tr.first_ideal_ms < 0:
                tr.first_ideal_ms = ts_ms
            best_full = matches_plate_pattern(tr.best_plate, allow_missing=False)
            best_partial = matches_plate_pattern(tr.best_plate, allow_missing=True)
            improved = (int(full_now), int(partial_now), conf_sum) > (int(best_full), int(best_partial), tr.best_conf)
            if improved:
                tr.best_plate = agg
                tr.best_conf = conf_sum
                tr.best_ms = ts_ms
                lock_now = False
                if full_now and '#' not in agg:
                    lock_now = True
                elif partial_now and agg.count('#') <= 2 and conf_sum >= self.lock_conf_thresh:
                    lock_now = True
                if lock_now:
                    tr.locked = True
            return improved

        # 1) обновляем существующие треки
        for tid, tr in list(self.tracks.items()):
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(detections):
                if j in assigned_det:
                    continue
                iou = self.iou(tr.bbox, d)
                if iou > best_iou:
                    best_iou, best_j = iou, j

            if best_iou >= self.iou_match:
                det = detections[best_j]
                tr.bbox = det[:4]
                det_conf = float(det[4])
                tr.last_seen_frame = frame_idx
                tr.last_seen_ms = ts_ms
                assigned_det.add(best_j)

                if (frame_idx % ocr_every_n) == 0:
                    text_raw, ocr_conf, meta = ocr_fn(reader, frame_bgr, (*tr.bbox, det_conf))
                    if text_raw:
                        meta_dict = meta if isinstance(meta, dict) else {}
                        char_probs = meta_dict.get('char_probs') if meta_dict else None
                        improved = update_votes(tr, text_raw, ocr_conf, det_conf, char_probs)
                        if improved:
                            save_best_artifacts(tr, meta_dict)

        # 2) создаём новые треки
        for j, d in enumerate(detections):
            if j in assigned_det:
                continue
            tid = self._next_id; self._next_id += 1
            det_conf = float(d[4])
            tr = Track(id=tid, bbox=d[:4], last_seen_frame=frame_idx, start_ms=ts_ms, last_seen_ms=ts_ms)
            text_raw, ocr_conf, meta = ocr_fn(reader, frame_bgr, (*d[:4], det_conf))
            if text_raw:
                meta_dict = meta if isinstance(meta, dict) else {}
                char_probs = meta_dict.get('char_probs') if meta_dict else None
                improved = update_votes(tr, text_raw, ocr_conf, det_conf, char_probs)
                if improved:
                    save_best_artifacts(tr, meta_dict)
            self.tracks[tid] = tr

        # 3) закрываем «потерянные» треки
        finished, to_delete = [], []
        for tid, tr in self.tracks.items():
            if frame_idx - tr.last_seen_frame > self.max_misses:
                finished.append(tr)
                to_delete.append(tid)
        for tid in to_delete:
            del self.tracks[tid]
        return finished


class CropSaver:
    def __init__(self, root_dir: str, min_conf: float = 1.5, max_items: int = 0):
        self.root = Path(root_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.min_conf = float(min_conf)
        self.max_items = max(0, int(max_items))
        self.saved = 0
        self.manifest_path = self.root / 'manifest.csv'
        if self.manifest_path.exists():
            self.manifest_initialized = self.manifest_path.stat().st_size > 0
        else:
            self.manifest_initialized = False

    @staticmethod
    def _sanitize_plate(plate: str) -> str:
        cleaned = ''.join(ch if ch.isalnum() else '_' for ch in plate)
        cleaned = cleaned.strip('_')
        return cleaned or 'unknown'

    def _next_filename(self, row: Dict[str, str], track_id: int, plate: str) -> Path:
        base = f"{row['time_detect'].replace(':','-')}_{self._sanitize_plate(plate)}_{track_id:04d}"
        out_path = self.root / f"{base}.jpg"
        suffix = 1
        while out_path.exists():
            out_path = self.root / f"{base}_{suffix:02d}.jpg"
            suffix += 1
        return out_path

    def _append_manifest(self, filename: str, row: Dict[str, str], conf: float, source_ref: str | None):
        headers = ['filename', 'plate_num', 'conf', 'time_start', 'time_detect', 'time_end', 'source']
        line = [
            filename,
            row.get('plate_num', ''),
            f"{conf:.3f}",
            row.get('time_start', ''),
            row.get('time_detect', ''),
            row.get('time_end', ''),
            source_ref or '',
        ]
        with self.manifest_path.open('a', newline='', encoding='utf-8') as fh:
            writer = csv.writer(fh, delimiter=';')
            if not self.manifest_initialized:
                writer.writerow(headers)
                self.manifest_initialized = True
            writer.writerow(line)

    def save(self, track: Track, row: Dict[str, str], source_ref: str | None = None):
        if track.best_crop is None:
            return
        if track.best_conf < self.min_conf:
            return
        if self.max_items and self.saved >= self.max_items:
            return
        out_path = self._next_filename(row, track.id, track.best_plate or 'unknown')
        crop = np.ascontiguousarray(track.best_crop)
        ok = cv2.imwrite(str(out_path), crop)
        if not ok:
            print(f"[WARN] �� 㤠���� ���������� ����� ᮧ���: {out_path}")
            return
        self._append_manifest(out_path.name, row, track.best_conf, source_ref)
        self.saved += 1


class LatencyMeter:
    def __init__(self):
        self.count = 0
        self.total_ms = 0.0
        self.max_ms = 0.0

    def record(self, elapsed_ms: float):
        self.count += 1
        self.total_ms += elapsed_ms
        if elapsed_ms > self.max_ms:
            self.max_ms = elapsed_ms

    def summary(self) -> Dict[str, float | int]:
        if not self.count:
            return {}
        avg_ms = self.total_ms / self.count
        return {'avg_ms': avg_ms, 'max_ms': self.max_ms, 'count': self.count}


class EmissionLimiter:
    def __init__(self, min_interval_ms: int = 200):
        self.min_interval_ms = max(0, int(min_interval_ms))
        self._state: Dict[str, Dict[str, object]] = {}

    @staticmethod
    def _plate_key(plate_text: str) -> str:
        if not plate_text:
            return ''
        norm = normalize_plate_pattern(plate_text)
        return norm if norm else plate_text

    @staticmethod
    def _missing_chars(plate_text: str) -> int:
        return plate_text.count('#') if plate_text else len(plate_text)

    def allow(self, row: Dict[str, str], out_rows: List[Dict[str, str]]) -> Tuple[bool, bool]:
        detect_ms = row.get('time_ms')
        if detect_ms is None:
            return True, False
        plate = row.get('plate_num', '')
        key = self._plate_key(plate)
        if not key:
            return True, False
        info = self._state.get(key)
        if info is None:
            return True, False
        prev_time = int(info.get('time_ms', -1))
        if prev_time < 0 or (int(detect_ms) - prev_time) >= self.min_interval_ms:
            return True, False
        prev_idx = int(info.get('index', -1))
        prev_plate = info.get('plate', '')
        updated_existing = False
        if 0 <= prev_idx < len(out_rows):
            prev_missing = self._missing_chars(str(prev_plate))
            new_missing = self._missing_chars(plate)
            if new_missing < prev_missing:
                out_rows[prev_idx]['plate_num'] = plate
                info['plate'] = plate
                updated_existing = True
        return False, updated_existing

    def register(self, row: Dict[str, str], index: int):
        detect_ms = int(row.get('time_ms', -1))
        plate = row.get('plate_num', '')
        key = self._plate_key(plate)
        if not key:
            return
        self._state[key] = {'time_ms': detect_ms, 'index': index, 'plate': plate}


def build_track_row(tr: Track, plate_text: str) -> Dict[str, str]:
    detect_ms = tr.first_ideal_ms if tr.first_ideal_ms >= 0 else (tr.best_ms if tr.best_ms > 0 else tr.start_ms)
    return {
        'time_start': mmssms_from_ms(tr.start_ms),
        'time_detect': mmssms_from_ms(detect_ms),
        'time_end': mmssms_from_ms(tr.last_seen_ms),
        'plate_num': plate_text,
        'is_vehicle': 1,
        'time_ms': detect_ms,
    }


def emit_track(
    tr: Track,
    plate_text: str,
    out_rows: List[Dict[str, str]],
    crop_saver: CropSaver | None,
    default_source: str | None,
    limiter: EmissionLimiter | None = None,
    suffix: str = '',
):
    row = build_track_row(tr, plate_text)
    tr.emitted = True
    if limiter:
        should_emit, updated = limiter.allow(row, out_rows)
        if not should_emit:
            if updated:
                print(f"[UPDATE] track {tr.id}{suffix} -> plate refined within {limiter.min_interval_ms}ms window: {plate_text}")
            else:
                print(f"[SKIP] track {tr.id}{suffix} suppressed by {limiter.min_interval_ms}ms window: {plate_text}")
            return
    out_rows.append(row)
    idx = len(out_rows) - 1
    if limiter:
        limiter.register(row, idx)
    print(f"EMIT(track {tr.id}{suffix}) {row['time_start']}->{row['time_end']} {plate_text} conf={tr.best_conf:.2f}")
    if crop_saver:
        crop_saver.save(tr, row, tr.best_source or default_source)


def process_frames(
    frame_iter: Iterable[Tuple[int, int, np.ndarray, Dict[str, object] | None]],
    tracker: SimpleTracker,
    plate_model: YOLO,
    reader,
    ocr_fn,
    args,
    device: str,
    crop_saver: CropSaver | None,
    default_source: str | None,
    emission_limiter: EmissionLimiter | None,
    latency_meter: LatencyMeter | None = None,
) -> List[Dict[str, str]]:
    out_rows: List[Dict[str, str]] = []
    vote_min = TRACK_VOTE_MIN
    require_full = REQUIRE_FULL_PLATE
    for frame_idx, ts_ms, frame, meta in frame_iter:
        frame_t0 = time.perf_counter() if latency_meter else None
        dets = detect_plate_bboxes(plate_model, frame, conf_thres=args.conf, iou_thres=args.iou, device=device)

        if args.show:
            vis = frame.copy()
            for (x1, y1, x2, y2, cf) in dets:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(vis, f"{cf:.2f}", (x1, max(0, y1-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1, cv2.LINE_AA)
            for tr in tracker.tracks.values():
                x1, y1, x2, y2 = tr.bbox
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 128, 0), 1)
                if tr.best_plate:
                    cv2.putText(vis, tr.best_plate, (x1, min(vis.shape[0]-5, y2+16)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50,200,255), 2, cv2.LINE_AA)
            cv2.imshow('RU ANPR', vis)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        finished = tracker.update(
            dets,
            frame_idx,
            ts_ms,
            ocr_fn,
            reader,
            frame,
            ocr_every_n=args.ocr_every_n,
            frame_meta=meta,
        )

        for tr in finished:
            if tr.emitted:
                continue
            if tr.best_conf < vote_min:
                continue
            if require_full:
                if not tr.best_plate or "#" in tr.best_plate:
                    continue
                if not matches_plate_pattern(tr.best_plate, allow_missing=False):
                    continue
            else:
                if not matches_plate_pattern(tr.best_plate, allow_missing=True):
                    continue
            emit_track(
                tr,
                tr.best_plate,
                out_rows,
                crop_saver,
                default_source,
                limiter=emission_limiter,
            )
        if latency_meter and frame_t0 is not None:
            latency_meter.record((time.perf_counter() - frame_t0) * 1000.0)

    for tr in list(tracker.tracks.values()):
        if tr.emitted:
            continue
        plate_out = tr.best_plate if tr.best_plate else '########'
        if require_full:
            if not plate_out or '#' in plate_out:
                continue
            if not matches_plate_pattern(plate_out, allow_missing=False):
                continue
        else:
            if not matches_plate_pattern(plate_out, allow_missing=True):
                continue
        emit_track(
            tr,
            plate_out,
            out_rows,
            crop_saver,
            default_source,
            limiter=emission_limiter,
            suffix=', end',
        )
    return out_rows


# ========================
# Main
# ========================

def main():
    ap = argparse.ArgumentParser(description='RU ANPR MVP (per-track best emit)')
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default='out.csv')
    ap.add_argument('--input_type', choices=['auto', 'video', 'images'], default='auto', help='Форсировать тип входа (auto/video/images)')
    ap.add_argument('--model_plate', default=DEFAULT_PLATE_MODEL, help='Path to YOLO plate detector weights (default: models/yolo11_plate.pt)')
    ap.add_argument('--device', default='auto', choices=['auto','cuda','cpu'])
    ap.add_argument('--target_fps', type=float, default=15.0)
    ap.add_argument('--conf', type=float, default=0.45)
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--match_iou', type=float, default=0.3, help='IoU для трекера')
    ap.add_argument('--max_misses', type=int, default=12, help='через сколько пропусков закрывать трек')
    ap.add_argument('--ocr_every_n', type=int, default=2, help='делать OCR не на каждом кадре')
    ap.add_argument('--show', action='store_true', help='показывать окно с детекциями')
    ap.add_argument('--ocr_lang', type=str, default='en')
    ap.add_argument('--ocr_engine', choices=['easy','paddle','crnn'], default='easy')
    ap.add_argument('--paddle_rec_dir', type=str, default=None)
    ap.add_argument('--paddle_char_dict', type=str, default='configs/ocr_chars.txt')
    ap.add_argument('--crnn_checkpoint', type=str, default='models/ocr_crnn.pt', help='Path to custom CRNN checkpoint')
    ap.add_argument('--char_conf_thr', type=float, default=0.0, help='Базовый порог уверенности символа для маски в #')
    ap.add_argument('--char_conf_thr_digits', type=float, default=None, help='Порог для цифр (если не задан, используется base)')
    ap.add_argument('--char_conf_thr_letters', type=float, default=None, help='Порог для букв (если не задан, используется base)')
    ap.add_argument('--save_crops_dir', type=str, default=None, help='Каталог для экспорта кропов номеров (JPEG + manifest.csv)')
    ap.add_argument('--save_crops_min_conf', type=float, default=1.5, help='Минимальная уверенность для сохранения кропа')
    ap.add_argument('--save_crops_limit', type=int, default=0, help='Ограничение по числу кропов (0 = без лимита)')
    ap.add_argument('--emit_min_interval_ms', type=int, default=200, help='Minimum interval between identical plate emissions (ms)')
    ap.add_argument('--profile_latency', action='store_true', help='Collect and print per-frame latency statistics')
    args = ap.parse_args()
    if not os.path.exists(args.model_plate):
        candidate = FALLBACK_PLATE_MODEL if os.path.exists(FALLBACK_PLATE_MODEL) else None
        if candidate:
            print(f"[WARN] plate model '{args.model_plate}' not found, falling back to '{candidate}'")
            args.model_plate = candidate
        else:
            print(f"[ERR] plate model '{args.model_plate}' not found and fallback '{FALLBACK_PLATE_MODEL}' is absent")
            sys.exit(1)

    device = args.device
    try:
        import torch
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        elif device == 'cuda' and not torch.cuda.is_available():
            print('CUDA not available -> falling back to CPU')
            device = 'cpu'
    except Exception:
        device = 'cpu'

    plate_model, reader = load_models(
        args.model_plate,
        device,
        ocr_lang=args.ocr_lang,
        ocr_engine=args.ocr_engine,
        paddle_rec_dir=args.paddle_rec_dir,
        paddle_char_dict=args.paddle_char_dict,
        crnn_checkpoint=args.crnn_checkpoint,
    )
    need_crop_meta = bool(args.save_crops_dir)
    ocr_fn = partial(
        ocr_plate,
        char_conf_thr=args.char_conf_thr,
        char_conf_thr_digits=args.char_conf_thr_digits,
        char_conf_thr_letters=args.char_conf_thr_letters,
        return_crop=need_crop_meta,
    )
    if args.show:
        cv2.namedWindow('RU ANPR', cv2.WINDOW_NORMAL)

    crop_saver = CropSaver(
        args.save_crops_dir,
        min_conf=args.save_crops_min_conf,
        max_items=args.save_crops_limit,
    ) if args.save_crops_dir else None

    emission_limiter = EmissionLimiter(args.emit_min_interval_ms) if (args.emit_min_interval_ms and args.emit_min_interval_ms > 0) else None
    latency_meter = LatencyMeter() if args.profile_latency else None

    tracker = SimpleTracker(iou_match=args.match_iou, max_misses=args.max_misses, lock_conf_thresh=TRACK_LOCK_CONF)
    input_kind = infer_input_type(args.input, args.input_type)

    if input_kind == 'images':
        image_paths = collect_image_paths(args.input)
        if not image_paths:
            print(f"[ERR] Не удалось найти изображения по пути: {args.input}")
            sys.exit(1)
        interval_ms = 1000.0 / max(1e-3, args.target_fps)
        print(f"[INFO] image_seq count={len(image_paths)}, interval_ms={interval_ms:.1f}, device={device}")
        frame_iter = iter_image_frames(image_paths, interval_ms)
        out_rows = process_frames(
            frame_iter,
            tracker,
            plate_model,
            reader,
            ocr_fn,
            args,
            device,
            crop_saver,
            default_source=args.input,
            emission_limiter=emission_limiter,
            latency_meter=latency_meter,
        )
    else:
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            print(f"[ERR] Не удалось открыть видео: {args.input}")
            sys.exit(1)
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(src_fps / max(1e-6, args.target_fps))))
        print(f"[INFO] src_fps={src_fps:.2f}, target_fps={args.target_fps}, stride={stride}, device={device}")
        frame_iter = iter_video_frames(cap, src_fps, stride)
        try:
            out_rows = process_frames(
                frame_iter,
                tracker,
                plate_model,
                reader,
                ocr_fn,
                args,
                device,
                crop_saver,
                default_source=args.input,
                emission_limiter=emission_limiter,
                latency_meter=latency_meter,
            )
        finally:
            cap.release()

    if args.show:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    df = pd.DataFrame(out_rows)
    if not df.empty:
        if 'time_detect' in df.columns:
            df_out = pd.DataFrame({'time': df['time_detect'], 'plate_num': df['plate_num']})
        elif {'time', 'plate_num'}.issubset(df.columns):
            df_out = df[['time', 'plate_num']].copy()
        else:
            df_out = pd.DataFrame({
                'time': df.get('time', pd.Series(dtype=str)),
                'plate_num': df.get('plate_num', pd.Series(dtype=str)),
            })
    else:
        df_out = pd.DataFrame(columns=['time', 'plate_num'])
    df_out.to_csv(args.out, index=False, sep=';')
    print(f"[OK] ��������: {args.out} ({len(df_out)} �����)")
    if latency_meter and latency_meter.count:
        stats = latency_meter.summary()
        if stats:
            print(f"[INFO] latency_ms avg={stats['avg_ms']:.1f} max={stats['max_ms']:.1f} (frames={int(stats['count'])})")
    if crop_saver:
        print(f"[INFO] Plate crops saved: {crop_saver.saved} -> {crop_saver.root}")


if __name__ == '__main__':
    main()
