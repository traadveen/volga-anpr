"""Label-free two-row preprocessing of rectified type1a plate crops.

The caller supplies the plate type. This module does not classify layouts,
rectify images, repair OCR strings, or accept ground-truth character boxes.
Unreliable row evidence falls back to the unchanged whole crop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

MODES = ('whole_plate', 'split_and_stitch', 'split_and_ocr_separately')
METHODS = ('fixed_ratio', 'horizontal_projection', 'connected_components', 'geometry')


@dataclass(frozen=True)
class MultilineConfig:
    fixed_ratio: float = .5
    search_min: float = .38
    search_max: float = .62
    border_x: float = .045
    border_y: float = .03
    min_component_height: float = .12
    max_component_height: float = .55
    min_component_width: float = .012
    min_row_components: int = 2
    min_gap: float = .02
    row_padding: float = .035
    stitch_height: int = 64
    stitch_gap: int = 10
    stitch_padding: int = 4

    def __post_init__(self):
        if not 0 < self.search_min < self.fixed_ratio < self.search_max < 1:
            raise ValueError('Split ratios must be ordered inside (0,1)')
        if not 0 <= self.border_x < .2 or not 0 <= self.border_y < .2:
            raise ValueError('Invalid border margins')
        if not 0 < self.min_component_height < self.max_component_height < 1:
            raise ValueError('Invalid component heights')
        if not 0 < self.min_component_width < .2 or not 0 <= self.row_padding < .2:
            raise ValueError('Invalid widths/padding')
        if not 0 < self.min_gap < .2 or self.min_row_components < 2:
            raise ValueError('Invalid row evidence thresholds')
        for key in ('stitch_height', 'stitch_gap', 'stitch_padding', 'min_row_components'):
            if type(getattr(self, key)) is not int:
                raise ValueError(f'{key} must be an integer')
        if self.stitch_height < 16 or self.stitch_gap < 0 or self.stitch_padding < 0:
            raise ValueError('Invalid stitch dimensions')

    @classmethod
    def load(cls, path: str | Path):
        return cls(**json.loads(Path(path).read_text(encoding='utf-8')))


@dataclass(frozen=True)
class SplitResult:
    requested_method: str
    used_method: str
    cut_y: int | None
    reliable: bool
    confidence: float  # structural score, not calibrated OCR probability
    fallback: tuple[str, ...] = ()


@dataclass
class PreparedPlate:
    requested_mode: str
    effective_mode: str
    split: SplitResult | None
    inputs: tuple[np.ndarray, ...]
    row_boxes: tuple[tuple[int, int, int, int], ...] = ()  # exclusive xyxy in input
    # Row crop -> stitched image homographies, for geometric verification only.
    stitch_transforms: tuple[np.ndarray, ...] = ()

    def diagnostics(self):
        return {'requested_mode': self.requested_mode, 'effective_mode': self.effective_mode,
                'split': asdict(self.split) if self.split else None,
                'row_boxes': self.row_boxes,
                'input_shapes': [list(im.shape) for im in self.inputs],
                'stitch_transforms': [m.tolist() for m in self.stitch_transforms]}


def _gray(image):
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError('Expected uint8 numpy image')
    if image.ndim not in (2, 3) or (image.ndim == 3 and image.shape[2] != 3):
        raise ValueError('Expected gray or BGR image')
    if min(image.shape[:2]) < 8:
        raise ValueError('Crop is too small or empty')
    return image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def _evidence(image, config):
    gray = _gray(image)
    h, w = gray.shape
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    mx, my = max(1, round(w*config.border_x)), max(1, round(h*config.border_y))
    mask[:my] = mask[-my:] = 0
    mask[:, :mx] = mask[:, -mx:] = 0
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    components = []
    for x, y, cw, ch, area in stats[1:count]:
        if (config.min_component_height*h <= ch <= config.max_component_height*h
                and cw >= config.min_component_width*w and cw <= .28*w
                and area >= .001*h*w and cw/ch > .06
                and x > mx and x+cw < w-mx and y > my and y+ch < h-my):
            components.append((int(x), int(y), int(x+cw), int(y+ch)))
    return mask, components


def _groups(components, h, config):
    if len(components) < 2*config.min_row_components:
        return None
    centers = np.array([(b[1]+b[3])/2 for b in components])
    lo, hi = np.quantile(centers, [.25, .75])
    for _ in range(12):
        top = np.abs(centers-lo) <= np.abs(centers-hi)
        if top.all() or not top.any():
            return None
        new_lo, new_hi = float(np.median(centers[top])), float(np.median(centers[~top]))
        if new_lo == lo and new_hi == hi:
            break
        lo, hi = new_lo, new_hi
    a = [b for b, flag in zip(components, top) if flag]
    b = [b for b, flag in zip(components, top) if not flag]
    if min(len(a), len(b)) < config.min_row_components or hi-lo < .22*h:
        return None
    end, start = max(x[3] for x in a), min(x[1] for x in b)
    if start-end < config.min_gap*h:
        return None
    return a, b, end, start


def find_split(image, method='horizontal_projection', config=None):
    config = config or MultilineConfig()
    if method not in METHODS:
        raise ValueError(f'Unknown split method: {method}')
    mask, components = _evidence(image, config)
    h, w = mask.shape
    groups = _groups(components, h, config)
    # Require two independent bands even for fixed baseline: no forced split of
    # a blank, single-row, or merged/occluded crop.
    if groups is None:
        return SplitResult(method, 'whole_plate', None, False, 0., ('insufficient_separated_row_evidence',))
    a, b, end, start = groups
    low, high = round(h*config.search_min), round(h*config.search_max)
    valid_low, valid_high = max(low, end), min(high, start)
    if valid_high <= valid_low:
        return SplitResult(method, 'whole_plate', None, False, 0., ('row_gap_outside_search_window',))
    density = np.count_nonzero(mask, axis=1).astype(float)
    k = max(1, round(.025*h))
    smooth = np.convolve(density, np.ones(k)/k, mode='same')
    candidates = np.arange(valid_low, valid_high+1)
    # The valley, then distance from center, then y make tie handling stable.
    projection = int(min(candidates, key=lambda y: (smooth[y], abs(y-h*.5), y)))
    contrast = 1-float(smooth[projection]/max(1., np.percentile(smooth, 80)))
    fallbacks = []
    used = method
    if method == 'fixed_ratio':
        cut = round(h*config.fixed_ratio)
        if not end <= cut <= start:
            # Deliberately keep the fixed baseline fixed; reject unsafe cuts.
            return SplitResult(method, 'whole_plate', None, False, 0., ('fixed_cut_crosses_characters',))
    elif method == 'connected_components':
        cut = round((end+start)/2)
    elif method == 'geometry':
        # A rectified type1a has a vertical divider beside its region field.
        vertical = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                    np.ones((max(3, round(.30*h)), 1), np.uint8))
        n, _, stats, _ = cv2.connectedComponentsWithStats(vertical, 8)
        dividers = [(int(y), int(ch)) for x,y,cw,ch,area in stats[1:n]
                    if .42*w < x < .64*w and cw < .04*w and .35*h < y < .65*h
                    and y+ch > .80*h and ch > .28*h]
        if 1.2 < w/h < 2.5 and dividers:
            proposed = round(min(dividers)[0]-.035*h)
            cut = min(max(proposed, valid_low), valid_high)
        else:
            cut = projection
            used = 'horizontal_projection'
            fallbacks.append('region_divider_not_found')
    else:
        cut = projection
    # Main connected-component envelopes are a conservative cross-check. They
    # do not supply any label characters or expected character count.
    if not end <= cut <= start:
        cut, used = projection, 'horizontal_projection'
        fallbacks.append('unsafe_candidate_cut')
    return SplitResult(method, used, int(cut), True, max(0., min(1., contrast)), tuple(fallbacks))


def _row_boxes(image, cut, config):
    _, components = _evidence(image, config)
    h, w = image.shape[:2]
    groups = _groups(components, h, config)
    if groups is None:
        raise ValueError('Row evidence changed unexpectedly')
    pad = max(1, round(h*config.row_padding))
    # Preserve full horizontal extent: narrow region 1s and disconnected ink
    # must survive even if a component was filtered from split evidence.
    boxes = []
    for group, y0, y1 in ((groups[0], 0, cut), (groups[1], cut, h)):
        top = max(y0, min(b[1] for b in group)-pad)
        bottom = min(y1, max(b[3] for b in group)+pad)
        boxes.append((0, top, w, bottom))
    return tuple(boxes)


def stitch_rows(rows, config=None):
    config = config or MultilineConfig()
    if len(rows) != 2:
        raise ValueError('Exactly two rows required')
    if rows[0].ndim != rows[1].ndim:
        raise ValueError('Rows must have matching channels')
    resized, sizes = [], []
    for row in rows:
        _gray(row)
        height, width = row.shape[:2]
        rw = max(1, round(width*config.stitch_height/height))
        resized.append(cv2.resize(row, (rw, config.stitch_height), interpolation=cv2.INTER_LINEAR))
        sizes.append((rw, height, width))
    p = config.stitch_padding
    shape = (config.stitch_height+2*p, sum(s[0] for s in sizes)+config.stitch_gap+2*p)
    if rows[0].ndim == 3:
        shape += (3,)
    background = np.median(np.concatenate([r.reshape(-1, *r.shape[2:]) for r in rows]), axis=0)
    canvas = np.empty(shape, dtype=np.uint8)
    canvas[...] = background
    transforms = []
    x = p
    for row, (rw, rh, ow) in zip(resized, sizes):
        canvas[p:p+config.stitch_height, x:x+rw] = row
        transforms.append(np.array([[rw/ow,0,x],[0,config.stitch_height/rh,p],[0,0,1]], dtype=float))
        x += rw+config.stitch_gap
    return canvas, tuple(transforms)


def prepare_multiline(image, plate_type='type1a', mode='split_and_stitch',
                      method='horizontal_projection', config=None):
    config = config or MultilineConfig()
    _gray(image)
    if mode not in MODES or method not in METHODS:
        raise ValueError('Unknown multiline mode/method')
    if plate_type not in ('type1', 'type1a', 'type1b', 'other'):
        raise ValueError('Unknown plate type')
    if mode == 'whole_plate' or plate_type != 'type1a':
        return PreparedPlate(mode, 'whole_plate', None, (image,))
    split = find_split(image, method, config)
    if not split.reliable:
        return PreparedPlate(mode, 'whole_plate', split, (image,))
    boxes = _row_boxes(image, split.cut_y, config)
    rows = tuple(image[y0:y1,x0:x1].copy() for x0,y0,x1,y1 in boxes)
    if any(min(row.shape[:2]) < 8 for row in rows):
        failed = SplitResult(method, 'whole_plate', None, False, 0., ('row_crop_too_small',))
        return PreparedPlate(mode, 'whole_plate', failed, (image,))
    if mode == 'split_and_stitch':
        stitched, transforms = stitch_rows(rows, config)
        return PreparedPlate(mode, mode, split, (stitched,), boxes, transforms)
    return PreparedPlate(mode, mode, split, rows, boxes)


def recognize_prepared(prepared: PreparedPlate, recognize: Callable):
    """Read input(s) in top/bottom order with an injected frozen recognizer.

    The callback returns (text, sequence_confidence, character_probabilities).
    No format correction, threshold selection, or label hint is applied.
    """
    results = [recognize(image) for image in prepared.inputs]
    text = ''.join(result[0] for result in results)
    probabilities = [float(p) for result in results for p in result[2]]
    confidence = float(np.mean(probabilities)) if probabilities else 0.
    return text, confidence, probabilities
