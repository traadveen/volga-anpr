"""Observed plate geometry; fixed type canvases before the frozen OCR helper."""
from __future__ import annotations
import cv2
import numpy as np

CANONICAL = {'type1': (520, 112), 'type1a': (290, 170), 'type1b': (520, 112)}


def validate_quad(points, image_shape=None):
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError('Quad must contain four finite TL,TR,BR,BL points')
    edges = np.roll(q, -1, axis=0) - q
    cross = edges[:, 0] * np.roll(edges, -1, axis=0)[:, 1] - edges[:, 1] * np.roll(edges, -1, axis=0)[:, 0]
    if not np.all(cross > 1e-4) or cv2.contourArea(q) < 8:
        raise ValueError('Crossing, degenerate or incorrectly ordered quad')
    if np.min(np.linalg.norm(edges, axis=1)) < 2:
        raise ValueError('Quad edge too short')
    if image_shape is not None:
        h, w = image_shape[:2]
        if np.any(q < 0) or np.any(q[:, 0] >= w) or np.any(q[:, 1] >= h):
            raise ValueError('Quad leaves source image')
    return q


def order_obb(points):
    """Unoriented rectangle has no reading direction; assume upright road scenes.

    Order the polygon cyclically, then choose the upper-left endpoint. This is
    only for OBB output, never a repair of semantic pose keypoint predictions.
    """
    q = np.asarray(points, dtype=np.float32)
    if q.shape != (4, 2) or not np.isfinite(q).all():
        raise ValueError('Invalid OBB points')
    center = q.mean(axis=0)
    q = q[np.argsort(np.arctan2(q[:, 1]-center[1], q[:, 0]-center[0]))]
    q = np.roll(q, -int(np.argmin(q.sum(axis=1))), axis=0)
    return validate_quad(q)


def bbox_from_quad(points):
    q = validate_quad(points)
    return np.r_[q.min(axis=0), q.max(axis=0)].tolist()


def canonical_rectify(frame, points, plate_type, provenance):
    if not provenance or plate_type not in CANONICAL:
        raise ValueError('Observed quad provenance and target type required')
    q = validate_quad(points, frame.shape)
    # The declared semantic top edge must run rightwards in an upright road
    # image. A cyclic shift is convex but would rotate the OCR canvas by90deg.
    # Plates with a top edge outside +/-45deg fall back to bbox in Phase F.
    horizontal=((q[1]-q[0])+(q[2]-q[3]))/2
    if horizontal[0] <= 0 or abs(horizontal[1]) > horizontal[0]:
        raise ValueError('Semantic top edge is not TL->TR in the supported upright orientation')
    w, h = CANONICAL[plate_type]
    destination = np.float32([[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]])
    matrix = cv2.getPerspectiveTransform(q, destination)
    if not np.isfinite(matrix).all() or abs(np.linalg.det(matrix)) < 1e-10:
        raise ValueError('Singular homography')
    return cv2.warpPerspective(frame, matrix, (w, h), flags=cv2.INTER_LINEAR), matrix


def bbox_iou(a, b):
    a, b = np.asarray(a), np.asarray(b)
    area_a = np.maximum(0, a[2:]-a[:2]).prod()
    area_b = np.maximum(0, b[2:]-b[:2]).prod()
    intersection = np.maximum(0, np.minimum(a[2:], b[2:])-np.maximum(a[:2], b[:2])).prod()
    union = area_a + area_b - intersection
    return float(intersection / union) if union > 0 else 0.


def quad_iou(a, b):
    a, b = validate_quad(a), validate_quad(b)
    intersection, _ = cv2.intersectConvexConvex(a, b)
    union = cv2.contourArea(a) + cv2.contourArea(b) - intersection
    return float(max(0., intersection)/union) if union > 0 else 0.


def quad_errors(predicted, target):
    p, t = validate_quad(predicted), validate_quad(target)
    error = np.linalg.norm(p-t, axis=1)
    diagonal = np.linalg.norm(t.max(axis=0)-t.min(axis=0))
    return {'corner_pixel_error': float(error.mean()),
            'normalized_corner_error': float(error.mean()/diagonal),
            'quad_iou': quad_iou(p, t)}
