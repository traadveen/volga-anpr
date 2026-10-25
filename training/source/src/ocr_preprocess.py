"""One train/inference OCR preprocessing path, without label inputs.

Synthetic augmentation is already present in stored frames. Additional
photometric augmentation, if requested, happens before this rectification.
Rectification geometry is supplied by the caller, never predicted here.
"""
from dataclasses import dataclass
import cv2
import numpy as np
from ocr.crnn import _normalize_image
from src.multiline import MultilineConfig, prepare_multiline


@dataclass
class OCRInput:
    crop: np.ndarray
    normalized: np.ndarray
    diagnostics: dict


def prepare_ocr_input(image, plate_type, *, homography=None, canonical_size=None,
                      multiline_config=None, image_height=32, image_width=160):
    if homography is not None:
        matrix=np.asarray(homography,dtype=np.float64)
        if matrix.shape!=(3,3) or not np.isfinite(matrix).all() or abs(np.linalg.det(matrix))<1e-12:
            raise ValueError('Invalid rectification homography')
        if canonical_size is None or min(canonical_size)<8:
            raise ValueError('Canonical size required for synthetic rectification')
        w,h=map(int,canonical_size)
        restored=cv2.warpPerspective(image,np.linalg.inv(matrix),(w,h),flags=cv2.INTER_LINEAR)
        image=cv2.resize(restored,(w//2,h//2),interpolation=cv2.INTER_AREA)
    prepared=prepare_multiline(image,plate_type,mode='split_and_stitch',method='horizontal_projection',
                               config=multiline_config or MultilineConfig())
    crop=prepared.inputs[0]
    return OCRInput(crop,_normalize_image(crop,image_height,image_width),prepared.diagnostics())
