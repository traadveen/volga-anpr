"""AY000554 OCR filenames -> reviewed OCR-crop annotations, no downloads."""
from pathlib import Path

from ..schema import Annotation, TARGETS, normalize_plate, plate_error

SOURCE_ID = 'AY000554/Car_plate_OCR_dataset'
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def prepare(source_root: Path, splits, review: dict, plate_type: str):
    if plate_type not in TARGETS:
        raise ValueError('Explicit reviewed OCR crop plate type is required')
    pairs = []
    for split in splits:
        folder = source_root / split
        if not folder.is_dir():
            raise ValueError(f'Unpacked source split not found: {folder}')
        for image in sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in EXTENSIONS):
            if not image.resolve().is_relative_to(source_root.resolve()):
                raise ValueError('Source image resolves outside source-root')
            # This specific source documents filename == OCR label. Do not strip suffixes.
            text = normalize_plate(image.stem)
            error = plate_error(text)
            if error:
                raise ValueError(f'{image}: filename label invalid: {error}; annotate explicitly, never guess')
            from ..validation import fingerprint
            shape = fingerprint(image)
            name = f'images/real/hf_ocr/{split}/' + image.relative_to(folder).as_posix()
            record = Annotation(name, text, plate_type, (0., 0., float(shape['width']), float(shape['height'])),
                                None, None, False, review['source'], review['license'], (), 'ocr_crop', split)
            pairs.append((record, image))
    return pairs
