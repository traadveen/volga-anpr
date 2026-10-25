import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ocr.utils import sanitize_plate_label

def sanitize(text: str) -> str:
    # Partial labels are not complete OCR targets; never delete their slots.
    normalized = sanitize_plate_label(text, allow_partial=True)
    return normalized if "#" not in normalized else ""


def build_labels(split_dir: Path) -> dict:
    ann_dir = split_dir / 'ann'
    img_dir = split_dir / 'img'
    rows = []
    blank = missing = 0
    for ann_path in ann_dir.glob('*.json'):
        data = json.loads(ann_path.read_text(encoding='utf-8'))
        text = sanitize(data.get('description', ''))
        if not text:
            blank += 1
            continue
        stem = ann_path.stem
        candidates = list(img_dir.glob(f'{stem}.*'))
        if not candidates:
            missing += 1
            continue
        rel_path = Path('img') / candidates[0].name
        rows.append((rel_path.as_posix(), text))
    rows.sort()
    out_path = split_dir / 'labels.tsv'
    out_path.write_text('\n'.join(f"{img}\t{text}" for img, text in rows), encoding='utf-8')
    return {'rows': len(rows), 'blank': blank, 'missing': missing, 'out': out_path}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True, help='Path to autoriaNumberplateOcrRu root (with train/val/test)')
    args = ap.parse_args()
    base = Path(args.base)
    if not base.exists():
        raise SystemExit(f'Base path not found: {base}')
    for split in ('train', 'val', 'test'):
        split_dir = base / split
        if not split_dir.exists():
            continue
        stats = build_labels(split_dir)
        print(f"{split}: {stats['rows']} rows (blank={stats['blank']}, missing={stats['missing']}) -> {stats['out']}")


if __name__ == '__main__':
    main()
