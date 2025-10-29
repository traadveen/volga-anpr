import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ocr.utils import matches_plate_pattern, sanitize_plate_label  # noqa: E402


def sanitize_label(name: str) -> str:
    """
    Convert a folder name into a canonical RU licence-plate label.
    Returns an empty string when the name cannot be sanitised or does not match
    the expected pattern (letter-digit-digit-digit-letter-letter-digit-digit[digit]).
    """
    cleaned = sanitize_plate_label(name, allow_partial=True)
    if not cleaned:
        return ""
    return cleaned if matches_plate_pattern(cleaned, allow_partial=False) else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', required=True, help='Folder with per-label subfolders (custom crops)')
    ap.add_argument('--dataset-root', required=True, help='Nomeroff dataset root (with train/img & labels.tsv)')
    ap.add_argument('--subset', default='train', help='Subset to augment (default: train)')
    ap.add_argument('--dest-folder', default='custom_img', help='Subfolder name inside subset for new images')
    args = ap.parse_args()

    src = Path(args.src)
    root = Path(args.dataset_root)
    subset = root / args.subset
    labels_path = subset / 'labels.tsv'
    dest_img = subset / args.dest_folder
    dest_img.mkdir(parents=True, exist_ok=True)

    if not labels_path.exists():
        raise SystemExit(f'labels.tsv not found: {labels_path}')

    added = 0
    skipped_invalid = 0
    skipped_empty = 0

    rows = [line for line in labels_path.read_text(encoding='utf-8').splitlines() if line]
    existing = set(rows)
    skipped_duplicates = 0

    for label_dir in sorted(src.iterdir()):
        if not label_dir.is_dir():
            continue
        label = sanitize_label(label_dir.name)
        if not label:
            skipped_invalid += 1
            continue
        harvested = 0
        for img_path in sorted(label_dir.glob('*')):
            if not img_path.is_file():
                continue
            new_name = f"{label}_{img_path.name}"
            row = f"{args.dest_folder}/{new_name}\t{label}"
            if row in existing:
                skipped_duplicates += 1
                continue
            dst = dest_img / new_name
            shutil.copy2(img_path, dst)
            rows.append(row)
            existing.add(row)
            added += 1
            harvested += 1
        if harvested == 0:
            skipped_empty += 1

    labels_path.write_text('\n'.join(rows), encoding='utf-8')
    print(
        f"Added {added} samples. "
        f"Skipped {skipped_invalid} label folders (invalid pattern), "
        f"{skipped_empty} without images, "
        f"{skipped_duplicates} duplicates already present. Updated {labels_path}."
    )


if __name__ == '__main__':
    main()
