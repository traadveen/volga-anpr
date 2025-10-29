import argparse
import shutil
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import hf_hub_download
from tqdm import tqdm

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
YOLO_DATASET = "AY000554/Car_plate_detecting_dataset"
OCR_DATASET = "AY000554/Car_plate_OCR_dataset"

def _is_yolo_label_file(path: Path) -> bool:
    if path.suffix.lower() != ".txt":
        return False
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    return False
                float(parts[0])
                for value in parts[1:]:
                    float(value)
            return True
    except Exception:
        return False

def download_detection(target_root: Path, overwrite: bool = False):
    print(f"[INFO] Preparing YOLO detection dataset under {target_root}")
    for split in ["train", "val", "test"]:
        dest_images = target_root / split / "images"
        dest_labels = target_root / split / "labels"
        if dest_images.exists() and dest_labels.exists() and not overwrite:
            print(f"[SKIP] {split} already prepared")
            continue
        if overwrite:
            if dest_images.exists():
                shutil.rmtree(dest_images)
            if dest_labels.exists():
                shutil.rmtree(dest_labels)
        dest_images.mkdir(parents=True, exist_ok=True)
        dest_labels.mkdir(parents=True, exist_ok=True)

        fn = f"{split}.zip"
        print(f"[INFO] Downloading {YOLO_DATASET}::{fn}")
        zip_path = hf_hub_download(repo_id=YOLO_DATASET, filename=fn, repo_type="dataset")
        with TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            tmp_path = Path(tmp)
            images = [
                p for p in tmp_path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
                and "images" in [part.lower() for part in p.parts]
            ]
            labels = [
                p for p in tmp_path.rglob("*.txt")
                if p.is_file() and _is_yolo_label_file(p)
            ]

            if not images:
                raise RuntimeError(f"No images found in archive {fn}")
            if not labels:
                raise RuntimeError(f"No YOLO label files detected in archive {fn}")

            for src in tqdm(images, desc=f"{split} images", unit="img"):
                shutil.copy2(src, dest_images / src.name)
            for src in tqdm(labels, desc=f"{split} labels", unit="lbl"):
                shutil.copy2(src, dest_labels / src.name)
    print("[OK] Detection dataset ready.")

def _guess_manifest_file(base: Path):
    candidates = []
    for p in base.rglob("*"):
        if p.suffix.lower() in {".txt", ".csv"} and "license" not in p.name.lower():
            if "label" in p.name.lower() or "annotation" in p.name.lower():
                candidates.append(p)
    return sorted(candidates, key=lambda x: len(x.parts))

def _parse_manifest(path: Path):
    entries = []
    with path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if "," in line:
                name, text = line.split(",", 1)
            elif "\t" in line:
                name, text = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    continue
                name, text = parts
            entries.append((name.strip(), text.strip()))
    return entries

def download_ocr(target_root: Path, overwrite: bool = False):
    print(f"[INFO] Preparing OCR dataset under {target_root}")
    for split in ["train", "val", "test"]:
        dest_split = target_root / split
        dest_images = dest_split / "images"
        manifest_out = dest_split / "labels.tsv"

        if dest_images.exists() and manifest_out.exists() and not overwrite:
            print(f"[SKIP] {split} already prepared")
            continue

        if overwrite and dest_split.exists():
            shutil.rmtree(dest_split)
        dest_images.mkdir(parents=True, exist_ok=True)

        fn = f"{split}.zip"
        print(f"[INFO] Downloading {OCR_DATASET}::{fn}")
        zip_path = hf_hub_download(repo_id=OCR_DATASET, filename=fn, repo_type="dataset")
        with TemporaryDirectory() as tmp:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(tmp)
            tmp_path = Path(tmp)
            image_index = {
                p.name: p for p in tmp_path.rglob("*")
                if p.is_file()
                and p.suffix.lower() in IMAGE_EXTS
            }
            manifest_candidates = _guess_manifest_file(tmp_path)
            entries = []
            if manifest_candidates:
                manifest = manifest_candidates[0]
                entries = _parse_manifest(manifest)
                if not entries:
                    print(f"[WARN] Manifest {manifest} empty, will derive labels from filenames.")
            if not entries:
                print("[INFO] Deriving labels from image filenames.")
                entries = [(name, Path(name).stem) for name in image_index.keys()]
                entries.sort()

            with manifest_out.open("w", encoding="utf-8") as out_f:
                for name, text in tqdm(entries, desc=f"{split} pairs", unit="pair"):
                    src = image_index.get(name)
                    if src is None:
                        continue
                    shutil.copy2(src, dest_images / name)
                    rel_path = Path("images") / name
                    out_f.write(f"{rel_path.as_posix()}\t{text}\n")
    print("[OK] OCR dataset ready.")

def main():
    parser = argparse.ArgumentParser(description="Download and prepare HuggingFace datasets for RU-ANPR.")
    parser.add_argument(
        "--target-root",
        default="data",
        help="Base directory to store prepared datasets (default: data).",
    )
    parser.add_argument("--skip-detection", action="store_true", help="Skip the detection dataset.")
    parser.add_argument("--skip-ocr", action="store_true", help="Skip the OCR dataset.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing prepared data.")
    args = parser.parse_args()

    root = Path(args.target_root)
    if not args.skip_detection:
        download_detection(root / "hf_detection", overwrite=args.overwrite)
    if not args.skip_ocr:
        download_ocr(root / "hf_ocr", overwrite=args.overwrite)

if __name__ == "__main__":
    main()
