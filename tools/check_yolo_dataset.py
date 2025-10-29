# C:\ru-anpr-mvp\tools\check_yolo_dataset.py
import os, glob
from PIL import Image

ROOT = r"C:\ru-anpr-mvp\data"
for split in ["train","val"]:
    imgs = sorted(glob.glob(os.path.join(ROOT,"images",split,"*.*")))
    labs = sorted(glob.glob(os.path.join(ROOT,"labels",split,"*.txt")))
    print(f"[{split}] images={len(imgs)} labels={len(labs)}")
    bad = 0
    for p in imgs[:50]:  # выборочно
        try: Image.open(p).verify()
        except Exception: bad += 1
    print(f"[{split}] bad_images={bad}")
print("OK")
