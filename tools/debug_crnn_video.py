"""
Quick diagnostic script to inspect raw CRNN predictions on video frames
without tracker/voting.

Example:
    python tools/debug_crnn_video.py \\
        --video video.mp4 \\
        --crnn models/ocr_crnn_nomeroff.pt \\
        --detector models/yolo11_plate.pt \\
        --stride 2 --max-frames 500 --conf 0.45
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ocr.crnn import CRNNRecognizer  # type: ignore

ALPH = "ABEKMHOPCTYX"
CYR_EQ = "\u0410\u0412\u0415\u041a\u041c\u041d\u041e\u0420\u0421\u0422\u0423\u0425"  # АВЕКМНОРСТУХ
KIR2LAT = str.maketrans({c: l for c, l in zip(CYR_EQ, ALPH)} | {c.lower(): l for c, l in zip(CYR_EQ, ALPH)})


def sanitize_text(t: str) -> str:
    if not t:
        return ""
    t = t.translate(KIR2LAT).upper()
    return "".join(ch for ch in t if ch.isdigit() or ch in ALPH)


def iter_video_frames(cap: cv2.VideoCapture, stride: int, max_frames: int | None):
    frame_idx = 0
    yielded = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if (frame_idx % stride) == 0:
            ts_ms = int(frame_idx / max(cap.get(cv2.CAP_PROP_FPS) or 25.0, 1e-6) * 1000)
            yield frame_idx, ts_ms, frame
            yielded += 1
            if max_frames and yielded >= max_frames:
                break
        frame_idx += 1


def run(args: argparse.Namespace):
    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Video not found: {video_path}")

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Failed to open video: {video_path}")

    yolo = YOLO(args.detector)
    recognizer = CRNNRecognizer(args.crnn, device=args.device)

    out_rows: List[List[str]] = []
    out_headers = ["frame_idx", "time_ms", "bbox", "det_conf", "raw_text", "sanitized", "char_probs"]
    out_path = Path(args.output) if args.output else None

    print("frame_idx\tms\tconf\tbbox\traw\tclean")
    for frame_idx, ts_ms, frame in iter_video_frames(cap, args.stride, args.max_frames):
        dets = yolo.predict(frame, conf=args.conf, iou=0.5, verbose=False)
        if not dets:
            continue
        pred = dets[0]
        if pred.boxes is None or len(pred.boxes) == 0:
            continue
        xyxy = pred.boxes.xyxy.cpu().numpy()
        confs = pred.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), det_conf in zip(xyxy, confs):
            crop = frame[int(y1):int(y2), int(x1):int(x2)]
            if crop.size == 0:
                continue
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
            res = recognizer.readtext(gray, detail=1)
            if not res:
                continue
            _, raw_text, _, extras = res[0]
            char_probs = extras.get("char_probs") if isinstance(extras, dict) else None
            sanitized = sanitize_text(raw_text)
            bbox_str = f"{int(x1)},{int(y1)},{int(x2)},{int(y2)}"

            print(f"{frame_idx}\t{ts_ms}\t{det_conf:.2f}\t{bbox_str}\t{raw_text}\t{sanitized}")
            if args.show:
                vis = crop.copy()
                cv2.putText(vis, sanitized or raw_text, (5, max(18, vis.shape[0] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("CRNN debug", vis)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            if out_path:
                out_rows.append([
                    str(frame_idx),
                    str(ts_ms),
                    bbox_str,
                    f"{det_conf:.4f}",
                    raw_text,
                    sanitized,
                    ",".join(f"{float(p):.4f}" for p in (char_probs or [])),
                ])

    if out_path and out_rows:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh, delimiter=";")
            writer.writerow(out_headers)
            writer.writerows(out_rows)
        print(f"[INFO] Saved diagnostics to {out_path}")

    cap.release()
    if args.show:
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Inspect raw CRNN predictions on video frames")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--crnn", required=True, help="Path to CRNN checkpoint (.pt)")
    ap.add_argument("--detector", default="models/yolo11_plate.pt", help="YOLO plate detector weights")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device for CRNN inference")
    ap.add_argument("--stride", type=int, default=2, help="Process every N-th frame")
    ap.add_argument("--max-frames", type=int, default=0, help="Limit number of frames (0 = all)")
    ap.add_argument("--conf", type=float, default=0.45, help="YOLO confidence threshold")
    ap.add_argument("--show", action="store_true", help="Display cropped plates as they are processed")
    ap.add_argument("--output", default=None, help="Optional CSV file to dump predictions")
    return ap.parse_args()


if __name__ == "__main__":
    run(parse_args())
