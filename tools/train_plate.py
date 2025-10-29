import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a YOLO detector for license plates.")
    parser.add_argument(
        "--model",
        default="yolo11n.pt",
        help="Base YOLO checkpoint to fine-tune (e.g. yolo11n.pt, yolo11m.pt).",
    )
    parser.add_argument(
        "--data",
        default="configs/hf_detection.yaml",
        help="Path to YOLO data.yaml describing the detection dataset.",
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to train on: 'auto', 'cpu', 'cuda', or explicit index like 'cuda:0'.",
    )
    parser.add_argument("--lr0", type=float, default=0.003, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.0005)
    parser.add_argument(
        "--warmup_epochs",
        type=float,
        default=3.0,
        help="Number of warmup epochs before switching to cosine schedule.",
    )
    parser.add_argument("--project", default="runs/train", help="Directory for Ultralytics training logs.")
    parser.add_argument("--name", default="yolo11-plates-hf", help="Run name inside the project directory.")
    parser.add_argument(
        "--mosaic",
        type=float,
        default=1.0,
        help="Probability for mosaic augmentation (set 0.0 to disable).",
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Enable train/val plotting (uses extra RAM). Disabled by default for stability.",
    )
    parser.add_argument(
        "--resume",
        default=None,
        help="Path to a checkpoint to resume from. If set, --model is ignored.",
    )
    return parser.parse_args()


def resolve_device(device: str):
    if device == "auto":
        return None
    if device == "cuda":
        return "cuda:0"
    return device


def main():
    args = parse_args()

    if args.resume:
        model = YOLO(args.resume)
    else:
        if not Path(args.model).exists():
            raise FileNotFoundError(f"Pretrained checkpoint '{args.model}' not found.")
        model = YOLO(args.model)

    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "workers": args.workers,
        "lr0": args.lr0,
        "weight_decay": args.weight_decay,
        "warmup_epochs": args.warmup_epochs,
        "device": resolve_device(args.device),
        "project": args.project,
        "name": args.name,
        "cos_lr": True,
        "optimizer": "SGD",
        "patience": 20,
        "close_mosaic": 10,
        "mosaic": args.mosaic,
        "augment": True,
        "val": True,
        "plots": args.plots,
    }

    # Filter out None to avoid Ultralytics complaining.
    train_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}

    model.train(**train_kwargs)


if __name__ == "__main__":
    main()
