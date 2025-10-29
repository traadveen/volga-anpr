"""
Train a CRNN plate recognizer on the cropped license plate dataset.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from ocr import (  # noqa:E402
    CTCLabelConverter,
    PlateOCRDataset,
    alphabet_from_tsv,
    create_crnn_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRNN OCR for RU license plates.")
    parser.add_argument("--data_root", default="data/hf_ocr", help="Root directory containing train/val/test folders.")
    parser.add_argument("--train_dir", default="train", help="Relative path to the training split.")
    parser.add_argument("--val_dir", default="val", help="Relative path to the validation split.")
    parser.add_argument("--alphabet", default=None, help="Explicit alphabet string; defaults to characters from train TSV.")
    parser.add_argument("--img_height", type=int, default=32, help="Input image height after resizing.")
    parser.add_argument("--img_width", type=int, default=160, help="Input image width after resizing.")
    parser.add_argument("--hidden_size", type=int, default=256, help="Hidden size of BiLSTM layers.")
    parser.add_argument("--lstm_layers", type=int, default=2, help="Number of stacked BiLSTM layers.")
    parser.add_argument("--batch_size", type=int, default=128, help="Mini-batch size.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay for AdamW.")
    parser.add_argument("--clip_grad", type=float, default=5.0, help="Gradient clipping value.")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Training device selection.")
    parser.add_argument("--num_workers", type=int, default=4, help="Dataloader worker threads.")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from.")
    parser.add_argument("--output", default="models/ocr_crnn.pt", help="Where to store the best checkpoint.")
    parser.add_argument("--save_every", type=int, default=0, help="Optional periodic checkpointing every N epochs.")
    parser.add_argument("--patience", type=int, default=0, help="Early stopping patience (epochs without word_acc improvement).")
    return parser.parse_args()


def collate_batch(batch):
    images = torch.stack([item[0] for item in batch])
    targets = torch.cat([item[1] for item in batch])
    lengths = torch.tensor([int(item[2]) for item in batch], dtype=torch.long)
    texts = [item[3] for item in batch]
    return images, targets, lengths, texts


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    prev_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        row = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            row.append(min(row[-1] + 1, prev_row[j] + 1, prev_row[j - 1] + cost))
        prev_row = row
    return prev_row[-1]


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    converter: CTCLabelConverter,
    criterion: nn.CTCLoss,
    device: torch.device,
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_samples = 0
    total_words = 0
    correct_words = 0
    total_chars = 0
    matched_chars = 0
    with torch.inference_mode():
        for images, targets, lengths, texts in loader:
            images = images.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            logits = model(images)  # T x B x C
            log_probs = F.log_softmax(logits, dim=2)
            input_lengths = torch.full(
                (images.size(0),),
                logits.size(0),
                dtype=torch.long,
                device=device,
            )
            loss = criterion(log_probs, targets, input_lengths, lengths)
            total_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            pred_texts, _, _ = converter.decode_logits(logits.cpu())
            for pred, gt in zip(pred_texts, texts):
                total_words += 1
                if pred == gt:
                    correct_words += 1
                total_chars += len(gt)
                dist = levenshtein(pred, gt)
                matched_chars += max(0, len(gt) - dist)

    avg_loss = total_loss / max(1, total_samples)
    word_acc = correct_words / max(1, total_words)
    char_acc = matched_chars / max(1, total_chars)
    return {"loss": avg_loss, "word_acc": word_acc, "char_acc": char_acc}


def main():
    args = parse_args()
    if args.patience < 0:
        raise ValueError("--patience must be >= 0.")

    device_str = args.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)

    target_total_epochs = args.epochs
    if target_total_epochs <= 0:
        raise ValueError("--epochs must be positive.")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        alphabet = checkpoint["alphabet"]
        cfg = checkpoint.get("config", {})
        args.img_height = cfg.get("image_height", args.img_height)
        args.img_width = cfg.get("image_width", args.img_width)
        args.hidden_size = cfg.get("hidden_size", args.hidden_size)
        args.lstm_layers = cfg.get("lstm_layers", args.lstm_layers)
        converter = CTCLabelConverter(alphabet)
        model = create_crnn_model(
            alphabet=alphabet,
            hidden_size=args.hidden_size,
            lstm_layers=args.lstm_layers,
        )
        model.load_state_dict(checkpoint["model"])
        model.to(device)
        last_epoch = checkpoint.get("epoch", 0)
        if target_total_epochs <= last_epoch:
            print(
                f"[Resume] Checkpoint '{args.resume}' already reached epoch {last_epoch}. "
                f"Increase --epochs to continue training."
            )
            return
        start_epoch = last_epoch + 1
        total_epochs = target_total_epochs
        best_word_acc = checkpoint.get("metrics", {}).get("word_acc", 0.0)
    else:
        train_labels = os.path.join(args.data_root, args.train_dir, "labels.tsv")
        if args.alphabet:
            alphabet = args.alphabet
        else:
            alphabet = alphabet_from_tsv(train_labels)
        converter = CTCLabelConverter(alphabet)
        model = create_crnn_model(
            alphabet=alphabet,
            hidden_size=args.hidden_size,
            lstm_layers=args.lstm_layers,
        )
        model.to(device)
        start_epoch = 1
        total_epochs = target_total_epochs
        best_word_acc = 0.0

    if args.resume:
        print(
            f"[Resume] Loaded checkpoint '{args.resume}' (last_epoch={start_epoch-1}). "
            f"Continuing up to epoch {total_epochs}."
        )

    train_dataset = PlateOCRDataset(
        root_dir=os.path.join(args.data_root, args.train_dir),
        tsv_file="labels.tsv",
        converter=converter,
        image_height=args.img_height,
        image_width=args.img_width,
    )
    val_dataset = PlateOCRDataset(
        root_dir=os.path.join(args.data_root, args.val_dir),
        tsv_file="labels.tsv",
        converter=converter,
        image_height=args.img_height,
        image_width=args.img_width,
    )

    def _describe_dataset(name: str, dataset: PlateOCRDataset):
        stats = getattr(dataset, "stats", None)
        if stats:
            print(
                f"[Dataset] {name}: kept {stats.get('kept', 0)}/{stats.get('rows', 0)} "
                f"(missing={stats.get('missing_files', 0)}, "
                f"invalid_label={stats.get('invalid_label', 0)}, "
                f"invalid_pattern={stats.get('invalid_pattern', 0)})"
            )
        warnings = getattr(dataset, "warnings", None) or []
        if warnings:
            for msg in warnings[:5]:
                print(f"[Dataset][{name}] warning: {msg}")
            extra = len(warnings) - 5
            if extra > 0:
                print(f"[Dataset][{name}] ... {extra} more warnings suppressed")

    _describe_dataset("train", train_dataset)
    _describe_dataset("val", val_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_batch,
    )

    criterion = nn.CTCLoss(blank=converter.blank_idx, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    if args.resume and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
        # Move optimizer state tensors to the correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        for group in optimizer.param_groups:
            group["lr"] = args.lr

    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    epochs_no_improve = 0

    for epoch in range(start_epoch, total_epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(
            train_loader,
            desc=f"Epoch {epoch}/{total_epochs}",
            leave=False,
        )
        for images, targets, lengths, _ in progress:
            images = images.to(device)
            targets = targets.to(device)
            lengths = lengths.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(images)  # T x B x C
                log_probs = F.log_softmax(logits, dim=2)
                input_lengths = torch.full(
                    (images.size(0),),
                    logits.size(0),
                    dtype=torch.long,
                    device=device,
                )
                loss = criterion(log_probs, targets, input_lengths, lengths)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item() * images.size(0)
            progress.set_postfix(loss=loss.item())

        train_loss = running_loss / max(1, len(train_dataset))
        val_metrics = evaluate(model, val_loader, converter, criterion, device)
        print(
            f"[Epoch {epoch:03d}] train_loss={train_loss:.4f} "
            f"val_loss={val_metrics['loss']:.4f} "
            f"word_acc={val_metrics['word_acc']*100:.2f}% "
            f"char_acc={val_metrics['char_acc']*100:.2f}%"
        )

        improved = val_metrics["word_acc"] > best_word_acc
        if improved:
            best_word_acc = val_metrics["word_acc"]
            ckpt = {
                "alphabet": converter.alphabet,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "config": {
                    "image_height": args.img_height,
                    "image_width": args.img_width,
                    "hidden_size": args.hidden_size,
                    "lstm_layers": args.lstm_layers,
                },
                "epoch": epoch,
                "metrics": val_metrics,
            }
            torch.save(ckpt, args.output)
            print(f"[Info] Saved best checkpoint to {args.output}")
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if args.save_every and (epoch % args.save_every == 0):
            path = os.path.splitext(args.output)[0] + f"_epoch{epoch}.pt"
            torch.save(
                {
                    "alphabet": converter.alphabet,
                    "model": model.state_dict(),
                    "config": {
                        "image_height": args.img_height,
                        "image_width": args.img_width,
                        "hidden_size": args.hidden_size,
                        "lstm_layers": args.lstm_layers,
                    },
                    "epoch": epoch,
                },
                path,
            )
            print(f"[Info] Saved periodic checkpoint to {path}")

        if args.patience and epochs_no_improve >= args.patience:
            print(f"[EarlyStopping] No word_acc improvement for {args.patience} epoch(s). Stopping.")
            break


if __name__ == "__main__":
    main()
