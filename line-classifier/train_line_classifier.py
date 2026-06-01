"""
Train a simple CNN to classify text lines as handwritten (H) or typewritten (L).

Global average pooling makes the model resolution-agnostic — no resizing needed.
Images are only normalised to [0, 1] and converted to grayscale.

Usage:
    conda activate laypa
    python scripts/train_line_classifier.py \
        --manifest /data/storage/users/tockier/laypa_classify/manifest.json \
        --labels   /data/storage/users/tockier/laypa_classify/labels.json \
        --output   /data/storage/users/tockier/laypa_classify/model \
        --epochs   30 \
        --batch    16

The best checkpoint (by val accuracy) is saved as <output>/best.pth.
"""

import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


# ── dataset ───────────────────────────────────────────────────────────────────

class LineDataset(Dataset):
    def __init__(self, entries: list[dict]):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = cv2.imread(e["crop"], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((38, 64), dtype=np.uint8)
        # normalise to [0, 1], add channel dim → (1, H, W)
        x = torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.tensor(e["label"], dtype=torch.long)
        return x, y


def collate_pad(batch):
    """Pad images in a batch to the same (H, W) with zeros on the right/bottom."""
    xs, ys = zip(*batch)
    max_h = max(x.shape[1] for x in xs)
    max_w = max(x.shape[2] for x in xs)
    padded = torch.zeros(len(xs), 1, max_h, max_w)
    for i, x in enumerate(xs):
        padded[i, :, :x.shape[1], :x.shape[2]] = x
    return padded, torch.stack(ys)


# ── model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Sequential):
    def __init__(self, in_ch, out_ch, pool=True):
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2, 2))
        super().__init__(*layers)


class LineCNN(nn.Module):
    """
    Tiny CNN ending in global average pooling → fully connected.
    Accepts any (C, H, W) input; pool makes it resolution-agnostic.
    """
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),   # H/2, W/2
            ConvBlock(32, 64, pool=True),   # H/4, W/4
            ConvBlock(64, 128, pool=True),  # H/8, W/8
            ConvBlock(128, 256, pool=False), # same spatial, deeper features
        )
        self.gap = nn.AdaptiveAvgPool2d(1)  # (B, 256, 1, 1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        return self.classifier(x)


# ── training ──────────────────────────────────────────────────────────────────

def build_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--labels",   required=True)
    p.add_argument("--output",   required=True)
    p.add_argument("--epochs",   type=int, default=30)
    p.add_argument("--batch",    type=int, default=16)
    p.add_argument("--lr",       type=float, default=1e-3)
    p.add_argument("--val-split", type=float, default=0.15,
                   help="Fraction of data held out for validation")
    p.add_argument("--seed",     type=int, default=42)
    return p.parse_args()


def load_entries(manifest_path: str, labels_path: str) -> list[dict]:
    manifest = json.load(open(manifest_path))
    labels   = json.load(open(labels_path))
    label_map = {"H": 0, "L": 1}
    entries = []
    for entry in manifest:
        lbl = labels.get(str(entry["id"]))
        if lbl not in label_map:
            continue
        entries.append({**entry, "label": label_map[lbl]})
    return entries


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train(train)
    total_loss = correct = seen = 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct    += (logits.argmax(1) == y).sum().item()
            seen       += len(y)
    return total_loss / seen, correct / seen


def main():
    args = build_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_entries(args.manifest, args.labels)
    print(f"Dataset: {len(entries)} labeled samples")

    random.shuffle(entries)
    val_n   = max(1, int(len(entries) * args.val_split))
    val_set = entries[:val_n]
    trn_set = entries[val_n:]

    h_train = sum(1 for e in trn_set if e["label"] == 0)
    l_train = sum(1 for e in trn_set if e["label"] == 1)
    print(f"Train: {len(trn_set)} (H={h_train}, L={l_train})  |  Val: {len(val_set)}")

    # weighted sampler to handle class imbalance
    class_counts = [h_train, l_train]
    weights = [1.0 / class_counts[e["label"]] for e in trn_set]
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(trn_set))

    trn_loader = DataLoader(
        LineDataset(trn_set), batch_size=args.batch,
        sampler=sampler, collate_fn=collate_pad, num_workers=2,
    )
    val_loader = DataLoader(
        LineDataset(val_set), batch_size=args.batch,
        shuffle=False, collate_fn=collate_pad, num_workers=2,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model     = LineCNN(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        trn_loss, trn_acc = run_epoch(model, trn_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({"epoch": epoch, "state_dict": model.state_dict(),
                        "val_acc": val_acc}, output_dir / "best.pth")
            marker = "  ← best"

        print(f"Epoch {epoch:3d}/{args.epochs}  "
              f"trn loss={trn_loss:.4f} acc={trn_acc:.3f}  |  "
              f"val loss={val_loss:.4f} acc={val_acc:.3f}{marker}")

    print(f"\nBest val accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {output_dir / 'best.pth'}")


if __name__ == "__main__":
    main()
