"""
Benchmark LineCNN inference throughput on a single GPU.

Runs two scenarios:
  1. Pure GPU  — all crops pre-loaded into VRAM, measures raw model throughput
  2. Realistic — loads crops from disk each batch, measures end-to-end throughput

Extrapolates both to 10 million lines.

Usage:
    conda activate laypa
    CUDA_VISIBLE_DEVICES=7 python scripts/benchmark_inference.py \
        --manifest   /data/storage/users/tockier/laypa_annotate/manifest.json \
        --checkpoint /data/storage/users/tockier/laypa_train/model/best.pth
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn


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
    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(1,  32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=False),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        return self.classifier(self.gap(self.features(x)))


# ── helpers ───────────────────────────────────────────────────────────────────

def load_model(checkpoint, device):
    model = LineCNN().to(device)
    ckpt = torch.load(checkpoint, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def read_gray(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return np.zeros((40, 200), dtype=np.uint8)
    return img


def pad_batch(tensors):
    """Stack variable-size (1,H,W) tensors into one batch by zero-padding."""
    max_h = max(t.shape[1] for t in tensors)
    max_w = max(t.shape[2] for t in tensors)
    out = torch.zeros(len(tensors), 1, max_h, max_w)
    for i, t in enumerate(tensors):
        out[i, :, :t.shape[1], :t.shape[2]] = t
    return out


def vram_used_gb(device):
    return torch.cuda.memory_allocated(device) / 1024**3


def vram_total_gb(device):
    return torch.cuda.get_device_properties(device).total_memory / 1024**3


def fmt_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f} min"
    if seconds < 86400:
        return f"{seconds/3600:.1f} h"
    return f"{seconds/86400:.1f} days"


# ── scenario 1: pure GPU throughput ───────────────────────────────────────────

def benchmark_gpu(model, crops_gray, device, target_lines=10_000_000):
    print("\n" + "="*60)
    print("SCENARIO 1 — Pure GPU throughput (data pre-loaded in VRAM)")
    print("="*60)

    total_vram = vram_total_gb(device)
    print(f"GPU VRAM total : {total_vram:.1f} GB")

    # Convert a representative set to tensors, find max batch that fits
    sample_tensors = [
        torch.from_numpy(img.astype(np.float32) / 255.0).unsqueeze(0)
        for img in crops_gray[:2000]
    ]

    # Binary search for max batch size
    lo, hi = 64, 16384
    best_batch = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        torch.cuda.empty_cache()
        try:
            batch = pad_batch(sample_tensors[:mid]).to(device)
            with torch.no_grad():
                _ = model(batch)
            del batch
            torch.cuda.empty_cache()
            best_batch = mid
            lo = mid + 1
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            hi = mid - 1

    print(f"Max batch size : {best_batch} lines")

    # Load that batch once, measure sustained throughput
    batch = pad_batch(sample_tensors[:best_batch]).to(device)
    used = vram_used_gb(device)
    print(f"VRAM used      : {used:.2f} GB / {total_vram:.1f} GB ({used/total_vram:.1%})")

    # Warmup
    with torch.no_grad():
        for _ in range(5):
            _ = model(batch)
    torch.cuda.synchronize(device)

    # Timed run
    n_iters = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iters):
            _ = model(batch)
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - t0

    lines_per_sec = best_batch * n_iters / elapsed
    total_time    = target_lines / lines_per_sec

    print(f"\nThroughput     : {lines_per_sec:,.0f} lines/sec")
    print(f"                 {lines_per_sec*3600:,.0f} lines/hour")
    print(f"\n→ {target_lines:,} lines would take : {fmt_time(total_time)}")

    del batch, sample_tensors
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    return lines_per_sec


# ── scenario 2: realistic disk-to-GPU throughput ───────────────────────────────

def benchmark_realistic(model, crop_paths, device, batch_size=512, target=10_000_000):
    target_lines = target
    print("\n" + "="*60)
    print("SCENARIO 2 — Realistic throughput (disk → CPU → GPU)")
    print("="*60)
    print(f"Batch size     : {batch_size}")

    n_lines   = len(crop_paths)
    n_batches = max(1, n_lines // batch_size)
    # Use at most 50 batches for the timing run
    n_time    = min(50, n_batches)

    # Warmup — first batch
    imgs = [read_gray(p) for p in crop_paths[:batch_size]]
    tensors = [torch.from_numpy(i.astype(np.float32)/255.0).unsqueeze(0) for i in imgs]
    batch = pad_batch(tensors).to(device)
    with torch.no_grad():
        _ = model(batch)
    torch.cuda.synchronize(device)
    del batch

    # Timed
    t0 = time.perf_counter()
    processed = 0
    for b in range(n_time):
        start = (b * batch_size) % n_lines
        end   = min(start + batch_size, n_lines)
        chunk = crop_paths[start:end]
        imgs    = [read_gray(p) for p in chunk]
        tensors = [torch.from_numpy(i.astype(np.float32)/255.0).unsqueeze(0) for i in imgs]
        batch   = pad_batch(tensors).to(device)
        with torch.no_grad():
            _ = model(batch)
        torch.cuda.synchronize(device)
        processed += len(chunk)
        del batch

    elapsed = time.perf_counter() - t0
    lines_per_sec = processed / elapsed
    total_time    = target_lines / lines_per_sec

    print(f"Measured over  : {processed:,} lines ({n_time} batches)")
    print(f"\nThroughput     : {lines_per_sec:,.0f} lines/sec")
    print(f"                 {lines_per_sec*3600:,.0f} lines/hour")
    print(f"\n→ {target_lines:,} lines would take : {fmt_time(total_time)}")
    return lines_per_sec


# ── main ──────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest",   required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--target",     type=int, default=10_000_000,
                   help="Total lines to extrapolate to")
    return p.parse_args()


def main():
    args   = get_args()
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES selects the card

    props = torch.cuda.get_device_properties(device)
    print(f"GPU            : {props.name}")
    print(f"VRAM           : {props.total_memory/1024**3:.1f} GB")
    print(f"Target         : {args.target:,} lines")

    model = load_model(args.checkpoint, device)
    # torch.compile not supported on Python 3.14+

    manifest   = json.load(open(args.manifest))
    crop_paths = [e["crop"] for e in manifest]
    print(f"Dataset        : {len(crop_paths)} crops available for benchmarking")

    crops_gray = [read_gray(p) for p in crop_paths]
    print(f"Loaded {len(crops_gray)} images into CPU RAM")

    gpu_tput  = benchmark_gpu(model, crops_gray, device, args.target)
    real_tput = benchmark_realistic(model, crop_paths, device,
                                    batch_size=256, target=args.target)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Scenario':<30} {'Throughput':>16} {'Time for 10M lines':>20}")
    print("-"*60)
    for label, tput in [("Pure GPU (max VRAM fill)", gpu_tput),
                         ("Realistic (disk → GPU)",   real_tput)]:
        print(f"{label:<30} {tput:>12,.0f} l/s  {fmt_time(args.target/tput):>18}")
    print("="*60)


if __name__ == "__main__":
    main()
