#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from dataset import CropImageDataset, collect_crop_rows, collate_images_and_rows
from vae_model import ConvVAE, vae_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata", default="outputs/object_crops_raw/metadata.jsonl")
    parser.add_argument("--review-log", default="outputs/review_logs/review_log.jsonl")
    parser.add_argument("--output-dir", default="outputs/vae")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--beta", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-reviewed-paths", action="store_true")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    metadata = project_root / args.metadata
    review_log = project_root / args.review_log
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_crop_rows(project_root, metadata, review_log, include_reviewed_paths=not args.no_reviewed_paths)
    if not rows:
        raise SystemExit(f"No crop rows found from {metadata}")

    dataset = CropImageDataset(project_root, rows, image_size=args.image_size)
    val_size = max(1, int(len(dataset) * 0.1)) if len(dataset) > 10 else 0
    train_size = len(dataset) - val_size
    if val_size > 0:
        train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    else:
        train_ds, val_ds = dataset, None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_images_and_rows)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_images_and_rows) if val_ds else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ConvVAE(image_size=args.image_size, latent_dim=args.latent_dim, in_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    history = []
    best_val = float("inf")
    best_path = output_dir / "vae_best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for images, _rows in train_loader:
            images = images.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(images)
            loss, recon_loss, kl_loss = vae_loss(recon, images, mu, logvar, beta=args.beta)
            loss.backward()
            optimizer.step()
            train_loss += float(loss.item()) * images.size(0)
        train_loss /= max(1, len(train_ds))

        val_loss = None
        if val_loader:
            model.eval()
            total = 0.0
            with torch.no_grad():
                for images, _rows in val_loader:
                    images = images.to(device)
                    recon, mu, logvar = model(images)
                    loss, _, _ = vae_loss(recon, images, mu, logvar, beta=args.beta)
                    total += float(loss.item()) * images.size(0)
            val_loss = total / max(1, len(val_ds))
            if val_loss < best_val:
                best_val = val_loss
                torch.save({"model_state": model.state_dict(), "config": vars(args)}, best_path)
        else:
            torch.save({"model_state": model.state_dict(), "config": vars(args)}, best_path)

        item = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        history.append(item)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f}" + (f" val_loss={val_loss:.6f}" if val_loss is not None else ""))

    final_path = output_dir / "vae_final.pt"
    torch.save({"model_state": model.state_dict(), "config": vars(args)}, final_path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    (output_dir / "config.json").write_text(json.dumps(vars(args), indent=2), encoding="utf-8")
    print(f"✅ Saved VAE final model: {final_path}")
    print(f"✅ Saved VAE best model:  {best_path}")


if __name__ == "__main__":
    main()
