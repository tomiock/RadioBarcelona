#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

try:
    import faiss
except ImportError as exc:
    raise SystemExit("faiss is not installed. Install faiss-cpu or faiss-gpu first.") from exc

from dataset import CropImageDataset, collect_crop_rows, collate_images_and_rows
from vae_model import ConvVAE


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norms, eps)


def load_model(model_path: Path, image_size: int, latent_dim: int, device: torch.device) -> ConvVAE:
    checkpoint = torch.load(model_path, map_location=device)
    cfg = checkpoint.get("config", {})
    image_size = int(cfg.get("image_size", image_size))
    latent_dim = int(cfg.get("latent_dim", latent_dim))
    model = ConvVAE(image_size=image_size, latent_dim=latent_dim, in_channels=1).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--metadata", default="outputs/object_crops_raw/metadata.jsonl")
    parser.add_argument("--review-log", default="outputs/review_logs/review_log.jsonl")
    parser.add_argument("--model", default="outputs/vae/vae_best.pt")
    parser.add_argument("--output-dir", default="outputs/faiss/vae/global")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--latent-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--by-type", action="store_true", help="Also build class-specific FAISS indexes under output-dir/../by_type/")
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    metadata = project_root / args.metadata
    review_log = project_root / args.review_log
    model_path = project_root / args.model
    output_dir = project_root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_crop_rows(project_root, metadata, review_log, include_reviewed_paths=False)
    if not rows:
        raise SystemExit(f"No crop rows found from {metadata}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(model_path, args.image_size, args.latent_dim, device)
    dataset = CropImageDataset(project_root, rows, image_size=model.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_images_and_rows)

    embeddings = []
    metadata_rows = []
    with torch.no_grad():
        for images, batch_rows in loader:
            images = images.to(device)
            mu, _logvar = model.encode(images)
            z = mu.detach().cpu().numpy().astype(np.float32)
            embeddings.append(z)
            metadata_rows.extend(batch_rows)

    emb = np.concatenate(embeddings, axis=0).astype(np.float32)
    emb = l2_normalize(emb)

    index = faiss.IndexFlatIP(emb.shape[1])
    index.add(emb)

    faiss.write_index(index, str(output_dir / "visual_index.faiss"))
    np.save(output_dir / "embeddings.npy", emb)
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for row in metadata_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    (output_dir / "config.json").write_text(json.dumps({
        "model": str(model_path.relative_to(project_root) if model_path.is_relative_to(project_root) else model_path),
        "metadata": str(metadata.relative_to(project_root) if metadata.is_relative_to(project_root) else metadata),
        "review_log": str(review_log.relative_to(project_root) if review_log.is_relative_to(project_root) else review_log),
        "num_vectors": int(index.ntotal),
        "dim": int(emb.shape[1]),
        "metric": "cosine_similarity_via_inner_product_on_l2_normalized_vectors",
    }, indent=2), encoding="utf-8")
    print(f"✅ Global VAE FAISS index written to: {output_dir}")

    if args.by_type:
        parent = output_dir.parent
        by_type_root = parent / "by_type"
        by_type_root.mkdir(parents=True, exist_ok=True)
        types = {}
        for i, row in enumerate(metadata_rows):
            cls = row.get("effective_type") or row.get("type") or "unknown"
            cls = str(cls).replace("/", "_")
            types.setdefault(cls, []).append(i)
        for cls, indices in types.items():
            cls_dir = by_type_root / cls
            cls_dir.mkdir(parents=True, exist_ok=True)
            cls_emb = emb[indices]
            cls_index = faiss.IndexFlatIP(cls_emb.shape[1])
            cls_index.add(cls_emb)
            faiss.write_index(cls_index, str(cls_dir / "visual_index.faiss"))
            with (cls_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
                for i in indices:
                    f.write(json.dumps(metadata_rows[i], ensure_ascii=False) + "\n")
            print(f"✅ Type index {cls}: {len(indices)} vectors")


if __name__ == "__main__":
    main()
